"""HTTP client for API Football v3 with caching and quota control.

API Football reports most failures as HTTP 200 with a non empty ``errors``
object (plan restrictions, exhausted quota, bad token). Those responses are
classified and raised, and never cached, so a refused request can't be saved
as if it were real data.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from scout.ratelimit import QuotaGuard
from scout.storage import CacheKey, RawMeta, RawStore

log = logging.getLogger(__name__)

SOURCE = "api_football"
_KEPT_HEADERS = {
    "x-ratelimit-requests-limit",
    "x-ratelimit-requests-remaining",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "date",
}


class ApiFootballError(RuntimeError):
    def __init__(self, message: str, errors: Any = None) -> None:
        super().__init__(message)
        self.errors = errors


class PlanRestrictionError(ApiFootballError):
    """The current plan does not cover this request (for example an old season)."""


class QuotaExceededError(ApiFootballError):
    """The server says the daily or per minute quota is used up."""


class AuthError(ApiFootballError):
    """Missing or invalid API key."""


@dataclass(frozen=True)
class FetchResult:
    payload: dict[str, Any]
    from_cache: bool
    key: CacheKey


def _int_header(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _is_minute_rate_limited(response: httpx.Response) -> bool:
    try:
        errors = response.json().get("errors")
    except ValueError:
        return False
    return isinstance(errors, dict) and any(str(k).lower() == "ratelimit" for k in errors)


def classify_errors(errors: Any) -> ApiFootballError | None:
    """Map the ``errors`` field of a response to an exception, or None if empty."""
    if not errors:
        return None
    items = errors.items() if isinstance(errors, dict) else [("error", e) for e in errors]
    text = "; ".join(f"{k}: {v}" for k, v in items)
    keys = {str(k).lower() for k, _ in items}
    lowered = text.lower()
    if keys & {"token", "access"} and "plan" not in lowered and "season" not in lowered:
        return AuthError(text, errors)
    if keys & {"requests", "ratelimit"} or "request limit" in lowered:
        return QuotaExceededError(text, errors)
    if "plan" in keys or "plan" in lowered or "season" in lowered:
        return PlanRestrictionError(text, errors)
    return ApiFootballError(text, errors)


class ApiFootballClient:
    def __init__(
        self,
        *,
        api_key: str,
        store: RawStore,
        guard: QuotaGuard,
        base_url: str = "https://v3.football.api-sports.io",
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.guard = guard
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            headers={"x-apisports-key": api_key},
            timeout=timeout,
            transport=transport,
        )
        self.requests_made = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ApiFootballClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_cached(self, endpoint: str, params: dict[str, Any]) -> bool:
        return self.store.exists(CacheKey.of(SOURCE, endpoint, params))

    def get(self, endpoint: str, params: dict[str, Any]) -> FetchResult:
        """Return the response for ``endpoint`` and ``params``, from cache when possible."""
        key = CacheKey.of(SOURCE, endpoint, params)
        if self.store.exists(key):
            return FetchResult(json.loads(self.store.get(key)), from_cache=True, key=key)

        response = self._request_with_retries(endpoint, dict(key.params))
        payload = response.json()
        error = classify_errors(payload.get("errors"))
        if error is not None:
            log.warning("api_football %s %s refused: %s", endpoint, dict(key.params), error)
            raise error

        meta = RawMeta(
            url=str(response.request.url),
            params=dict(key.params),
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() in _KEPT_HEADERS},
        )
        self.store.put(key, response.content, meta)
        return FetchResult(payload, from_cache=False, key=key)

    def _request_with_retries(self, endpoint: str, params: dict[str, str]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(min(2**attempt, 30))
            self.guard.acquire()
            self.requests_made += 1
            try:
                response = self._http.get(f"/{endpoint.strip('/')}", params=params)
            except httpx.TransportError as exc:
                last_exc = exc
                log.warning("transport error on %s (attempt %d): %s", endpoint, attempt + 1, exc)
                continue
            self.guard.observe_server_remaining(
                daily_remaining=_int_header(response.headers, "x-ratelimit-requests-remaining"),
                minute_remaining=_int_header(response.headers, "x-ratelimit-remaining"),
            )
            if response.status_code in (401, 403):
                raise AuthError(f"HTTP {response.status_code}: check API_FOOTBALL_KEY")
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = ApiFootballError(f"HTTP {response.status_code}")
                log.warning(
                    "HTTP %d on %s (attempt %d)", response.status_code, endpoint, attempt + 1
                )
                continue
            if _is_minute_rate_limited(response):
                # Per minute limit hit despite our window (clock skew, other clients):
                # mark the minute as full and retry instead of failing the run.
                last_exc = QuotaExceededError("per minute rate limit")
                self.guard.observe_server_remaining(daily_remaining=None, minute_remaining=0)
                log.warning("per minute rate limit on %s (attempt %d)", endpoint, attempt + 1)
                continue
            response.raise_for_status()
            return response
        raise ApiFootballError(
            f"{endpoint} failed after {self.max_retries + 1} attempts"
        ) from last_exc
