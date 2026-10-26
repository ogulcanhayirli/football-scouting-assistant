"""The chat's caps, kept in one small S3 object (prompt 20).

* **Per visitor**: 20 messages an hour by IP. The IP is never stored; the
  state keeps a salted hash of it, and only for the current hour.
* **Budget**: 1 USD a day and 5 USD a month (UTC), computed from the usage
  fields of every reply (``scout.chat.agent.cost_usd``).
* **Length**: 500 characters a message (checked by the handler).

The evaluation (``scout chat-eval``) sends a token derived from the Claude
key (``eval_token``), which only a holder of the key can make: its
questions skip the per visitor limit and are counted against their own cap
of 5 USD a day, so a run never eats the visitors' budget.

Two Lambda instances may run at once (reserved concurrency 2), so every
change is a read, modify and conditional write (``If-Match`` on the ETag),
retried when the other instance wrote first. The budget is checked before
each model call, not reserved ahead, so two turns running at the same time
can pass a cap by at most what those two turns cost; the spend limit of the
Claude Console workspace is the hard backstop (docs/DEPLOY.md).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

STATE_KEY = "state/chat.json"
TRIES = 6


@dataclass(frozen=True)
class Caps:
    messages_per_hour: int = 20
    day_usd: float = 1.0
    month_usd: float = 5.0
    eval_day_usd: float = 5.0
    max_chars: int = 500


def eval_token(api_key: str) -> str:
    """The evaluation's token: an HMAC of a fixed text under the Claude key."""
    return hmac.new(api_key.encode(), b"scout-chat-eval", hashlib.sha256).hexdigest()


def is_eval(header: str | None, api_key: str) -> bool:
    return bool(header) and bool(api_key) and hmac.compare_digest(header, eval_token(api_key))


class ConflictError(Exception):
    """Another instance wrote the state first."""


class MemoryStore:
    """The state in memory, for tests and local runs."""

    def __init__(self) -> None:
        self.body: dict | None = None
        self.version = 0

    def get(self) -> tuple[dict | None, Any]:
        return (json.loads(json.dumps(self.body)) if self.body else None), self.version

    def put(self, state: dict, tag: Any) -> None:
        if tag != self.version:
            raise ConflictError
        self.body = state
        self.version += 1


class S3Store:
    """The state object in the chat bucket, written only if unchanged since read."""

    def __init__(self, client: Any, bucket: str, key: str = STATE_KEY) -> None:
        self.client, self.bucket, self.key = client, bucket, key
        self.conditional = True

    def get(self) -> tuple[dict | None, Any]:
        try:
            got = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except self.client.exceptions.NoSuchKey:
            return None, None
        return json.loads(got["Body"].read()), got["ETag"]

    def put(self, state: dict, tag: Any) -> None:
        args = {
            "Bucket": self.bucket,
            "Key": self.key,
            "Body": json.dumps(state, separators=(",", ":")).encode(),
            "ContentType": "application/json",
        }
        if self.conditional:
            if tag is None:
                args["IfNoneMatch"] = "*"
            else:
                args["IfMatch"] = tag
        try:
            self.client.put_object(**args)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise ConflictError from exc
            if type(exc).__name__ == "ParamValidationError" and self.conditional:
                # A boto3 too old for conditional writes: last writer wins.
                log.warning("chat state: conditional writes unsupported, writing plainly")
                self.conditional = False
                return self.put(state, tag)
            raise


def _periods(now: datetime) -> dict[str, str]:
    return {
        "day": now.strftime("%Y-%m-%d"),
        "month": now.strftime("%Y-%m"),
        "hour": now.strftime("%Y-%m-%dT%H"),
    }


def _fresh(state: dict | None, now: datetime) -> dict:
    """The state rolled over to the current hour, day and month."""
    p = _periods(now)
    s = dict(state or {})
    s.setdefault("salt", secrets.token_hex(16))
    if s.get("hour") != p["hour"]:
        s["hour"], s["visitors"] = p["hour"], {}
    if s.get("day") != p["day"]:
        s["day"], s["day_usd"], s["eval_day_usd"] = p["day"], 0.0, 0.0
    if s.get("month") != p["month"]:
        s["month"], s["month_usd"], s["eval_month_usd"] = p["month"], 0.0, 0.0
    s.setdefault("totals", {})
    return s


@dataclass
class Admission:
    ok: bool
    reason: str = ""
    day_usd: float = 0.0
    month_usd: float = 0.0
    evaluation: bool = False


CAPS = Caps()


class Guard:
    def __init__(self, store, caps: Caps = CAPS, clock=None, sleep=time.sleep) -> None:
        self.store, self.caps = store, caps
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep

    def _update(self, change) -> dict:
        """Read, apply ``change`` (which may raise to refuse) and write back."""
        for attempt in range(TRIES):
            state, tag = self.store.get()
            state = _fresh(state, self.clock())
            result = change(state)
            try:
                self.store.put(state, tag)
                return result
            except ConflictError:
                self.sleep(0.05 * (attempt + 1))
        raise RuntimeError("chat state: too many conflicting writes")

    def visitor(self, state: dict, ip: str) -> str:
        return hashlib.sha256(f"{state['salt']}|{ip}".encode()).hexdigest()[:16]

    def admit(self, ip: str, evaluation: bool = False) -> Admission:
        """May this message run? Counts it against the visitor when it may."""
        caps = self.caps

        def change(state: dict) -> Admission:
            totals = state["totals"]
            if evaluation:
                if state.get("eval_day_usd", 0.0) >= caps.eval_day_usd:
                    return Admission(False, "budget", evaluation=True)
                totals["eval_turns"] = totals.get("eval_turns", 0) + 1
                return Admission(
                    True, day_usd=state["eval_day_usd"], month_usd=0.0, evaluation=True
                )
            if (
                state.get("day_usd", 0.0) >= caps.day_usd
                or state.get("month_usd", 0.0) >= caps.month_usd
            ):
                totals["refused_budget"] = totals.get("refused_budget", 0) + 1
                return Admission(False, "budget")
            who = self.visitor(state, ip)
            if state["visitors"].get(who, 0) >= caps.messages_per_hour:
                totals["refused_rate"] = totals.get("refused_rate", 0) + 1
                return Admission(False, "rate")
            state["visitors"][who] = state["visitors"].get(who, 0) + 1
            totals["turns"] = totals.get("turns", 0) + 1
            return Admission(True, day_usd=state["day_usd"], month_usd=state["month_usd"])

        return self._update(change)

    def room(self, admission: Admission, spent_this_turn: float) -> bool:
        """Is there budget left for one more model call in this turn?"""
        caps = self.caps
        if admission.evaluation:
            return admission.day_usd + spent_this_turn < caps.eval_day_usd
        return (
            admission.day_usd + spent_this_turn < caps.day_usd
            and admission.month_usd + spent_this_turn < caps.month_usd
        )

    def record(self, cost: float, usage: dict, evaluation: bool = False) -> dict:
        """Add a finished turn's cost and tokens; returns the totals after it."""

        def change(state: dict) -> dict:
            if evaluation:
                state["eval_day_usd"] = state.get("eval_day_usd", 0.0) + cost
                state["eval_month_usd"] = state.get("eval_month_usd", 0.0) + cost
            else:
                state["day_usd"] = state.get("day_usd", 0.0) + cost
                state["month_usd"] = state.get("month_usd", 0.0) + cost
            totals = state["totals"]
            prefix = "eval_" if evaluation else ""
            totals[prefix + "usd"] = round(totals.get(prefix + "usd", 0.0) + cost, 6)
            for k, v in usage.items():
                totals[prefix + k] = totals.get(prefix + k, 0) + v
            return {
                k: state.get(k) for k in ("day", "day_usd", "month", "month_usd", "eval_day_usd")
            }

        return self._update(change)
