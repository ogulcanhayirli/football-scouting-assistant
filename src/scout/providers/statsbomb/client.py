"""StatsBomb open data download (prompt 18b).

The repository (github.com/statsbomb/open-data) serves JSON exported from
the StatsBomb API: ``competitions.json``, one match list per competition
season, and one lineup and one event file per match. Every file is cached
byte for byte under ``data/raw/statsbomb`` through the raw store, with its
``.meta.json`` sidecar, and is never fetched again (rule 4). The user
agreement (``LICENSE.pdf``) and the logo it asks every publication to carry
are kept next to them under ``files/``.

The data may be used for research and shared analysis, not commercially and
not passed on as data (User Agreement 1.2); any published analysis carries
the StatsBomb logo (1.4). So nothing from these files goes to the page or
the downloads except results computed from them, with the logo.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scout.config import StatsBombConfig
from scout.fsutil import atomic_write_bytes, atomic_write_text
from scout.storage import CacheKey, RawMeta, RawStore

log = logging.getLogger(__name__)

SOURCE = "statsbomb"
_KEPT_HEADERS = {"etag", "date", "content-length", "last-modified"}
# Repository paths of the user agreement and the logo (its README's Media Pack lockup).
FILES = {
    "LICENSE.pdf": "LICENSE.pdf",
    "statsbomb-logo.png": "img/SB - Icon Lockup - Colour positive.png",
}


def remote_path(endpoint: str, params: dict[str, Any]) -> str:
    """Path of a cached endpoint in the repository."""
    if endpoint == "competitions":
        return "data/competitions.json"
    if endpoint == "matches":
        return f"data/matches/{params['competition_id']}/{params['season_id']}.json"
    if endpoint in ("lineups", "events"):
        return f"data/{endpoint}/{params['match_id']}.json"
    raise ValueError(f"unknown StatsBomb endpoint {endpoint!r}")


class StatsBombClient:
    def __init__(
        self,
        *,
        store: RawStore,
        base_url: str = StatsBombConfig().base_url,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self.requests_made = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> StatsBombClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """The parsed JSON of one endpoint, from the cache when it is there."""
        params = params or {}
        key = CacheKey.of(SOURCE, endpoint, params)
        if not self.store.exists(key):
            path = remote_path(endpoint, params)
            response = self._request(path)
            json.loads(response.content)  # a body that is not JSON is never cached
            self.store.put(key, response.content, self._meta(response, params))
        return json.loads(self.store.get(key))

    def get_file(self, root: Path, name: str) -> Path:
        """A repository file (the user agreement, the logo) under ``root/files``,
        fetched once; the body is written last."""
        target = Path(root) / SOURCE / "files" / name
        if not target.is_file():
            response = self._request(FILES[name])
            meta = target.with_name(f"{name}.meta.json")
            atomic_write_text(meta, json.dumps(asdict(self._meta(response, {})), indent=2))
            atomic_write_bytes(target, response.content)
        return target

    def _meta(self, response: httpx.Response, params: dict[str, Any]) -> RawMeta:
        return RawMeta(
            url=str(response.request.url),
            params={str(k): str(v) for k, v in params.items()},
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() in _KEPT_HEADERS},
        )

    def _request(self, path: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(min(2**attempt, 30))
            self.requests_made += 1
            try:
                response = self._http.get(f"/{path}")
            except httpx.TransportError as exc:
                last = exc
                log.warning(
                    "statsbomb %s: transport error (attempt %d): %s", path, attempt + 1, exc
                )
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                log.warning(
                    "statsbomb %s: HTTP %d (attempt %d)", path, response.status_code, attempt + 1
                )
                continue
            response.raise_for_status()  # a refusal (404) is raised and never cached
            return response
        raise RuntimeError(
            f"statsbomb {path} failed after {self.max_retries + 1} attempts"
        ) from last


@dataclass
class SeasonFiles:
    league_key: str
    season: int
    competition_id: int
    season_id: int
    match_ids: list[int] = field(default_factory=list)


def season_ids(competitions: list[dict], config: StatsBombConfig) -> list[SeasonFiles]:
    """The competition and season ids of every configured season; a season
    the repository does not hold is logged and skipped."""
    found = {
        (c["competition_name"], c["season_name"]): (int(c["competition_id"]), int(c["season_id"]))
        for c in competitions
    }
    out = []
    for comp in config.competitions:
        for year in comp.seasons:
            ids = found.get((comp.competition, f"{year}/{year + 1}"))
            if ids is None:
                log.warning(
                    "statsbomb: no %s %d/%d in competitions.json", comp.competition, year, year + 1
                )
                continue
            out.append(SeasonFiles(comp.league_key, year, *ids))
    return out


def body_bytes(raw_dir: Path) -> tuple[int, int]:
    """Files and bytes of the cached bodies under ``raw/statsbomb`` (sidecars left out)."""
    files = [
        p
        for p in (Path(raw_dir) / SOURCE).rglob("*")
        if p.is_file() and not p.name.endswith(".meta.json") and not p.name.startswith(".")
    ]
    return len(files), sum(p.stat().st_size for p in files)


def download(
    store: RawStore,
    raw_dir: Path,
    config: StatsBombConfig,
    workers: int = 8,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Competitions, match lists, lineups and events of every configured
    season, plus the user agreement and the logo. Resumable: whatever is
    cached is not fetched again."""
    with StatsBombClient(store=store, base_url=config.base_url, transport=transport) as client:
        seasons = season_ids(client.get("competitions"), config)
        for s in seasons:
            matches = client.get(
                "matches", {"competition_id": s.competition_id, "season_id": s.season_id}
            )
            s.match_ids = sorted(int(m["match_id"]) for m in matches)
        jobs = [(e, m) for s in seasons for m in s.match_ids for e in ("lineups", "events")]
        done = 0

        def fetch(job: tuple[str, int]) -> None:
            nonlocal done
            client.get(job[0], {"match_id": job[1]})
            done += 1
            if done % 200 == 0:
                log.info("statsbomb: %d of %d files", done, len(jobs))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, jobs))
        for name in FILES:
            client.get_file(raw_dir, name)
        requests = client.requests_made
    files, size = body_bytes(raw_dir)
    summary = {
        "seasons": [(s.league_key, s.season, len(s.match_ids)) for s in seasons],
        "matches": sum(len(s.match_ids) for s in seasons),
        "requests": requests,
        "files": files,
        "bytes": size,
    }
    log.info(
        "statsbomb: %d matches, %d requests this run, %d files, %.1f MB cached",
        summary["matches"],
        requests,
        files,
        size / 1e6,
    )
    return summary
