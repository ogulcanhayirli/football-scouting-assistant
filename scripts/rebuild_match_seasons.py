"""Rebuild per match player stat partitions from the raw cache, offline.

Usage:
    uv run python scripts/rebuild_match_seasons.py --config configs/backfill.ultra.yaml \
        segunda_division:2016 eerste_divisie:2020 [--from-s3]

For a repair added to the adapter after a season was written (rule 7): the
season is parsed again from the cached ``/fixtures`` list and
``/fixtures/players`` responses and written only if it passes the contract.
The client has no network: a response missing from the cache stops the run
instead of being requested. ``--from-s3`` first copies just the raw files
those seasons need from ``$SCOUT_S3_BUCKET`` (a fresh container has none).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from scout.backfill.matches import _backfill_match_season
from scout.config import Settings, load_settings
from scout.providers.api_football.client import SOURCE, ApiFootballClient
from scout.providers.api_football.matches import ApiFootballMatchProvider
from scout.ratelimit import QuotaGuard
from scout.storage import CacheKey, LocalRawStore

log = logging.getLogger("rebuild_match_seasons")


class OfflineError(RuntimeError):
    pass


def _refuse(request: httpx.Request) -> httpx.Response:
    raise OfflineError(f"not in the cache: {request.url}")


def offline_client(settings: Settings) -> ApiFootballClient:
    guard = QuotaGuard(daily_limit=1, per_minute_limit=1)
    return ApiFootballClient(
        api_key="offline",
        store=LocalRawStore(settings.paths.raw_dir),
        guard=guard,
        transport=httpx.MockTransport(_refuse),
        max_retries=0,
    )


def _competition(settings: Settings, key: str):
    for item in [*settings.leagues, *settings.competitions]:
        if item.key == key:
            return item
    raise KeyError(f"{key} is neither a league nor a competition in the config")


def fetch_from_s3(settings: Settings, league, season: int, bucket: str) -> None:
    import boto3

    from scout.providers.api_football.matches import finished_fixtures

    s3 = boto3.client("s3")
    raw = settings.paths.raw_dir

    def get(key: CacheKey) -> None:
        for suffix in (".json", ".meta.json"):
            local = raw / key.relative_path(suffix)
            if local.exists():
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, f"raw/{key.relative_path(suffix).as_posix()}", str(local))

    listing = CacheKey.of(SOURCE, "fixtures", {"league": league.api_football_id, "season": season})
    get(listing)
    payload = LocalRawStore(raw).get(listing)
    ids = [f["fixture"]["id"] for f in finished_fixtures(json.loads(payload))]
    keys = [CacheKey.of(SOURCE, "fixtures/players", {"fixture": i}) for i in ids]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, keys))
    log.info("%s %s: %d match responses in the cache", league.key, season, len(keys))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/backfill.ultra.yaml")
    ap.add_argument("seasons", nargs="+", help="league_key:season")
    ap.add_argument("--from-s3", action="store_true")
    args = ap.parse_args()
    settings = load_settings(Path(args.config))
    failed = 0
    with offline_client(settings) as client:
        provider = ApiFootballMatchProvider(client)
        for item in args.seasons:
            key, season = item.split(":")
            league = _competition(settings, key)
            if args.from_s3:
                fetch_from_s3(settings, league, int(season), os.environ["SCOUT_S3_BUCKET"])
            outcome = _backfill_match_season(
                provider, league, int(season), settings, probe_first=False
            )
            log.info(
                "%s %s: %s, %d rows %s", key, season, outcome.status, outcome.rows, outcome.detail
            )
            failed += outcome.status != "complete"
        assert client.requests_made == 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
