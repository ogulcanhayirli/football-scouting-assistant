"""Download the decision 6 competitions in the owner's order, resumably.

Usage:
    uv run python scripts/run_expansion.py --config configs/backfill.expansion.yaml \
        --deadline-minutes 100

Per competition: season totals (``/players`` pages), player stats per match
(``/fixtures/players``) and lineups (``/fixtures/lineups``); ``/fixtures`` is
fetched once per season by the match levels. Player stats per match are
fetched for every season the API flags as covered, without the five match
probe (owner, 16 October). For the youth competitions the ``/players`` pages
are cached raw only, as they are context, not league rows.

The run stops cleanly at the daily cap of the config (the quota guard), at
``--deadline-minutes``, or when everything is done; the raw cache is the
resume state, so the same command continues. New raw files go to S3 after
each competition and when the run stops. On a fresh container (no local
probe responses), ``--restore`` first copies this expansion's raw files back
from S3, so nothing is downloaded twice.

Progress is in ``data/state/expansion_progress.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scout.backfill import run_backfill
from scout.backfill.matches import (
    LINEUP_LEVEL,
    PLAYER_LEVEL,
    _backfill_match_season,
    run_match_backfill,
)
from scout.cli import build_client
from scout.config import LeagueConfig, Settings, load_settings
from scout.fsutil import atomic_write_text
from scout.providers.api_football import ApiFootballStatsProvider, QuotaExceededError
from scout.providers.api_football.matches import ApiFootballMatchProvider, finished_fixtures
from scout.ratelimit import DailyQuotaExhaustedError

log = logging.getLogger("run_expansion")

# Owner's order of 16 October: youth first, then leagues with player stats per
# match from 2016 or earlier, then the rest.
ORDER = [
    "uefa_youth_league", "uefa_u21_championship", "fifa_u20_world_cup",
    "fifa_u17_world_cup", "olympics_men",
    "norwegian_eliteserien", "liga_mx", "saudi_pro_league", "segunda_division",
    "bundesliga_2", "ligue_2", "mls", "austrian_bundesliga",
    "swedish_allsvenskan", "j1_league", "swiss_super_league", "serie_b",
    "colombian_primera_a", "polish_ekstraklasa", "croatian_hnl", "eerste_divisie",
    "k_league_1", "serbian_superliga", "czech_first_league", "uruguayan_primera",
]  # fmt: skip

S3_RAW = "raw/"
MANIFEST = "expansion_uploaded.txt"


class DeadlineReachedError(Exception):
    pass


# S3


def _s3():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(max_pool_connections=24))


def _parallel(fn, items, workers: int = 16) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fn, items))  # list() surfaces the first error


def upload_new(settings: Settings, bucket: str) -> int:
    """Upload raw files not yet in the manifest: sidecar first, body last, as on disk.

    The manifest lists files known to be in S3. It is created by a full listing
    of the bucket, so files restored from or synced to S3 earlier are not resent.
    """
    raw = settings.paths.raw_dir
    manifest = settings.paths.state_dir / MANIFEST
    s3 = _s3()
    if manifest.exists():
        done = set(manifest.read_text(encoding="utf-8").split())
    else:
        from scout.sync import list_remote

        remote = list_remote(s3, bucket, S3_RAW)
        done = {k[len(S3_RAW) :] for k in remote}
    local = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file())
    todo = [rel for rel in local if rel not in done and not rel.endswith(".part")]
    metas = [r for r in todo if r.endswith(".meta.json")]
    bodies = [r for r in todo if not r.endswith(".meta.json")]

    def put(rel: str) -> None:
        with (raw / rel).open("rb") as body:
            s3.put_object(Bucket=bucket, Key=S3_RAW + rel, Body=body)

    _parallel(put, metas)
    _parallel(put, bodies)
    done.update(todo)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(manifest, "\n".join(sorted(done)) + "\n")
    log.info("S3: %d raw files uploaded, %d known in S3", len(todo), len(done))
    return len(todo)


def restore(settings: Settings, bucket: str) -> int:
    """Copy back from S3 the raw files of this expansion's competitions only."""
    from scout.sync import list_remote

    s3 = _s3()
    raw = settings.paths.raw_dir
    remote = list_remote(s3, bucket, S3_RAW + "api_football/")
    ids = {str(c.api_football_id) for c in [*settings.leagues, *settings.competitions]}
    by_league = re.compile(r"api_football/(leagues/id|fixtures/league|players/league)=(\d+)[/.]")
    wanted = [k for k in remote if (m := by_league.match(k[len(S3_RAW) :])) and m[2] in ids]
    _download(s3, bucket, raw, wanted)
    fixture_ids: set[str] = set()
    for path in (raw / "api_football" / "fixtures").glob("league=*/season=*.json"):
        if path.name.endswith(".meta.json") or path.parent.name.split("=")[1] not in ids:
            continue
        for f in json.loads(path.read_bytes()).get("response") or []:
            fixture_ids.add(str(f["fixture"]["id"]))
    per_match = re.compile(r"api_football/fixtures/(players|lineups)/fixture=(\d+)\.")
    extra = [k for k in remote if (m := per_match.match(k[len(S3_RAW) :])) and m[2] in fixture_ids]
    _download(s3, bucket, raw, extra)
    log.info("restored %d raw files from S3", len(wanted) + len(extra))
    return len(wanted) + len(extra)


def _download(s3, bucket: str, raw: Path, keys: list[str]) -> None:
    def get(key: str) -> None:
        path = raw / key[len(S3_RAW) :]
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        s3.download_file(bucket, key, str(tmp))
        tmp.replace(path)

    # Sidecars first, so a body never sits locally without its sidecar.
    _parallel(get, sorted(keys, key=lambda k: not k.endswith(".meta.json")))


# Download


def _cache_player_pages(provider: ApiFootballStatsProvider, comp: LeagueConfig) -> str | None:
    """Raw ``/players`` pages of a competition; returns a stop reason or None."""
    try:
        for s in provider.list_seasons(comp):
            if s.has_player_stats and not s.current:
                for _ in provider.iter_player_pages(comp, s.season):
                    pass
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        return f"quota: {exc}"
    return None


def _ended_tournament_seasons(client, stats, comp: LeagueConfig) -> list[int]:
    """Seasons the API still calls current although they are over: a tournament's
    latest edition stays current until the next one (U21 2025, Olympics 2024).
    Over means the end date has passed and every listed fixture is finished, so
    the cached responses are final."""
    today = datetime.now(UTC).date()
    ended = []
    for s in stats.list_seasons(comp):
        if not s.current or s.end is None or s.end >= today:
            continue
        params = {"league": comp.api_football_id, "season": s.season}
        fixtures = client.get("fixtures", params).payload.get("response") or []
        if fixtures and len(finished_fixtures({"response": fixtures})) == len(fixtures):
            ended.append(s.season)
    return ended


def _fetch_ended_seasons(settings, client, stats, matches, comp) -> tuple[dict, str | None]:
    summary: dict[str, Any] = {}
    try:
        for season in _ended_tournament_seasons(client, stats, comp):
            for _ in stats.iter_player_pages(comp, season):
                pass
            player = _backfill_match_season(
                matches, comp, season, settings, PLAYER_LEVEL, probe_first=False
            )
            lineup = _backfill_match_season(matches, comp, season, settings, LINEUP_LEVEL)
            summary[str(season)] = {
                "player_match": {"status": player.status, "rows": player.rows},
                "lineup": {"status": lineup.status, "rows": lineup.rows},
            }
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        return summary, f"quota: {exc}"
    return summary, None


def run_competition(settings, client, key: str, check_deadline) -> tuple[bool, str | None, dict]:
    """Returns (complete, stop reason, per step summary)."""
    stats = ApiFootballStatsProvider(client)
    matches = ApiFootballMatchProvider(client)
    youth = key in {c.key for c in settings.competitions}
    scope = "competitions" if youth else "leagues"
    summary: dict[str, Any] = {}
    complete = True

    steps = [
        ("player_match", dict(level=PLAYER_LEVEL, probe_first=False)),
        ("lineup", dict(level=LINEUP_LEVEL)),
    ]
    check_deadline()
    if youth:
        comp = next(c for c in settings.competitions if c.key == key)
        stop = _cache_player_pages(stats, comp)
        summary["season_pages"] = "cached" if stop is None else stop
    else:
        result = run_backfill(settings, stats, league_keys=[key])
        stop = result.stopped_reason
        complete &= result.complete
        summary["season"] = _outcomes(result)
    if stop:
        return False, stop, summary
    for name, kwargs in steps:
        check_deadline()
        result = run_match_backfill(
            settings, stats, matches, league_keys=[key], scope=scope, **kwargs
        )
        summary[name] = _outcomes(result)
        complete &= result.complete
        if result.stopped_reason:
            return False, result.stopped_reason, summary
    if youth:
        check_deadline()
        summary["ended_current"], stop = _fetch_ended_seasons(
            settings, client, stats, matches, comp
        )
        if stop:
            return False, stop, summary
        complete &= all(
            v[level]["status"] in {"complete", "no_player_data"}
            for v in summary["ended_current"].values()
            for level in ("player_match", "lineup")
        )
    return complete, None, summary


def _outcomes(result) -> dict[str, Any]:
    return {
        "requests": result.requests_made,
        "seasons": {
            str(o.season): {"status": o.status, "rows": o.rows, "detail": o.detail}
            for o in result.outcomes
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/backfill.expansion.yaml")
    ap.add_argument("--deadline-minutes", type=float, default=100)
    ap.add_argument("--restore", action="store_true", help="Copy raw files back from S3 first")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    settings = load_settings(args.config)
    bucket = os.environ["SCOUT_S3_BUCKET"]
    keys = {lg.key for lg in [*settings.leagues, *settings.competitions]}
    if set(ORDER) != keys:
        raise SystemExit(f"ORDER and config differ: {sorted(set(ORDER) ^ keys)}")

    if args.restore:
        restore(settings, bucket)

    progress_path = settings.paths.state_dir / "expansion_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    deadline = time.monotonic() + args.deadline_minutes * 60

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise DeadlineReachedError

    stop: str | None = None
    with build_client(settings) as client:
        start = client.requests_made
        for key in ORDER:
            if progress.get(key, {}).get("complete"):
                continue
            log.info("== %s", key)
            try:
                complete, stop, summary = run_competition(settings, client, key, check_deadline)
            except DeadlineReachedError:
                stop = "deadline"
                break
            finally:
                if not args.no_upload:
                    upload_new(settings, bucket)
            progress[key] = {"complete": complete, **summary}
            atomic_write_text(progress_path, json.dumps(progress, indent=1))
            if stop:
                break
        used = client.guard.used_today
        made = client.requests_made - start
    done = [k for k in ORDER if progress.get(k, {}).get("complete")]
    left = [k for k in ORDER if k not in done]
    print(f"requests this run: {made:,}; used today (UTC): {used:,}")
    print(f"complete ({len(done)}/{len(ORDER)}): {', '.join(done) or 'none'}")
    print(f"stopped: {stop or 'all done'}; next: {', '.join(left[:3]) or 'nothing'}")
    return 0 if not left else 2


if __name__ == "__main__":
    raise SystemExit(main())
