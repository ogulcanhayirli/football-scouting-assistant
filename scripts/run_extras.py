"""Download sidelined periods, coaches and match events, resumably (owner, 17 October).

Usage:
    uv run python scripts/run_extras.py --config configs/backfill.expansion.yaml \
        --deadline-minutes 100

Ids come from the processed tables of every league (old and new): players
from season totals and per match stats, teams from the same, fixtures from
per match player stats. Order: sidelined, coaches, events.

Batched endpoints take 20 ids per request. Ids already inside a cached batch
are skipped and only the rest are chunked, so a changed id list or a new
container never repeats a request. The run stops at the config's daily cap,
at ``--deadline-minutes``, or when done; the same command resumes. New raw
files go to S3 after each part and when the run stops (``upload_new`` of
scripts/run_expansion.py). ``--restore`` first copies this task's raw files
and the processed id tables back from S3. Tables: ``sidelined/sidelined.parquet``,
``coaches/coach_spells.parquet`` and ``fixture_events/league=<key>/season=<year>.parquet``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from run_expansion import _download, _s3, upload_new
from scout.cli import build_client
from scout.config import Settings, load_settings
from scout.contracts import COACH_SPELLS, FIXTURE_EVENTS, SIDELINED, validate
from scout.fsutil import atomic_write_text
from scout.providers.api_football import ApiFootballError, QuotaExceededError
from scout.providers.api_football.extras import (
    ApiFootballExtrasProvider,
    chunks,
    coach_rows_to_frame,
    event_rows_to_frame,
    parse_coaches,
    parse_fixture_events,
    parse_sidelined,
    sidelined_rows_to_frame,
)
from scout.ratelimit import DailyQuotaExhaustedError
from scout.sync import list_remote

log = logging.getLogger("run_extras")
PARTS = ["sidelined", "coaches", "events"]
SOURCE_TABLES = ["player_season_stats", "player_match_stats"]


class DeadlineReachedError(Exception):
    pass


def _read(settings: Settings, table: str, columns: list[str]) -> pd.DataFrame:
    paths = glob.glob(str(settings.paths.processed_dir / table / "league=*" / "season=*.parquet"))
    frames = [pd.read_parquet(p, columns=columns) for p in paths]
    return pd.concat(frames, ignore_index=True)


def load_ids(settings: Settings) -> tuple[list[int], list[int], pd.DataFrame]:
    season = _read(settings, "player_season_stats", ["source_player_id", "team_id"])
    match = _read(
        settings,
        "player_match_stats",
        ["source_player_id", "team_id", "fixture_id", "league_key", "season"],
    )
    players = pd.concat([season.source_player_id, match.source_player_id]).dropna().astype(int)
    teams = pd.concat([season.team_id, match.team_id]).dropna().astype(int)
    fixtures = match[["fixture_id", "league_key", "season"]].drop_duplicates("fixture_id")
    return sorted(set(players[players > 0])), sorted(set(teams[teams > 0])), fixtures


def _batch_files(settings: Settings, endpoint: str, param: str) -> list[Path]:
    """Cached responses of ``endpoint`` asked with ``param`` alone, read straight
    from the cache folder (the store's key listing would walk every per match file)."""
    base = settings.paths.raw_dir / "api_football" / endpoint
    return [p for p in sorted(base.glob(f"{param}=*.json")) if not p.name.endswith(".meta.json")]


def cached_batch_ids(settings: Settings, endpoint: str, param: str) -> set[int]:
    done: set[int] = set()
    for path in _batch_files(settings, endpoint, param):
        done.update(int(x) for x in path.stem.split("=", 1)[1].split("-"))
    return done


def fetch_batched(fetch, ids: list[int], cached: set[int], check_deadline, what: str) -> int:
    todo = chunks(i for i in ids if i not in cached)
    log.info(
        "%s: %d ids, %d already cached, %d requests to make",
        what,
        len(ids),
        len(set(ids) & cached),
        len(todo),
    )
    for n, chunk in enumerate(todo, 1):
        check_deadline()
        try:
            fetch(chunk)
        except (DailyQuotaExhaustedError, QuotaExceededError):
            raise
        except ApiFootballError as exc:
            log.error("%s chunk starting %s refused: %s", what, chunk[0], exc)
        if n % 200 == 0 or n == len(todo):
            log.info("%s: %d/%d requests", what, n, len(todo))
    return len(todo)


def _payloads(settings: Settings, endpoint: str, param: str):
    for path in _batch_files(settings, endpoint, param):
        yield json.loads(path.read_bytes())


def build_sidelined(settings, client, today) -> int:
    rows = []
    for payload in _payloads(settings, "sidelined", "players"):
        rows += parse_sidelined(payload)
    frame = validate(SIDELINED, sidelined_rows_to_frame(rows, today))
    path = settings.paths.processed_dir / "sidelined" / "sidelined.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return len(frame)


def build_coaches(settings, client, today) -> int:
    rows = []
    for payload in _payloads(settings, "coachs", "team"):
        rows += parse_coaches(payload)
    frame = validate(COACH_SPELLS, coach_rows_to_frame(rows, today))
    path = settings.paths.processed_dir / "coaches" / "coach_spells.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return len(frame)


def build_events(settings, client, fixtures: pd.DataFrame) -> int:
    where = fixtures.set_index("fixture_id")[["league_key", "season"]]
    rows = []
    for payload in _payloads(settings, "fixtures", "ids"):
        for fixture in payload.get("response") or []:
            rows += parse_fixture_events(fixture)
    frame = event_rows_to_frame(rows).drop_duplicates(["fixture_id", "event_index"])
    frame = frame[frame["fixture_id"].isin(where.index)]
    frame = frame.join(where, on="fixture_id", rsuffix="_stats")
    total = 0
    for (league_key, season), part in frame.groupby(["league_key", "season_stats"]):
        part = validate(FIXTURE_EVENTS, part.drop(columns="season_stats").reset_index(drop=True))
        path = (
            settings.paths.processed_dir
            / "fixture_events"
            / f"league={league_key}"
            / f"season={season}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        total += len(part)
    return total


def restore(settings: Settings, bucket: str) -> None:
    s3 = _s3()
    raw_keys = []
    for prefix in ["raw/api_football/sidelined/", "raw/api_football/coachs/"]:
        raw_keys += list(list_remote(s3, bucket, prefix))
    raw_keys += [
        k for k in list_remote(s3, bucket, "raw/api_football/fixtures/") if "/fixtures/ids=" in k
    ]
    _download(s3, bucket, settings.paths.raw_dir, raw_keys)
    processed = []
    for table in SOURCE_TABLES:
        processed += list(list_remote(s3, bucket, f"processed/{table}/"))
    for key in processed:
        path = settings.paths.processed_dir / key[len("processed/") :]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(path))
    log.info("restored %d raw files and %d processed tables", len(raw_keys), len(processed))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/backfill.expansion.yaml")
    ap.add_argument("--deadline-minutes", type=float, default=100)
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "botocore", "boto3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings(args.config)
    bucket = os.environ["SCOUT_S3_BUCKET"]
    if args.restore:
        restore(settings, bucket)
    players, teams, fixtures = load_ids(settings)
    log.info("ids: %d players, %d teams, %d fixtures", len(players), len(teams), len(fixtures))

    progress_path = settings.paths.state_dir / "extras_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    deadline = time.monotonic() + args.deadline_minutes * 60
    today = datetime.now(UTC).date()

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise DeadlineReachedError

    stop = None
    with build_client(settings) as client:
        start = client.requests_made
        extras = ApiFootballExtrasProvider(client)
        for part in PARTS:
            if progress.get(part, {}).get("complete"):
                continue
            log.info("== %s", part)
            try:
                if part == "sidelined":
                    cached = cached_batch_ids(settings, "sidelined", "players")
                    fetch_batched(extras.sidelined, players, cached, check_deadline, part)
                    rows = build_sidelined(settings, client, today)
                elif part == "coaches":
                    for n, team in enumerate(teams, 1):
                        check_deadline()
                        try:
                            extras.coaches(team)
                        except (DailyQuotaExhaustedError, QuotaExceededError):
                            raise
                        except ApiFootballError as exc:
                            log.error("coaches for team %s refused: %s", team, exc)
                        if n % 200 == 0:
                            log.info("coaches: %d/%d teams", n, len(teams))
                    rows = build_coaches(settings, client, today)
                else:
                    cached = cached_batch_ids(settings, "fixtures", "ids")
                    fetch_batched(
                        extras.fixtures,
                        fixtures.fixture_id.astype(int).tolist(),
                        cached,
                        check_deadline,
                        part,
                    )
                    rows = build_events(settings, client, fixtures)
                progress[part] = {"complete": True, "rows": rows}
                log.info("%s complete: %d rows", part, rows)
            except DeadlineReachedError:
                stop = "deadline"
            except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
                stop = f"quota: {exc}"
            finally:
                upload_new(settings, bucket)
                atomic_write_text(progress_path, json.dumps(progress, indent=1))
            if stop:
                break
        used, made = client.guard.used_today, client.requests_made - start
    done = [p for p in PARTS if progress.get(p, {}).get("complete")]
    print(f"requests this run: {made:,}; used today (UTC): {used:,}")
    print(f"complete: {', '.join(f'{p} ({progress[p]["rows"]:,} rows)' for p in done) or 'none'}")
    print(f"stopped: {stop or 'all done'}")
    return 0 if len(done) == len(PARTS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
