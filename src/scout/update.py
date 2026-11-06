"""``scout update``: the free tier update (from 7 November 2026, 100 requests a day).

One run, for the match days from yesterday back ``update.days`` days:

1. ``/fixtures?date=`` for each day: every match in the world that day, so a
   week costs seven requests whatever the number of leagues. A past day is
   final once its matches are played, so it is cached once.
2. The finished matches of our leagues and competitions not yet in a cached
   ``/fixtures?ids=`` batch, in ``update.priority`` order (league strength),
   fetched 20 per request while the run's budget lasts. Each answer holds the
   player stats, lineups and events of the match. What the budget leaves
   is fetched by the next run, still first in line.
3. The season in progress tables under ``processed/current/``
   (``player_match_stats`` and ``match_lineups``, the partitions of the
   leagues that got new matches) get the new rows, are validated and are
   uploaded to S3 next to the raw responses.

The runner starts empty (GitHub Actions), so the raw cache lives in S3
(``S3BackedRawStore``): a response already in S3 is copied down, never asked
for again (rule 4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from scout.backfill.current import _cached_batches, current_dir
from scout.config import LeagueConfig, Settings
from scout.contracts import MATCH_LINEUPS, PLAYER_MATCH_STATS, validate
from scout.providers.api_football.client import SOURCE, ApiFootballClient, QuotaExceededError
from scout.providers.api_football.extras import joined
from scout.providers.api_football.matches import (
    FINISHED,
    lineup_rows_to_frame,
    match_rows_to_frame,
    parse_fixture_lineups,
    parse_fixture_players,
)
from scout.ratelimit import DailyQuotaExhaustedError
from scout.storage import CacheKey

log = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    days: list[str]
    in_window: int = 0
    already_cached: int = 0
    fetched: int = 0
    queued: int = 0  # finished matches in the window left for the next run
    requests_made: int = 0
    stopped_reason: str | None = None
    partitions: dict[str, int] = field(default_factory=dict)
    per_league: dict[str, dict[str, int]] = field(default_factory=dict)


def ordered_competitions(settings: Settings) -> list[LeagueConfig]:
    """Leagues and competitions in fetch order: ``update.priority`` first, the
    rest in config order (leagues, then competitions)."""
    every = [*settings.leagues, *settings.competitions]
    rank = {key: i for i, key in enumerate(settings.update.priority)}
    order = {c.key: i for i, c in enumerate(every)}
    return sorted(every, key=lambda c: (rank.get(c.key, len(rank)), order[c.key]))


def window(today: date, days: int) -> list[date]:
    return [today - timedelta(days=n) for n in range(1, days + 1)]


def s3_cached_batches(s3, bucket: str) -> dict[int, CacheKey]:
    """Fixture id -> the batch key, for every ``/fixtures?ids=`` answer in S3."""
    found: dict[int, CacheKey] = {}
    prefix = f"raw/{SOURCE}/fixtures/ids="
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[1]
            if name.endswith(".meta.json") or not name.endswith(".json"):
                continue
            value = name[len("ids=") : -len(".json")]
            key = CacheKey(SOURCE, "fixtures", (("ids", value),))
            for fid in value.split("-"):
                found[int(fid)] = key
    return found


def run_update(
    settings: Settings,
    client: ApiFootballClient,
    today: date,
    *,
    s3=None,
    bucket: str | None = None,
) -> UpdateResult:
    cfg = settings.update
    days = window(today, cfg.days)
    result = UpdateResult(days=[d.isoformat() for d in days])
    start = client.requests_made
    comps = ordered_competitions(settings)
    by_id = {c.api_football_id: c for c in comps}
    rank = {c.key: i for i, c in enumerate(comps)}

    def budget_left() -> int:
        return cfg.max_requests - (client.requests_made - start)

    finished: dict[int, dict[str, Any]] = {}
    fetched: dict[int, dict[str, Any]] = {}
    batches: dict[int, CacheKey] = {}
    have: list[int] = []
    try:
        for day in days:
            if budget_left() <= 0 and not client.is_cached("fixtures", {"date": day.isoformat()}):
                result.stopped_reason = "budget"
                break
            for fixture in (
                client.get("fixtures", {"date": day.isoformat()}).payload.get("response") or []
            ):
                info = fixture.get("fixture") or {}
                league_id = (fixture.get("league") or {}).get("id")
                if league_id in by_id and (info.get("status") or {}).get("short") in FINISHED:
                    finished[int(info["id"])] = fixture
        result.in_window = len(finished)

        batches.update(_cached_batches(client))
        if s3 is not None and bucket:
            batches.update(
                {k: v for k, v in s3_cached_batches(s3, bucket).items() if k not in batches}
            )
        have = [fid for fid in finished if fid in batches]
        result.already_cached = len(have)

        def priority(fid: int) -> tuple:
            fixture = finished[fid]
            comp = by_id[fixture["league"]["id"]]
            return (rank[comp.key], fixture["fixture"].get("timestamp") or 0, fid)

        queue = sorted((fid for fid in finished if fid not in batches), key=priority)
        position = 0
        while position < len(queue) and result.stopped_reason is None:
            if budget_left() <= 0:
                result.stopped_reason = "budget"
                break
            batch = sorted(queue[position : position + 20])
            payload = client.get("fixtures", {"ids": joined(batch)}).payload
            for fixture in payload.get("response") or []:
                fetched[int(fixture["fixture"]["id"])] = fixture
            position += 20
        result.fetched = len(fetched)
        result.queued = len(queue) - min(position, len(queue))
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. The next run goes on from the queue.", exc)

    # Matches already in a cached batch: read them back (copied down from S3 if needed).
    for key in {batches[fid] for fid in have}:
        if client.store.exists(key):
            for fixture in json.loads(client.store.get(key)).get("response") or []:
                fid = int(fixture["fixture"]["id"])
                if fid in finished and fid not in fetched:
                    fetched[fid] = fixture

    _write_partitions(settings, fetched, by_id, result)
    for fid, fixture in finished.items():
        comp = by_id[fixture["league"]["id"]]
        row = result.per_league.setdefault(comp.key, {"finished": 0, "held": 0})
        row["finished"] += 1
        row["held"] += fid in fetched
    result.requests_made = client.requests_made - start
    return result


def _write_partitions(
    settings: Settings,
    fixtures: dict[int, dict[str, Any]],
    by_id: dict[int, LeagueConfig],
    result: UpdateResult,
) -> None:
    """Add the matches to the season in progress tables, one partition per
    league season, keeping what the partition already holds."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for fixture in fixtures.values():
        comp = by_id[fixture["league"]["id"]]
        groups.setdefault((comp.key, int(fixture["league"]["season"])), []).append(fixture)
    out = current_dir(settings)
    for (key, season), group in sorted(groups.items()):
        comp = next(c for c in by_id.values() if c.key == key)
        players, lineups = [], []
        for fixture in group:
            players += parse_fixture_players(
                fixture, {"response": fixture.get("players") or []}, comp
            )
            lineups += parse_fixture_lineups(
                fixture, {"response": fixture.get("lineups") or []}, comp
            )
        for table, rows, to_frame, contract, unique in (
            (
                "player_match_stats",
                players,
                match_rows_to_frame,
                PLAYER_MATCH_STATS,
                ["fixture_id", "source_player_id"],
            ),
            (
                "match_lineups",
                lineups,
                lineup_rows_to_frame,
                MATCH_LINEUPS,
                ["fixture_id", "source_player_id"],
            ),
        ):
            if not rows:
                continue
            path = out / table / f"league={key}" / f"season={season}.parquet"
            frame = to_frame(rows)
            if path.exists():
                old = pd.read_parquet(path)
                old = old[~old["fixture_id"].isin(frame["fixture_id"].unique())]
                frame = pd.concat([old, frame], ignore_index=True)
            frame = validate(
                contract, frame.drop_duplicates(unique, keep="last").reset_index(drop=True)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            result.partitions[str(path.relative_to(settings.paths.processed_dir))] = len(frame)


def upload_partitions(settings: Settings, result: UpdateResult, s3, bucket: str) -> list[str]:
    """Copy the partitions this run wrote to ``processed/`` in the bucket."""
    keys = []
    for rel in result.partitions:
        key = f"processed/{Path(rel).as_posix()}"
        s3.upload_file(str(settings.paths.processed_dir / rel), bucket, key)
        keys.append(key)
    return keys
