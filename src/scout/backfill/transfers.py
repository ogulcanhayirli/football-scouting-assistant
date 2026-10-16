"""Transfer history backfill: ``/transfers`` once per club.

The clubs are every team in the downloaded league seasons
(``player_season_stats``), so run the season backfill first. Each club's
response lists the whole career of moves of every player it has had; the
union over all clubs is deduplicated and written as one table,
``transfers/transfers.parquet``, only once every club is in the cache and the
contract passes. Like every backfill it resumes from the raw cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.errors

from scout.config import Settings
from scout.contracts import TRANSFERS, validate
from scout.providers.api_football.client import QuotaExceededError
from scout.providers.api_football.transfers import (
    ApiFootballTransferProvider,
    parse_transfers,
    transfer_rows_to_frame,
)
from scout.ratelimit import DailyQuotaExhaustedError

log = logging.getLogger(__name__)


@dataclass
class TransferResult:
    clubs_total: int = 0
    clubs_done: int = 0
    rows: int = 0
    requests_made: int = 0
    status: str = "pending"  # complete | partial | contract_failed
    detail: str = ""
    stopped_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == "complete"


def transfers_path(settings: Settings) -> Path:
    return settings.paths.processed_dir / "transfers" / "transfers.parquet"


def club_ids(settings: Settings, league_keys: list[str] | None = None) -> list[int]:
    """Every team id in the downloaded season stats of the configured leagues."""
    root = settings.paths.processed_dir / "player_season_stats"
    ids: set[int] = set()
    for league in settings.leagues:
        if league_keys and league.key not in league_keys:
            continue
        for path in sorted((root / f"league={league.key}").glob("season=*.parquet")):
            teams = pd.read_parquet(path, columns=["team_id"])["team_id"].dropna()
            ids.update(int(t) for t in teams)
    return sorted(ids)


def run_transfer_backfill(
    settings: Settings,
    provider: ApiFootballTransferProvider,
    *,
    league_keys: list[str] | None = None,
) -> TransferResult:
    clubs = club_ids(settings, league_keys)
    result = TransferResult(clubs_total=len(clubs))
    client = provider.client
    start_requests = client.requests_made
    rows: list[dict] = []
    try:
        for n, team_id in enumerate(clubs, start=1):
            rows.extend(parse_transfers(provider.team_transfers(team_id)))
            result.clubs_done = n
            if n % 50 == 0:
                log.info("transfers: club %d/%d", n, len(clubs))
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. Run the same command again to resume.", exc)
    finally:
        result.requests_made = client.requests_made - start_requests
    if result.clubs_done < result.clubs_total:
        result.status = "partial"
        return result
    frame = transfer_rows_to_frame(rows)
    result.rows = len(frame)
    try:
        validate(TRANSFERS, frame)
    except pandera.errors.SchemaErrors as exc:
        result.status = "contract_failed"
        result.detail = f"{len(exc.failure_cases)} contract violations"
        log.error("transfers failed the contract: %s\n%s", result.detail, exc.failure_cases)
        return result
    path = transfers_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    result.status = "complete"
    log.info(
        "transfers: %d clubs, %d transfers, %d players, written to %s",
        len(clubs),
        len(frame),
        frame["source_player_id"].nunique(),
        path,
    )
    return result
