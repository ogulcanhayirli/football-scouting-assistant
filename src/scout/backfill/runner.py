"""Quota aware, resumable backfill of API Football player season stats.

Resume and idempotency come from the raw cache: a page that is cached is never
requested again, so a stopped run continues where it left off. A season is
written to Parquet only once every page is cached and the contract passes.

Stopping conditions are explicit:

* plan restriction on a season: recorded, the run moves to the next season;
* daily budget spent (ours or the server's): the run stops cleanly and says
  when to resume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandera.errors

from scout.config import LeagueConfig, Settings
from scout.contracts import PLAYER_SEASON_STATS, validate
from scout.providers.api_football import (
    ApiFootballStatsProvider,
    PlanRestrictionError,
    QuotaExceededError,
    parse_players_page,
)
from scout.providers.api_football.adapter import rows_to_frame
from scout.ratelimit import DailyQuotaExhaustedError

log = logging.getLogger(__name__)


@dataclass
class SeasonOutcome:
    league_key: str
    season: int
    status: str  # complete | partial | refused_by_plan | contract_failed
    pages_total: int | None = None
    pages_cached: int = 0
    rows: int = 0
    detail: str = ""


@dataclass
class BackfillResult:
    outcomes: list[SeasonOutcome] = field(default_factory=list)
    requests_made: int = 0
    stopped_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.stopped_reason is None and all(
            o.status in {"complete", "refused_by_plan"} for o in self.outcomes
        )


def target_seasons(
    provider: ApiFootballStatsProvider, league: LeagueConfig, settings: Settings
) -> list[int]:
    """Seasons to download. ``all`` means every finished season with player coverage.

    The current season is left out of ``all`` on purpose: its data still
    changes, and the cache treats responses as final.
    """
    wanted = settings.backfill.seasons
    if wanted != "all":
        return list(wanted)
    seasons = provider.list_seasons(league)
    return [s.season for s in seasons if s.has_player_stats and not s.current]


def parquet_path(settings: Settings, league_key: str, season: int) -> Path:
    return (
        settings.paths.processed_dir
        / "player_season_stats"
        / f"league={league_key}"
        / f"season={season}.parquet"
    )


def _backfill_season(
    provider: ApiFootballStatsProvider,
    league: LeagueConfig,
    season: int,
    settings: Settings,
) -> SeasonOutcome:
    outcome = SeasonOutcome(league.key, season, status="partial")
    rows: list[dict] = []
    for payload in provider.iter_player_pages(league, season):
        paging = payload.get("paging") or {}
        outcome.pages_total = int(paging.get("total") or 1)
        outcome.pages_cached += 1
        rows.extend(parse_players_page(payload, league, season))
        if outcome.pages_cached % 10 == 0 or outcome.pages_cached == outcome.pages_total:
            log.info(
                "%s %s: page %d/%d, quota left today %d",
                league.key,
                season,
                outcome.pages_cached,
                outcome.pages_total,
                provider.client.guard.remaining_today,
            )

    frame = rows_to_frame(rows)
    outcome.rows = len(frame)
    try:
        validate(PLAYER_SEASON_STATS, frame)
    except pandera.errors.SchemaErrors as exc:
        outcome.status = "contract_failed"
        outcome.detail = f"{len(exc.failure_cases)} contract violations"
        log.error("%s %s failed the contract: %s", league.key, season, outcome.detail)
        return outcome

    path = parquet_path(settings, league.key, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    outcome.status = "complete"
    return outcome


def run_backfill(
    settings: Settings,
    provider: ApiFootballStatsProvider,
    *,
    league_keys: list[str] | None = None,
    seasons: list[int] | None = None,
) -> BackfillResult:
    result = BackfillResult()
    leagues = [lg for lg in settings.leagues if not league_keys or lg.key in league_keys]
    client = provider.client
    start_requests = client.requests_made
    try:
        for league in leagues:
            for season in seasons or target_seasons(provider, league, settings):
                log.info("%s %s: starting", league.key, season)
                try:
                    outcome = _backfill_season(provider, league, season, settings)
                except PlanRestrictionError as exc:
                    outcome = SeasonOutcome(league.key, season, "refused_by_plan", detail=str(exc))
                    log.warning("%s %s refused by plan: %s", league.key, season, exc)
                    result.outcomes.append(outcome)
                    if settings.backfill.stop_on_plan_error:
                        result.stopped_reason = f"plan: {exc}"
                        return result
                    continue
                except (DailyQuotaExhaustedError, QuotaExceededError):
                    result.outcomes.append(SeasonOutcome(league.key, season, "partial"))
                    raise
                result.outcomes.append(outcome)
                log.info("%s %s: %s, %d rows", league.key, season, outcome.status, outcome.rows)
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. Run the same command again to resume.", exc)
    finally:
        result.requests_made = client.requests_made - start_requests
        log.info(
            "Backfill finished: %d requests made, %d left today, stopped=%s",
            result.requests_made,
            client.guard.remaining_today,
            result.stopped_reason,
        )
    return result
