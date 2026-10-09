"""Quota aware, resumable backfill of per match stats: player level
(``/fixtures/players``) or team level (``/fixtures/statistics``).

Per league and season: one ``/fixtures`` request lists the matches, then one
``/fixtures/players`` request per finished match. As in the season backfill,
the raw cache is the resume state, so a stopped run continues where it left
off, and a season is written to Parquet only when every match is cached and
the contract passes.

Only seasons the API flags with ``statistics_players`` coverage are fetched:
for older seasons the endpoint returns nothing, and asking would waste quota.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandera.errors

from scout.backfill.runner import BackfillResult, SeasonOutcome
from scout.config import LeagueConfig, Settings
from scout.contracts import (
    INJURIES,
    MATCH_LINEUPS,
    PLAYER_MATCH_STATS,
    TEAM_MATCH_STATS,
    validate,
)
from scout.fsutil import atomic_write_text
from scout.providers.api_football import (
    ApiFootballStatsProvider,
    PlanRestrictionError,
    QuotaExceededError,
)
from scout.providers.api_football.matches import (
    ApiFootballMatchProvider,
    injury_rows_to_frame,
    lineup_rows_to_frame,
    match_rows_to_frame,
    parse_fixture_lineups,
    parse_fixture_players,
    parse_fixture_statistics,
    parse_injuries,
    team_rows_to_frame,
)
from scout.ratelimit import DailyQuotaExhaustedError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Level:
    """One kind of per match download: endpoint, parser, contract and table."""

    name: str
    table: str
    fetch: Callable  # (provider, fixture_id) -> payload
    parse: Callable  # (fixture, payload, league) -> rows
    to_frame: Callable  # rows -> DataFrame
    contract: object
    covered: Callable  # SeasonInfo -> bool, the API's own coverage flag


PLAYER_LEVEL = Level(
    "player",
    "player_match_stats",
    lambda provider, fid: provider.fixture_players(fid),
    parse_fixture_players,
    match_rows_to_frame,
    PLAYER_MATCH_STATS,
    lambda info: info.has_match_player_stats,
)
TEAM_LEVEL = Level(
    "team",
    "team_match_stats",
    lambda provider, fid: provider.fixture_statistics(fid),
    parse_fixture_statistics,
    team_rows_to_frame,
    TEAM_MATCH_STATS,
    lambda info: info.has_match_team_stats,
)
LINEUP_LEVEL = Level(
    "lineup",
    "match_lineups",
    lambda provider, fid: provider.fixture_lineups(fid),
    parse_fixture_lineups,
    lineup_rows_to_frame,
    MATCH_LINEUPS,
    lambda info: info.has_lineups,
)
LEVELS = {level.name: level for level in (PLAYER_LEVEL, TEAM_LEVEL, LINEUP_LEVEL)}


def match_parquet_path(
    settings: Settings, league_key: str, season: int, table: str = "player_match_stats"
) -> Path:
    return (
        settings.paths.processed_dir / table / f"league={league_key}" / f"season={season}.parquet"
    )


def match_target_seasons(
    seasons_provider: ApiFootballStatsProvider,
    league: LeagueConfig,
    settings: Settings,
    level: Level = PLAYER_LEVEL,
) -> list[int]:
    """Finished seasons with per match player stats, or the configured list."""
    wanted = settings.backfill.seasons
    listed = seasons_provider.list_seasons(league)
    covered = {s.season for s in listed if level.covered(s) and not s.current}
    if wanted == "all":
        return sorted(covered)
    skipped = sorted(set(wanted) - covered)
    if skipped:
        log.info("%s: no per match player stats for seasons %s; skipped", league.key, skipped)
    return [s for s in wanted if s in covered]


PROBE_MATCHES = 5


def _season_has_player_data(
    provider: ApiFootballMatchProvider, fixtures: list[dict], level: Level = PLAYER_LEVEL
) -> bool:
    """Fetch a few matches spread over the season before committing to all of them.

    The coverage flag is not reliable: EPL 2014 is flagged as covered, yet 377
    of its 380 matches return no players. Probed responses are cached, so the
    full pass reuses them.
    """
    if not fixtures:
        return False
    step = max(len(fixtures) // PROBE_MATCHES, 1)
    sample = fixtures[::step][:PROBE_MATCHES]
    return any(level.fetch(provider, f["fixture"]["id"]).get("response") for f in sample)


def _backfill_match_season(
    provider: ApiFootballMatchProvider,
    league: LeagueConfig,
    season: int,
    settings: Settings,
    level: Level = PLAYER_LEVEL,
) -> SeasonOutcome:
    outcome = SeasonOutcome(league.key, season, status="partial")
    fixtures = provider.fixtures(league, season)
    outcome.pages_total = len(fixtures)
    if not _season_has_player_data(provider, fixtures, level):
        outcome.status = "no_player_data"
        outcome.detail = f"none of {PROBE_MATCHES} probed matches has player data; skipped"
        log.warning("%s %s: %s", league.key, season, outcome.detail)
        return outcome
    rows: list[dict] = []
    empty = 0
    for fixture in fixtures:
        payload = level.fetch(provider, fixture["fixture"]["id"])
        outcome.pages_cached += 1
        parsed = level.parse(fixture, payload, league)
        empty += not parsed
        rows.extend(parsed)
        if outcome.pages_cached % 50 == 0 or outcome.pages_cached == outcome.pages_total:
            log.info(
                "%s %s: match %d/%d, quota left today %d",
                league.key,
                season,
                outcome.pages_cached,
                outcome.pages_total,
                provider.client.guard.remaining_today,
            )

    frame = level.to_frame(rows)
    outcome.rows = len(frame)
    outcome.detail = f"{empty} matches without player data" if empty else ""
    try:
        validate(level.contract, frame)
    except pandera.errors.SchemaErrors as exc:
        outcome.status = "contract_failed"
        outcome.detail = f"{len(exc.failure_cases)} contract violations"
        log.error("%s %s failed the contract: %s", league.key, season, outcome.detail)
        log.error("%s", exc.failure_cases.head(20).to_string())
        return outcome

    path = match_parquet_path(settings, league.key, season, level.table)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    outcome.status = "complete"
    return outcome


def run_match_backfill(
    settings: Settings,
    seasons_provider: ApiFootballStatsProvider,
    provider: ApiFootballMatchProvider,
    *,
    league_keys: list[str] | None = None,
    seasons: list[int] | None = None,
    level: Level = PLAYER_LEVEL,
    scope: str = "leagues",
) -> BackfillResult:
    """``scope`` is ``leagues`` or ``competitions`` (cups and national teams)."""
    result = BackfillResult()
    pool = settings.competitions if scope == "competitions" else settings.leagues
    leagues = [lg for lg in pool if not league_keys or lg.key in league_keys]
    client = provider.client
    start_requests = client.requests_made
    try:
        for league in leagues:
            targets = match_target_seasons(seasons_provider, league, settings, level)
            if seasons:
                targets = [s for s in seasons if s in targets]
            for season in targets:
                log.info("%s %s: matches starting", league.key, season)
                try:
                    outcome = _backfill_match_season(provider, league, season, settings, level)
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
                log.info(
                    "%s %s: %s, %d matches, %d rows %s",
                    league.key,
                    season,
                    outcome.status,
                    outcome.pages_total or 0,
                    outcome.rows,
                    outcome.detail,
                )
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. Run the same command again to resume.", exc)
    finally:
        result.requests_made = client.requests_made - start_requests
        log.info(
            "Match backfill finished: %d requests made, %d left today, stopped=%s",
            result.requests_made,
            client.guard.remaining_today,
            result.stopped_reason,
        )
    return result


def write_match_report(result: BackfillResult, reports_dir: Path, level: str = "player") -> Path:
    """A short Markdown and JSON summary of the match backfill."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "league": o.league_key,
            "season": o.season,
            "status": o.status,
            "matches": f"{o.pages_cached}/{o.pages_total}",
            "rows": o.rows,
            "detail": o.detail,
        }
        for o in result.outcomes
    ]
    lines = [
        "# Match backfill report",
        "",
        f"Requests made this run: {result.requests_made}. "
        f"Stopped: {result.stopped_reason or 'no'}.",
        "",
        "| League | Season | Status | Matches | Rows | Note |",
        "|---|---|---|---|---|---|",
        *(
            f"| {r['league']} | {r['season']} | {r['status']} | {r['matches']} "
            f"| {r['rows']} | {r['detail']} |"
            for r in records
        ),
        "",
    ]
    stem = "match_coverage_report" if level == "player" else f"match_{level}_coverage_report"
    md = reports_dir / f"{stem}.md"
    atomic_write_text(md, "\n".join(lines))
    atomic_write_text(
        reports_dir / f"{stem}.json",
        json.dumps(
            {
                "requests_made": result.requests_made,
                "stopped": result.stopped_reason,
                "seasons": records,
            },
            indent=2,
        ),
    )
    return md


def run_injury_backfill(
    settings: Settings,
    seasons_provider: ApiFootballStatsProvider,
    provider: ApiFootballMatchProvider,
    *,
    league_keys: list[str] | None = None,
    scope: str = "leagues",
) -> BackfillResult:
    """``/injuries`` for every covered, finished season: one request each (the
    API returns a whole league season in one response)."""
    result = BackfillResult()
    pool = settings.competitions if scope == "competitions" else settings.leagues
    client = provider.client
    start_requests = client.requests_made
    try:
        for league in [lg for lg in pool if not league_keys or lg.key in league_keys]:
            listed = seasons_provider.list_seasons(league)
            for season in sorted(s.season for s in listed if s.has_injuries and not s.current):
                frame = injury_rows_to_frame(
                    parse_injuries(provider.injuries(league, season), league, season)
                )
                outcome = SeasonOutcome(
                    league.key, season, "complete", pages_total=1, pages_cached=1, rows=len(frame)
                )
                try:
                    validate(INJURIES, frame)
                except pandera.errors.SchemaErrors as exc:
                    outcome.status = "contract_failed"
                    outcome.detail = f"{len(exc.failure_cases)} contract violations"
                    log.error(
                        "%s %s injuries failed the contract: %s", league.key, season, outcome.detail
                    )
                else:
                    path = match_parquet_path(settings, league.key, season, "injuries")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path, index=False)
                result.outcomes.append(outcome)
                log.info(
                    "%s %s injuries: %s, %d rows", league.key, season, outcome.status, outcome.rows
                )
    except (DailyQuotaExhaustedError, QuotaExceededError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. Run the same command again to resume.", exc)
    finally:
        result.requests_made = client.requests_made - start_requests
    return result
