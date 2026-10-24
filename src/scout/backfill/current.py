"""``scout current``: the present day pull (prompt 16).

For one snapshot day: the clubs of every configured league's current season
and of the club cups (their current season, plus the clubs in the cup
matches we hold for the newest season), the league clubs of the newest
feature season (a relegated club still holds players on our lists), each
club's squad and transfer list, and the season in progress of every league:
its fixture list, the finished matches and its injuries.

Tables, under ``data/processed/current/`` (display only, never a model input):

* ``squads/as_of=<day>.parquet``: one row per club and squad player
* ``transfers/as_of=<day>.parquet``: every move the snapshot lists, the
  ``TRANSFERS`` table's columns
* ``current_club.parquet``: one row per player, his club today
* ``club_leagues.parquet``: team id, name, league key and name of the clubs
  in our leagues' current seasons
* ``player_match_stats/league=<key>/season=<year>.parquet`` and
  ``injuries/league=<key>/season=<year>.parquet``: the season in progress,
  the same columns and contracts as the finished seasons' tables

Like every backfill it resumes from the raw cache, and it stops cleanly at
``max_requests`` or when the quota guard says the day is spent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.errors

from scout.config import LeagueConfig, Settings
from scout.contracts import (
    CURRENT_CLUB,
    INJURIES,
    PLAYER_MATCH_STATS,
    SQUADS,
    TRANSFERS,
    validate,
)
from scout.providers.api_football.adapter import ApiFootballStatsProvider
from scout.providers.api_football.client import SOURCE, QuotaExceededError
from scout.providers.api_football.current import (
    ApiFootballCurrentProvider,
    finished_ids,
    parse_squad,
    squad_rows_to_frame,
    team_ids,
)
from scout.providers.api_football.extras import chunks
from scout.providers.api_football.matches import (
    injury_rows_to_frame,
    match_rows_to_frame,
    parse_fixture_players,
    parse_injuries,
)
from scout.providers.api_football.transfers import parse_transfers, transfer_rows_to_frame
from scout.ratelimit import DailyQuotaExhaustedError

log = logging.getLogger(__name__)

FORECAST_DATE = pd.Timestamp("2026-06-30")  # the registered forecast's feature date


class RequestCapReachedError(RuntimeError):
    """The run's own request cap (below the day's quota) is spent."""


@dataclass
class CurrentResult:
    as_of: str
    clubs: int = 0
    squads_done: int = 0
    transfers_done: int = 0
    matches: int = 0
    requests_made: int = 0
    stopped_reason: str | None = None
    tables: dict[str, int] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)  # league seasons that failed a contract

    @property
    def complete(self) -> bool:
        return self.stopped_reason is None and not self.failed


def current_dir(settings: Settings) -> Path:
    return settings.paths.processed_dir / "current"


def current_seasons(
    seasons_provider: ApiFootballStatsProvider, leagues: list[LeagueConfig]
) -> dict[str, int]:
    """League key -> the season the API flags as current."""
    out = {}
    for league in leagues:
        current = [s.season for s in seasons_provider.list_seasons(league) if s.current]
        if current:
            out[league.key] = max(current)
    return out


def held_clubs(settings: Settings, newest: int | None = None) -> set[int]:
    """Clubs of the newest season we hold: league season totals and club cup matches."""
    processed = settings.paths.processed_dir
    ids: set[int] = set()
    league_files = sorted((processed / "player_season_stats").glob("league=*/season=*.parquet"))
    keys = {lg.key for lg in settings.leagues}
    seasons = [int(f.stem.split("=")[1]) for f in league_files if f.parent.name[7:] in keys]
    newest = newest if newest is not None else max(seasons, default=0)
    for league in settings.leagues:
        path = (
            processed / "player_season_stats" / f"league={league.key}" / f"season={newest}.parquet"
        )
        if path.exists():
            ids.update(
                int(t) for t in pd.read_parquet(path, columns=["team_id"])["team_id"].dropna()
            )
    for comp in settings.competitions:
        if comp.kind != "club":
            continue
        path = processed / "player_match_stats" / f"league={comp.key}" / f"season={newest}.parquet"
        if path.exists():
            ids.update(
                int(t) for t in pd.read_parquet(path, columns=["team_id"])["team_id"].dropna()
            )
    return ids


def resolve_current_club(
    squads: pd.DataFrame, transfers: pd.DataFrame, club_leagues: pd.DataFrame
) -> pd.DataFrame:
    """One club per player today.

    A player in one squad is at that club. A player listed by several clubs
    (a loan listed by both, a squad not yet updated) is at the club his
    latest move went to when that club lists him, else at the club in one
    of our leagues, else at the lowest club id (logged). A player in no
    squad whose latest move after the forecast date went to a club is at
    that club, marked ``club_source = transfer``.
    """
    league_of = club_leagues.set_index("team_id")
    latest = (
        transfers.dropna(subset=["team_in_id"])
        .sort_values(["source_player_id", "transfer_date"])
        .drop_duplicates("source_player_id", keep="last")
        .set_index("source_player_id")
    )
    sq = squads.copy()
    sq["latest_in"] = sq["source_player_id"].map(latest["team_in_id"])
    sq["in_league"] = sq["team_id"].isin(league_of.index)
    sq["rank"] = (sq["team_id"] == sq["latest_in"]).fillna(False).astype(int) * 2 + sq[
        "in_league"
    ].astype(int)
    sq["listed_by"] = sq.groupby("source_player_id")["team_id"].transform("size")
    sq = sq.sort_values(["source_player_id", "rank", "team_id"], ascending=[True, False, True])
    ambiguous = sq[(sq["listed_by"] > 1)].drop_duplicates("source_player_id")
    if len(ambiguous):
        log.info("current club: %d players listed by more than one squad", len(ambiguous))
    best = sq.drop_duplicates("source_player_id", keep="first")
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "source_player_id": best["source_player_id"].astype("Int64"),
            "player_name": best["player_name"],
            "current_club_id": best["team_id"].astype("Int64"),
            "current_club": best["team_name"],
            "squad_position": best["squad_position"],
            "shirt_number": best["shirt_number"].astype("Int64"),
            "squads_listing": best["listed_by"].astype("Int64"),
            "club_source": "squad",
            "squad_date": best["as_of"],
        }
    )
    moved = latest[latest["transfer_date"] > FORECAST_DATE]
    missing = moved.index.difference(pd.Index(out["source_player_id"].astype(int)))
    if len(missing):
        extra = moved.loc[missing]
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {
                        "source": SOURCE,
                        "source_player_id": pd.array(extra.index, dtype="Int64"),
                        "player_name": extra["player_name"].to_numpy(),
                        "current_club_id": extra["team_in_id"].astype("Int64").to_numpy(),
                        "current_club": extra["team_in_name"].to_numpy(),
                        "squad_position": None,
                        "shirt_number": pd.array([pd.NA] * len(extra), dtype="Int64"),
                        "squads_listing": pd.array([0] * len(extra), dtype="Int64"),
                        "club_source": "transfer",
                        "squad_date": squads["as_of"].max(),
                    }
                ),
            ],
            ignore_index=True,
        )
    ids = out["current_club_id"]
    out["current_league_key"] = ids.map(league_of["league_key"]).astype(object)
    out["current_league"] = ids.map(league_of["league_name"]).astype(object)
    out["squad_date"] = pd.to_datetime(out["squad_date"]).astype("datetime64[ns]")
    return out.sort_values("source_player_id").reset_index(drop=True)


def run_current(
    settings: Settings,
    seasons_provider: ApiFootballStatsProvider,
    provider: ApiFootballCurrentProvider,
    *,
    max_requests: int | None = None,
    steps: tuple[str, ...] = ("clubs", "squads", "transfers", "matches", "injuries"),
) -> CurrentResult:
    client = provider.client
    result = CurrentResult(as_of=provider.as_of)
    start = client.requests_made
    out = current_dir(settings)

    def check_cap() -> None:
        if max_requests is not None and client.requests_made - start >= max_requests:
            raise RequestCapReachedError(f"{max_requests} requests made")

    seasons = current_seasons(seasons_provider, settings.leagues)
    cups = [c for c in settings.competitions if c.kind == "club"]
    cup_seasons = current_seasons(seasons_provider, cups)
    try:
        # Clubs
        club_rows = []
        clubs = held_clubs(settings)
        for league in settings.leagues:
            if league.key not in seasons:
                continue
            check_cap()
            payload = provider.teams(league, seasons[league.key])
            for item in payload.get("response") or []:
                team = item.get("team") or {}
                if team.get("id"):
                    club_rows.append(
                        {
                            "team_id": int(team["id"]),
                            "team_name": team.get("name"),
                            "league_key": league.key,
                            "league_name": league.name,
                            "season": seasons[league.key],
                        }
                    )
        for comp in cups:
            if comp.key in cup_seasons:
                check_cap()
                clubs.update(team_ids(provider.teams(comp, cup_seasons[comp.key])))
        club_leagues = pd.DataFrame(
            club_rows, columns=["team_id", "team_name", "league_key", "league_name", "season"]
        ).drop_duplicates("team_id")
        club_leagues["team_id"] = club_leagues["team_id"].astype("Int64")
        clubs.update(int(t) for t in club_leagues["team_id"])
        clubs_sorted = sorted(clubs)
        result.clubs = len(clubs_sorted)
        out.mkdir(parents=True, exist_ok=True)
        club_leagues.to_parquet(out / "club_leagues.parquet", index=False)
        log.info("current: %d clubs (%d in our leagues now)", len(clubs_sorted), len(club_leagues))

        # Squads and transfers
        squad_rows, transfer_rows = [], []
        for n, team_id in enumerate(clubs_sorted, start=1):
            if "squads" in steps:
                check_cap()
                squad_rows.extend(parse_squad(provider.squad(team_id), provider.as_of))
                result.squads_done = n
            if "transfers" in steps:
                check_cap()
                transfer_rows.extend(parse_transfers(provider.team_transfers(team_id)))
                result.transfers_done = n
            if n % 100 == 0:
                log.info(
                    "current: club %d/%d, quota left today %d",
                    n,
                    len(clubs_sorted),
                    client.guard.remaining_today,
                )
        day = f"as_of={provider.as_of}.parquet"
        squads = transfers = None
        if "squads" in steps:
            squads = validate(SQUADS, squad_rows_to_frame(squad_rows))
            (out / "squads").mkdir(exist_ok=True)
            squads.to_parquet(out / "squads" / day, index=False)
            result.tables["squads"] = len(squads)
        if "transfers" in steps:
            transfers = validate(TRANSFERS, transfer_rows_to_frame(transfer_rows))
            (out / "transfers").mkdir(exist_ok=True)
            transfers.to_parquet(out / "transfers" / day, index=False)
            result.tables["transfers"] = len(transfers)
        if squads is not None and transfers is not None:
            current = validate(CURRENT_CLUB, resolve_current_club(squads, transfers, club_leagues))
            current.to_parquet(out / "current_club.parquet", index=False)
            result.tables["current_club"] = len(current)

        # The season in progress
        for league in settings.leagues:
            season = seasons.get(league.key)
            if season is None:
                continue
            if "matches" in steps:
                check_cap()
                ids = finished_ids(provider.fixtures(league, season))
                fixtures = []
                for batch in uncached_chunks(client, ids):
                    check_cap()
                    fixtures.extend(provider.fixtures_by_id(batch).get("response") or [])
                fixtures.extend(
                    cached_fixtures(client, ids, have={f["fixture"]["id"] for f in fixtures})
                )
                rows = []
                for fixture in fixtures:
                    players = {"response": fixture.get("players") or []}
                    rows.extend(parse_fixture_players(fixture, players, league))
                result.matches += len(ids)
                frame = match_rows_to_frame(rows) if rows else None
                try:
                    if frame is not None:
                        validate(PLAYER_MATCH_STATS, frame)
                except pandera.errors.SchemaErrors as exc:
                    result.failed.append(league.key)
                    log.error(
                        "current %s %s failed the contract: %d violations\n%s",
                        league.key,
                        season,
                        len(exc.failure_cases),
                        exc.failure_cases.head(20).to_string(),
                    )
                    frame = None
                if frame is not None:
                    path = (
                        out
                        / "player_match_stats"
                        / f"league={league.key}"
                        / f"season={season}.parquet"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path, index=False)
                log.info(
                    "current %s %s: %d finished matches, %d rows",
                    league.key,
                    season,
                    len(ids),
                    len(rows),
                )
            if "injuries" in steps:
                check_cap()
                frame = injury_rows_to_frame(
                    parse_injuries(provider.injuries(league, season), league, season)
                )
                if len(frame):
                    validate(INJURIES, frame)
                    path = out / "injuries" / f"league={league.key}" / f"season={season}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path, index=False)
    except (DailyQuotaExhaustedError, QuotaExceededError, RequestCapReachedError) as exc:
        result.stopped_reason = f"quota: {exc}"
        log.warning("Stopping: %s. Run the same command again to resume.", exc)
    finally:
        result.requests_made = client.requests_made - start
    return result


# Finished matches are cached by batches of ids; a later day's new matches
# form new batches, so the batches already cached are found by their ids.


def _cached_batches(client) -> dict[int, tuple[tuple[str, str], ...]]:
    found = {}
    for key in client.store.keys(SOURCE, "fixtures"):
        params = dict(key.params)
        if set(params) != {"ids"}:
            continue
        for fid in params["ids"].split("-"):
            found[int(fid)] = key
    return found


def uncached_chunks(client, ids: list[int]) -> list[list[int]]:
    have = _cached_batches(client)
    return chunks([i for i in ids if i not in have])


def cached_fixtures(client, ids: list[int], have: set[int]) -> list[dict]:
    import json

    batches = _cached_batches(client)
    wanted = {i for i in ids if i not in have and i in batches}
    out = []
    for key in {batches[i] for i in wanted}:
        for fixture in json.loads(client.store.get(key)).get("response") or []:
            if fixture["fixture"]["id"] in wanted:
                out.append(fixture)
    return out
