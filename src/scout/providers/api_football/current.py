"""The present: current squads, a fresh transfer list and the season in
progress (prompt 16, 23 October 2026).

* ``/teams?league=&season=``: the clubs of a league season. Fixed once the
  season has started, so cached like any other request.
* ``/players/squads?team=``: the squad a club lists today (id, name, age,
  shirt number, one of Goalkeeper, Defender, Midfielder, Attacker).
* ``/transfers?team=``: as in ``scout.providers.api_football.transfers``,
  asked again for the moves since the first download.
* ``/fixtures?league=&season=`` of the season in progress, then
  ``/fixtures?ids=`` (up to 20) for the finished matches: the full match,
  of which the ``players`` block is the ``/fixtures/players`` answer.
* ``/injuries?league=&season=`` of the season in progress.

Squads, transfers, the fixture list and the injuries of a season in progress
change from day to day, so they are cached per snapshot day (the ``as_of``
key of ``ApiFootballClient.get``): the same day never asks twice, a later
day asks again. A finished match does not change and is cached once by id.
Everything here describes today and is for display only; no feature table
may carry it (``PRESENT_ONLY_COLUMNS`` in ``scout.contracts.schemas``).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scout.config import LeagueConfig
from scout.providers.api_football.client import SOURCE, ApiFootballClient
from scout.providers.api_football.extras import joined
from scout.providers.api_football.matches import FINISHED

log = logging.getLogger(__name__)

SQUAD_POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Attacker"]
MAX_SHIRT = 99
MIN_AGE, MAX_AGE = 14, 46


class ApiFootballCurrentProvider:
    source = SOURCE

    def __init__(self, client: ApiFootballClient, as_of: str) -> None:
        self.client = client
        self.as_of = as_of  # the snapshot day, YYYY-MM-DD

    def teams(self, league: LeagueConfig, season: int) -> dict[str, Any]:
        params = {"league": league.api_football_id, "season": season}
        return self.client.get("teams", params).payload

    def squad(self, team_id: int) -> dict[str, Any]:
        return self.client.get("players/squads", {"team": team_id}, snapshot=self.as_of).payload

    def team_transfers(self, team_id: int) -> dict[str, Any]:
        return self.client.get("transfers", {"team": team_id}, snapshot=self.as_of).payload

    def fixtures(self, league: LeagueConfig, season: int) -> dict[str, Any]:
        params = {"league": league.api_football_id, "season": season}
        return self.client.get("fixtures", params, snapshot=self.as_of).payload

    def fixtures_by_id(self, fixture_ids: list[int]) -> dict[str, Any]:
        return self.client.get("fixtures", {"ids": joined(fixture_ids)}).payload

    def injuries(self, league: LeagueConfig, season: int) -> dict[str, Any]:
        params = {"league": league.api_football_id, "season": season}
        return self.client.get("injuries", params, snapshot=self.as_of).payload


def team_ids(payload: dict[str, Any]) -> list[int]:
    ids = [((item.get("team") or {}).get("id")) for item in payload.get("response") or []]
    return sorted({int(i) for i in ids if i})


def finished_ids(payload: dict[str, Any]) -> list[int]:
    out = []
    for item in payload.get("response") or []:
        info = item.get("fixture") or {}
        if (info.get("status") or {}).get("short") in FINISHED and info.get("id"):
            out.append(int(info["id"]))
    return sorted(set(out))


# Squads

SQUAD_COLUMNS = [
    "source",
    "as_of",
    "team_id",
    "team_name",
    "source_player_id",
    "player_name",
    "age",
    "shirt_number",
    "squad_position",
]


def parse_squad(payload: dict[str, Any], as_of: str) -> list[dict]:
    rows = []
    for block in payload.get("response") or []:
        team = block.get("team") or {}
        for player in block.get("players") or []:
            rows.append(
                {
                    "source": SOURCE,
                    "as_of": as_of,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "source_player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "age": player.get("age"),
                    "shirt_number": player.get("number"),
                    "squad_position": player.get("position"),
                }
            )
    return rows


def squad_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Typed squad rows. Source errors are logged and repaired: a row without
    a player id is dropped; an age outside 14 to 46, a shirt number outside
    1 to 99 and an unknown position become missing; a player listed twice
    in one squad is kept once."""
    frame = pd.DataFrame(rows, columns=SQUAD_COLUMNS)
    for column in ["team_id", "source_player_id", "age", "shirt_number"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["as_of"] = pd.to_datetime(frame["as_of"]).astype("datetime64[ns]")
    no_id = frame["source_player_id"].isna() | (frame["source_player_id"] <= 0).fillna(True)
    if no_id.any():
        log.warning("squads: dropping %d rows without a player id", int(no_id.sum()))
    frame = frame[~no_id].copy()
    repairs = {
        "age": ~frame["age"].between(MIN_AGE, MAX_AGE).fillna(True),
        "shirt_number": ~frame["shirt_number"].between(1, MAX_SHIRT).fillna(True),
        "squad_position": frame["squad_position"].notna()
        & ~frame["squad_position"].isin(SQUAD_POSITIONS),
    }
    for column, bad in repairs.items():
        for idx in frame.index[bad]:
            log.warning(
                "squads: %s %r of player %s at team %s is impossible; stored as missing",
                column,
                frame.at[idx, column],
                frame.at[idx, "source_player_id"],
                frame.at[idx, "team_id"],
            )
        frame.loc[bad, column] = pd.NA
    dup = frame.duplicated(["team_id", "source_player_id"], keep="first")
    if dup.any():
        log.warning("squads: dropping %d players listed twice in one squad", int(dup.sum()))
    return frame[~dup].reset_index(drop=True)
