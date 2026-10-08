"""API Football match level adapter: ``/fixtures`` and ``/fixtures/players`` to the
canonical player_match_stats table.

One ``/fixtures`` request lists a season's matches; one ``/fixtures/players``
request per finished match returns every player in both squads, bench
included. Like the season adapter, parsing is a pure function of cached JSON.

Field differences from the season level data: ``passes.accuracy`` is a count
of accurate passes here (it is a percentage in season stats), so it becomes
``passes_accurate``; ``games.number`` is the shirt number and positions are
the one letter codes G, D, M and F.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scout.config import LeagueConfig
from scout.providers.api_football.adapter import _drop_negative_counts
from scout.providers.api_football.client import SOURCE, ApiFootballClient

log = logging.getLogger(__name__)

# Statuses of a match that was played to the end. Postponed, cancelled,
# abandoned and awarded matches have no player stats.
FINISHED = {"FT", "AET", "PEN"}

MATCH_STAT_FIELDS: dict[str, tuple[str, str]] = {
    "position": ("games", "position"),
    "shirt_number": ("games", "number"),
    "minutes": ("games", "minutes"),
    "rating": ("games", "rating"),
    "captain": ("games", "captain"),
    "substitute": ("games", "substitute"),
    "shots_total": ("shots", "total"),
    "shots_on": ("shots", "on"),
    "goals": ("goals", "total"),
    "goals_conceded": ("goals", "conceded"),
    "assists": ("goals", "assists"),
    "saves": ("goals", "saves"),
    "passes_total": ("passes", "total"),
    "passes_key": ("passes", "key"),
    "passes_accurate": ("passes", "accuracy"),
    "tackles_total": ("tackles", "total"),
    "blocks": ("tackles", "blocks"),
    "interceptions": ("tackles", "interceptions"),
    "duels_total": ("duels", "total"),
    "duels_won": ("duels", "won"),
    "dribbles_attempts": ("dribbles", "attempts"),
    "dribbles_success": ("dribbles", "success"),
    "dribbled_past": ("dribbles", "past"),
    "fouls_drawn": ("fouls", "drawn"),
    "fouls_committed": ("fouls", "committed"),
    "cards_yellow": ("cards", "yellow"),
    "cards_red": ("cards", "red"),
    "penalty_won": ("penalty", "won"),
    "penalty_committed": ("penalty", "commited"),
    "penalty_scored": ("penalty", "scored"),
    "penalty_missed": ("penalty", "missed"),
    "penalty_saved": ("penalty", "saved"),
}
MATCH_NUMERIC_STATS = [
    c for c in MATCH_STAT_FIELDS if c not in {"position", "shirt_number", "captain", "substitute"}
] + ["offsides"]

MATCH_ID_COLUMNS = [
    "source",
    "fixture_id",
    "kickoff",
    "league_key",
    "league_id",
    "season",
    "round",
    "team_id",
    "team_name",
    "opponent_id",
    "opponent_name",
    "is_home",
    "team_goals",
    "opponent_goals",
    "source_player_id",
    "player_name",
]
MATCH_COLUMNS = [*MATCH_ID_COLUMNS, *MATCH_STAT_FIELDS, "offsides"]


def finished_fixtures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixtures from a ``/fixtures`` response that were played to the end."""
    return [
        f
        for f in payload.get("response") or []
        if ((f.get("fixture") or {}).get("status") or {}).get("short") in FINISHED
    ]


def parse_fixture_players(
    fixture: dict[str, Any], payload: dict[str, Any], league: LeagueConfig
) -> list[dict]:
    """Rows for one match: one per player listed for either team, bench included."""
    info = fixture.get("fixture") or {}
    teams = fixture.get("teams") or {}
    goals = fixture.get("goals") or {}
    home = (teams.get("home") or {}).get("id")
    sides = {
        (teams.get("home") or {}).get("id"): ("home", "away"),
        (teams.get("away") or {}).get("id"): ("away", "home"),
    }
    rows: list[dict] = []
    for team_block in payload.get("response") or []:
        team = team_block.get("team") or {}
        own, other = sides.get(team.get("id"), (None, None))
        opponent = (teams.get(other) or {}) if other else {}
        for item in team_block.get("players") or []:
            player = item.get("player") or {}
            stat = (item.get("statistics") or [{}])[0]
            row: dict[str, Any] = {
                "source": SOURCE,
                "fixture_id": info.get("id"),
                "kickoff": info.get("date"),
                "league_key": league.key,
                "league_id": league.api_football_id,
                "season": (fixture.get("league") or {}).get("season"),
                "round": (fixture.get("league") or {}).get("round"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "opponent_id": opponent.get("id"),
                "opponent_name": opponent.get("name"),
                "is_home": team.get("id") == home if own else None,
                "team_goals": goals.get(own) if own else None,
                "opponent_goals": goals.get(other) if other else None,
                "source_player_id": player.get("id"),
                "player_name": player.get("name"),
            }
            for column, (group, field) in MATCH_STAT_FIELDS.items():
                row[column] = (stat.get(group) or {}).get(field)
            row["offsides"] = stat.get("offsides")
            rows.append(row)
    return rows


def match_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    for column in MATCH_NUMERIC_STATS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in [
        "fixture_id",
        "league_id",
        "season",
        "team_id",
        "opponent_id",
        "team_goals",
        "opponent_goals",
        "source_player_id",
        "shirt_number",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["captain", "substitute", "is_home"]:
        frame[column] = frame[column].astype("boolean")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce").dt.as_unit("ns")
    _drop_negative_counts(frame, MATCH_NUMERIC_STATS)
    return frame


class ApiFootballMatchProvider:
    """Fetches a season's fixture list and each finished match's player stats."""

    source = SOURCE

    def __init__(self, client: ApiFootballClient) -> None:
        self.client = client

    def fixtures(self, league: LeagueConfig, season: int) -> list[dict[str, Any]]:
        params = {"league": league.api_football_id, "season": season}
        return finished_fixtures(self.client.get("fixtures", params).payload)

    def fixture_players(self, fixture_id: int) -> dict[str, Any]:
        return self.client.get("fixtures/players", {"fixture": fixture_id}).payload
