"""API Football adapter: raw ``/players`` pages to canonical player_season_stats.

Parsing is a pure function of the cached JSON, so tables can be rebuilt
offline and tested without HTTP.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from scout.config import LeagueConfig
from scout.providers.api_football.client import SOURCE, ApiFootballClient
from scout.providers.base import SeasonInfo

log = logging.getLogger(__name__)

# Canonical column -> path in one ``statistics`` entry. The source spells
# "appearences" and "commited" this way; the canonical names fix that.
STAT_FIELDS: dict[str, tuple[str, str]] = {
    "position": ("games", "position"),
    "appearances": ("games", "appearences"),
    "lineups": ("games", "lineups"),
    "minutes": ("games", "minutes"),
    "rating": ("games", "rating"),
    "captain": ("games", "captain"),
    "sub_in": ("substitutes", "in"),
    "sub_out": ("substitutes", "out"),
    "bench": ("substitutes", "bench"),
    "shots_total": ("shots", "total"),
    "shots_on": ("shots", "on"),
    "goals": ("goals", "total"),
    "goals_conceded": ("goals", "conceded"),
    "assists": ("goals", "assists"),
    "saves": ("goals", "saves"),
    "passes_total": ("passes", "total"),
    "passes_key": ("passes", "key"),
    "passes_accuracy": ("passes", "accuracy"),
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
    "cards_yellowred": ("cards", "yellowred"),
    "cards_red": ("cards", "red"),
    "penalty_won": ("penalty", "won"),
    "penalty_committed": ("penalty", "commited"),
    "penalty_scored": ("penalty", "scored"),
    "penalty_missed": ("penalty", "missed"),
    "penalty_saved": ("penalty", "saved"),
}

NUMERIC_STATS = [c for c in STAT_FIELDS if c not in {"position", "captain"}]

ID_COLUMNS = [
    "source",
    "source_player_id",
    "player_name",
    "firstname",
    "lastname",
    "birth_date",
    "nationality",
    "height_cm",
    "weight_kg",
    "team_id",
    "team_name",
    "league_key",
    "league_id",
    "season",
]
COLUMNS = ID_COLUMNS + list(STAT_FIELDS)

# The API labels most forwards "Attacker" but some entries "Forward" (seen in
# EPL 2023). Map them to one label so positions form four groups.
POSITION_ALIASES = {"Forward": "Attacker"}


def _parse_cm_kg(value: Any) -> float | None:
    parts = str(value).split() if value is not None else []
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_players_page(payload: dict[str, Any], league: LeagueConfig, season: int) -> list[dict]:
    """Rows for one ``/players`` page.

    A player who changed club mid season has one ``statistics`` entry per club;
    each becomes its own row. Entries for other competitions are dropped.
    """
    rows: list[dict] = []
    for item in payload.get("response") or []:
        player = item.get("player") or {}
        birth = player.get("birth") or {}
        for stat in item.get("statistics") or []:
            stat_league = stat.get("league") or {}
            if stat_league.get("id") != league.api_football_id:
                continue
            team = stat.get("team") or {}
            row: dict[str, Any] = {
                "source": SOURCE,
                "source_player_id": player.get("id"),
                "player_name": player.get("name"),
                "firstname": player.get("firstname"),
                "lastname": player.get("lastname"),
                "birth_date": _parse_date(birth.get("date")),
                "nationality": player.get("nationality"),
                "height_cm": _parse_cm_kg(player.get("height")),
                "weight_kg": _parse_cm_kg(player.get("weight")),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "league_key": league.key,
                "league_id": league.api_football_id,
                "season": stat_league.get("season", season),
            }
            for column, (group, field) in STAT_FIELDS.items():
                row[column] = (stat.get(group) or {}).get(field)
            row["position"] = POSITION_ALIASES.get(row["position"], row["position"])
            rows.append(row)
    return rows


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in [*NUMERIC_STATS, "height_cm", "weight_kg"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in ["source_player_id", "team_id", "league_id", "season"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["captain"] = frame["captain"].astype("boolean")
    frame["birth_date"] = pd.to_datetime(frame["birth_date"], errors="coerce")
    return frame


def parse_seasons(payload: dict[str, Any], league: LeagueConfig) -> list[SeasonInfo]:
    seasons: list[SeasonInfo] = []
    for item in payload.get("response") or []:
        for s in item.get("seasons") or []:
            coverage = s.get("coverage") or {}
            fixtures = coverage.get("fixtures") or {}
            seasons.append(
                SeasonInfo(
                    league_key=league.key,
                    season=int(s["year"]),
                    start=_parse_date(s.get("start")),
                    end=_parse_date(s.get("end")),
                    current=bool(s.get("current")),
                    has_player_stats=bool(coverage.get("players"))
                    or bool(fixtures.get("statistics_players")),
                )
            )
    return sorted(seasons, key=lambda x: x.season)


class ApiFootballStatsProvider:
    source = SOURCE

    def __init__(self, client: ApiFootballClient) -> None:
        self.client = client

    def list_seasons(self, league: LeagueConfig) -> list[SeasonInfo]:
        result = self.client.get("leagues", {"id": league.api_football_id})
        return parse_seasons(result.payload, league)

    def iter_player_pages(self, league: LeagueConfig, season: int):
        """Yield every ``/players`` page payload, fetching only what is not cached."""
        params = {"league": league.api_football_id, "season": season}
        first = self.client.get("players", params | {"page": 1})
        yield first.payload
        total = int((first.payload.get("paging") or {}).get("total") or 1)
        for page in range(2, total + 1):
            yield self.client.get("players", params | {"page": page}).payload

    def player_season_stats(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        rows: list[dict] = []
        for payload in self.iter_player_pages(league, season):
            rows.extend(parse_players_page(payload, league, season))
        return rows_to_frame(rows)
