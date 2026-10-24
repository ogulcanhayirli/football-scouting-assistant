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
MAX_MATCH_MINUTES = 150  # extra time plus long stoppages (World Cup 2022 ran past 130)
# A loan and a recall make two club switches in a season; three or more, back
# and forth between the same clubs, is two players sharing one id.
MAX_CLUB_SWITCHES = 2
# Success counts and their attempts: successes above attempts, or successes
# with no attempts recorded, can't both be right.
SUCCESS_PAIRS = {
    "shots_on": "shots_total",
    "passes_accurate": "passes_total",
    "duels_won": "duels_total",
    "dribbles_success": "dribbles_attempts",
}

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

# Matches whose player stats are wrong throughout, not just in the values the
# contract catches. Their rows keep the players and the result; every stat is missing.
CORRUPT_FIXTURES = {
    # Ekstraklasa 2023, Korona Kielce against Rakow: 154 minutes, more accurate
    # passes than passes, and implausible values in the rest of the match.
    1035823: "player stats wrong throughout",
}

# A team's detailed stats for a match sent as zeros throughout: every
# player who played has 0 passes and 0 duels (Ukrainian Premier League
# 2026/27, 98% of team matches; some K League 1 2026). Only goals, assists,
# cards, penalties and minutes are real there.
PLACEHOLDER_MIN_PLAYERS = 10
PLACEHOLDER_STATS = [
    "shots_total",
    "shots_on",
    "passes_total",
    "passes_key",
    "passes_accurate",
    "tackles_total",
    "blocks",
    "interceptions",
    "duels_total",
    "duels_won",
    "dribbles_attempts",
    "dribbles_success",
    "dribbled_past",
    "fouls_drawn",
    "fouls_committed",
    "saves",
    "goals_conceded",
    "offsides",
    "rating",
]


def placeholder_blocks(frame: pd.DataFrame) -> pd.Series:
    """True for rows of a team match whose detailed stats are placeholders."""
    played = frame["minutes"].fillna(0) > 0
    zero = (frame["passes_total"].fillna(0) == 0) & (frame["duels_total"].fillna(0) == 0)
    keys = [frame["fixture_id"], frame["team_id"]]
    players = played.groupby(keys).transform("sum")
    all_zero = (zero | ~played).groupby(keys).transform("all")
    some_value = (frame["passes_total"].notna() & played).groupby(keys).transform("any")
    return (players >= PLACEHOLDER_MIN_PLAYERS) & all_zero & some_value


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
    blocks = payload.get("response") or []
    claimed = {(b.get("team") or {}).get("id") for b in blocks} & set(sides)
    unclaimed = [tid for tid in sides if tid not in claimed]
    for team_block in blocks:
        team = team_block.get("team") or {}
        if team.get("id") not in sides and len(unclaimed) == 1 and len(blocks) == 2:
            # Source error: a block under an unknown team id (Chapecoense as 22722
            # with no name, Brasileirão 2026); it can only be the other side.
            side = "home" if unclaimed[0] == home else "away"
            log.warning(
                "fixture %s: players listed under unknown team %s; taken as %s",
                info.get("id"),
                team.get("id"),
                (teams.get(side) or {}).get("name"),
            )
            team = teams.get(side) or {}
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
    return _fix_source_errors(frame)


def _fix_source_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair errors in the source that would otherwise fail the contract.

    Each case was seen in real responses; each is logged, and the raw cache
    keeps the original:

    * a player without an id (sent as 0 or null) can't be linked to anyone,
      so the row is dropped;
    * two players with the same id in one match (brothers, usually, such as
      Halil İbrahim and Yasin Pehlivan): the row whose name the id carries most
      often in this frame is kept, the other dropped;
    * a player credited with more goals than the team scored: the goal count
      becomes missing, because either number could be the wrong one;
    * more successes than attempts (dribbles, Brasileirão 2022, two players;
      passes, Ekstraklasa 2023, fixture 1035823), or successes with the
      attempts left blank (duels and shots, Eerste Divisie and Allsvenskan
      2020): both counts become missing, because either could be the wrong
      one; filling the blank attempts with 0 would give rates above 1;
    * one id switching clubs back and forth more than ``MAX_CLUB_SWITCHES``
      times within the frame (a league season) belongs to two players
      (Segunda División 2016: a Lugo defender and an Oviedo keeper, 70
      matches): the rows of every club but the one with most rows are
      dropped, since the other player can't be linked to anyone;
    * a position other than G, D, M or F ("SUB", "-") becomes missing;
    * every stat of a match in ``CORRUPT_FIXTURES`` becomes missing; the rows,
      players and result stay;
    * minutes above ``MAX_MATCH_MINUTES`` become missing (Serie B 2020, Venezia
      against Empoli: 1,306 minutes for the starters, 90 plus a constant 1,216);
    * a team match whose detailed stats are zeros throughout (``placeholder_blocks``,
      Ukrainian Premier League 2026/27): those stats and the rating become
      missing; minutes, goals, assists, cards and penalties stay.
    """
    placeholder = placeholder_blocks(frame)
    if placeholder.any():
        blocks = frame.loc[placeholder, ["fixture_id", "team_id"]].drop_duplicates()
        log.warning(
            "%d team matches with placeholder zeros for the detailed stats (fixture %s, ...); "
            "stored as missing",
            len(blocks),
            blocks["fixture_id"].iloc[0],
        )
    frame.loc[placeholder, PLACEHOLDER_STATS] = pd.NA
    corrupt = frame["fixture_id"].isin(list(CORRUPT_FIXTURES))
    for fixture_id in frame.loc[corrupt, "fixture_id"].unique():
        log.warning(
            "Fixture %s: %s; all player stats stored as missing",
            fixture_id,
            CORRUPT_FIXTURES[int(fixture_id)],
        )
    frame.loc[corrupt, MATCH_NUMERIC_STATS] = pd.NA
    no_id = frame["source_player_id"].isna() | (frame["source_player_id"] <= 0)
    for _, row in frame[no_id].iterrows():
        log.warning(
            "Dropping %s (%s) in fixture %s: no player id",
            row["player_name"],
            row["team_name"],
            row["fixture_id"],
        )
    frame = frame[~no_id]

    dup = frame.duplicated(["fixture_id", "source_player_id"], keep=False)
    if dup.any():
        usual = frame.groupby("source_player_id")["player_name"].agg(
            lambda names: names.value_counts().index[0]
        )
        clash = frame[dup]
        keep = clash["player_name"] == clash["source_player_id"].map(usual)
        # Ties (same name twice) keep the row with the most minutes.
        kept = (
            clash[keep]
            .sort_values("minutes", ascending=False, na_position="last")
            .drop_duplicates(["fixture_id", "source_player_id"])
        )
        for _, row in clash.drop(kept.index).iterrows():
            log.warning(
                "Dropping %s in fixture %s: shares id %s with %s",
                row["player_name"],
                row["fixture_id"],
                row["source_player_id"],
                usual[row["source_player_id"]],
            )
        frame = frame[~dup | frame.index.isin(kept.index)]

    frame = frame.copy()
    too_many = frame["goals"].notna() & frame["team_goals"].notna()
    too_many &= frame["goals"] > frame["team_goals"]
    for idx in frame.index[too_many.fillna(False)]:
        log.warning(
            "Goals=%s for %s exceed the team's %s in fixture %s; stored as missing",
            frame.at[idx, "goals"],
            frame.at[idx, "player_name"],
            frame.at[idx, "team_goals"],
            frame.at[idx, "fixture_id"],
        )
    frame.loc[too_many.fillna(False), "goals"] = pd.NA
    for success, attempts in SUCCESS_PAIRS.items():
        made, tried = frame[success], frame[attempts]
        bad = ((made > tried) | ((made > 0) & tried.isna())).fillna(False)
        for idx in frame.index[bad]:
            log.warning(
                "%s=%s with %s=%s for %s in fixture %s; both missing",
                success,
                frame.at[idx, success],
                attempts,
                frame.at[idx, attempts],
                frame.at[idx, "player_name"],
                frame.at[idx, "fixture_id"],
            )
        frame.loc[bad, [success, attempts]] = pd.NA
    frame = _drop_shared_ids(frame)
    too_long = (frame["minutes"] > MAX_MATCH_MINUTES).fillna(False)
    for idx in frame.index[too_long]:
        log.warning(
            "Minutes=%s for %s in fixture %s exceed %s; stored as missing",
            frame.at[idx, "minutes"],
            frame.at[idx, "player_name"],
            frame.at[idx, "fixture_id"],
            MAX_MATCH_MINUTES,
        )
    frame.loc[too_long, "minutes"] = pd.NA
    unknown = frame["position"].notna() & ~frame["position"].isin(["G", "D", "M", "F"])
    frame.loc[unknown, "position"] = None  # "SUB" and "-" are placeholders, not positions
    return frame.reset_index(drop=True)


def _drop_shared_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of an id that switches clubs back and forth: see ``_fix_source_errors``."""
    if frame.empty:
        return frame
    ordered = frame.sort_values("kickoff")
    switches = ordered.groupby("source_player_id")["team_id"].agg(
        lambda teams: int((teams != teams.shift()).sum()) - 1
    )
    shared = switches[switches > MAX_CLUB_SWITCHES].index
    drop = pd.Series(False, index=frame.index)
    for player in shared:
        mine = frame["source_player_id"] == player
        counts = frame.loc[mine, "team_id"].value_counts()
        keep = counts.index[0]
        others = mine & (frame["team_id"] != keep)
        log.warning(
            "Id %s switches clubs %d times: %s keeps the %d rows for team %s, "
            "%d rows for other teams (%s) dropped",
            player,
            switches[player],
            frame.loc[mine & (frame["team_id"] == keep), "player_name"].mode().iloc[0],
            int(counts.iloc[0]),
            keep,
            int(others.sum()),
            ", ".join(sorted(set(frame.loc[others, "player_name"].dropna()))),
        )
        drop |= others
    return frame[~drop]


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

    def fixture_statistics(self, fixture_id: int) -> dict[str, Any]:
        return self.client.get("fixtures/statistics", {"fixture": fixture_id}).payload

    def fixture_lineups(self, fixture_id: int) -> dict[str, Any]:
        return self.client.get("fixtures/lineups", {"fixture": fixture_id}).payload

    def injuries(self, league: LeagueConfig, season: int) -> dict[str, Any]:
        params = {"league": league.api_football_id, "season": season}
        return self.client.get("injuries", params).payload


# Team level: /fixtures/statistics, one row per team per match.

# API label -> canonical column. xG ("expected_goals") and "goals_prevented"
# only exist from 2023/24.
TEAM_STAT_FIELDS = {
    "Shots on Goal": "shots_on",
    "Shots off Goal": "shots_off",
    "Total Shots": "shots_total",
    "Blocked Shots": "shots_blocked",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corners",
    "Offsides": "offsides",
    "Ball Possession": "possession",
    "Yellow Cards": "cards_yellow",
    "Red Cards": "cards_red",
    "Goalkeeper Saves": "saves",
    "Total passes": "passes_total",
    "Passes accurate": "passes_accurate",
    "Passes %": "pass_accuracy",
    "expected_goals": "xg",
    "goals_prevented": "goals_prevented",
}
TEAM_NUMERIC = list(TEAM_STAT_FIELDS.values())
TEAM_COLUMNS = [
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
    "is_home",
    "goals_for",
    "goals_against",
    *TEAM_NUMERIC,
]


def _stat_value(value: Any) -> Any:
    """Percentages come as "55%"; missing as null."""
    if isinstance(value, str) and value.endswith("%"):
        return value[:-1]
    return value


def parse_fixture_statistics(
    fixture: dict[str, Any], payload: dict[str, Any], league: LeagueConfig
) -> list[dict]:
    info = fixture.get("fixture") or {}
    teams = fixture.get("teams") or {}
    goals = fixture.get("goals") or {}
    side = {
        (teams.get("home") or {}).get("id"): "home",
        (teams.get("away") or {}).get("id"): "away",
    }
    rows = []
    for block in payload.get("response") or []:
        team = block.get("team") or {}
        own = side.get(team.get("id"))
        other = {"home": "away", "away": "home"}.get(own)
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
            "opponent_id": (teams.get(other) or {}).get("id") if other else None,
            "is_home": own == "home" if own else None,
            "goals_for": goals.get(own) if own else None,
            "goals_against": goals.get(other) if other else None,
        }
        values = {s.get("type"): s.get("value") for s in block.get("statistics") or []}
        for label, column in TEAM_STAT_FIELDS.items():
            row[column] = _stat_value(values.get(label))
        rows.append(row)
    return rows


def team_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=TEAM_COLUMNS)
    for column in TEAM_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in [
        "fixture_id",
        "league_id",
        "season",
        "team_id",
        "opponent_id",
        "goals_for",
        "goals_against",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["is_home"] = frame["is_home"].astype("boolean")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce").dt.as_unit("ns")
    _drop_negative_counts_team(frame)
    # Some 2021 Asian qualifiers send a total of 0 next to real on target,
    # off target and blocked counts. That total is wrong; it becomes missing.
    bad_total = (frame["shots_on"] > frame["shots_total"]).fillna(False)
    for idx in frame.index[bad_total]:
        log.warning(
            "Shots on target %s exceed the total %s for %s in fixture %s; total missing",
            frame.at[idx, "shots_on"],
            frame.at[idx, "shots_total"],
            frame.at[idx, "team_name"],
            frame.at[idx, "fixture_id"],
        )
    frame.loc[bad_total, "shots_total"] = pd.NA
    # In a few older matches the inside/outside split adds up to more than the
    # total (which always equals on target + off target + blocked). The total
    # is kept; the split becomes missing for that row.
    bad_split = (frame["shots_inside_box"] > frame["shots_total"]).fillna(False)
    for idx in frame.index[bad_split]:
        log.warning(
            "Inside the box shots %s exceed the total %s for %s in fixture %s; split missing",
            frame.at[idx, "shots_inside_box"],
            frame.at[idx, "shots_total"],
            frame.at[idx, "team_name"],
            frame.at[idx, "fixture_id"],
        )
    frame.loc[bad_split, ["shots_inside_box", "shots_outside_box"]] = pd.NA
    return frame


def _drop_negative_counts_team(frame: pd.DataFrame) -> None:
    columns = [c for c in TEAM_NUMERIC if c != "goals_prevented"]  # may be negative
    for column in columns:
        negative = (frame[column] < 0).fillna(False)
        for idx in frame.index[negative]:
            log.warning(
                "Negative team %s=%s for %s in fixture %s; stored as missing",
                column,
                frame.at[idx, column],
                frame.at[idx, "team_name"],
                frame.at[idx, "fixture_id"],
            )
        frame.loc[negative, column] = pd.NA


# Lineups: /fixtures/lineups, one row per player listed (starters and bench).
# Formation and the pitch grid ("row:column") only exist in later seasons.

LINEUP_COLUMNS = [
    "source",
    "fixture_id",
    "kickoff",
    "league_key",
    "league_id",
    "season",
    "team_id",
    "team_name",
    "formation",
    "coach_id",
    "coach_name",
    "source_player_id",
    "player_name",
    "shirt_number",
    "position",
    "grid_row",
    "grid_col",
    "starter",
]


def parse_fixture_lineups(
    fixture: dict[str, Any], payload: dict[str, Any], league: LeagueConfig
) -> list[dict]:
    info = fixture.get("fixture") or {}
    rows = []
    for block in payload.get("response") or []:
        team = block.get("team") or {}
        coach = block.get("coach") or {}
        for starter, key in ((True, "startXI"), (False, "substitutes")):
            for entry in block.get(key) or []:
                player = entry.get("player") or {}
                grid = str(player.get("grid") or "")
                row_col = grid.split(":") if ":" in grid else [None, None]
                rows.append(
                    {
                        "source": SOURCE,
                        "fixture_id": info.get("id"),
                        "kickoff": info.get("date"),
                        "league_key": league.key,
                        "league_id": league.api_football_id,
                        "season": (fixture.get("league") or {}).get("season"),
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "formation": block.get("formation"),
                        "coach_id": coach.get("id"),
                        "coach_name": coach.get("name"),
                        "source_player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "shirt_number": player.get("number"),
                        "position": player.get("pos"),
                        "grid_row": row_col[0],
                        "grid_col": row_col[1],
                        "starter": starter,
                    }
                )
    return rows


def lineup_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=LINEUP_COLUMNS)
    for column in [
        "fixture_id",
        "league_id",
        "season",
        "team_id",
        "coach_id",
        "source_player_id",
        "shirt_number",
        "grid_row",
        "grid_col",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["starter"] = frame["starter"].astype(bool)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce").dt.as_unit("ns")
    frame.loc[~frame["position"].isin(["G", "D", "M", "F"]), "position"] = None
    # Same source errors as the player stats: no id, or two players sharing one.
    no_id = frame["source_player_id"].isna() | (frame["source_player_id"] <= 0)
    if no_id.any():
        log.warning("Dropping %d lineup rows without a player id", int(no_id.sum()))
    frame = frame[~no_id]
    dup = frame.duplicated(["fixture_id", "team_id", "source_player_id"], keep="first")
    if dup.any():
        log.warning("Dropping %d lineup rows repeating a player id in a match", int(dup.sum()))
    return frame[~dup].reset_index(drop=True)


# Injuries: /injuries per league and season, one row per player and match.

INJURY_COLUMNS = [
    "source",
    "league_key",
    "season",
    "fixture_id",
    "fixture_date",
    "team_id",
    "team_name",
    "source_player_id",
    "player_name",
    "status",
    "reason",
]


def parse_injuries(payload: dict[str, Any], league: LeagueConfig, season: int) -> list[dict]:
    rows = []
    for item in payload.get("response") or []:
        player = item.get("player") or {}
        team = item.get("team") or {}
        fixture = item.get("fixture") or {}
        rows.append(
            {
                "source": SOURCE,
                "league_key": league.key,
                "season": season,
                "fixture_id": fixture.get("id"),
                "fixture_date": fixture.get("date"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "source_player_id": player.get("id"),
                "player_name": player.get("name"),
                "status": player.get("type"),  # "Missing Fixture" or "Questionable"
                "reason": player.get("reason"),
            }
        )
    return rows


def injury_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    for column in ["season", "fixture_id", "team_id", "source_player_id"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["fixture_date"] = pd.to_datetime(
        frame["fixture_date"], utc=True, errors="coerce"
    ).dt.as_unit("ns")
    no_id = frame["source_player_id"].isna() | (frame["source_player_id"] <= 0)
    frame = frame[~no_id]
    dup = frame.duplicated(["fixture_id", "source_player_id"], keep="first")
    if dup.any():
        log.warning("Dropping %d repeated injury rows", int(dup.sum()))
    return frame[~dup].reset_index(drop=True)
