from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from scout.providers.api_football.adapter import NUMERIC_STATS
from scout.providers.api_football.extras import MAX_ELAPSED as MAX_EVENT_MINUTE
from scout.providers.api_football.matches import (
    MATCH_NUMERIC_STATS,
    MAX_CLUB_SWITCHES,
    MAX_MATCH_MINUTES,
    TEAM_NUMERIC,
    placeholder_blocks,
)
from scout.providers.api_football.names import NAME_COLUMNS, garbled

MAX_SEASON_MINUTES = 5000  # 38 league matches of 90 minutes is 3,420; leaves room for stoppage
RATIO_PAIRS = [
    ("shots_on", "shots_total"),
    ("duels_won", "duels_total"),
    ("dribbles_success", "dribbles_attempts"),
    ("lineups", "appearances"),
]


def _not_greater(part: str, whole: str) -> Check:
    def check(df: pd.DataFrame) -> pd.Series:
        both = df[part].notna() & df[whole].notna()
        return ~both | (df[part] <= df[whole])

    return Check(check, name=f"{part}_le_{whole}", error=f"{part} must not exceed {whole}")


def _with_attempts(part: str, whole: str) -> Check:
    """Successes need their attempts: 1 duel won with the duels left blank
    becomes more than all duels once the blank is read as 0."""

    def check(df: pd.DataFrame) -> pd.Series:
        return ~((df[part] > 0) & df[whole].isna()).fillna(False)

    return Check(check, name=f"{part}_has_{whole}", error=f"{part} without {whole}")


def _one_player_per_id(df: pd.DataFrame) -> bool:
    ordered = df.sort_values("kickoff")
    switches = ordered.groupby(["league_key", "season", "source_player_id"])["team_id"].agg(
        lambda teams: int((teams != teams.shift()).sum()) - 1
    )
    return bool((switches <= MAX_CLUB_SWITCHES).all())


def _names_repaired(df: pd.DataFrame) -> bool:
    """No name the adapter's repair would still change (rule 7: UTF-8 read as
    Latin-1, HTML entities)."""
    return not any(garbled(df[c]).any() for c in NAME_COLUMNS if c in df)


NAMES_REPAIRED = Check(
    _names_repaired,
    name="names_repaired",
    error="a name is garbled (UTF-8 read as Latin-1, or an HTML entity): repair it in the adapter",
)

_count = {"nullable": True, "checks": Check.ge(0)}

PLAYER_SEASON_STATS = DataFrameSchema(
    {
        "source": Column(str),
        "source_player_id": Column("Int64", Check.gt(0)),
        "player_name": Column(str, Check.str_length(min_value=1)),
        "birth_date": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("1950-01-01"), pd.Timestamp("2015-01-01")),
            nullable=True,
        ),
        "team_id": Column("Int64", nullable=False),
        "league_key": Column(str),
        "league_id": Column("Int64"),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "position": Column(
            str, Check.isin(["Goalkeeper", "Defender", "Midfielder", "Attacker"]), nullable=True
        ),
        **{c: Column("Float64", **_count) for c in NUMERIC_STATS},
        "minutes": Column("Float64", Check.in_range(0, MAX_SEASON_MINUTES), nullable=True),
        "rating": Column("Float64", Check.in_range(0, 10), nullable=True),
    },
    checks=[NAMES_REPAIRED, *[_not_greater(p, w) for p, w in RATIO_PAIRS]],
    unique=["source_player_id", "team_id", "league_id", "season"],
    strict=False,
    coerce=False,
    name="player_season_stats",
)

PLAYERS = DataFrameSchema(
    {
        "source": Column(str),
        "source_player_id": Column("Int64", Check.gt(0), unique=True),
        "player_name": Column(str, Check.str_length(min_value=1)),
        "birth_date": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("1950-01-01"), pd.Timestamp("2015-01-01")),
            nullable=True,
        ),
        "position": Column(str, nullable=True),
        "nationality": Column(str, nullable=True),
    },
    strict=False,
    name="players",
)

VALUATIONS = DataFrameSchema(
    {
        "source": Column(str),
        "source_player_id": Column("Int64", Check.gt(0)),
        "valuation_date": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("2000-01-01"), pd.Timestamp("2030-01-01")),
        ),
        "market_value_eur": Column(float, Check.in_range(0, 500_000_000)),
    },
    unique=["source_player_id", "valuation_date"],
    strict=False,
    name="valuations",
)


MATCH_RATIO_PAIRS = [
    ("shots_on", "shots_total"),
    ("duels_won", "duels_total"),
    ("dribbles_success", "dribbles_attempts"),
    ("passes_accurate", "passes_total"),
    ("goals", "team_goals"),
]

PLAYER_MATCH_STATS = DataFrameSchema(
    {
        "source": Column(str),
        "fixture_id": Column("Int64", Check.gt(0)),
        "kickoff": Column("datetime64[ns, UTC]"),
        "league_key": Column(str),
        "league_id": Column("Int64"),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "team_id": Column("Int64"),
        "opponent_id": Column("Int64"),
        "is_home": Column("boolean"),
        "source_player_id": Column("Int64", Check.gt(0)),
        "player_name": Column(str, Check.str_length(min_value=1), nullable=True),
        "position": Column(str, Check.isin(["G", "D", "M", "F"]), nullable=True),
        **{c: Column("Float64", **_count) for c in MATCH_NUMERIC_STATS},
        "minutes": Column("Float64", Check.in_range(0, MAX_MATCH_MINUTES), nullable=True),
        "rating": Column("Float64", Check.in_range(0, 10), nullable=True),
    },
    checks=[
        NAMES_REPAIRED,
        *[_not_greater(p, w) for p, w in MATCH_RATIO_PAIRS],
        *[_with_attempts(p, w) for p, w in MATCH_RATIO_PAIRS if w != "team_goals"],
        Check(
            _one_player_per_id,
            name="one_player_per_id",
            error="an id switches clubs back and forth: two players share it",
        ),
        Check(
            lambda df: not placeholder_blocks(df).any(),
            name="no_placeholder_zeros",
            error="a team match has zeros for every detailed stat: placeholders, not data",
        ),
    ],
    unique=["fixture_id", "source_player_id"],
    strict=False,
    coerce=False,
    name="player_match_stats",
)


def _on_or_before(column: str, limit: str) -> Check:
    """Point in time rule: ``column`` (a date the row used) is not after ``limit``."""

    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna() | (df[column] <= df[limit])

    return Check(check, name=f"{column}_le_{limit}", error=f"{column} after {limit}: leak")


def _before(column: str, limit: str) -> Check:
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna() | (df[column] < df[limit])

    return Check(check, name=f"{column}_lt_{limit}", error=f"{column} not before {limit}: leak")


def _after(column: str, start: str) -> Check:
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna() | (df[column] > df[start])

    return Check(check, name=f"{column}_after_{start}", error=f"{column} not after {start}")


_rate = {"nullable": True, "checks": Check.in_range(0, 1)}
_per90 = {"nullable": True, "checks": Check.ge(0)}

# Columns that describe a player as the Transfermarkt snapshot sees him today.
# They live only in the player profile table, for display in the current
# season's lists; a feature table carrying one would break the point in time
# rule, so the feature contract rejects them.
# The current club (squads of today, prompt 16) and the season so far block
# describe the present the same way.
PRESENT_ONLY_COLUMNS = (
    "contract_expiry",
    "agent",
    "highest_market_value_eur",
    "current_club_id",
    "current_club",
    "current_league",
    "current_league_key",
    "shirt_number",
    "squad_position",
    "squad_date",
    "club_source",
    "moved_since_forecast",
)
PRESENT_ONLY_PREFIX = "so_far_"  # every column of the season so far block


def _no_present_only_columns(df: pd.DataFrame) -> bool:
    return not set(PRESENT_ONLY_COLUMNS) & set(df.columns) and not any(
        str(c).startswith(PRESENT_ONLY_PREFIX) for c in df.columns
    )


PLAYER_SEASON_FEATURES = DataFrameSchema(
    {
        "league_key": Column(str, Check.str_matches(r"^[a-z0-9_]+$")),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "source_player_id": Column("Int64", Check.gt(0)),
        "tm_player_id": Column("Int64", nullable=True),
        "feature_date": Column("datetime64[ns]"),
        "tier": Column(str, Check.isin(["match", "season"])),
        "regular": Column(bool),
        "age": Column(float, Check.in_range(14, 46), nullable=True),
        "minutes": Column("Float64", Check.in_range(0, MAX_SEASON_MINUTES)),
        "minutes_share": Column(float, **_rate),
        "goals_per90": Column("Float64", **_per90),
        "shots_total_per90": Column("Float64", **_per90),
        "passes_key_per90": Column("Float64", **_per90),
        "rating": Column("Float64", Check.in_range(0, 10), nullable=True),
        "shot_accuracy": Column("Float64", **_rate),
        "pass_accuracy": Column("Float64", **_rate),
        "duel_win_rate": Column("Float64", **_rate),
        "dribble_success_rate": Column("Float64", **_rate),
        "market_value": Column(float, Check.gt(0), nullable=True),
    },
    checks=[
        _on_or_before("last_match_date", "feature_date"),
        _on_or_before("value_date", "feature_date"),
        _before("prev_feature_date", "feature_date"),
        _before("last_comp_match_date", "feature_date"),
        _before("last_injury_date", "feature_date"),
        _before("last_transfer_date", "feature_date"),
        _before("last_coach_change_date", "feature_date"),
        Check(
            _no_present_only_columns,
            name="no_present_only_columns",
            error=f"feature tables must not carry {PRESENT_ONLY_COLUMNS} or {PRESENT_ONLY_PREFIX}*",
        ),
    ],
    unique=["league_key", "season", "source_player_id"],
    strict=False,
    name="player_season_features",
)

PLAYER_PROFILES = DataFrameSchema(
    {
        "af_player_id": Column("Int64", Check.gt(0), unique=True),
        "tm_player_id": Column("Int64", Check.gt(0)),
        "contract_expiry": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("2000-01-01"), pd.Timestamp("2040-12-31")),
            nullable=True,
        ),
        "agent": Column("string", Check.str_length(min_value=1), nullable=True),
        "highest_market_value_eur": Column(
            float, [Check.gt(0), Check.le(500_000_000)], nullable=True
        ),
        "as_of": Column("datetime64[ns]"),
    },
    strict=False,
    name="player_profiles",
)

PLAYER_SEASON_LABELS = DataFrameSchema(
    {
        "league_key": Column(str),
        "season": Column("Int64"),
        "source_player_id": Column("Int64", Check.gt(0)),
        "feature_date": Column("datetime64[ns]"),
        "value_now": Column(float, Check.gt(0), nullable=True),
        "value_24m": Column(float, Check.gt(0), nullable=True),
        "value_36m": Column(float, Check.gt(0), nullable=True),
    },
    checks=[_after("value_24m_date", "feature_date"), _after("value_36m_date", "feature_date")],
    unique=["league_key", "season", "source_player_id"],
    strict=False,
    name="player_season_labels",
)


TEAM_MATCH_STATS = DataFrameSchema(
    {
        "source": Column(str),
        "fixture_id": Column("Int64", Check.gt(0)),
        "kickoff": Column("datetime64[ns, UTC]"),
        "league_key": Column(str),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "team_id": Column("Int64", Check.gt(0)),
        "opponent_id": Column("Int64", nullable=True),
        "is_home": Column("boolean", nullable=True),
        **{c: Column("Float64", **_count) for c in TEAM_NUMERIC if c != "goals_prevented"},
        "goals_prevented": Column("Float64", nullable=True),
        "possession": Column("Float64", Check.in_range(0, 100), nullable=True),
        "pass_accuracy": Column("Float64", Check.in_range(0, 100), nullable=True),
        "xg": Column("Float64", Check.in_range(0, 15), nullable=True),
    },
    checks=[
        _not_greater("shots_on", "shots_total"),
        _not_greater("shots_inside_box", "shots_total"),
        _not_greater("passes_accurate", "passes_total"),
    ],
    unique=["fixture_id", "team_id"],
    strict=False,
    coerce=False,
    name="team_match_stats",
)


MATCH_LINEUPS = DataFrameSchema(
    {
        "fixture_id": Column("Int64", Check.gt(0)),
        "kickoff": Column("datetime64[ns, UTC]"),
        "league_key": Column(str),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "team_id": Column("Int64", Check.gt(0)),
        "source_player_id": Column("Int64", Check.gt(0)),
        "position": Column(str, Check.isin(["G", "D", "M", "F"]), nullable=True),
        "grid_row": Column("Int64", Check.in_range(1, 7), nullable=True),
        "grid_col": Column("Int64", Check.in_range(1, 7), nullable=True),
        "starter": Column(bool),
    },
    checks=[NAMES_REPAIRED],
    unique=["fixture_id", "team_id", "source_player_id"],
    strict=False,
    coerce=False,
    name="match_lineups",
)

INJURIES = DataFrameSchema(
    {
        "league_key": Column(str),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "fixture_id": Column("Int64", Check.gt(0), nullable=True),
        "fixture_date": Column("datetime64[ns, UTC]", nullable=True),
        "source_player_id": Column("Int64", Check.gt(0)),
        "status": Column(str, nullable=True),
        "reason": Column(str, nullable=True),
    },
    checks=[NAMES_REPAIRED],
    unique=["fixture_id", "source_player_id"],
    strict=False,
    coerce=False,
    name="injuries",
)

TRANSFERS = DataFrameSchema(
    {
        "source_player_id": Column("Int64", Check.gt(0)),
        "transfer_date": Column("datetime64[ns]"),
        "team_out_id": Column("Int64", Check.gt(0), nullable=True),
        "team_in_id": Column("Int64", Check.gt(0), nullable=True),
        "type_raw": Column(str, nullable=True),
        "kind": Column(
            str, Check.isin(["transfer", "loan", "loan_return", "free", "other", "unknown"])
        ),
        "fee_eur": Column("Float64", Check.ge(0), nullable=True),
    },
    checks=[NAMES_REPAIRED],
    unique=["source_player_id", "transfer_date", "team_out_id", "team_in_id"],
    strict=False,
    coerce=False,
    name="transfers",
)


SQUADS = DataFrameSchema(
    {
        "source": Column(str),
        "as_of": Column("datetime64[ns]"),
        "team_id": Column("Int64", Check.gt(0)),
        "source_player_id": Column("Int64", Check.gt(0)),
        "player_name": Column(str, nullable=True),
        "age": Column("Int64", Check.in_range(14, 46), nullable=True),
        "shirt_number": Column("Int64", Check.in_range(1, 99), nullable=True),
        "squad_position": Column(
            str, Check.isin(["Goalkeeper", "Defender", "Midfielder", "Attacker"]), nullable=True
        ),
    },
    checks=[NAMES_REPAIRED],
    unique=["team_id", "source_player_id"],
    strict=False,
    coerce=False,
    name="squads",
)

CURRENT_CLUB = DataFrameSchema(
    {
        "source_player_id": Column("Int64", Check.gt(0), unique=True),
        "current_club_id": Column("Int64", Check.gt(0)),
        "current_club": Column(str, nullable=True),
        "current_league": Column(str, nullable=True),
        "shirt_number": Column("Int64", Check.in_range(1, 99), nullable=True),
        "club_source": Column(str, Check.isin(["squad", "transfer"])),
        "squad_date": Column("datetime64[ns]"),
    },
    strict=False,
    coerce=False,
    name="current_club",
)


EARLIEST_DATE = pd.Timestamp("1950-01-01")
_dated = {"nullable": True, "checks": Check.ge(EARLIEST_DATE)}

SIDELINED = DataFrameSchema(
    {
        "source": Column(str),
        "source_player_id": Column("Int64", Check.gt(0)),
        "type": Column(str, nullable=True),
        "start_date": Column("datetime64[ns]", Check.ge(EARLIEST_DATE)),
        "end_date": Column("datetime64[ns]", **_dated),
    },
    checks=[_not_greater("start_date", "end_date")],
    unique=["source_player_id", "type", "start_date", "end_date"],
    strict=False,
    coerce=False,
    name="sidelined",
)

COACH_SPELLS = DataFrameSchema(
    {
        "source": Column(str),
        "coach_id": Column("Int64", Check.gt(0)),
        "coach_name": Column(str, nullable=True),
        "birth_date": Column("datetime64[ns]", **_dated),
        "team_id": Column("Int64", Check.gt(0)),
        "start_date": Column("datetime64[ns]", **_dated),
        "end_date": Column("datetime64[ns]", **_dated),
    },
    checks=[NAMES_REPAIRED, _not_greater("start_date", "end_date")],
    unique=["coach_id", "team_id", "start_date", "end_date"],
    strict=False,
    coerce=False,
    name="coach_spells",
)

FIXTURE_EVENTS = DataFrameSchema(
    {
        "source": Column(str),
        "fixture_id": Column("Int64", Check.gt(0)),
        "kickoff": Column("datetime64[ns, UTC]"),
        "season": Column("Int64", Check.in_range(2000, 2100)),
        "event_index": Column("Int64", Check.ge(0)),
        "elapsed": Column("Int64", Check.in_range(0, MAX_EVENT_MINUTE), nullable=True),
        "extra": Column("Int64", Check.ge(0), nullable=True),
        "team_id": Column("Int64", nullable=True),
        "source_player_id": Column("Int64", Check.gt(0), nullable=True),
        "assist_id": Column("Int64", Check.gt(0), nullable=True),
        "type": Column(str, nullable=True),
        "detail": Column(str, nullable=True),
    },
    checks=[NAMES_REPAIRED],
    unique=["fixture_id", "event_index"],
    strict=False,
    coerce=False,
    name="fixture_events",
)


# The players table (prompt 20f, ``scout.players``): one row per player of the
# newest season, published next to the two lists.


def _blank_has_reason(value: str, reason: str) -> Check:
    """A chance is blank exactly when its reason is given."""

    def check(df: pd.DataFrame) -> pd.Series:
        return df[value].isna() == df[reason].notna()

    return Check(check, name=f"{reason}_iff_blank", error=f"{reason} must explain a blank {value}")


def _off_list_has_reason(member: str, reason: str) -> Check:
    """A player is on the list exactly when no reason says why not."""

    def check(df: pd.DataFrame) -> pd.Series:
        return df[member].astype(bool) == df[reason].isna()

    return Check(check, name=f"{reason}_iff_off", error=f"{reason} must explain not {member}")


def _regular_only(column: str) -> Check:
    """Ability and fair value exist only for regular players (450+ minutes)."""

    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna() | (df["minutes"] >= 450)

    return Check(check, name=f"{column}_regulars_only", error=f"{column} under 450 minutes")


def _reports_on_lists(df: pd.DataFrame) -> pd.Series:
    listed = df["in_candidates"].astype(bool) | df["in_bargains"].astype(bool)
    return ~df["report_available"].astype(bool) | listed


_pct = {"nullable": True, "checks": Check.in_range(0, 100)}
_chance_count = {"nullable": True, "checks": Check.ge(30)}

PLAYERS_TABLE = DataFrameSchema(
    {
        "season": Column(int, Check.in_range(2000, 2100)),
        "player_id": Column("Int64", Check.gt(0), unique=True),
        "player": Column(str, Check.str_length(min_value=1)),
        "age": Column(float, Check.in_range(14, 50), nullable=True),
        "nationality": Column(str, nullable=True),
        "position": Column("string", nullable=True),
        "club": Column(str),
        "league": Column(str),
        "minutes": Column("Int64", Check.in_range(1, MAX_SEASON_MINUTES)),
        "minutes_share": Column(float, Check.ge(0), nullable=True),
        "ability_score": Column(float, nullable=True),
        "ability_pct": Column(float, **_pct),
        "value_eur": Column(float, Check.gt(0), nullable=True),
        "value_date": Column("datetime64[ns]", nullable=True),
        "value_rank": Column("Int64", Check.ge(1), nullable=True),
        "fair_value_eur": Column(float, Check.ge(0), nullable=True),
        "value_to_fair": Column(float, Check.gt(0), nullable=True),
        "elite_decile": Column("Int64", Check.in_range(1, 10), nullable=True),
        "elite_chance_pct": Column("Int64", **_pct),
        "elite_chance_players": Column("Int64", **_chance_count),
        "elite_chance_why_blank": Column(str, nullable=True),
        "big_move_decile": Column("Int64", Check.in_range(1, 10), nullable=True),
        "big_move_chance_pct": Column("Int64", **_pct),
        "big_move_chance_players": Column("Int64", **_chance_count),
        "big_move_chance_why_blank": Column(str, nullable=True),
        "flag_count": Column(int, Check.in_range(0, 3)),
        "flags": Column(str),
        "in_candidates": Column(bool),
        "in_bargains": Column(bool),
        "why_not_candidate": Column(str, nullable=True),
        "why_not_bargain": Column(str, nullable=True),
        "report_available": Column(bool),
        "data_date": Column("datetime64[ns]"),
    },
    checks=[
        NAMES_REPAIRED,
        _on_or_before("value_date", "data_date"),
        _blank_has_reason("elite_chance_pct", "elite_chance_why_blank"),
        _blank_has_reason("big_move_chance_pct", "big_move_chance_why_blank"),
        _off_list_has_reason("in_candidates", "why_not_candidate"),
        _off_list_has_reason("in_bargains", "why_not_bargain"),
        _regular_only("ability_pct"),
        _regular_only("fair_value_eur"),
        Check(_reports_on_lists, name="reports_on_lists", error="a report off the two lists"),
    ],
    strict=True,
    coerce=False,
    name="players_table",
)


# The bargains table (``scout.lists``): the prime and experienced lists rank
# every eligible player by value against fair value; only a player priced
# below his fair value is called a bargain (prompt 20g).


def _below_fair(df: pd.DataFrame) -> pd.Series:
    return df["below_fair_value"].astype(bool) == (df["value_to_fair"] < 1)


def _one_rank_per_place(df: pd.DataFrame) -> bool:
    return not df.duplicated(["list", "list_rank"]).any()


BARGAINS_TABLE = DataFrameSchema(
    {
        "season": Column(int, Check.in_range(2000, 2100)),
        "list": Column(str, Check.isin(["prime", "experienced"])),
        "list_rank": Column(int, Check.ge(1)),
        "player_id": Column("Int64", Check.gt(0)),
        "player": Column(str, Check.str_length(min_value=1)),
        "age": Column(float, Check.in_range(14, 34)),
        "value_eur": Column(float, Check.gt(0)),
        "fair_value_eur": Column(float, Check.ge(0)),
        "value_to_fair": Column(float, Check.gt(0)),
        "value_to_fair_pct": Column(float, Check.in_range(0, 100)),
        "below_fair_value": Column(bool),
    },
    checks=[
        NAMES_REPAIRED,
        Check(_below_fair, name="below_fair_value", error="below_fair_value is value_to_fair < 1"),
        Check(_one_rank_per_place, name="one_rank_per_place", error="a list rank used twice"),
    ],
    strict=False,  # the present (club today, season so far) joins later
    coerce=False,
    name="bargains_table",
)


def validate(schema: DataFrameSchema, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily so one run reports every violation, not just the first."""
    return schema.validate(frame, lazy=True)


__all__ = [
    "BARGAINS_TABLE",
    "COACH_SPELLS",
    "CURRENT_CLUB",
    "FIXTURE_EVENTS",
    "INJURIES",
    "MATCH_LINEUPS",
    "PLAYERS",
    "PLAYERS_TABLE",
    "PLAYER_MATCH_STATS",
    "PLAYER_PROFILES",
    "PLAYER_SEASON_FEATURES",
    "PLAYER_SEASON_LABELS",
    "PLAYER_SEASON_STATS",
    "PRESENT_ONLY_COLUMNS",
    "PRESENT_ONLY_PREFIX",
    "SIDELINED",
    "SQUADS",
    "TEAM_MATCH_STATS",
    "TRANSFERS",
    "VALUATIONS",
    "pa",
    "validate",
]
