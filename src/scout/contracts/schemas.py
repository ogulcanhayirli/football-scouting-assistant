from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from scout.providers.api_football.adapter import NUMERIC_STATS
from scout.providers.api_football.matches import (
    MATCH_NUMERIC_STATS,
    MAX_MATCH_MINUTES,
    TEAM_NUMERIC,
)

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
    checks=[_not_greater(p, w) for p, w in RATIO_PAIRS],
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
    checks=[_not_greater(p, w) for p, w in MATCH_RATIO_PAIRS],
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
    ],
    unique=["league_key", "season", "source_player_id"],
    strict=False,
    name="player_season_features",
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
    unique=["source_player_id", "transfer_date", "team_out_id", "team_in_id"],
    strict=False,
    coerce=False,
    name="transfers",
)


def validate(schema: DataFrameSchema, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily so one run reports every violation, not just the first."""
    return schema.validate(frame, lazy=True)


__all__ = [
    "INJURIES",
    "MATCH_LINEUPS",
    "PLAYERS",
    "PLAYER_MATCH_STATS",
    "PLAYER_SEASON_FEATURES",
    "PLAYER_SEASON_LABELS",
    "PLAYER_SEASON_STATS",
    "TEAM_MATCH_STATS",
    "TRANSFERS",
    "VALUATIONS",
    "pa",
    "validate",
]
