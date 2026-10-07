from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from scout.providers.api_football.adapter import NUMERIC_STATS

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


def validate(schema: DataFrameSchema, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily so one run reports every violation, not just the first."""
    return schema.validate(frame, lazy=True)


__all__ = ["PLAYERS", "PLAYER_SEASON_STATS", "VALUATIONS", "pa", "validate"]
