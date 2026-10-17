"""Data contracts for the canonical tables.

Every adapter's output is validated here before it is written as Parquet.
The contracts encode facts that hold for any source (counts are not negative,
shots on target never exceed shots), so a broken parser or a changed upstream
schema fails at the boundary instead of silently skewing a model.
"""

from scout.contracts.schemas import (
    COACH_SPELLS,
    FIXTURE_EVENTS,
    INJURIES,
    MATCH_LINEUPS,
    PLAYER_MATCH_STATS,
    PLAYER_PROFILES,
    PLAYER_SEASON_FEATURES,
    PLAYER_SEASON_LABELS,
    PLAYER_SEASON_STATS,
    PLAYERS,
    PRESENT_ONLY_COLUMNS,
    SIDELINED,
    TEAM_MATCH_STATS,
    TRANSFERS,
    VALUATIONS,
    validate,
)

__all__ = [
    "COACH_SPELLS",
    "FIXTURE_EVENTS",
    "INJURIES",
    "MATCH_LINEUPS",
    "PLAYERS",
    "PLAYER_MATCH_STATS",
    "PLAYER_PROFILES",
    "PLAYER_SEASON_FEATURES",
    "PLAYER_SEASON_LABELS",
    "PLAYER_SEASON_STATS",
    "PRESENT_ONLY_COLUMNS",
    "SIDELINED",
    "TEAM_MATCH_STATS",
    "TRANSFERS",
    "VALUATIONS",
    "validate",
]
