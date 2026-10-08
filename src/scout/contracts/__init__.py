"""Data contracts for the canonical tables.

Every adapter's output is validated here before it is written as Parquet.
The contracts encode facts that hold for any source (counts are not negative,
shots on target never exceed shots), so a broken parser or a changed upstream
schema fails at the boundary instead of silently skewing a model.
"""

from scout.contracts.schemas import (
    PLAYER_MATCH_STATS,
    PLAYER_SEASON_STATS,
    PLAYERS,
    VALUATIONS,
    validate,
)

__all__ = ["PLAYERS", "PLAYER_MATCH_STATS", "PLAYER_SEASON_STATS", "VALUATIONS", "validate"]
