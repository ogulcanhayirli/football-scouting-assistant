"""Provider interfaces.

Each data source gets one adapter that turns its raw responses into the
canonical tables below. The pandera contracts in ``scout.contracts`` check
those tables, so a new source (for example Sportmonks) only needs a new
adapter that passes the same contracts.

Canonical tables
----------------
player_season_stats
    One row per player, team, league and season with season totals.
    Built by a ``StatsProvider``.
players
    One row per player with biographical fields. Built by a ``ValuationProvider``.
valuations
    One row per player and valuation date. Built by a ``ValuationProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from scout.config import LeagueConfig


@dataclass(frozen=True)
class SeasonInfo:
    league_key: str
    season: int  # start year: 2023 means 2023/24
    start: date | None
    end: date | None
    current: bool
    has_player_stats: bool
    has_match_player_stats: bool = False  # per match player stats (/fixtures/players)
    has_match_team_stats: bool = False  # per match team stats (/fixtures/statistics)


class StatsProvider(Protocol):
    source: str

    def list_seasons(self, league: LeagueConfig) -> list[SeasonInfo]: ...

    def player_season_stats(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Canonical player_season_stats rows for one league and season."""
        ...


class ValuationProvider(Protocol):
    source: str

    def players(self, leagues: list[LeagueConfig]) -> pd.DataFrame:
        """Canonical players table for everyone who appeared in these leagues."""
        ...

    def valuations(self, leagues: list[LeagueConfig]) -> pd.DataFrame:
        """Canonical valuations for the players returned by ``players``."""
        ...
