"""Typed configuration loaded from YAML.

Every run is driven by one config file, so switching from the free tier to Pro
is a config change, not a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class LeagueConfig(BaseModel):
    """A league we track, with its id in each source."""

    key: str = Field(description="Short internal key, for example 'epl' or 'super_lig'.")
    name: str
    api_football_id: int
    transfermarkt_id: str = Field(description="Transfermarkt competition code, e.g. GB1, TR1.")
    region: Literal["europe", "south_america", "north_america", "asia"] = Field(
        default="europe",
        description="Confederation; decides which club cups cover its players.",
    )
    calendar_year: bool = Field(
        default=False,
        description="Season runs within one calendar year (Brazil, Argentina), not August to May.",
    )
    skip_seasons: list[int] = Field(
        default_factory=list,
        description="Seasons left out of the features, e.g. a short transitional season.",
    )


class QuotaConfig(BaseModel):
    """Request limits for the API Football plan in use."""

    daily_limit: int = Field(gt=0)
    per_minute_limit: int = Field(gt=0)
    safety_margin: int = Field(
        default=0,
        ge=0,
        description="Requests per day left unused, so manual checks still work.",
    )

    @model_validator(mode="after")
    def _margin_below_limit(self) -> QuotaConfig:
        if self.safety_margin >= self.daily_limit:
            raise ValueError("safety_margin must be smaller than daily_limit")
        return self


class ApiFootballConfig(BaseModel):
    base_url: str = "https://v3.football.api-sports.io"
    plan: Literal["free", "pro", "ultra", "mega"] = "free"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    quota: QuotaConfig


class TransfermarktConfig(BaseModel):
    """transfermarkt-datasets snapshot (github.com/dcaribou/transfermarkt-datasets)."""

    base_url: str = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
    files: list[str] = Field(
        default_factory=lambda: [
            "players.csv.gz",
            "player_valuations.csv.gz",
            "appearances.csv.gz",
            "clubs.csv.gz",
            "transfers.csv.gz",
            "games.csv.gz",
        ]
    )


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    state_dir: Path = Path("data/state")
    reports_dir: Path = Path("reports")


class BackfillConfig(BaseModel):
    """What the backfill downloads.

    ``seasons`` lists explicit start years (2023 means 2023/24). ``"all"``
    means every season the API lists for the league that has player coverage.
    """

    seasons: list[int] | Literal["all"] = "all"
    endpoints: list[Literal["players"]] = Field(default_factory=lambda: ["players"])
    stop_on_plan_error: bool = Field(
        default=False,
        description="Stop on a plan error instead of skipping the season. "
        "Off for the free tier, where old seasons are expected to be refused.",
    )


class CompetitionConfig(LeagueConfig):
    """A cup, national team or youth competition. Its matches add columns to the
    players' league rows (European, international and youth minutes); it never
    gets rows of its own."""

    transfermarkt_id: str = ""
    kind: Literal["club", "national", "youth"]


class Settings(BaseModel):
    api_football: ApiFootballConfig
    transfermarkt: TransfermarktConfig = TransfermarktConfig()
    leagues: list[LeagueConfig]
    competitions: list[CompetitionConfig] = Field(default_factory=list)
    paths: PathsConfig = PathsConfig()
    backfill: BackfillConfig = BackfillConfig()

    def league(self, key: str) -> LeagueConfig:
        for league in self.leagues:
            if league.key == key:
                return league
        raise KeyError(f"Unknown league key {key!r}")


def load_settings(path: str | Path) -> Settings:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(raw)
