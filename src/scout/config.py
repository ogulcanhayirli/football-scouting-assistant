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


class StatsBombCompetition(BaseModel):
    """Competition seasons of the StatsBomb open data, by their names in
    ``competitions.json`` and our league key."""

    league_key: str
    competition: str = Field(description="Name in competitions.json, e.g. 'La Liga'.")
    seasons: list[int] = Field(description="Start years: 2004 means 2004/2005.")


class StatsBombConfig(BaseModel):
    """StatsBomb open data (github.com/statsbomb/open-data): non commercial
    use, attributed with the StatsBomb logo (prompt 18b)."""

    base_url: str = "https://raw.githubusercontent.com/statsbomb/open-data/master"
    competitions: list[StatsBombCompetition] = Field(
        default_factory=lambda: [
            StatsBombCompetition(
                league_key="la_liga", competition="La Liga", seasons=list(range(2004, 2021))
            ),
            StatsBombCompetition(league_key="epl", competition="Premier League", seasons=[2003]),
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


class ModelPrice(BaseModel):
    """Standard Claude API rates in USD per million tokens (prompts up to 100K tokens)."""

    input: float = Field(gt=0)
    output: float = Field(gt=0)


class ReportsConfig(BaseModel):
    """Generated scout reports (prompt 19, ``scout reports``): the writer and
    the judge, the hard token cap of one run and the rates its cost is
    computed with."""

    writer_model: str = "claude-haiku-5-5"
    writer_effort: Literal["low", "medium", "high"] = "low"
    writer_max_tokens: int = Field(
        default=6000,
        gt=0,
        description="Thinking counts toward it; 2,000 truncated 7 of about 8,000 requests "
        "in the first real run (prompt 19b).",
    )
    judge_model: str = "claude-sonnet-5-5"
    judge_effort: Literal["low", "medium", "high"] = "medium"
    judge_max_tokens: int = Field(default=4000, gt=0)
    judge_sample: int = Field(default=50, ge=0)
    retries: int = Field(default=2, ge=0, description="Regenerations after a failed check.")
    # The hard check (prompt 19b): the prompt asks for about 130 and never more
    # than 150 (``WORDS_TARGET``, ``WORDS_ASKED`` in ``scout.reports_text``); a
    # faithful report a few words over is kept, not thrown away.
    words_min: int = 90
    words_max: int = 170
    template_runs: int = Field(
        default=2,
        ge=1,
        description="Runs in a row a player's report may end on the template under one "
        "prompt version before the next runs stop asking for it until the prompt changes.",
    )
    max_tokens_per_run: int = Field(
        default=30_000_000,
        gt=0,
        description="Every token one run may be billed for, writer and judge together "
        "(input, cache writes and reads, output); requests that could pass it are not sent.",
    )
    poll_seconds: float = Field(default=30.0, gt=0)
    max_run_minutes: float = Field(
        default=20.0,
        gt=0,
        description="The longest the step waits on the Message Batches API (prompt 19c: run 12 "
        "waited 100 minutes on one batch). A batch still running then is kept in pending.json "
        "and collected by the next run; nothing is cancelled.",
    )
    judge_min_days: float = Field(
        default=6.0,
        ge=0,
        description="Days between two judged samples: daily runs (from 6 November) judge once a "
        "week, not 50 reports a day; 0 judges every run.",
    )
    prices: dict[str, ModelPrice] = Field(
        default_factory=lambda: {
            "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
            "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00),
        }
    )
    batch_discount: float = Field(default=0.5, gt=0, le=1)
    cache_read: float = 0.1  # of the input rate
    cache_write_5m: float = 1.25
    cache_write_1h: float = 2.0

    @model_validator(mode="after")
    def _priced(self) -> ReportsConfig:
        for model in (self.writer_model, self.judge_model):
            if model not in self.prices:
                raise ValueError(f"reports_text.prices has no rate for {model}")
        if not 0 < self.words_min <= self.words_max:
            raise ValueError("reports_text: words_min must be positive and at most words_max")
        return self


class ChatConfig(BaseModel):
    """The Ask tab (prompt 20): the Function URL of the ``ScoutChat`` stack
    (its ``ChatUrl`` output). Without it the tab says the chat is not
    switched on. The URL is public on the page anyway, so it is committed."""

    url: str | None = None
    model: str = "Claude Sonnet 5.5"
    # Where ``scout chat-eval`` reads the published tables (the page's own
    # address), so its answers come from the files the Lambda reads.
    site_url: str | None = None


class UpdateConfig(BaseModel):
    """The free tier update (``scout update``, from 7 November 2026)."""

    days: int = Field(default=7, ge=1, le=30, description="Match days back from yesterday.")
    max_requests: int = Field(default=90, ge=1, description="Requests a run may make.")
    priority: list[str] = Field(
        default_factory=list,
        description="League and competition keys in the order their matches are fetched; "
        "keys not listed follow in config order.",
    )
    s3_raw: bool = Field(
        default=True, description="Read and write the raw cache in S3 ($SCOUT_S3_BUCKET)."
    )


class Settings(BaseModel):
    api_football: ApiFootballConfig
    transfermarkt: TransfermarktConfig = TransfermarktConfig()
    statsbomb: StatsBombConfig = StatsBombConfig()
    leagues: list[LeagueConfig]
    competitions: list[CompetitionConfig] = Field(default_factory=list)
    paths: PathsConfig = PathsConfig()
    backfill: BackfillConfig = BackfillConfig()
    reports_text: ReportsConfig = ReportsConfig()
    chat: ChatConfig = ChatConfig()
    update: UpdateConfig = UpdateConfig()

    def league(self, key: str) -> LeagueConfig:
        for league in self.leagues:
            if league.key == key:
                return league
        raise KeyError(f"Unknown league key {key!r}")


def _read_yaml(path: Path) -> dict:
    """A config may name a ``base`` config (relative to itself) and override its
    top level keys; mappings one level down are merged key by key."""
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    merged = (
        _read_yaml(path.parent / base) if not Path(base).is_absolute() else _read_yaml(Path(base))
    )
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(_read_yaml(Path(path)))
