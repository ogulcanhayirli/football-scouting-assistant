"""``scout backtest``: run the expanding window backtest for each target and
write the results.

Outputs:

* ``reports/backtest_<target>.md``: metrics overall, by league, by tier and
  by season, for every model
* ``reports/backtest_summary.json``: the overall numbers, machine readable
* ``data/processed/models/backtest_<target>.parquet``: every out of time
  prediction, for error analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from scout.config import Settings
from scout.fsutil import atomic_write_text
from scout.models.backtest import run_backtest, summarize
from scout.models.estimators import models_for
from scout.models.targets import TARGETS, build_dataset
from scout.tables import read_feature_table

log = logging.getLogger(__name__)


def _table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in ("mae_log", "rmse_log", "median_ape", "spearman"):
        if column in shown:
            shown[column] = shown[column].round(3)
    return shown.to_markdown(index=False)


def run_target(
    features: pd.DataFrame, labels: pd.DataFrame, name: str, tuned: dict | None = None
) -> pd.DataFrame:
    target = TARGETS[name]
    data, columns = build_dataset(features, labels, target)
    log.info("%s: %d rows, %d feature columns", name, len(data), len(columns))
    return run_backtest(data, columns, target, models_for(name, tuned))


def write_report(name: str, predictions: pd.DataFrame, reports: Path) -> dict:
    overall = summarize(predictions)
    seasons = predictions["season"]
    parts = [
        f"# Backtest: {name}",
        "",
        f"Expanding window, out of time predictions for seasons {seasons.min()} to "
        f"{seasons.max()}; regular players only. Errors are in log units "
        "(0.10 is about 10%); median_ape is the median absolute percentage error "
        "of the implied value; spearman is the rank correlation.",
        "",
        "## Overall",
        "",
        _table(overall),
        "",
        "## By league",
        "",
        _table(summarize(predictions, ["league_key"])),
        "",
        "## By tier",
        "",
        _table(summarize(predictions, ["tier"])),
        "",
        "## By season",
        "",
        _table(summarize(predictions, ["season"])),
        "",
    ]
    atomic_write_text(reports / f"backtest_{name}.md", "\n".join(parts))
    return {
        row["model"]: {k: v for k, v in row.items() if k != "model"}
        for row in overall.to_dict("records")
    }


def run_all(settings: Settings, names: list[str], tuned: dict | None = None) -> dict:
    root = settings.paths.processed_dir / "features"
    features = read_feature_table(root / "player_season_features")
    labels = read_feature_table(root / "player_season_labels")
    reports = settings.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    out_dir = settings.paths.processed_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        predictions = run_target(features, labels, name, tuned)
        predictions.to_parquet(out_dir / f"backtest_{name}.parquet", index=False)
        summary[name] = write_report(name, predictions, reports)
    atomic_write_text(reports / "backtest_summary.json", json.dumps(summary, indent=2))
    return summary
