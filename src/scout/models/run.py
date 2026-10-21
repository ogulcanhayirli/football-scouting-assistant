"""``scout backtest``: run the expanding window backtest for each target and
write the results.

Outputs:

* ``reports/backtest_<target>.md``: metrics overall, by league, by tier and
  by season, for every model
* ``reports/backtest_summary.json``: the overall numbers, machine readable
* ``data/processed/models/backtest_<target>.parquet``: every out of time
  prediction, for error analysis
* ``data/processed/models/stored/fair_value_fold_<season>.txt``: the fold
  models that predicted the newest season, which ``scout score`` reuses
  (``scout.models.store``)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from scout.config import Settings
from scout.fsutil import atomic_write_text
from scout.models.backtest import fold_season, run_backtest, summarize
from scout.models.estimators import models_for
from scout.models.store import FAIR_VALUE, save_booster, store_dir
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
    features: pd.DataFrame,
    labels: pd.DataFrame,
    name: str,
    tuned: dict | None = None,
    keep: dict | None = None,
) -> pd.DataFrame:
    target = TARGETS[name]
    data, columns = build_dataset(features, labels, target)
    log.info("%s: %d rows, %d feature columns", name, len(data), len(columns))
    return run_backtest(data, columns, target, models_for(name, tuned, keep))


def newest_folds(features: pd.DataFrame) -> list[int]:
    """The folds that hold rows of the newest season: European rows sit in
    the newest fold, calendar year leagues in the one before."""
    newest = features[features["season"] == features["season"].max()]
    return sorted(int(f) for f in fold_season(newest["feature_date"]).unique())


def store_fair_value(keep: dict, features: pd.DataFrame, folder: Path) -> list[Path]:
    """Save the fold models that scored the newest season."""
    season = int(features["season"].max())
    paths = []
    for fold in newest_folds(features):
        booster = keep.get(fold)
        if booster is None:  # no valued rows in that fold: nothing to score
            log.warning("fair value: no model for fold %d, nothing stored", fold)
            continue
        paths.append(
            save_booster(
                booster,
                folder,
                FAIR_VALUE.format(fold=fold),
                booster.feature_name(),
                model="fair_value",
                fold=fold,
                trained_on=f"folds up to {fold - 1}",
                newest_season=season,
                created=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
            )
        )
    return paths


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
        keep: dict = {}
        predictions = run_target(features, labels, name, tuned, keep)
        predictions.to_parquet(out_dir / f"backtest_{name}.parquet", index=False)
        for path in store_fair_value(keep, features, store_dir(settings.paths.processed_dir)):
            log.info("Stored %s", path)
        summary[name] = write_report(name, predictions, reports)
    atomic_write_text(reports / "backtest_summary.json", json.dumps(summary, indent=2))
    return summary


def replace_season(old: pd.DataFrame, new: pd.DataFrame, season: int) -> pd.DataFrame:
    """``old`` with the rows of ``season`` replaced by ``new``: a player still
    there keeps his place in the file, a new one comes last, one who is gone
    is dropped. Unchanged rows give an unchanged file."""
    keys = ["league_key", "season", "source_player_id"]

    def index(frame: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [frame["league_key"].astype(str), frame["season"].astype(int), frame[keys[2]]]
        )

    place = pd.Series(range(len(old)), index=index(old), dtype=float)
    kept = old[old["season"].astype(int) != season]
    kept = kept.assign(_place=place.to_numpy()[(old["season"].astype(int) != season).to_numpy()])
    new = new.assign(_place=place.reindex(index(new)).to_numpy())
    out = pd.concat([kept, new], ignore_index=True)
    out = out.sort_values("_place", kind="stable", na_position="last")
    return out.drop(columns="_place").reset_index(drop=True)


def score_stored_fair_value(settings: Settings) -> pd.DataFrame:
    """The weekly run: the newest season's fair value from the stored fold
    models, replacing that season's rows of ``backtest_fair_value.parquet``.
    Each row takes the model of its own fold (or the newest stored fold
    before it), so it never sees a row from its own season or later. The
    baseline is the same lookup of medians as in the backtest. Nothing is
    trained."""
    from scout.models.estimators import group_median
    from scout.models.store import load_booster, stored_folds

    root = settings.paths.processed_dir / "features"
    features = read_feature_table(root / "player_season_features")
    labels = read_feature_table(root / "player_season_labels")
    data, columns = build_dataset(features, labels, TARGETS["fair_value"])
    folder = store_dir(settings.paths.processed_dir)
    available = stored_folds(folder)
    season = int(features["season"].max())
    folds = fold_season(data["feature_date"])
    newest = data["season"] == season
    frames = []
    for fold in sorted(folds[newest].unique()):
        usable = [f for f in available if f <= fold]
        if not usable:
            raise FileNotFoundError(
                f"no stored fair value model for fold {fold} or earlier in {folder}: "
                "run `scout backtest` once (docs/DEPLOY.md, Retraining)"
            )
        stored = load_booster(folder, FAIR_VALUE.format(fold=usable[-1]))
        if stored.columns != columns:
            raise ValueError(
                "the fair value columns changed since the stored model was trained: "
                "retrain with `scout backtest` (docs/DEPLOY.md, Retraining)"
            )
        test = data[newest & (folds == fold)]
        train = data[folds <= stored.meta["fold"] - 1]
        frame = test[["league_key", "season", "source_player_id", "player_name", "tier", "y"]]
        frame = frame.assign(
            train_rows=len(train),
            pred_group_median=group_median(train, test, columns),
            pred_lightgbm=stored.predict(test),
        )
        frames.append(frame)
    path = settings.paths.processed_dir / "models" / "backtest_fair_value.parquet"
    old = pd.read_parquet(path)
    out = replace_season(old, pd.concat(frames, ignore_index=True), season)
    out.to_parquet(path, index=False)
    log.info("fair value: %d rows of season %d scored with the stored models", newest.sum(), season)
    return out
