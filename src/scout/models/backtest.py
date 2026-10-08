"""Expanding window backtest: every season is predicted by a model that only
saw earlier seasons.

For a test season ``t`` the model trains on seasons up to
``t - max(1, horizon_years)``. A growth label 24 months ahead is observed two
seasons after its row, so a 2016 row's label (June 2019) would already lie
after the 2018 feature date; the gap keeps every training label in the past
of every test row. Seasons that ran late (2019/20) are handled by ``purge``,
and ``check_no_leak`` then verifies the rule on the actual label dates, not
just the season numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scout.models.targets import Target

log = logging.getLogger(__name__)

MIN_TRAIN_SEASONS = 3


@dataclass(frozen=True)
class Fold:
    test_season: int
    train_seasons: list[int]


def expanding_folds(
    seasons: list[int], target: Target, min_train_seasons: int = MIN_TRAIN_SEASONS
) -> Iterator[Fold]:
    gap = max(1, target.horizon_years)
    ordered = sorted(set(seasons))
    for test in ordered:
        train = [s for s in ordered if s <= test - gap]
        if len(train) >= min_train_seasons:
            yield Fold(test, train)


class LeakError(AssertionError):
    pass


def check_no_leak(train: pd.DataFrame, test: pd.DataFrame, target: Target) -> None:
    """Every training row, label included, must be known before the earliest test row."""
    first_test = test["feature_date"].min()
    if (train["feature_date"] >= first_test).any():
        raise LeakError("a training row's feature date is not before the test season")
    if target.label_date:
        observed = train[target.label_date].dropna()
        if (observed > first_test).any():
            raise LeakError(
                f"{int((observed > first_test).sum())} training labels observed after "
                f"the test feature date {first_test.date()}"
            )


def purge(train: pd.DataFrame, test: pd.DataFrame, target: Target) -> pd.DataFrame:
    """Drop training rows whose label was observed after the test feature date.

    The season gap handles the normal case; purging handles seasons that ran
    late. 2019/20 ended in July 2020, so its 24 month labels can date from
    late July 2022, after the 30 June 2022 feature date of season 2021.
    """
    if not target.label_date:
        return train
    first_test = test["feature_date"].min()
    late = train[target.label_date] > first_test
    if late.any():
        log.info(
            "%s: purged %d training rows labelled after %s",
            target.name,
            int(late.sum()),
            first_test.date(),
        )
    return train[~late]


# A model is a function: (train rows, test rows, feature columns) -> predictions.
Model = Callable[[pd.DataFrame, pd.DataFrame, list[str]], np.ndarray]


def run_backtest(
    data: pd.DataFrame, columns: list[str], target: Target, models: dict[str, Model]
) -> pd.DataFrame:
    """Out of time predictions for every fold and model, one row per test row."""
    out = []
    for fold in expanding_folds(data["season"].astype(int).tolist(), target):
        train = data[data["season"].isin(fold.train_seasons)]
        test = data[data["season"] == fold.test_season]
        train = purge(train, test, target)
        check_no_leak(train, test, target)
        frame = test[
            ["league_key", "season", "source_player_id", "player_name", "tier", "y"]
        ].copy()
        frame["train_rows"] = len(train)
        for name, model in models.items():
            frame[f"pred_{name}"] = model(train, test, columns)
        out.append(frame)
        log.info(
            "%s: test %d (%d rows) on %d to %d (%d rows)",
            target.name,
            fold.test_season,
            len(test),
            fold.train_seasons[0],
            fold.train_seasons[-1],
            len(train),
        )
    return pd.concat(out, ignore_index=True)


# Metrics, on the log scale


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation; undefined (NaN) for fewer than 3 rows or a constant
    prediction such as ``no_change``."""
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    """MAE and RMSE in log units; median absolute percentage error of the
    implied value (or growth factor); Spearman rank correlation, which is what
    matters when a scout sorts a list."""
    err = pred - y
    ratio = np.exp(err)
    return {
        "rows": len(y),
        "mae_log": float(err.abs().mean()),
        "rmse_log": float(np.sqrt((err**2).mean())),
        "median_ape": float((ratio - 1).abs().median()),
        "spearman": _spearman(y, pred),
    }


def summarize(predictions: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per model, optionally per group (season, league, tier)."""
    models = [c.removeprefix("pred_") for c in predictions.columns if c.startswith("pred_")]
    groups = predictions.groupby(by) if by else [((), predictions)]
    rows = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        for model in models:
            row = dict(zip(by or [], key, strict=True))
            row["model"] = model
            row.update(metrics(group["y"], group[f"pred_{model}"]))
            rows.append(row)
    return pd.DataFrame(rows)
