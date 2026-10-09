"""The models the backtest compares. Each takes training rows, test rows and
the feature columns, and returns predictions for the test rows.

* ``no_change`` (growth only): the value stays where it is. The bar any growth
  model has to clear first.
* ``group_median``: the median target of similar players in the training
  seasons: same league, position and age band, falling back to coarser groups
  when a group is too small. The simple baseline the README promises.
* ``lightgbm``: gradient boosted trees on every feature column, untuned. The
  settings automatic model tuning will search later are the arguments here.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

AGE_BANDS = [0, 20, 23, 26, 29, 32, 50]
MIN_GROUP = 15

# LightGBM's own parameter names; automatic model tuning will search these.
DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
}
NUM_BOOST_ROUND = 600


def no_change(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.zeros(len(test))


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, right=False, labels=False)


def group_median(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Median target of the finest group with at least ``MIN_GROUP`` training rows."""
    levels = [
        ["league_key", "position", "age_band"],
        ["league_key", "position"],
        ["league_key"],
    ]
    train = train.assign(age_band=_age_band(train["age"]))
    test = test.assign(age_band=_age_band(test["age"]))
    pred = pd.Series(np.nan, index=test.index)
    for keys in levels:
        stats = train.groupby(keys, observed=True)["y"].agg(["median", "size"])
        stats = stats[stats["size"] >= MIN_GROUP]["median"]
        found = (
            test[keys].apply(tuple, axis=1).map(lambda k, s=stats: s.get(k if len(k) > 1 else k[0]))
        )
        pred = pred.fillna(found.astype(float))
    return pred.fillna(train["y"].median()).to_numpy()


def make_lightgbm(params: dict | None = None, rounds: int = NUM_BOOST_ROUND, seed: int = 7):
    settings = {
        "objective": "regression",
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
        **DEFAULT_PARAMS,
        **(params or {}),
    }

    def lightgbm_model(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
        dataset = lgb.Dataset(train[columns], label=train["y"], free_raw_data=False)
        booster = lgb.train(settings, dataset, num_boost_round=rounds)
        return booster.predict(test[columns])

    return lightgbm_model


def models_for(target_name: str, tuned: dict | None = None) -> dict:
    """The models to compare. ``tuned`` holds settings found by automatic model
    tuning (LightGBM parameters plus ``num_boost_round``), added as ``lightgbm_tuned``."""
    models = {"group_median": group_median, "lightgbm": make_lightgbm()}
    if target_name.startswith("growth"):
        models = {"no_change": no_change, **models}
    if tuned:
        params = dict(tuned)
        rounds = int(params.pop("num_boost_round", NUM_BOOST_ROUND))
        models["lightgbm_tuned"] = make_lightgbm(params, rounds=rounds)
    return models
