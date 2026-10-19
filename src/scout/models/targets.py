"""Modelling datasets, built from the feature and label tables.

* ``fair_value``: log of the market value at the feature date, so errors are
  relative (0.1 is about 10%, for a 1m and a 100m player alike). Every
  ``market_value*`` column is left out of its features, because
  ``market_value`` is the target itself.

The value growth targets (24 and 36 months) were retired on 19 October 2026
(decision 5): they barely beat their age and position baseline (MODELS.md).
The growth labels stay in the label table for the bargains backtest.

Only regular players (450 minutes or more) are modelled: below that a season
says little about a player, and those are not the players a scout ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scout.contracts import PRESENT_ONLY_COLUMNS

KEYS = ["league_key", "season", "source_player_id"]

# Never inputs: identifiers, names, dates, columns that are only there to
# audit the point in time rule, and the snapshot's present day profile fields
# (the feature contract already rejects those; this is the second lock).
NOT_FEATURES = {
    *PRESENT_ONLY_COLUMNS,
    "source_player_id",
    "tm_player_id",
    "link_rule",
    "player_name",
    "team_id",
    "team_name",
    "nationality",
    "feature_date",
    "last_match_date",
    "value_date",
    "prev_feature_date",
    "last_comp_match_date",
    "last_injury_date",
    "last_transfer_date",
    "last_coach_change_date",
    "injury_source",
    "season",
}
CATEGORICAL = [
    "league_key",
    "tier",
    "position",
    "sub_position",
    "foot",
    "prev_league",
    "last_move_kind",
]
VALUE_COLUMNS = [
    "market_value",
    "market_value_1y_before",
    "value_change_1y",
    "market_value_max_to_date",
]


@dataclass(frozen=True)
class Target:
    name: str
    label: str  # column in the label table
    horizon_years: int  # how far after the feature date the label is observed
    label_date: str | None  # label table column holding the date it was observed
    uses_value_features: bool


TARGETS = {"fair_value": Target("fair_value", "value_now", 0, None, False)}


def feature_columns(features: pd.DataFrame, target: Target) -> list[str]:
    columns = [c for c in features.columns if c not in NOT_FEATURES]
    if not target.uses_value_features:
        columns = [c for c in columns if c not in VALUE_COLUMNS]
    return columns


def build_dataset(
    features: pd.DataFrame, labels: pd.DataFrame, target: Target
) -> tuple[pd.DataFrame, list[str]]:
    """Rows with a label, the target ``y``, and the list of feature columns."""
    label_cols = [*KEYS, target.label] + ([target.label_date] if target.label_date else [])
    data = features.merge(labels[label_cols], on=KEYS, how="inner")
    data = data[data["regular"].astype(bool)]
    if target.name == "fair_value":
        data = data.assign(y=np.log(data[target.label]))
    else:
        data = data.assign(y=data[target.label])
    data = data[np.isfinite(data["y"].astype(float))].reset_index(drop=True)
    columns = feature_columns(features, target)
    for column in CATEGORICAL:
        if column in data:
            data[column] = data[column].astype("category")
    return data, columns
