"""Modelling datasets: one per target, built from the feature and label tables.

Targets are on a log scale, so errors are relative: an error of 0.1 means
about 10%, whether the player is worth 1m or 100m.

* ``fair_value``: log of the market value at the feature date. Every
  ``market_value*`` column is left out of its features, because
  ``market_value`` is the target itself.
* ``growth_24m`` and ``growth_36m``: log of the value 24 or 36 months later
  divided by the value now. The current value is a legitimate input here.

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


TARGETS = {
    "fair_value": Target("fair_value", "value_now", 0, None, False),
    "growth_24m": Target("growth_24m", "log_growth_24m", 2, "value_24m_date", True),
    "growth_36m": Target("growth_36m", "log_growth_36m", 3, "value_36m_date", True),
}


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
