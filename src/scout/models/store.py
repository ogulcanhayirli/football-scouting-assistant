"""Stored models: the boosters behind the newest season's scores.

``scout backtest``, ``scout moves`` and ``scout scouting`` train them and save
them here (``data/processed/models/stored/``); ``scout score``, which the
weekly run calls, loads them and scores the newest season without training
anything. Retraining stays a manual command (docs/DEPLOY.md).

Each model is two files: ``<name>.txt``, LightGBM's own text format (it keeps
the category lists, so a column's codes are mapped back by value when the
newest rows bring a category the model never saw), and ``<name>.json`` with
the feature columns in order and what the model was trained on. The ability
score version 2 fit (measures and weights, league effects, team coefficient,
prior minutes) is stored the same way in ``ability_v2.json``, so the newest
season is scored on the scale the models learned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scout.fsutil import atomic_write_text

STORE = "stored"

# The fair value model of each fold that holds rows of the newest season:
# European rows (feature date 30 June) sit in the newest fold, calendar year
# leagues (1 January) in the fold before it (scout.models.backtest.fold_season).
FAIR_VALUE = "fair_value_fold_{fold}"
BIG_MOVE = "big_move_24m"


def store_dir(processed_dir: Path) -> Path:
    return Path(processed_dir) / "models" / STORE


def save_booster(booster: lgb.Booster, folder: Path, name: str, columns: list[str], **meta) -> Path:
    """Write ``<name>.txt`` and ``<name>.json``; returns the model file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.txt"
    atomic_write_text(path, booster.model_to_string())
    info = {"name": name, "columns": list(columns), **meta}
    atomic_write_text(folder / f"{name}.json", json.dumps(info, indent=2, default=str) + "\n")
    return path


@dataclass(frozen=True)
class Stored:
    booster: lgb.Booster
    columns: list[str]
    meta: dict

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(rows[self.columns])


def load_booster(folder: Path, name: str) -> Stored:
    path = Path(folder) / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: train it once with the manual commands (docs/DEPLOY.md, "
            "Retraining) and sync it up"
        )
    meta = json.loads((Path(folder) / f"{name}.json").read_text())
    booster = lgb.Booster(model_str=path.read_text())
    return Stored(booster, meta.pop("columns"), meta)


def stored_folds(folder: Path, pattern: str = FAIR_VALUE) -> list[int]:
    """Fold seasons with a stored fair value model, oldest first."""
    prefix, suffix = pattern.split("{fold}")
    folds = []
    for path in Path(folder).glob(f"{prefix}*{suffix}.txt"):
        middle = path.stem[len(prefix) : len(path.stem) - len(suffix)]
        if middle.isdigit():
            folds.append(int(middle))
    return sorted(folds)


# The ability score version 2 fit


def _native(value):
    """numpy scalars as plain Python numbers, for json."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


@dataclass(frozen=True)
class AbilityFit:
    """Everything ability version 2 learns from the data, so a later season
    is scored on the same scale: composite measures and weights per position
    group, league effects, the Elo team coefficient and the prior minutes."""

    weights: pd.DataFrame
    strength: pd.DataFrame
    team_coef: float
    prior: dict[str, float]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "weights": self.weights.to_dict("records"),
                    "strength": self.strength.to_dict("records"),
                    "team_coef": self.team_coef,
                    "prior": self.prior,
                },
                indent=2,
                default=_native,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> AbilityFit:
        data = json.loads(text)
        return cls(
            pd.DataFrame(data["weights"]),
            pd.DataFrame(data["strength"]),
            float(data["team_coef"]),
            {k: float(v) for k, v in data["prior"].items()},
        )


ABILITY = "ability_v2.json"


def save_ability(fit: AbilityFit, folder: Path) -> Path:
    path = Path(folder) / ABILITY
    atomic_write_text(path, fit.to_json())
    return path


def load_ability(folder: Path) -> AbilityFit:
    path = Path(folder) / ABILITY
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `scout scouting` once and sync it up")
    return AbilityFit.from_json(path.read_text())
