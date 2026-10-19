"""One set of warning flags for both lists (decisions 2 and 4).

* **Core flags** count towards the fall back risk and are shown on the
  candidates and the bargains: low minutes, injured, value falling. Each
  carries its raw number; a missing number raises no flag.
* **The ability flag** was dropped from the count on 19 October 2026 (prompt
  12): on the elite pool it raised the fall back risk 1.26 times, under the
  owner's ``ABILITY_MIN_RATIO`` of 1.3, so the ability percentile is shown
  as a plain column instead. ``ABILITY`` stays here so the report retests
  it every run.
* **Context tags** are shown on the bargains only and never counted: on loan
  (the parent club sells, at its price) and stale value (the market value is
  old, so the cheap price may simply be out of date).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Flag:
    column: str
    above: bool  # True: raised at or above the threshold; False: below it
    threshold: float
    rule: str

    def raised(self, values: pd.Series) -> pd.Series:
        values = values.astype(float)
        hit = values >= self.threshold if self.above else values < self.threshold
        return hit & values.notna()


FLAGS = {
    "low_minutes": Flag("minutes_share", False, 0.47, "under 47% of team minutes"),
    "injured": Flag("injury_matches_missed_12m", True, 5.0, "5 or more matches missed injured"),
    "value_falling": Flag("value_change_1y", False, 1.0, "value lower than a year before"),
}
ABILITY = Flag("ability_pct", False, 55.0, "ability below the 55th percentile")
ABILITY_MIN_RATIO = 1.3  # the owner's rule: a flag must raise the risk at least this much

STALE_DAYS = 180  # market value dated this long before the feature date
TAGS = {
    "on_loan": "on loan: the parent club decides the sale and the price",
    "stale_value": f"stale value: dated more than {STALE_DAYS} days before the feature date",
}


def warning_flags(data: pd.DataFrame, flags: dict[str, Flag] = FLAGS) -> pd.DataFrame:
    """One boolean column per flag."""
    return pd.DataFrame({name: f.raised(data[f.column]) for name, f in flags.items()})


def raw_text(name: str, value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if name == "low_minutes":
        return f"{value:.0%} of team minutes"
    if name == "injured":
        return f"{value:.0f} matches missed injured"
    if name == "value_falling":
        return f"value {value - 1:+.0%} on a year before"
    return f"ability percentile {value:.0f}"


def describe(data: pd.DataFrame, flags: dict[str, Flag] = FLAGS) -> pd.DataFrame:
    """``flag_count`` and ``flags``, a readable list of the raised flags with
    their raw numbers ("none" when no flag is raised)."""
    raised = warning_flags(data, flags)
    texts = []
    for idx in data.index:
        parts = [
            f"{flag.rule} ({raw_text(name, data.at[idx, flag.column])})"
            for name, flag in flags.items()
            if raised.at[idx, name]
        ]
        texts.append("; ".join(parts) or "none")
    return pd.DataFrame(
        {"flag_count": raised.sum(axis=1).astype(int), "flags": texts}, index=data.index
    )


def context_tags(data: pd.DataFrame) -> pd.DataFrame:
    """``on_loan`` and ``stale_value`` as booleans, and ``stale_days``."""
    days = (data["feature_date"] - data["value_date"]).dt.days
    return pd.DataFrame(
        {
            "on_loan": (data["on_loan"].astype(float) == 1).fillna(False).astype(bool),
            "stale_value": (days > STALE_DAYS).fillna(False).astype(bool),
            "stale_days": days,
        },
        index=data.index,
    )
