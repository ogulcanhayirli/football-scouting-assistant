"""Elite chance and fall back risk for the elite candidates (decisions 1, 2).

The elite pool (``scout.scouting.elite_pool``): regular players ranked #101
to #1,000 by value and 23 or under at the feature date. Each candidate is
shown with

* **elite chance:** the share of past candidates in the same decile of the
  elite model's score (out of sample, from the season by season backtest)
  who entered the top 100 within 2 seasons;
* **fall back risk:** the share of past candidates with the same number of
  warning flags whose value rank was at least twice as bad 2 seasons later
  (``fall_back`` in ``scout.scouting``);
* **the flags** with their raw numbers (``scout.flags``).

``scout.lists`` puts these on the candidates table.

Rates are historical frequencies, not model probabilities: a bucket with
fewer than 30 players is never shown (the lowest deciles are merged until
each holds 30), every rate is rounded to the nearest 5%, and the number of
players behind it is carried along. Flags are counted only from 2015, when
per match data (minutes share, ability) starts; a missing raw number raises
no flag. The three year fall back was a check run once and is not shipped
(decision 1; MODELS.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scout.flags import ABILITY, ABILITY_MIN_RATIO, FLAGS, warning_flags

log = logging.getLogger(__name__)

MIN_BUCKET = 30
ROUND_TO = 5
DECILES = 10
FLAG_START = 2015  # minutes share and ability come from per match data
COUNT_BUCKETS = ["0", "1", "2", "3+"]


def count_bucket(count: pd.Series) -> pd.Series:
    return count.clip(upper=3).astype(int).astype(str).replace("3", "3+")


def rounded(rate: float) -> int:
    """Nearest 5%."""
    return int(ROUND_TO * np.floor(rate * 100 / ROUND_TO + 0.5))


def show_rate(rate: float | None, players: int) -> str:
    if rate is None or players < MIN_BUCKET:
        return f"fewer than {MIN_BUCKET} (n={players})"
    pct = rounded(rate)
    text = "under 5%" if pct == 0 and rate > 0 else f"{pct}%"
    return f"{text} (n={players:,})"


def rate_table(y: pd.Series, bucket: pd.Series, season: pd.Series) -> pd.DataFrame:
    """Rate of ``y`` per bucket, per season and pooled (season ``"all"``):
    columns season, bucket, players, rate, pct. ``rate`` and ``pct`` are
    missing for buckets under ``MIN_BUCKET`` players."""
    frame = pd.DataFrame({"y": y.astype(float), "bucket": bucket, "season": season}).dropna()
    per = frame.groupby(["season", "bucket"])["y"].agg(players="size", rate="mean").reset_index()
    pooled = frame.groupby("bucket")["y"].agg(players="size", rate="mean").reset_index()
    out = pd.concat([per, pooled.assign(season="all")], ignore_index=True)
    out["season"] = out["season"].astype(str).str.replace(r"\.0$", "", regex=True)
    small = out["players"] < MIN_BUCKET
    out.loc[small, "rate"] = np.nan
    out["pct"] = out["rate"].map(lambda r: np.nan if pd.isna(r) else rounded(r))
    return out


def wide(table: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    """Seasons as rows, buckets as columns, cells such as ``45% (n=123)``."""
    cells = table.assign(
        cell=[
            show_rate(None if pd.isna(r) else r, int(n))
            for r, n in zip(table["rate"], table["players"], strict=True)
        ]
    )
    out = cells.pivot(index="season", columns="bucket", values="cell")
    out = out.reindex(columns=[b for b in buckets if b in out.columns]).fillna("none")
    order = sorted(i for i in out.index if i != "all") + (["all"] if "all" in out.index else [])
    return out.reindex(order)


def flag_effects(y: pd.Series, flags: pd.DataFrame, rules: dict[str, str] | None = None):
    """Per flag: how often it is raised and the fall back rate with and
    without it."""
    rules = rules or {name: f.rule for name, f in FLAGS.items()} | {"low_ability": ABILITY.rule}
    rows = []
    for name in flags.columns:
        on = flags[name]
        with_, without = y[on].mean(), y[~on].mean()
        rows.append(
            {
                "flag": name,
                "rule": rules.get(name, name),
                "players_with": int(on.sum()),
                "share_with": on.mean(),
                "rate_with": with_,
                "rate_without": without,
                "ratio": with_ / without if without > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ability_decision(y: pd.Series, ability: pd.Series) -> tuple[float, bool]:
    """The ability flag's risk ratio and whether it passes the owner's rule."""
    ratio = flag_effects(y, pd.DataFrame({"low_ability": ABILITY.raised(ability)}))["ratio"].iloc[0]
    return float(ratio), bool(ratio >= ABILITY_MIN_RATIO)


def score_deciles(score: pd.Series, season: pd.Series) -> pd.Series:
    """1 (lowest score) to 10 (highest) within each season."""
    pct = score.groupby(season).rank(pct=True, method="first")
    return np.ceil(pct * DECILES).clip(1, DECILES).astype(int)


def merge_low(counts: pd.Series) -> dict[int, str]:
    """Bucket label per decile: the lowest deciles are merged until each
    bucket holds ``MIN_BUCKET`` players ("1 to 3"); a short top remainder
    joins the bucket below it."""
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0
    for decile in range(1, DECILES + 1):
        current.append(decile)
        total += int(counts.get(decile, 0))
        if total >= MIN_BUCKET:
            groups.append(current)
            current, total = [], 0
    if current:
        if groups:
            groups[-1] += current
        else:
            groups.append(current)
    return {d: str(g[0]) if len(g) == 1 else f"{g[0]} to {g[-1]}" for g in groups for d in g}


@dataclass
class ChanceTable:
    """Historical rate per score decile, merged at the bottom."""

    mapping: dict[int, str]  # decile -> bucket label
    table: pd.DataFrame  # rate_table rows, per season and pooled

    @property
    def pooled(self) -> pd.DataFrame:
        return self.table[self.table["season"] == "all"].set_index("bucket")

    @property
    def buckets(self) -> list[str]:
        return list(dict.fromkeys(self.mapping[d] for d in range(1, DECILES + 1)))

    def lookup(self, deciles: pd.Series) -> pd.DataFrame:
        """bucket, rate, pct and players for each decile."""
        bucket = deciles.map(self.mapping)
        pooled = self.pooled
        return pd.DataFrame(
            {
                "bucket": bucket,
                "rate": bucket.map(pooled["rate"]),
                "pct": bucket.map(pooled["pct"]),
                "players": bucket.map(pooled["players"]).fillna(0).astype(int),
            },
            index=deciles.index,
        )


def chance_table(predictions: pd.DataFrame, score: str = "score") -> ChanceTable:
    """From backtest predictions (``y``, ``season`` and the score column)."""
    deciles = score_deciles(predictions[score], predictions["season"])
    mapping = merge_low(deciles.value_counts())
    table = rate_table(predictions["y"], deciles.map(mapping), predictions["season"])
    return ChanceTable(mapping, table)


def fall_back_rows(data: pd.DataFrame, label: str = "fall_back") -> pd.DataFrame:
    rows = data[data[label].notna() & data["regular"].astype(bool) & (data["season"] >= FLAG_START)]
    return rows.assign(fall_back=rows[label])


# The fall back pool


def pool_summary(data: pd.DataFrame) -> dict:
    """The fall back pool, its base rate, the flag effects and the count
    table, with the ability flag retested against the owner's rule."""
    rows = fall_back_rows(data)
    y = rows["fall_back"]
    flags = warning_flags(rows)
    with_ability = flags.assign(low_ability=ABILITY.raised(rows[ABILITY.column]))
    ratio, passes = ability_decision(y, rows[ABILITY.column])
    return {
        "rows": rows,
        "y": y,
        "effects": flag_effects(y, with_ability),
        "count": rate_table(y, count_bucket(flags.sum(axis=1)), rows["season"]),
        "ability_ratio": ratio,
        "ability_passes": passes,
    }
