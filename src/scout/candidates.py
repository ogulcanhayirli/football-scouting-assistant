"""Elite chance and fall back risk for the elite candidates (decisions 1, 2).

The elite pool (``scout.scouting.elite_pool``): regular players ranked #101
to #1,000 by value and 23 or under at the feature date. Each candidate is
shown with

* **elite chance:** the share of past candidates in the same decile of the
  elite model's score who entered the top 100 within 2 seasons;
* **fall back risk:** the share of past candidates with the same number of
  warning flags whose value rank was at least twice as bad 2 seasons later
  (``fall_back`` in ``scout.scouting``);
* **the flags** with their raw numbers (``scout.flags``).

``scout.lists`` puts these on the candidates table.

Two ways to measure a bucket's rate live here. ``chance_table`` pools the
backtest's out of sample rows of every test season (the shown rate until
prompt 14b, kept for comparison). ``realized_tables`` (prompt 14b, the shown
rate since) reads the past forecasts rebuilt by ``scout forecast history``,
each built with the models and buckets available at its time and scored two
seasons on: the rate a user of those tables would actually have seen come
true. The second is what the table promises; the first ran high at the top
of the elite chance (40% shown, 30.5% happened).

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
# Two or more flags is one bucket (prompt 14): the 3+ bucket held few players
# and its rate dipped below the bucket of 2.
COUNT_BUCKETS = ["0", "1", "2 or more"]


def count_bucket(count: pd.Series) -> pd.Series:
    return count.clip(upper=2).astype(int).astype(str).replace("2", "2 or more")


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
    """Historical rate per score decile, merged at the bottom (or per flag
    count bucket)."""

    mapping: dict  # decile (or flag count bucket) -> bucket label, in order
    table: pd.DataFrame  # rate_table rows, per season and pooled

    @property
    def pooled(self) -> pd.DataFrame:
        return self.table[self.table["season"] == "all"].set_index("bucket")

    @property
    def buckets(self) -> list[str]:
        return list(dict.fromkeys(self.mapping.values()))

    def lookup(self, deciles: pd.Series) -> pd.DataFrame:
        """bucket, rate, pct and players for each decile (or flag count
        bucket)."""
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


def count_table(y: pd.Series, bucket: pd.Series, season: pd.Series) -> ChanceTable:
    """Rate per flag count bucket (0, 1, 2 or more)."""
    return ChanceTable({b: b for b in COUNT_BUCKETS}, rate_table(y, bucket, season))


# Realized rates of past forecasts (prompt 14b)

# promise -> (bucket column of a scored forecast, outcome column)
REALIZED = {
    "elite": ("elite_decile", "elite"),
    "step_up": ("step_up_decile", "step_up"),
    "big_move": ("big_move_decile", "big_move"),
    "fall_back": ("fall_back_bucket", "fall_back"),
}


def realized_tables(outcomes: pd.DataFrame, through: int | None = None) -> dict[str, ChanceTable]:
    """Per promise, the share of past forecast players in each bucket whose
    outcome came true, among those whose outcome is known. ``outcomes``: the
    scored rows of ``scout forecast history`` (season, the bucket and
    outcome columns of ``REALIZED``); ``through``: forecast seasons up to it
    only (a forecast for season S uses S minus 2, the last one scored by
    then). The lowest deciles are merged until each bucket holds
    ``MIN_BUCKET`` known outcomes."""
    rows = outcomes if through is None else outcomes[outcomes["season"] <= through]
    out = {}
    for name, (bucket, outcome) in REALIZED.items():
        known = rows[rows[bucket].notna() & rows[outcome].notna()]
        if name == "fall_back":
            out[name] = count_table(known[outcome], known[bucket], known["season"])
            continue
        deciles = known[bucket].astype(int)
        mapping = merge_low(deciles.value_counts())
        out[name] = ChanceTable(
            mapping, rate_table(known[outcome], deciles.map(mapping), known["season"])
        )
    return out


def not_monotonic(table: ChanceTable) -> list[str]:
    """Sentences for each shown bucket whose pooled rate is below the one
    before it (the buckets run from least to most likely). Reported, never
    smoothed."""
    pooled = table.pooled.reindex(table.buckets).dropna(subset=["rate"])
    out = []
    for (low, a), (high, b) in zip(pooled.iterrows(), list(pooled.iterrows())[1:], strict=False):
        if b["rate"] < a["rate"]:
            out.append(
                f"{high} ({b['rate']:.1%}, n={int(b['players']):,}) is below {low} "
                f"({a['rate']:.1%}, n={int(a['players']):,})"
            )
    return out


def compare(old: ChanceTable, new: ChanceTable, keys: list) -> pd.DataFrame:
    """Old against new per decile (or flag count bucket): each table's
    bucket, its shown cell and, for the new one, the unrounded rate."""
    o, n = old.pooled, new.pooled
    rows = []
    for key in keys:
        ob, nb = old.mapping.get(key), new.mapping.get(key)
        row = {"decile" if isinstance(key, int | np.integer) else "flags": key}
        for label, b, pooled in (("old", ob, o), ("new", nb, n)):
            found = b is not None and b in pooled.index
            rate = pooled.loc[b, "rate"] if found else None
            players = int(pooled.loc[b, "players"]) if found else 0
            row[f"{label} bucket"] = b or ""
            row[label] = show_rate(None if rate is None or pd.isna(rate) else rate, players)
        nr = n.loc[nb, "rate"] if nb is not None and nb in n.index else np.nan
        row["new rate"] = "" if pd.isna(nr) else f"{nr:.1%}"
        rows.append(row)
    return pd.DataFrame(rows)


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
