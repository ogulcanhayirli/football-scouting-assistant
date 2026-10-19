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

Rates are historical frequencies, not model probabilities: a bucket with
fewer than 30 players is never shown (the lowest deciles are merged until
each holds 30), every rate is rounded to the nearest 5%, and the number of
players behind it is carried along. Flags are counted only from 2015, when
per match data (minutes share, ability) starts; a missing raw number raises
no flag. The three year fall back is a check run once, never shipped
(decision 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scout.flags import ABILITY, ABILITY_MIN_RATIO, FLAGS, describe, warning_flags

log = logging.getLogger(__name__)

MIN_BUCKET = 30
ROUND_TO = 5
DECILES = 10
FLAG_START = 2015  # minutes share and ability come from per match data
COUNT_BUCKETS = ["0", "1", "2", "3+"]
ABILITY_SCAN = [20, 30, 40, 50, 55, 60, 70]


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


def ability_scan(y: pd.Series, ability: pd.Series, thresholds=ABILITY_SCAN) -> pd.DataFrame:
    """The ability flag at other percentiles."""
    flags = pd.DataFrame({f"ability below {t}": ability.astype(float) < t for t in thresholds})
    return flag_effects(y, flags).drop(columns="rule")


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


@dataclass
class RiskTables:
    elite: ChanceTable
    fall_back: pd.DataFrame  # pooled rate_table rows by flag count bucket


def risk_tables(elite_predictions: pd.DataFrame, fall_back_rows: pd.DataFrame) -> RiskTables:
    """From the elite backtest predictions (``y``, ``score``, ``season``) and
    the candidates with a known fall back label and their flag columns."""
    count = warning_flags(fall_back_rows).sum(axis=1)
    fall = rate_table(fall_back_rows["fall_back"], count_bucket(count), fall_back_rows["season"])
    return RiskTables(
        chance_table(elite_predictions), fall[fall["season"] == "all"].set_index("bucket")
    )


def _bucket_entry(table: pd.DataFrame, bucket: str) -> dict:
    if bucket not in table.index:
        return {"pct": None, "rate": None, "players": 0, "text": show_rate(None, 0)}
    row = table.loc[bucket]
    shown = pd.notna(row["rate"])
    return {
        "pct": int(row["pct"]) if shown else None,
        "rate": float(row["rate"]) if shown else None,
        "players": int(row["players"]),
        "text": show_rate(row["rate"] if shown else None, int(row["players"])),
    }


def candidate_profile(candidate: pd.Series | dict, decile: int, tables: RiskTables) -> dict:
    """Elite chance from the candidate's score decile, fall back risk from
    his flag count, and every flag with its raw number."""
    from scout.flags import raw_text

    row = pd.Series(candidate)
    flags = []
    for name, flag in FLAGS.items():
        value = pd.to_numeric(pd.Series([row.get(flag.column)]), errors="coerce").iloc[0]
        flags.append(
            {
                "flag": name,
                "rule": flag.rule,
                "value": None if pd.isna(value) else float(value),
                "raised": bool(flag.raised(pd.Series([value])).iloc[0]),
                "text": raw_text(name, value),
            }
        )
    count = sum(f["raised"] for f in flags)
    bucket = count_bucket(pd.Series([count])).iloc[0]
    elite_bucket = tables.elite.mapping[int(decile)]
    return {
        "elite_chance": {
            "decile": int(decile),
            "bucket": elite_bucket,
            **_bucket_entry(tables.elite.pooled, elite_bucket),
        },
        "fall_back_risk": {"flags": count, **_bucket_entry(tables.fall_back, bucket)},
        "flags": flags,
    }


def profiles(candidates: pd.DataFrame, score: str, tables: RiskTables) -> pd.DataFrame:
    """Elite chance, fall back risk and flags for every candidate, flat."""
    decile = score_deciles(candidates[score], candidates["season"])
    elite = tables.elite.lookup(decile)
    flags = describe(candidates)
    fall = count_bucket(flags["flag_count"]).map(lambda b: _bucket_entry(tables.fall_back, b))
    return pd.DataFrame(
        {
            "elite_decile": decile,
            "elite_bucket": elite["bucket"],
            "elite_chance_rate": elite["rate"],
            "elite_chance_pct": elite["pct"],
            "elite_chance_n": elite["players"],
            "flag_count": flags["flag_count"],
            "fall_back_rate": fall.map(lambda e: e["rate"]),
            "fall_back_pct": fall.map(lambda e: e["pct"]),
            "fall_back_n": fall.map(lambda e: e["players"]),
            "flags": flags["flags"],
        },
        index=candidates.index,
    )


def fall_back_rows(data: pd.DataFrame, label: str = "fall_back") -> pd.DataFrame:
    rows = data[data[label].notna() & data["regular"].astype(bool) & (data["season"] >= FLAG_START)]
    return rows.assign(fall_back=rows[label])


# Report


def _effects_md(effects: pd.DataFrame) -> str:
    out = effects.copy()
    for c in ("share_with", "rate_with", "rate_without"):
        out[c] = out[c].map(lambda v: f"{v:.1%}")
    out["ratio"] = out["ratio"].round(2)
    return out.to_markdown(index=False)


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
        "count_with_ability": rate_table(y, count_bucket(with_ability.sum(axis=1)), rows["season"]),
        "ability_ratio": ratio,
        "ability_passes": passes,
    }


def write_report(
    data: pd.DataFrame,
    elite_predictions: pd.DataFrame,
    current: pd.DataFrame,
    reports,
    size: int = 25,
):
    """``reports/candidates.md``: flag retest, rate tables, the three year
    check and the current candidates with their profiles. ``data`` carries
    ``fall_back`` and ``fall_back_3y``; ``current`` the profiled candidates."""
    from scout.fsutil import atomic_write_text

    s = pool_summary(data)
    rows, y = s["rows"], s["y"]
    elite = chance_table(elite_predictions)
    three = fall_back_rows(data, "fall_back_3y")
    flags3 = warning_flags(three).assign(low_ability=ABILITY.raised(three[ABILITY.column]))
    injury_known = rows[FLAGS["injured"].column].notna()
    show = [
        "player_name",
        "league_key",
        "team_name",
        "age",
        "value_rank",
        "ability_pct",
        "elite_chance",
        "fall_back_risk",
        "flags",
    ]
    listed = current.sort_values("score_elite", ascending=False).head(size)
    listed = listed.assign(
        elite_chance=[
            f"decile {int(d)}: {show_rate(r if pd.notna(r) else None, int(n))}"
            for d, r, n in zip(
                listed["elite_decile"],
                listed["elite_chance_rate"],
                listed["elite_chance_n"],
                strict=True,
            )
        ],
        fall_back_risk=[
            f"{int(c)} flag{'' if c == 1 else 's'}: {show_rate(r if pd.notna(r) else None, int(n))}"
            for c, r, n in zip(
                listed["flag_count"], listed["fall_back_rate"], listed["fall_back_n"], strict=True
            )
        ],
    )
    verdict = "passes" if s["ability_passes"] else "fails"
    parts = [
        "# Elite candidates: elite chance and fall back risk",
        "",
        "Decisions 1 and 2; code in `src/scout/candidates.py` and `src/scout/flags.py`.",
        "Candidates are regular players ranked #101 to #1,000 by value and 23 or under",
        "at the feature date, the pool of the elite target. **Fall back:** two seasons",
        "later the value rank is at least twice as bad (#300 becomes #600 or worse).",
        f"Ability score version 2. Flags from {FLAG_START} (per match data); a missing raw",
        f"number raises no flag (injury history known for {injury_known.mean():.0%} of these",
        f"players). Buckets under {MIN_BUCKET} players are not shown; rates rounded to the",
        "nearest 5%.",
        "",
        f"Pool: {len(rows):,} candidates, seasons {int(rows['season'].min())} to "
        f"{int(rows['season'].max())}, fall back rate {y.mean():.1%}.",
        "",
        "## The flags, 2 years",
        "",
        _effects_md(s["effects"]),
        "",
        f"The ability flag raises the risk {s['ability_ratio']:.2f} times; the owner's rule "
        f"needs {ABILITY_MIN_RATIO}, so it {verdict}. It is "
        + ("counted." if s["ability_passes"] else "left out of the count and shown as a column."),
        "",
        "## The ability flag at other percentiles (version 2)",
        "",
        _effects_md(ability_scan(y, rows[ABILITY.column])),
        "",
        "## Fall back rate by flag count, 2 years",
        "",
        wide(s["count"], COUNT_BUCKETS).to_markdown(),
        "",
        "With the ability flag as a fourth flag (check):",
        "",
        wide(s["count_with_ability"], COUNT_BUCKETS).to_markdown(),
        "",
        "## Elite rate by model score decile (10 is the highest score)",
        "",
        "Out of sample scores from the season by season backtest of the elite model;",
        f"deciles within each season, the lowest merged until each holds {MIN_BUCKET} players.",
        "",
        wide(elite.table, elite.buckets).to_markdown(),
        "",
        "## Check only: 3 years (not shipped)",
        "",
        f"{len(three):,} candidates, fall back rate {three['fall_back'].mean():.1%}.",
        "",
        _effects_md(flag_effects(three["fall_back"], flags3)),
        "",
        f"## Current candidates, top {size} by elite model score",
        "",
        listed[show].round({"age": 1, "ability_pct": 0}).to_markdown(index=False),
        "",
    ]
    path = reports / "candidates.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
