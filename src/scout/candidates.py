"""Elite chance and fall back risk for the elite candidates (decisions 1, 2).

Each candidate (value rank #101 to #1,000, a regular player) is shown with

* **elite chance:** the share of past candidates in the same quintile of
  the elite model's score (out of sample, from the season by season
  backtest) who entered the top 100 within 2 seasons;
* **fall back risk:** the share of past candidates with the same number of
  warning flags whose value rank was at least twice as bad 2 seasons later
  (``fall_back`` in ``scout.scouting``);
* **the flags** with their raw numbers.

Rates are historical frequencies, not model probabilities: a bucket with
fewer than 30 players is never shown, every rate is rounded to the nearest
5%, and the number of players behind it is carried along. Flags are counted
only from 2015, when per match data (minutes share, ability) starts; a
missing raw number raises no flag. The three year fall back is a check run
once, never shipped (decision 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scout.fsutil import atomic_write_text

log = logging.getLogger(__name__)

MIN_BUCKET = 30
ROUND_TO = 5
QUINTILES = 5
FLAG_START = 2015  # minutes share and ability come from per match data
COUNT_BUCKETS = ["0", "1", "2", "3+"]
ABILITY_SCAN = [20, 30, 40, 50, 55, 60, 70]


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


# Under ability score version 2 the ability flag does not change the risk at
# any percentile (reports/candidates.md); it stays at 55 for the owner to
# decide, and the count table is also shown without it.
FLAGS = {
    "low_minutes": Flag("minutes_share", False, 0.47, "under 47% of team minutes"),
    "low_ability": Flag("ability_pct", False, 55.0, "ability below the 55th percentile"),
    "injured": Flag("injury_matches_missed_12m", True, 5.0, "5 or more matches missed injured"),
    "value_falling": Flag("value_change_1y", False, 1.0, "value lower than a year before"),
}


def warning_flags(data: pd.DataFrame, flags: dict[str, Flag] = FLAGS) -> pd.DataFrame:
    """One boolean column per flag."""
    return pd.DataFrame({name: f.raised(data[f.column]) for name, f in flags.items()})


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


def flag_effects(y: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    """Per flag: how often it is raised and the fall back rate with and
    without it."""
    rows = []
    for name in flags.columns:
        on = flags[name]
        with_, without = y[on].mean(), y[~on].mean()
        rows.append(
            {
                "flag": name,
                "rule": FLAGS[name].rule if name in FLAGS else name,
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


def score_quintiles(score: pd.Series, season: pd.Series) -> pd.Series:
    """1 (lowest score) to 5 (highest) within each season."""
    pct = score.groupby(season).rank(pct=True, method="first")
    return np.ceil(pct * QUINTILES).clip(1, QUINTILES).astype(int)


@dataclass
class RiskTables:
    elite: pd.DataFrame  # pooled rate_table rows by score quintile
    fall_back: pd.DataFrame  # pooled rate_table rows by flag count bucket

    @staticmethod
    def pooled(table: pd.DataFrame) -> pd.DataFrame:
        return table[table["season"] == "all"].set_index("bucket")


def risk_tables(elite_predictions: pd.DataFrame, fall_back_rows: pd.DataFrame) -> RiskTables:
    """Pooled tables from the elite backtest predictions (``y``, ``score``,
    ``season``) and the candidates with a known fall back label and their
    flag columns."""
    q = score_quintiles(elite_predictions["score"], elite_predictions["season"])
    elite = rate_table(elite_predictions["y"], q.astype(str), elite_predictions["season"])
    count = warning_flags(fall_back_rows).sum(axis=1)
    fall = rate_table(fall_back_rows["fall_back"], count_bucket(count), fall_back_rows["season"])
    return RiskTables(RiskTables.pooled(elite), RiskTables.pooled(fall))


def _bucket_entry(table: pd.DataFrame, bucket: str) -> dict:
    if bucket not in table.index:
        return {"pct": None, "players": 0, "text": show_rate(None, 0)}
    row = table.loc[bucket]
    shown = pd.notna(row["rate"])
    return {
        "pct": int(row["pct"]) if shown else None,
        "players": int(row["players"]),
        "text": show_rate(row["rate"] if shown else None, int(row["players"])),
    }


def _raw_text(name: str, value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if name == "low_minutes":
        return f"{value:.0%} of team minutes"
    if name == "low_ability":
        return f"ability percentile {value:.0f}"
    if name == "injured":
        return f"{value:.0f} matches missed injured"
    return f"value {value - 1:+.0%} on a year before"


def candidate_profile(candidate: pd.Series | dict, quintile: int, tables: RiskTables) -> dict:
    """Elite chance from the candidate's score quintile, fall back risk from
    his flag count, and every flag with its raw number."""
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
                "text": _raw_text(name, value),
            }
        )
    count = sum(f["raised"] for f in flags)
    bucket = count_bucket(pd.Series([count])).iloc[0]
    return {
        "elite_chance": {"quintile": int(quintile), **_bucket_entry(tables.elite, str(quintile))},
        "fall_back_risk": {"flags": count, **_bucket_entry(tables.fall_back, bucket)},
        "flags": flags,
    }


def profiles(candidates: pd.DataFrame, score: str, tables: RiskTables) -> pd.DataFrame:
    """``candidate_profile`` for every row, flattened for tables and Parquet."""
    quintile = score_quintiles(candidates[score], candidates["season"])
    rows = []
    for (_, row), q in zip(candidates.iterrows(), quintile, strict=True):
        p = candidate_profile(row, q, tables)
        raised = [f"{f['rule']} ({f['text']})" for f in p["flags"] if f["raised"]]
        rows.append(
            {
                "elite_quintile": q,
                "elite_chance_pct": p["elite_chance"]["pct"],
                "elite_chance_n": p["elite_chance"]["players"],
                "flag_count": p["fall_back_risk"]["flags"],
                "fall_back_pct": p["fall_back_risk"]["pct"],
                "fall_back_n": p["fall_back_risk"]["players"],
                "elite_chance": f"Q{q}: {p['elite_chance']['text']}",
                "fall_back_risk": f"{p['fall_back_risk']['flags']} "
                f"flag{'' if p['fall_back_risk']['flags'] == 1 else 's'}: "
                f"{p['fall_back_risk']['text']}",
                "flags": "; ".join(raised) or "none",
            }
        )
    return pd.DataFrame(rows, index=candidates.index)


# Report


def _effects_md(effects: pd.DataFrame) -> str:
    out = effects.copy()
    for c in ("share_with", "rate_with", "rate_without"):
        out[c] = out[c].map(lambda v: f"{v:.1%}")
    out["ratio"] = out["ratio"].round(2)
    return out.to_markdown(index=False)


def fall_back_rows(data: pd.DataFrame, label: str = "fall_back") -> pd.DataFrame:
    rows = data[data[label].notna() & data["regular"].astype(bool) & (data["season"] >= FLAG_START)]
    return rows.assign(fall_back=rows[label])


def write_report(
    data: pd.DataFrame,
    elite_predictions: pd.DataFrame,
    current: pd.DataFrame,
    reports: Path,
    size: int = 25,
) -> Path:
    """``reports/candidates.md``: flag retest, rate tables, the three year
    check and the current candidates with their profiles. ``data`` carries
    ``fall_back`` and ``fall_back_3y``; ``current`` the profiled candidates."""
    rows = fall_back_rows(data)
    y = rows["fall_back"]
    flags = warning_flags(rows)
    count = flags.sum(axis=1)
    no_ability = flags.drop(columns="low_ability").sum(axis=1)
    elite_q = score_quintiles(elite_predictions["score"], elite_predictions["season"])
    elite_table = rate_table(
        elite_predictions["y"], elite_q.astype(str), elite_predictions["season"]
    )
    three = fall_back_rows(data, "fall_back_3y")
    flags3 = warning_flags(three)
    injury_known = rows[FLAGS["injured"].column].notna()
    show = [
        "player_name",
        "league_key",
        "team_name",
        "age",
        "value_rank",
        "elite_chance",
        "fall_back_risk",
        "flags",
    ]
    listed = current.sort_values("score_elite", ascending=False).head(size)
    parts = [
        "# Elite candidates: elite chance and fall back risk",
        "",
        "Decisions 1 and 2; code in `src/scout/candidates.py`. Candidates are regular",
        "players ranked #101 to #1,000 by value, the pool of the elite target.",
        "**Fall back:** two seasons later the value rank is at least twice as bad",
        "(#300 becomes #600 or worse). Ability score version 2. Flags from",
        f"{FLAG_START} (per match data); a missing raw number raises no flag",
        f"(injury history known for {injury_known.mean():.0%} of these players).",
        f"Buckets under {MIN_BUCKET} players are not shown; rates rounded to the nearest 5%.",
        "",
        f"Pool: {len(rows):,} candidates, seasons {int(rows['season'].min())} to "
        f"{int(rows['season'].max())}, fall back rate {y.mean():.1%}.",
        "",
        "## The four flags, 2 years",
        "",
        _effects_md(flag_effects(y, flags)),
        "",
        "## The ability flag at other percentiles (version 2)",
        "",
        _effects_md(ability_scan(y, rows["ability_pct"])),
        "",
        "## Fall back rate by flag count, 2 years",
        "",
        wide(rate_table(y, count_bucket(count), rows["season"]), COUNT_BUCKETS).to_markdown(),
        "",
        "Without the ability flag (three flags):",
        "",
        wide(rate_table(y, count_bucket(no_ability), rows["season"]), COUNT_BUCKETS).to_markdown(),
        "",
        "## Elite rate by model score quintile (5 is the highest score)",
        "",
        "Out of sample scores from the season by season backtest of the elite model;",
        "quintiles within each season.",
        "",
        wide(elite_table, [str(q) for q in range(1, QUINTILES + 1)]).to_markdown(),
        "",
        "## Check only: 3 years (not shipped)",
        "",
        f"{len(three):,} candidates, fall back rate {three['fall_back'].mean():.1%}.",
        "",
        _effects_md(flag_effects(three["fall_back"], flags3)),
        "",
        wide(
            rate_table(three["fall_back"], count_bucket(flags3.sum(axis=1)), three["season"]),
            COUNT_BUCKETS,
        ).to_markdown(),
        "",
        f"## Current candidates, top {size} by elite model score",
        "",
        listed[show].round({"age": 1}).to_markdown(index=False),
        "",
    ]
    path = reports / "candidates.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
