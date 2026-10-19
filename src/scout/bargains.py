"""Bargains (decision 4): players already good whose market value sits well
below what the fair value model expects, and a backtest of whether such
lists paid off.

* **Score** is the actual market value divided by the fair value model's
  prediction (``ratio``; 0.5 means half of what the model expects). The
  prediction is the out of time one from ``scout backtest``
  (``backtest_fair_value.parquet``): every season, the current one included,
  is predicted by a model trained only on earlier seasons, so a player's own
  value never shapes his prediction.
* **Eligible**: a regular (450+ minutes) with ability score version 2 at the
  70th percentile of his position or better, and at least half of his team's
  minutes. One row per player and season (the league with most minutes).
* **Ranked within position group**: ``ratio_pct`` is the ratio's percentile
  among eligible players of the same position group and season (low is
  cheap), so goalkeepers, whom the market values low, do not fill the list.
* **Split** by age at the feature date: prime (29 and under) and experienced
  (30 to 33). Players of 34 and over are not listed: the market has written
  them off for a reason a value model cannot see.
* **Contract expiry** comes from ``player_profiles.parquet`` (the snapshot
  of the Transfermarkt release date) and is shown only when it ends after
  that date. It is display only, never an input.
* **Warning flags** point at reasons the market may be right; each carries
  its raw number. They are the core flags of the candidates (``scout.flags``),
  plus two context tags shown on this list only: on loan and stale value.
* **Backtest**: in each past season, the top 50 of each list against players
  of the same season, ability band and age band who passed the same filters
  but were not in the top 50: did their value rise over 24 months, and did
  they join a club with a higher Elo within two seasons?
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scout import flags
from scout.fsutil import atomic_write_text
from scout.scouting import AGE_BANDS, AGE_LABELS, KEYS

log = logging.getLogger(__name__)

MIN_ABILITY_PCT = 70.0
MIN_MINUTES_SHARE = 0.5
PRIME_MAX_AGE = 30  # prime: under 30 at the feature date, so 29 and under
EXPERIENCED_MAX_AGE = 34  # experienced: 30 to 33; older players are not listed
TOP_K = 50
HORIZON_SEASONS = 2  # stronger club within this many seasons
ABILITY_BANDS = [MIN_ABILITY_PCT, 80.0, 90.0, 100.01]
ABILITY_LABELS = ["70 to 79", "80 to 89", "90 to 100"]
LISTS = {"prime": "Prime (29 and under)", "experienced": "Experienced (30 to 33)"}


def warning_flags(data: pd.DataFrame) -> pd.DataFrame:
    """The shared core flags (``scout.flags``) with ``flag_count`` and the
    readable ``flags``, and the two context tags of this list."""
    raised = flags.warning_flags(data).add_prefix("flag_")
    return pd.concat([raised, flags.describe(data), flags.context_tags(data)], axis=1)


def eligible(data: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass the filters, with ``ratio``, ``ratio_pct``, ``list`` and
    the flags. ``data`` needs ``value`` and ``predicted`` (EUR), ``group`` and
    ``ability_pct`` besides the feature columns."""
    rows = data[data["regular"].astype(bool) & data["group"].notna()]
    rows = rows.sort_values("minutes").drop_duplicates(["season", "source_player_id"], keep="last")
    rows = rows[
        (rows["ability_pct"] >= MIN_ABILITY_PCT)
        & (rows["minutes_share"].astype(float) >= MIN_MINUTES_SHARE)
        & (rows["age"].astype(float) < EXPERIENCED_MAX_AGE)
        & (rows["value"] > 0)
        & (rows["predicted"] > 0)
    ].copy()
    rows["ratio"] = rows["value"] / rows["predicted"]
    rows["ratio_pct"] = (
        rows.groupby(["season", "group"], observed=True)["ratio"].rank(pct=True).mul(100)
    )
    rows["list"] = np.where(rows["age"].astype(float) < PRIME_MAX_AGE, "prime", "experienced")
    rows["ability_band"] = pd.cut(
        rows["ability_pct"], ABILITY_BANDS, right=False, labels=ABILITY_LABELS
    )
    rows["age_band"] = pd.cut(rows["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    flagged = warning_flags(rows)  # the on_loan tag replaces the 0 or 1 feature column
    rows = pd.concat([rows.drop(columns=flagged.columns, errors="ignore"), flagged], axis=1)
    return rows.sort_values(["season", "list", "ratio_pct", "ratio"]).reset_index(drop=True)


def ranked(rows: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Each season's top ``k`` of each list, ``list_rank`` from 1."""
    rows = rows.sort_values(["ratio_pct", "ratio"])
    rows = rows.assign(list_rank=rows.groupby(["season", "list"]).cumcount() + 1)
    return rows[rows["list_rank"] <= k]


# Outcomes


def outcomes(features: pd.DataFrame, labels: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Per feature row: ``gained_value`` (value 24 months later above the
    value now; missing where that value is not observed yet) and
    ``stronger_club`` (within two seasons he plays league minutes for a
    different club whose season Elo is above his club's Elo now; missing when
    his club has no Elo or the two seasons are not all in the data). A move to
    a league outside the data counts as no stronger club."""
    keys = ["league_key", "season", "team_id"]
    elo = teams[[*keys, "elo"]].assign(
        season=lambda d: d["season"].astype(int), team_id=lambda d: d["team_id"].astype("Int64")
    )
    rows = features[[*KEYS, "team_id"]].copy()
    rows["season"] = rows["season"].astype(int)
    rows["team_id"] = rows["team_id"].astype("Int64")
    rows = rows.merge(elo, on=keys, how="left")

    later = rows[["source_player_id", "season", "team_id", "elo"]].dropna(subset=["elo"])
    base = rows[[*KEYS, "team_id", "elo"]].rename(
        columns={"team_id": "base_team", "elo": "base_elo"}
    )
    pairs = base.dropna(subset=["base_elo"]).merge(
        later, on="source_player_id", suffixes=("", "_later")
    )
    ahead = pairs["season_later"] - pairs["season"]
    pairs = pairs[ahead.between(1, HORIZON_SEASONS)]
    hit = pairs[(pairs["team_id"] != pairs["base_team"]) & (pairs["elo"] > pairs["base_elo"])]
    hit = hit.drop_duplicates(KEYS).assign(stronger=True)[[*KEYS, "stronger"]]

    out = base.merge(hit, on=KEYS, how="left")
    last_season = int(rows["season"].max())
    known = out["base_elo"].notna() & (out["season"] + HORIZON_SEASONS <= last_season)
    out["stronger_club"] = np.where(known, out["stronger"].fillna(False).astype(float), np.nan)

    lab = labels[[*KEYS, "log_growth_24m"]].assign(season=lambda d: d["season"].astype(int))
    out = out.merge(lab, on=KEYS, how="left")
    growth = out["log_growth_24m"].astype(float)
    out["gained_value"] = np.where(growth.notna(), (growth > 0).astype(float), np.nan)
    return out[[*KEYS, "gained_value", "stronger_club"]]


OUTCOMES = {"gained_value": "value up after 24 months", "stronger_club": "club with higher Elo"}


def backtest(rows: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Per list, season and outcome: the top ``k``'s rate against the matched
    rate, the mean over the top ``k`` of the rate among the other eligible
    players of the same season, ability band and age band. A ``pooled`` row
    sums every season (``season`` is text); ``z`` compares hits with the
    matched expectation."""
    top = ranked(rows, k)
    is_top = rows.index.isin(top.index)
    cells = ["season", "ability_band", "age_band"]
    records = []
    for name in LISTS:
        for outcome in OUTCOMES:
            pool = rows[~is_top & rows[outcome].notna()]
            rate = pool.groupby(cells, observed=True)[outcome].agg(["mean", "size"])
            picked = top[(top["list"] == name) & top[outcome].notna()]
            picked = picked.join(rate, on=cells).dropna(subset=["mean"])
            for season, group in picked.groupby("season"):
                records.append(_record(name, outcome, str(int(season)), group, outcome))
            if len(picked):
                records.append(_record(name, outcome, "pooled", picked, outcome))
    return pd.DataFrame(records)


def _record(name: str, outcome: str, season, group: pd.DataFrame, column: str) -> dict:
    p = group["mean"].astype(float)
    hits, expected = float(group[column].sum()), float(p.sum())
    spread = float(np.sqrt((p * (1 - p)).sum()))
    return {
        "list": name,
        "outcome": outcome,
        "season": season,
        "players": len(group),
        "rate": hits / len(group),
        "matched_rate": expected / len(group),
        "difference": (hits - expected) / len(group),
        "z": (hits - expected) / spread if spread > 0 else np.nan,
    }


# Current lists


def current_lists(
    rows: pd.DataFrame, profiles: pd.DataFrame, snapshot: pd.Timestamp | None
) -> dict[str, pd.DataFrame]:
    """The newest season's eligible players of each list, cheapest first,
    with contract expiry where it ends after the snapshot date."""
    newest = rows[rows["season"] == rows["season"].max()]
    contract = profiles.set_index("af_player_id")["contract_expiry"]
    expiry = newest["source_player_id"].map(contract)
    expiry = expiry.where(expiry > snapshot) if snapshot is not None else expiry.where(False)
    newest = newest.assign(contract_expiry=expiry)
    out = {}
    for name in LISTS:
        frame = newest[newest["list"] == name].sort_values(["ratio_pct", "ratio"])
        out[name] = frame.assign(list_rank=np.arange(1, len(frame) + 1)).reset_index(drop=True)
    return out


SHOW = [
    "list_rank",
    "player_name",
    "league_key",
    "team_name",
    "group",
    "age",
    "ability_pct",
    "value_m",
    "predicted_m",
    "ratio",
    "ratio_pct",
    "minutes_share",
    "contract_expiry",
    "flag_count",
    "flags",
    "on_loan",
    "stale_value",
]


def readable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.assign(
        value_m=frame["value"] / 1e6,
        predicted_m=frame["predicted"] / 1e6,
        contract_expiry=frame["contract_expiry"].dt.strftime("%Y-%m-%d").fillna(""),
    )[SHOW]
    return out.round(
        {
            "age": 1,
            "ability_pct": 0,
            "value_m": 2,
            "predicted_m": 2,
            "ratio": 2,
            "ratio_pct": 1,
            "minutes_share": 2,
        }
    )


# Running it


def load(settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    """(rows with value, prediction, ability and outcomes; profiles; team
    Elo; snapshot date). Local files only: features, labels, the scouting and
    fair value outputs, the Elo tables."""
    from scout.elo import read_tables
    from scout.providers.transfermarkt import TransfermarktValuationProvider
    from scout.tables import read_feature_table

    processed = settings.paths.processed_dir
    root = processed / "features"
    features = read_feature_table(root / "player_season_features")
    labels = read_feature_table(root / "player_season_labels")
    profiles = pd.read_parquet(root / "player_profiles.parquet")
    models = processed / "models"
    fair = pd.read_parquet(models / "backtest_fair_value.parquet")
    scouting = pd.read_parquet(models / "scouting.parquet")
    teams, _ = read_tables(processed)
    snapshot = TransfermarktValuationProvider(
        settings.paths.raw_dir / "transfermarkt"
    ).snapshot_date()
    if snapshot is None:
        log.warning("no Transfermarkt release date in the cache; contract expiry left blank")

    data = features.assign(season=features["season"].astype(int))
    fair = fair.assign(
        season=fair["season"].astype(int),
        value=np.exp(fair["y"]),
        predicted=np.exp(fair["pred_lightgbm"]),
    )
    data = data.merge(fair[[*KEYS, "value", "predicted"]], on=KEYS, how="inner")
    scouting = scouting.assign(season=scouting["season"].astype(int))
    data = data.merge(scouting[[*KEYS, "group", "ability_pct"]], on=KEYS, how="left")
    data = data.merge(outcomes(features, labels, teams), on=KEYS, how="left")
    return data, profiles, teams, snapshot


def run_bargains(settings) -> Path:
    data, profiles, _, snapshot = load(settings)
    rows = eligible(data)
    log.info("bargains: %d eligible rows in %d seasons", len(rows), rows["season"].nunique())
    lists = current_lists(rows, profiles, snapshot)
    results = backtest(rows)

    out_dir = settings.paths.processed_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        *KEYS,
        "list_rank",
        "player_name",
        "team_id",
        "team_name",
        "group",
        "age",
        "ability_pct",
        "value",
        "predicted",
        "ratio",
        "ratio_pct",
        "minutes_share",
        "contract_expiry",
        "flag_count",
        "flags",
        "on_loan",
        "stale_value",
        "stale_days",
        *[c for c in rows.columns if c.startswith("flag_") and c != "flag_count"],
    ]
    for name, frame in lists.items():
        frame[keep].to_parquet(out_dir / f"bargains_{name}.parquet", index=False)
    results.to_parquet(out_dir / "bargains_backtest.parquet", index=False)
    path = write_report(rows, lists, results, snapshot, settings.paths.reports_dir)
    log.info("Bargains report: %s", path)
    return path


def write_report(
    rows: pd.DataFrame,
    lists: dict[str, pd.DataFrame],
    results: pd.DataFrame,
    snapshot: pd.Timestamp | None,
    reports: Path,
    size: int = 50,
) -> Path:
    season = int(rows["season"].max())
    snap = snapshot.date().isoformat() if snapshot is not None else "unknown"
    parts = [
        "# Bargains",
        "",
        "Decisions are in the module docstring of `src/scout/bargains.py` (decision 4).",
        "",
        "**How to read it.** `ratio` is the market value divided by what the fair value "
        "model expects for this player (0.40: the market asks 40% of the model's price). "
        "`ratio_pct` ranks that ratio among eligible players of the same position group "
        "(1 is the cheapest), and the lists are sorted by it. Only players at the 70th "
        f"ability percentile of their position or better, with at least {MIN_MINUTES_SHARE:.0%} "
        "of their team's minutes, are eligible. The prediction for each season comes from "
        "a model trained only on earlier seasons. Contract expiry is the Transfermarkt "
        f"snapshot of {snap}, shown only when it ends after that date. Warning flags are "
        "reasons the market may be right; a cheap player with several deserves a closer look "
        "before anything else.",
        "",
    ]
    for name, title in LISTS.items():
        frame = lists[name]
        parts += [
            f"## {title}, season {season}: {len(frame)} eligible, top {min(size, len(frame))}",
            "",
            readable(frame.head(size)).to_markdown(index=False),
            "",
        ]
    parts += backtest_section(results)
    path = reports / "bargains.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path


def backtest_section(results: pd.DataFrame) -> list[str]:
    parts = [
        "## Backtest: did past bargains pay off?",
        "",
        f"Each season's top {TOP_K} of each list against the matched baseline: players of the "
        "same season, ability band (70 to 79, 80 to 89, 90 to 100) and age band who passed "
        "the same filters but were not in the top 50. `rate` is the share of the top 50 "
        "with the outcome, `matched_rate` what players like them achieved, `z` the "
        "difference in standard errors (above 2 is unlikely to be luck). Value up: the "
        "market value 24 months later is higher (known for seasons up to 2022 and the "
        "2023 calendar year leagues). Club with higher Elo: within two seasons he plays "
        "league minutes for another club whose season Elo is above his club's now (known "
        "up to season 2023; a move to a league outside the data counts as no).",
        "",
        "Part of the value rise is the market correcting a price that was low against the "
        "model, which is the point of a bargain list; but a value that was low by chance "
        "(a late update) rises the same way, so the rise overstates what a buyer would "
        "have earned. The move to a stronger club is the cleaner test of whether the "
        "football was underrated.",
        "",
    ]
    if results.empty:
        return [*parts, "No season could be evaluated.", ""]
    pooled = results[results["season"] == "pooled"].drop(columns="season")
    parts += [
        "### Pooled over all seasons",
        "",
        pooled.assign(outcome=pooled["outcome"].map(OUTCOMES)).round(3).to_markdown(index=False),
        "",
    ]
    seasons = results[results["season"] != "pooled"]
    for name, title in LISTS.items():
        for outcome, label in OUTCOMES.items():
            table = seasons[(seasons["list"] == name) & (seasons["outcome"] == outcome)]
            parts += [
                f"### {title}: {label}, per season",
                "",
                table.drop(columns=["list", "outcome"]).round(3).to_markdown(index=False),
                "",
            ]
    return parts
