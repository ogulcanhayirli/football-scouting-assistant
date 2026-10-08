"""Load the processed inputs, build the feature and label tables, check and write them.

Outputs, under ``data/processed/features/``, partitioned like the other
tables so Glue and Athena can read them later:

* ``player_links.parquet``: API Football id to Transfermarkt id
* ``player_season_features/league=<key>/season=<year>/data.parquet``
* ``player_season_labels/league=<key>/season=<year>/data.parquet``

Building spends no API requests; it reads only the cache and Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scout.config import Settings
from scout.contracts import PLAYER_SEASON_FEATURES, PLAYER_SEASON_LABELS, validate
from scout.features.build import Inputs, build_features
from scout.fsutil import atomic_write_text
from scout.matching import match_players
from scout.providers.transfermarkt import TransfermarktValuationProvider
from scout.tables import PARTITION_FILE

log = logging.getLogger(__name__)

TM_DATA_END = pd.Timestamp("2026-06-12")  # last valuation in the snapshot


def _read_partitions(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("league=*/season=*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def build_links(season_stats: pd.DataFrame, tm_players: pd.DataFrame) -> pd.DataFrame:
    """One Transfermarkt id per API Football player, matched within each league
    (club names help there) and then de duplicated across leagues."""
    found = []
    for league_key, group in season_stats.groupby("league_key"):
        players = group.sort_values("minutes", ascending=False).drop_duplicates("source_player_id")
        result = match_players(players, tm_players)
        found.append(result[result["reason"] == "matched"].assign(league_key=league_key))
    links = pd.concat(found, ignore_index=True)
    conflicts = links.groupby("af_player_id")["tm_player_id"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts):
        log.warning("Dropping %d players linked to different targets per league", len(conflicts))
    links = links[~links["af_player_id"].isin(conflicts)]
    links = links.drop_duplicates("af_player_id")[["af_player_id", "tm_player_id", "rule"]]
    links["af_player_id"] = links["af_player_id"].astype("Int64")
    links["tm_player_id"] = links["tm_player_id"].astype("Int64")
    return links.reset_index(drop=True)


def load_inputs(settings: Settings) -> Inputs:
    processed = settings.paths.processed_dir
    season_stats = _read_partitions(processed / "player_season_stats")
    match_stats = _read_partitions(processed / "player_match_stats")
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    tm_players = tm.players(None)
    links = build_links(season_stats, tm_players)
    linked = set(links["tm_player_id"].dropna().astype(int))
    valuations = tm.valuations(None)
    valuations = valuations[valuations["source_player_id"].isin(linked)]
    log.info(
        "Inputs: %d season rows, %d match rows, %d links, %d valuations",
        len(season_stats),
        len(match_stats),
        len(links),
        len(valuations),
    )
    return Inputs(season_stats, match_stats, links, tm_players, valuations, TM_DATA_END)


def _write_partitioned(frame: pd.DataFrame, root: Path) -> int:
    """One folder per partition (``league=epl/season=2023/data.parquet``), the
    Hive layout Glue and Athena expect. The partition values live in the folder
    names, so ``season`` is dropped from the file: Athena does not allow a
    column and a partition key with the same name. Timestamps are written in
    milliseconds, which Athena reads; it can't read nanosecond timestamps."""
    count = 0
    for (league, season), part in frame.groupby(["league_key", "season"]):
        folder = root / f"league={league}" / f"season={int(season)}"
        folder.mkdir(parents=True, exist_ok=True)
        part.drop(columns="season").to_parquet(
            folder / PARTITION_FILE,
            index=False,
            coerce_timestamps="ms",
            allow_truncated_timestamps=True,
        )
        count += 1
    return count


def run_features(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = load_inputs(settings)
    features, label_table = build_features(inputs)
    validate(PLAYER_SEASON_FEATURES, features)
    validate(PLAYER_SEASON_LABELS, label_table)

    out = settings.paths.processed_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    inputs.links.to_parquet(out / "player_links.parquet", index=False)
    n = _write_partitioned(features, out / "player_season_features")
    _write_partitioned(label_table, out / "player_season_labels")
    write_feature_report(features, label_table, settings.paths.reports_dir)
    log.info("Wrote %d feature rows in %d league seasons to %s", len(features), n, out)
    return features, label_table


def write_feature_report(features: pd.DataFrame, label_table: pd.DataFrame, reports: Path) -> Path:
    """Row counts and coverage per league season, for a quick look after a build."""
    reports.mkdir(parents=True, exist_ok=True)
    table = features.groupby(["league_key", "season"]).agg(
        tier=("tier", "first"),
        feature_date=("feature_date", "first"),
        players=("source_player_id", "size"),
        regular=("regular", "sum"),
        linked=("tm_player_id", lambda s: s.notna().mean()),
        valued=("market_value", lambda s: s.notna().mean()),
        rating=("rating", lambda s: s.notna().mean()),
    )
    lab = label_table.groupby(["league_key", "season"]).agg(
        label_24m=("value_24m", "count"), label_36m=("value_36m", "count")
    )
    table = table.join(lab).reset_index()
    table["feature_date"] = table["feature_date"].dt.date
    for column in ("linked", "valued", "rating"):
        table[column] = (table[column] * 100).round(1)
    path = reports / "features_report.md"
    atomic_write_text(
        path,
        "# Feature build report\n\nShares are percentages of all players with minutes.\n\n"
        + table.to_markdown(index=False)
        + "\n",
    )
    return path
