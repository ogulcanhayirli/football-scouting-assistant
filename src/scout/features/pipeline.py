"""Load the processed inputs, build the feature and label tables, check and write them.

Outputs, under ``data/processed/features/``, partitioned like the other
tables so Glue and Athena can read them later:

* ``player_links.parquet``: API Football id to Transfermarkt id
* ``player_profiles.parquet``: contract expiry, agent and highest market
  value of each linked player, as the snapshot states them today. Display
  only, for the current season's lists; never joined into the features.
* ``player_season_features/league=<key>/season=<year>/data.parquet``
* ``player_season_labels/league=<key>/season=<year>/data.parquet``

Building spends no API requests; it reads only the cache and Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scout.backfill.transfers import transfers_path
from scout.config import Settings
from scout.contracts import (
    PLAYER_PROFILES,
    PLAYER_SEASON_FEATURES,
    PLAYER_SEASON_LABELS,
    validate,
)
from scout.features.build import Inputs, build_features
from scout.features.context import (
    club_calendar,
    injury_coverage,
    injury_source_agreement,
)
from scout.fsutil import atomic_write_text
from scout.matching import match_players
from scout.providers.transfermarkt import TransfermarktValuationProvider
from scout.tables import PARTITION_FILE, read_feature_table

log = logging.getLogger(__name__)


def _read_partitions(root: Path, keep=None) -> pd.DataFrame:
    """Every ``league=<key>/season=<year>.parquet`` under ``root``, or those
    for which ``keep(key, year)`` is true."""
    files = sorted(root.glob("league=*/season=*.parquet"))
    if keep is not None:
        files = [
            f for f in files if keep(f.parent.name.split("=", 1)[1], int(f.stem.split("=", 1)[1]))
        ]
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


def build_profiles(
    links: pd.DataFrame, tm_profiles: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """One row per linked player with the snapshot's present day profile fields.

    ``as_of`` is the snapshot's last valuation date: the fields describe the
    player around then, not at any feature date.
    """
    out = links[["af_player_id", "tm_player_id"]].merge(
        tm_profiles.rename(columns={"source_player_id": "tm_player_id"}),
        on="tm_player_id",
        how="left",
    )
    out["as_of"] = as_of
    return out.reset_index(drop=True)


def _split(frame: pd.DataFrame, keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of the given league keys, and the rest (cups and national teams)."""
    if frame.empty:
        return frame, frame
    mine = frame["league_key"].isin(keys)
    return frame[mine].reset_index(drop=True), frame[~mine].reset_index(drop=True)


def _configured(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rows of the configured leagues, without their skipped seasons. The
    processed folders also hold leagues downloaded but not yet configured."""
    if frame.empty:
        return frame
    skip = {(lg.key, s) for lg in settings.leagues for s in lg.skip_seasons}
    keep = frame["league_key"].isin({lg.key for lg in settings.leagues})
    if skip:
        pairs = zip(frame["league_key"], frame["season"].astype(int), strict=True)
        keep &= ~pd.Series([p in skip for p in pairs], index=frame.index)
    return frame[keep].reset_index(drop=True)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_inputs(
    settings: Settings, season: int | None = None, stored: pd.DataFrame | None = None
) -> Inputs:
    """Everything the build reads. With ``season`` (the weekly run), the
    league match and team stats of that season only, the cup and national
    team matches of every season (their windows and coverage reach back),
    and the stored links plus links for players the ``stored`` feature
    table has never seen."""
    processed = settings.paths.processed_dir
    league_keys = {lg.key for lg in settings.leagues}
    comp_keys = {c.key for c in settings.competitions}

    def wanted(key: str, year: int) -> bool:
        return season is None or key not in league_keys or year == season

    season_stats = _configured(_read_partitions(processed / "player_season_stats"), settings)
    match_stats, comp_matches = _split(
        _read_partitions(processed / "player_match_stats", wanted), league_keys
    )
    match_stats = _configured(match_stats, settings)
    comp_matches = comp_matches[comp_matches["league_key"].isin(comp_keys)]
    team_stats, _ = _split(_read_partitions(processed / "team_match_stats", wanted), league_keys)
    injuries = _read_partitions(processed / "injuries")
    sidelined = _read_optional(processed / "sidelined" / "sidelined.parquet")
    coaches = _read_optional(processed / "coaches" / "coach_spells.parquet")
    calendar = club_calendar(_read_optional(processed / "elo" / "elo_matches.parquet"))
    transfers_file = transfers_path(settings)
    transfers = pd.read_parquet(transfers_file) if transfers_file.exists() else pd.DataFrame()
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    tm_players = tm.players(None)
    if season is None:
        links = build_links(season_stats, tm_players)
    else:
        links = current_links(processed / "features", season_stats, tm_players, season, stored)
    linked = set(links["tm_player_id"].dropna().astype(int))
    valuations = tm.valuations(None)
    valuations = valuations[valuations["source_player_id"].isin(linked)]
    data_end = tm.data_end()
    log.info("Transfermarkt valuations end on %s", data_end.date())
    log.info(
        "Inputs: %d season rows, %d league and %d cup match rows, %d team rows, "
        "%d injuries, %d sidelined periods, %d coach spells, %d transfers, %d links, "
        "%d valuations",
        len(season_stats),
        len(match_stats),
        len(comp_matches),
        len(team_stats),
        len(injuries),
        len(sidelined),
        len(coaches),
        len(transfers),
        len(links),
        len(valuations),
    )
    return Inputs(
        season_stats,
        match_stats,
        links,
        tm_players,
        valuations,
        data_end,
        comp_matches=comp_matches,
        team_stats=team_stats,
        injuries=injuries,
        sidelined=sidelined,
        club_calendar=calendar,
        coaches=coaches,
        transfers=transfers,
        comps={c.key: (c.kind, c.region) for c in settings.competitions},
        league_regions={lg.key: lg.region for lg in settings.leagues},
        calendar_leagues=frozenset(lg.key for lg in settings.leagues if lg.calendar_year),
    )


def current_links(
    root: Path,
    season_stats: pd.DataFrame,
    tm_players: pd.DataFrame,
    season: int,
    stored: pd.DataFrame | None,
) -> pd.DataFrame:
    """The stored links, plus links for players who play in ``season`` and
    whom neither the stored links nor the stored feature table know (a new
    signing after an update). They are matched on all their season rows, as
    a full build would; the links of known players never change here."""
    links = pd.read_parquet(root / "player_links.parquet")
    known = set(links["af_player_id"].dropna().astype(int))
    if stored is not None and len(stored):
        known |= set(stored["source_player_id"].dropna().astype(int))
    played = season_stats[
        (season_stats["season"].astype(int) == season) & (season_stats["minutes"].fillna(0) > 0)
    ]
    new = set(played["source_player_id"].astype(int)) - known
    if not new:
        return links
    found = build_links(
        season_stats[season_stats["source_player_id"].astype(int).isin(new)], tm_players
    )
    log.info("Links: %d new players in season %d, %d linked", len(new), season, len(found))
    return pd.concat([links, found], ignore_index=True)


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
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    profiles = validate(
        PLAYER_PROFILES, build_profiles(inputs.links, tm.profiles(), inputs.data_end)
    )
    profiles.to_parquet(out / "player_profiles.parquet", index=False)
    n = _write_partitioned(features, out / "player_season_features")
    _write_partitioned(label_table, out / "player_season_labels")
    write_feature_report(features, label_table, settings.paths.reports_dir)
    write_injury_report(features, inputs, settings.paths.reports_dir)
    log.info("Wrote %d feature rows in %d league seasons to %s", len(features), n, out)
    return features, label_table


def run_current_features(
    settings: Settings, season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``scout features --current``: rebuild one season (the newest in the
    season stats when None) and rewrite only its partitions; every earlier
    season stays as stored. Reads the stored feature table for the history
    columns and the player links, so it needs a full build once. Then the
    season so far block from the cached season in progress
    (``scout.season_so_far``), so the weekly run refreshes it untrained."""
    out = settings.paths.processed_dir / "features"
    stored = read_feature_table(out / "player_season_features")
    if stored.empty:
        raise FileNotFoundError(f"no stored feature table in {out}: run `scout features` first")
    if season is None:
        season_stats = _configured(
            _read_partitions(settings.paths.processed_dir / "player_season_stats"), settings
        )
        season = int(season_stats["season"].max())
    inputs = load_inputs(settings, season, stored)
    features, label_table = build_features(inputs, season, stored)
    validate(PLAYER_SEASON_FEATURES, features)
    validate(PLAYER_SEASON_LABELS, label_table)

    if len(inputs.links) != len(pd.read_parquet(out / "player_links.parquet")):
        inputs.links.to_parquet(out / "player_links.parquet", index=False)  # new players
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    profiles = validate(
        PLAYER_PROFILES, build_profiles(inputs.links, tm.profiles(), inputs.data_end)
    )
    profiles.to_parquet(out / "player_profiles.parquet", index=False)
    n = _write_partitioned(features, out / "player_season_features")
    _write_partitioned(label_table, out / "player_season_labels")
    log.info("Wrote %d rows of season %d in %d leagues to %s", len(features), season, n, out)
    from scout.season_so_far import run_season_so_far

    run_season_so_far(settings, pd.concat([stored[stored["season"] != season], features]))
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


def write_injury_report(features: pd.DataFrame, inputs: Inputs, reports: Path) -> Path:
    """Where the injury columns come from per season, and how the two sources
    agree on the rows both cover (reports/injury_sources.md)."""
    source = features["injury_source"].fillna("none")
    coverage = pd.crosstab(features["season"], source, normalize="index").mul(100).round(1)
    coverage.columns = [f"{c} %" for c in coverage.columns]
    agreement = injury_source_agreement(
        features,
        inputs.injuries,
        injury_coverage(inputs.injuries),
        inputs.sidelined,
        inputs.club_calendar,
    )
    for column in ("injured_injuries", "injured_sidelined", "agree"):
        if column in agreement:
            agreement[column] = (agreement[column] * 100).round(1)
    path = reports / "injury_sources.md"
    atomic_write_text(
        path,
        "# Injury sources\n\n"
        "Share of feature rows per season whose injury columns come from the "
        "injuries table, the sidelined periods, or neither.\n\n"
        + coverage.reset_index().to_markdown(index=False)
        + "\n\n## Agreement where both sources cover a row\n\n"
        "Injured: one spell or more in the 12 months before the feature date "
        "(percent of rows). Missed: mean matches missed. Agree: percent of rows "
        "where both sources say injured or both say not.\n\n"
        + (agreement.round(3).to_markdown(index=False) if len(agreement) else "No overlap.")
        + "\n",
    )
    return path
