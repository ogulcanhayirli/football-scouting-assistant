"""The season so far (prompt 16): how each listed player is doing in the
season in progress, from the matches ``scout current`` has cached.

One row per player with league minutes this season, written by
``scout features --current`` to ``data/processed/current/season_so_far.parquet``
and joined into both tables by ``scout lists``. Display only: every column
starts with ``so_far_``, which the feature contract rejects, and no model
reads it. Nothing is trained or refitted:

* minutes and share of team minutes: as in the features
  (``match_tier_features``), up to the day after the league's last cached
  match;
* ability: the version 2 one season score with the stored fit
  (``ability_v2.json``: measures, weights, league effects, Elo team
  coefficient, prior minutes), as a percentile among this season's regulars
  (450 minutes or more) of the position group, blank in a league where under
  half the team matches carry detailed stats (Ukraine 2026/27, whose
  placeholder zeros the adapter blanks). The Elo team term is the
  club's rating in the league's newest Elo season up to this one (the season
  in progress where its fixtures are held, else the last finished one; a
  promoted club counts as its league's average);
* goals and assists per 90, blank for goalkeepers and centre backs, whose
  job they do not measure, and under 270 minutes;
* the warning flags as they stand on the pull day: low minutes on this
  season's share, injured on matches missed in the 12 months before it (the
  injuries table, finished seasons and the season in progress; blank where
  no league season in that window is covered), value falling as at the
  forecast, since the valuations end on 12 June 2026.

A player who played in two of our leagues this season is shown with the
league where he played most; ``so_far_minutes`` is that league's minutes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scout.features.build import REGULAR_MINUTES, match_tier_features
from scout.flags import FLAGS, describe

log = logging.getLogger(__name__)

NO_SCORING_GROUPS = {"goalkeeper", "centre_back"}  # goals and assists say little there
RATE_MIN_MINUTES = 270  # per 90 rates from fewer minutes swing on one goal
INJURY_WINDOW = pd.Timedelta(days=365)
COLUMNS = [
    "source_player_id",
    "so_far_date",
    "so_far_league_key",
    "so_far_season",
    "so_far_club",
    "so_far_matches",
    "so_far_minutes",
    "so_far_minutes_share",
    "so_far_ability_pct",
    "so_far_goals_per90",
    "so_far_assists_per90",
    "so_far_flag_count",
    "so_far_flags",
]


def _read(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("league=*/season=*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def so_far_dates(matches: pd.DataFrame) -> pd.DataFrame:
    """league_key, season, feature_date: the day after the last cached match."""
    last = matches.groupby(["league_key", "season"])["kickoff"].max().dt.tz_convert(None)
    dates = (last.dt.normalize() + pd.Timedelta(days=1)).rename("feature_date").reset_index()
    dates["feature_date"] = dates["feature_date"].astype("datetime64[ns]")
    return dates


DETAIL_SHARE = 0.5


def thin_leagues(matches: pd.DataFrame) -> list[str]:
    """Leagues where under half of the team matches carry detailed stats
    (passes), so a score would read their blanks as zeros."""
    played = matches[matches["minutes"].fillna(0) > 0]
    blocks = played.groupby(["league_key", "fixture_id", "team_id"])["passes_total"].agg(
        lambda s: s.notna().any()
    )
    share = blocks.groupby("league_key").mean()
    return sorted(share[share < DETAIL_SHARE].index)


def proxy_elo(
    teams: pd.DataFrame, leagues: pd.DataFrame, seasons: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Elo tables relabelled so each league's newest season up to the one
    in progress stands for the season in progress."""
    t_out, l_out = [], []
    for key, season in seasons.items():
        held = leagues[(leagues["league_key"] == key) & (leagues["season"] <= season)]
        if held.empty:
            continue
        use = int(held["season"].max())
        l_out.append(held[held["season"] == use].assign(season=season))
        t = teams[(teams["league_key"] == key) & (teams["season"] == use)]
        t_out.append(t.assign(season=season))
    # The Premier League row carries the reference rating of every season.
    return (
        pd.concat(t_out, ignore_index=True) if t_out else teams.iloc[:0],
        pd.concat(l_out, ignore_index=True) if l_out else leagues.iloc[:0],
    )


def injuries_missed(injuries: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Matches missed injured in the 12 months before ``day`` per player, and
    the league seasons that window covers."""
    if injuries.empty:
        return pd.DataFrame(columns=["source_player_id", "injury_matches_missed_12m"])
    i = injuries[injuries["status"] == "Missing Fixture"].copy()
    i["fixture_date"] = i["fixture_date"].dt.tz_convert(None)
    i = i[(i["fixture_date"] < day) & (i["fixture_date"] >= day - INJURY_WINDOW)]
    out = i.groupby("source_player_id")["fixture_id"].nunique().rename("injury_matches_missed_12m")
    return out.reset_index()


def season_so_far(
    matches: pd.DataFrame,
    positions: pd.DataFrame,
    injuries: pd.DataFrame,
    value_change: pd.DataFrame,
    fit=None,
    elo_tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    pull_day: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """The block for every player with league minutes this season.

    ``positions``: source_player_id, sub_position (his newest feature row).
    ``value_change``: source_player_id, value_change_1y (the forecast's).
    ``fit`` and ``elo_tables``: the stored ability fit and the Elo tables;
    without them the ability column stays blank.
    """
    from scout.scouting import ability_scores, position_group

    if matches.empty:
        return pd.DataFrame(columns=COLUMNS)
    dates = so_far_dates(matches)
    rows = match_tier_features(matches, dates)
    rows = rows.merge(dates, on=["league_key", "season"], how="left")
    real_season = rows["season"].astype(int)
    rows = rows.merge(positions, on="source_player_id", how="left")
    rows["tier"] = "match"
    rows["regular"] = rows["minutes"].astype(float) >= REGULAR_MINUTES
    # One season label, so the percentiles pool every league (J1 runs 2027).
    label = int(real_season.mode().iloc[0])
    rows["season"] = label
    thin = thin_leagues(matches)
    if thin:
        log.warning(
            "season so far: no ability in %s (detailed stats in under half the matches)", thin
        )
        from scout.scouting import COMPOSITE_CANDIDATES

        rows.loc[rows["league_key"].isin(thin), COMPOSITE_CANDIDATES] = np.nan
    if fit is not None and elo_tables is not None:
        seasons = dict(zip(rows["league_key"], real_season, strict=True))
        teams, leagues = proxy_elo(*elo_tables, seasons)
        teams, leagues = teams.assign(season=label), leagues.assign(season=label)
        if not (leagues["league_key"] == "epl").any():
            raise ValueError("no Premier League Elo row: the league term has no reference")
        scores, *_ = ability_scores(rows, "v2", (teams, leagues), pool=False, fit=fit)
        rows["ability"] = scores["ability_pct"]
    else:
        log.warning("season so far: no stored ability fit or Elo tables; ability left blank")
        rows["ability"] = np.nan
    group = position_group(rows)
    scoring = ~group.isin(NO_SCORING_GROUPS) & group.notna()
    scoring &= rows["minutes"].astype(float) >= RATE_MIN_MINUTES

    day = pull_day if pull_day is not None else rows["feature_date"].max()
    rows = rows.merge(injuries_missed(injuries, day), on="source_player_id", how="left")
    covered = _injury_coverage(injuries, day)
    rows["injury_matches_missed_12m"] = (
        rows["injury_matches_missed_12m"].fillna(0).where(rows["league_key"].isin(covered))
    )
    rows = rows.merge(value_change, on="source_player_id", how="left")
    flags = describe(rows, FLAGS)

    out = pd.DataFrame(
        {
            "source_player_id": rows["source_player_id"].astype("Int64"),
            "so_far_date": rows["feature_date"] - pd.Timedelta(days=1),
            "so_far_league_key": rows["league_key"],
            "so_far_season": real_season,
            "so_far_club": rows["team_name"],
            "so_far_matches": rows["appearances"].astype("Int64"),
            "so_far_minutes": rows["minutes"].astype(float).round(0),
            "so_far_minutes_share": rows["minutes_share"].astype(float).round(2),
            "so_far_ability_pct": rows["ability"].astype(float).round(0),
            "so_far_goals_per90": rows["goals_per90"].astype(float).where(scoring).round(2),
            "so_far_assists_per90": rows["assists_per90"].astype(float).where(scoring).round(2),
            "so_far_flag_count": flags["flag_count"],
            "so_far_flags": flags["flags"],
        }
    )
    out = out.sort_values("so_far_minutes", ascending=False, kind="stable")
    return out.drop_duplicates("source_player_id").sort_values("source_player_id")[COLUMNS]


def _injury_coverage(injuries: pd.DataFrame, day: pd.Timestamp) -> set[str]:
    """Leagues with an injuries table for a season inside the 12 month window."""
    if injuries.empty:
        return set()
    dates = injuries["fixture_date"].dt.tz_convert(None)
    inside = (dates < day) & (dates >= day - INJURY_WINDOW)
    return set(injuries.loc[inside, "league_key"])


def run_season_so_far(settings, stored_features: pd.DataFrame) -> Path | None:
    """Build and write the block from the cached season in progress
    (``scout current``); None when no match of it is cached."""
    from scout.elo import read_tables
    from scout.models.store import load_ability, store_dir

    processed = settings.paths.processed_dir
    current = processed / "current"
    configured = {lg.key for lg in settings.leagues}
    matches = _read(current / "player_match_stats")
    if matches.empty:
        log.warning("season so far: no match of the season in progress cached; run `scout current`")
        return None
    matches = matches[matches["league_key"].isin(configured)].reset_index(drop=True)
    newest = stored_features.sort_values("season").drop_duplicates("source_player_id", keep="last")
    positions = newest[["source_player_id", "sub_position"]]
    value_change = newest[["source_player_id", "value_change_1y"]]
    injuries = pd.concat([_read(processed / "injuries"), _read(current / "injuries")])
    if not injuries.empty:
        injuries = injuries.drop_duplicates(["fixture_id", "source_player_id"])
    fit = elo = None
    try:
        fit = load_ability(store_dir(processed))
        elo = read_tables(processed)
    except FileNotFoundError as exc:
        log.warning("season so far: %s", exc)
    pull = _pull_day(current)
    block = season_so_far(matches, positions, injuries, value_change, fit, elo, pull)
    path = current / "season_so_far.parquet"
    block.to_parquet(path, index=False)
    log.info(
        "season so far: %d players with minutes, %d with an ability percentile, as of %s",
        len(block),
        int(block["so_far_ability_pct"].notna().sum()),
        pull.date() if pull is not None else "unknown",
    )
    return path


def _pull_day(current: Path) -> pd.Timestamp | None:
    """The newest snapshot day of the squads (the day of the pull)."""
    days = sorted(p.stem.split("=", 1)[1] for p in (current / "squads").glob("as_of=*.parquet"))
    return pd.Timestamp(days[-1]) if days else None
