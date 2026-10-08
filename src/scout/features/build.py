"""Point in time feature and label tables, one row per player, league and season.

The rule behind every function here: a row may only use data dated on or
before its feature date. Each row keeps the dates it used (``last_match_date``
and ``value_date``) so the contract can check that rule on the output.

Feature date: the later of 30 June and the day after the season's last league
match. The 2019/20 seasons ran to 26 July 2020 after the COVID break, so a
fixed 30 June would have treated their last weeks as the future.

Two tiers (D10, D12): from 2015 the playing time and detailed stats come from
the per match data; for earlier seasons only the season endpoint's basic
counts exist, and detailed columns stay missing rather than imputed.

Deliberately not used: Transfermarkt's contract expiry, international caps
and current market value. They describe the player on the snapshot date
(2026), not on the feature date, and would leak the future into old rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MATCH_TIER_FROM = 2015  # first season with per match player stats in both leagues
REGULAR_MINUTES = 450
VALUE_MAX_AGE_DAYS = 365
FORM_MATCHES = 10
RECORDED_SHARE = 0.5  # a stat counts as recorded in a season above this share

# Counts summed over matches and turned into per 90 rates.
COUNT_STATS = [
    "goals",
    "assists",
    "shots_total",
    "shots_on",
    "passes_total",
    "passes_key",
    "passes_accurate",
    "tackles_total",
    "blocks",
    "interceptions",
    "duels_total",
    "duels_won",
    "dribbles_attempts",
    "dribbles_success",
    "dribbled_past",
    "fouls_drawn",
    "fouls_committed",
    "cards_yellow",
    "cards_red",
    "saves",
    "goals_conceded",
    "offsides",
    "penalty_won",
    "penalty_committed",
]
# Success rates: numerator, denominator.
RATES = {
    "shot_accuracy": ("shots_on", "shots_total"),
    "pass_accuracy": ("passes_accurate", "passes_total"),
    "duel_win_rate": ("duels_won", "duels_total"),
    "dribble_success_rate": ("dribbles_success", "dribbles_attempts"),
}


@dataclass(frozen=True)
class Inputs:
    season_stats: pd.DataFrame  # player_season_stats, all seasons
    match_stats: pd.DataFrame  # player_match_stats
    links: pd.DataFrame  # af_player_id, tm_player_id
    tm_players: pd.DataFrame  # Transfermarkt players (canonical columns)
    valuations: pd.DataFrame  # source_player_id, valuation_date, market_value_eur
    data_end: pd.Timestamp  # last date the valuation data covers


# Feature dates


def feature_dates(season_stats: pd.DataFrame, match_stats: pd.DataFrame) -> pd.DataFrame:
    """league_key, season, feature_date for every league season."""
    keys = season_stats[["league_key", "season"]].drop_duplicates()
    keys = keys.assign(
        feature_date=pd.to_datetime((keys["season"].astype(int) + 1).astype(str) + "-06-30")
    )
    if not match_stats.empty:
        last = (
            match_stats.groupby(["league_key", "season"])["kickoff"]
            .max()
            .dt.tz_convert(None)
            .dt.normalize()
            .add(pd.Timedelta(days=1))
            .rename("after_last_match")
            .reset_index()
        )
        keys = keys.merge(last, on=["league_key", "season"], how="left")
        keys["feature_date"] = keys[["feature_date", "after_last_match"]].max(axis=1)
        keys = keys.drop(columns="after_last_match")
    keys["feature_date"] = keys["feature_date"].astype("datetime64[ns]")
    return keys.reset_index(drop=True)


# Match tier


def fill_recorded_zeros(match_stats: pd.DataFrame) -> pd.DataFrame:
    """Blank counts become 0 for players who played, where the stat is recorded.

    The API leaves zero counts blank (FEASIBILITY.md, section 1). A blank is
    read as 0 only when most players who played in that league season have
    the stat; where nobody has it, it stays missing.
    """
    out = match_stats.copy()
    played = out["minutes"].fillna(0) > 0
    for column in COUNT_STATS:
        recorded = (
            out[played]
            .groupby(["league_key", "season"])[column]
            .agg(lambda s: s.notna().mean())
            .rename("_share")
        )
        share = out[["league_key", "season"]].join(recorded, on=["league_key", "season"])
        fill = played & out[column].isna() & (share["_share"] > RECORDED_SHARE)
        out.loc[fill, column] = 0
    return out


def team_results(match_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per match and team with goals for and against and points."""
    teams = match_stats.drop_duplicates(["fixture_id", "team_id"])[
        ["league_key", "season", "fixture_id", "kickoff", "team_id", "team_goals", "opponent_goals"]
    ].copy()
    gf, ga = teams["team_goals"], teams["opponent_goals"]
    teams["points"] = np.select([gf > ga, gf == ga], [3, 1], 0)
    teams.loc[gf.isna() | ga.isna(), "points"] = np.nan
    return teams


def team_context(results: pd.DataFrame) -> pd.DataFrame:
    """Per league, season and team: matches, points per game, goal difference per game."""
    grouped = results.groupby(["league_key", "season", "team_id"])
    out = grouped.agg(
        team_matches=("fixture_id", "nunique"),
        team_points=("points", "sum"),
        team_gf=("team_goals", "sum"),
        team_ga=("opponent_goals", "sum"),
    ).reset_index()
    out["team_ppg"] = out["team_points"] / out["team_matches"]
    out["team_gd_per_match"] = (out["team_gf"] - out["team_ga"]) / out["team_matches"]
    return out[["league_key", "season", "team_id", "team_matches", "team_ppg", "team_gd_per_match"]]


def match_tier_features(match_stats: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Season aggregates, per 90 rates, rates, team context and form from match rows."""
    m = match_stats.merge(dates, on=["league_key", "season"])
    m = m[m["kickoff"].dt.tz_convert(None) < m["feature_date"]]  # point in time guard
    m = fill_recorded_zeros(m)
    played = m[m["minutes"].fillna(0) > 0].copy()
    played["rating_x_min"] = played["rating"] * played["minutes"]
    played["rated_min"] = played["minutes"].where(played["rating"].notna())
    played["start"] = ~played["substitute"].fillna(False).astype(bool)

    keys = ["league_key", "season", "source_player_id"]
    agg = played.groupby(keys).agg(
        minutes=("minutes", "sum"),
        appearances=("fixture_id", "nunique"),
        starts=("start", "sum"),
        rating_x_min=("rating_x_min", "sum"),
        rated_min=("rated_min", "sum"),
        last_match_date=("kickoff", "max"),
        teams_played_for=("team_id", "nunique"),
        **{c: (c, lambda s: s.sum(min_count=1)) for c in COUNT_STATS},
    )
    agg["rating"] = agg["rating_x_min"] / agg["rated_min"].replace(0, np.nan)
    agg = agg.drop(columns=["rating_x_min", "rated_min"]).reset_index()
    agg["last_match_date"] = agg["last_match_date"].dt.tz_convert(None).astype("datetime64[ns]")

    # The team a player ended the season with.
    last_team = (
        played.sort_values("kickoff").groupby(keys)[["team_id", "team_name"]].last().reset_index()
    )
    agg = agg.merge(last_team, on=keys)

    for column in COUNT_STATS:
        agg[f"{column}_per90"] = agg[column] / agg["minutes"] * 90
    for name, (num, den) in RATES.items():
        agg[name] = agg[num] / agg[den].where(agg[den] > 0)

    results = team_results(m)
    context = team_context(results)
    agg = agg.merge(context, on=["league_key", "season", "team_id"], how="left")
    team_minutes = available_minutes(m, results, keys)
    agg = agg.join(team_minutes, on=keys)
    agg["minutes_share"] = (agg["minutes"] / agg["team_minutes"]).clip(upper=1.0)
    agg = agg.drop(columns="team_minutes")
    return agg.merge(form_features(m, results, keys), on=keys, how="left")


def available_minutes(m: pd.DataFrame, results: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Minutes a player could have played: 90 per match his clubs played while
    he belonged to them.

    A club's window runs from his first to his last squad listing for it (bench
    included), stretched to the start of the season for his first club and to
    the end for his last. Without the windows, a player who made one
    appearance for Manchester City before joining Chelsea (Palmer, 2023/24)
    would be measured against both clubs' full seasons.
    """
    spans = (
        m.groupby([*keys, "team_id"])["kickoff"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
        .sort_values("first_seen")
    )
    order = spans.groupby(keys).cumcount()
    count = spans.groupby(keys)["team_id"].transform("size")
    far_past = pd.Timestamp("1900-01-01", tz="UTC")
    far_future = pd.Timestamp("2200-01-01", tz="UTC")
    spans["start"] = spans["first_seen"].where(order > 0, far_past)
    spans["end"] = spans["last_seen"].where(order < count - 1, far_future)
    games = spans.merge(
        results[["league_key", "season", "team_id", "fixture_id", "kickoff"]].rename(
            columns={"kickoff": "kickoff_game"}
        ),
        on=["league_key", "season", "team_id"],
    )
    inside = (games["kickoff_game"] >= games["start"]) & (games["kickoff_game"] <= games["end"])
    return games[inside].groupby(keys)["fixture_id"].nunique().mul(90).rename("team_minutes")


def form_features(m: pd.DataFrame, results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The player's share of minutes, goal involvement and rating over the last
    ``FORM_MATCHES`` matches of the team he ended the season with."""
    played = m[m["minutes"].fillna(0) > 0]
    last_team = played.sort_values("kickoff").groupby(keys)["team_id"].last().rename("team_id")
    last_games = (
        results.sort_values("kickoff")
        .groupby(["league_key", "season", "team_id"])
        .tail(FORM_MATCHES)[["league_key", "season", "team_id", "fixture_id"]]
    )
    grid = last_team.reset_index().merge(last_games, on=["league_key", "season", "team_id"])
    rows = m[[*keys, "fixture_id", "minutes", "goals", "assists", "rating"]]
    grid = grid.merge(rows, on=[*keys, "fixture_id"], how="left")
    grid["minutes"] = grid["minutes"].fillna(0)  # not in the squad counts as 0 minutes
    grid["involvement"] = grid["goals"].fillna(0) + grid["assists"].fillna(0)
    form = grid.groupby(keys).agg(
        form_minutes=("minutes", "sum"),
        form_matches=("fixture_id", "nunique"),
        form_involvement=("involvement", "sum"),
        form_rating=("rating", "mean"),
    )
    form["form_minutes_share"] = (form["form_minutes"] / (form["form_matches"] * 90)).clip(
        upper=1.0
    )
    form["form_involvement_per90"] = (
        form["form_involvement"] / form["form_minutes"].where(form["form_minutes"] > 0) * 90
    )
    return form[["form_minutes_share", "form_involvement_per90", "form_rating"]].reset_index()


# Season tier


def season_tier_features(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Basic counts from the season endpoint, for seasons without match data."""
    keys = ["league_key", "season", "source_player_id"]
    s = season_stats.sort_values("minutes", ascending=False)
    agg = s.groupby(keys).agg(
        minutes=("minutes", lambda x: x.sum(min_count=1)),
        appearances=("appearances", lambda x: x.sum(min_count=1)),
        starts=("lineups", lambda x: x.sum(min_count=1)),
        goals=("goals", lambda x: x.sum(min_count=1)),
        cards_yellow=("cards_yellow", lambda x: x.sum(min_count=1)),
        cards_red=("cards_red", lambda x: x.sum(min_count=1)),
        teams_played_for=("team_id", "nunique"),
        team_id=("team_id", "first"),  # the club he played most minutes for
        team_name=("team_name", "first"),
    )
    played = agg["minutes"].fillna(0) > 0
    agg.loc[played, "goals"] = agg.loc[played, "goals"].fillna(0)
    for column in ["goals", "cards_yellow", "cards_red"]:
        agg[f"{column}_per90"] = agg[column] / agg["minutes"].where(played) * 90
    return agg.reset_index()


# Profile and values


def profile_features(inputs: Inputs, rows: pd.DataFrame) -> pd.DataFrame:
    """Name, age, position and physical data. Transfermarkt fills gaps for linked players."""
    keys = ["league_key", "season", "source_player_id"]
    s = inputs.season_stats
    af = (
        s.sort_values("minutes", ascending=False)
        .groupby("source_player_id")
        .agg(
            player_name=("player_name", "first"),
            af_birth_date=("birth_date", "first"),
            nationality=("nationality", "first"),
            af_height_cm=("height_cm", "first"),
        )
    )
    position = (
        s.sort_values("minutes", ascending=False)
        .groupby(keys)["position"]
        .first()
        .rename("position")
    )
    tm = inputs.tm_players.set_index("source_player_id")[
        ["birth_date", "sub_position", "foot", "height_cm"]
    ].rename(columns={"birth_date": "tm_birth_date", "height_cm": "tm_height_cm"})
    out = rows[[*keys, "feature_date"]].join(af, on="source_player_id").join(position, on=keys)
    out = out.merge(
        inputs.links, left_on="source_player_id", right_on="af_player_id", how="left"
    ).drop(columns="af_player_id")
    # Which matching rule linked him; always present so the schema is stable.
    out = out.rename(columns={"rule": "link_rule"})
    if "link_rule" not in out:
        out["link_rule"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out = out.join(tm, on="tm_player_id")
    birth = out["af_birth_date"].fillna(out["tm_birth_date"])
    out["age"] = (out["feature_date"] - birth).dt.days / 365.25
    out["height_cm"] = out["af_height_cm"].fillna(out["tm_height_cm"])
    return out.drop(columns=["af_birth_date", "tm_birth_date", "af_height_cm", "tm_height_cm"])


def value_asof(
    rows: pd.DataFrame,
    valuations: pd.DataFrame,
    date_column: str,
    max_age_days: int = VALUE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Latest valuation on or before each row's ``date_column``, at most
    ``max_age_days`` old. Returns value and its date, aligned to ``rows``."""
    left = rows[["tm_player_id", date_column]].reset_index()
    left = left.dropna(subset=["tm_player_id", date_column]).sort_values(date_column)
    left["tm_player_id"] = left["tm_player_id"].astype("int64")
    left[date_column] = left[date_column].astype("datetime64[ns]")
    right = valuations[["source_player_id", "valuation_date", "market_value_eur"]].dropna()
    right = right.rename(columns={"source_player_id": "tm_player_id"}).sort_values("valuation_date")
    right["tm_player_id"] = right["tm_player_id"].astype("int64")
    right["valuation_date"] = right["valuation_date"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        left,
        right,
        left_on=date_column,
        right_on="valuation_date",
        by="tm_player_id",
        direction="backward",
        tolerance=pd.Timedelta(days=max_age_days),
    ).set_index("index")
    return merged[["market_value_eur", "valuation_date"]].reindex(rows.index)


def value_features(rows: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Value at the feature date, a year earlier, the change, and the highest so far."""
    out = pd.DataFrame(index=rows.index)
    now = value_asof(rows, valuations, "feature_date")
    out["market_value"] = now["market_value_eur"]
    out["value_date"] = now["valuation_date"]
    year_ago = rows.assign(_d=rows["feature_date"] - pd.Timedelta(days=365))
    out["market_value_1y_before"] = value_asof(year_ago, valuations, "_d")["market_value_eur"]
    out["value_change_1y"] = out["market_value"] / out["market_value_1y_before"]

    vals = valuations.dropna(subset=["market_value_eur"]).rename(
        columns={"source_player_id": "tm_player_id"}
    )
    pairs = rows[["tm_player_id", "feature_date"]].dropna().reset_index()
    pairs = pairs.merge(vals[["tm_player_id", "valuation_date", "market_value_eur"]])
    pairs = pairs[pairs["valuation_date"] <= pairs["feature_date"]]
    out["market_value_max_to_date"] = pairs.groupby("index")["market_value_eur"].max()
    return out


def labels(rows: pd.DataFrame, valuations: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """Targets: value at the feature date (fair value) and 24 and 36 months later
    (growth). A horizon past ``data_end`` is missing, never a stale value."""
    keys = ["league_key", "season", "source_player_id"]
    out = rows[[*keys, "feature_date", "tm_player_id"]].copy()
    now = value_asof(rows, valuations, "feature_date")
    out["value_now"] = now["market_value_eur"]
    for months in (24, 36):
        target = rows.assign(_t=rows["feature_date"] + pd.DateOffset(months=months))
        found = value_asof(target, valuations, "_t")
        reachable = target["_t"] <= data_end
        out[f"value_{months}m"] = found["market_value_eur"].where(reachable)
        out[f"value_{months}m_date"] = found["valuation_date"].where(reachable)
        out[f"log_growth_{months}m"] = np.log(out[f"value_{months}m"] / out["value_now"])
    return out


# Assembly


def build_features(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The feature table and the label table, one row per league, season and player."""
    keys = ["league_key", "season", "source_player_id"]
    dates = feature_dates(inputs.season_stats, inputs.match_stats)

    match_part = inputs.match_stats[inputs.match_stats["season"] >= MATCH_TIER_FROM]
    detailed = match_tier_features(match_part, dates).assign(tier="match")
    early = inputs.season_stats[inputs.season_stats["season"] < MATCH_TIER_FROM]
    basic = season_tier_features(early).assign(tier="season")
    rows = pd.concat([detailed, basic], ignore_index=True)
    rows = rows[rows["minutes"].fillna(0) > 0]
    rows = rows.merge(dates, on=["league_key", "season"])

    profile = profile_features(inputs, rows)
    rows = rows.merge(profile, on=[*keys, "feature_date"])
    rows["regular"] = rows["minutes"] >= REGULAR_MINUTES
    rows = rows.reset_index(drop=True)
    rows = pd.concat([rows, value_features(rows, inputs.valuations)], axis=1)
    label_table = labels(rows, inputs.valuations, inputs.data_end)
    return _order(rows), label_table


FIRST_COLUMNS = [
    "league_key",
    "season",
    "source_player_id",
    "tm_player_id",
    "link_rule",
    "player_name",
    "feature_date",
    "tier",
    "regular",
    "team_id",
    "team_name",
    "position",
    "sub_position",
    "age",
    "height_cm",
    "foot",
    "nationality",
]


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    rest = [c for c in frame.columns if c not in FIRST_COLUMNS]
    frame = frame[FIRST_COLUMNS + rest]
    for column in ("source_player_id", "tm_player_id", "team_id", "season"):
        frame[column] = frame[column].astype("Int64")
    frame["regular"] = frame["regular"].astype(bool)
    frame["link_rule"] = frame["link_rule"].astype("string")
    return frame.sort_values(["league_key", "season", "source_player_id"]).reset_index(drop=True)
