"""Scouting views on top of the feature tables: league strength, an ability
score, three career targets, the traits behind them, and four shortlists.

Decisions, each argued in SCOUTING.md:

* **League strength** comes from players who changed leagues: the change in
  their score when they move, with the change in team strength and age held
  fixed, measured against the Premier League. Leagues with few moves are
  pulled towards the average of the smaller leagues. ``strength="elo"``
  takes league and team strength from our own team Elo (``scout.elo``): the
  league's mean club rating that season against the Premier League's, and
  the club's rating against its league's (``reports/elo.md``).
  ``strength="transfer_elo"`` keeps the transfer based league effect and
  takes the team term from the Elo tables.
* **Ability score, version 1** (``strength="transfer"``): the match rating
  with the league effect and team goal difference taken out, shrunk towards
  the position average for players with few minutes, as a percentile within
  the position group and season.
* **Ability score, version 2** (``strength="v2"``, the default since 19
  October 2026, decision 8): the match rating is replaced by a per 90
  composite for each position group. Its measures and weights are the per
  90 stats on which players who made the step up stood above same age peers
  (the trait comparison), each standardised within position group and
  season. League term from transfers and team term from Elo, both refitted
  in composite units; then the last two seasons pooled by minutes and
  shrunk towards the position mean by empirical Bayes, with the prior
  strength fitted from the data. It is more stable from season to season
  and predicts elite and step up at least as well (``scout ability``,
  ``reports/ability.md``, MODELS.md).
* **Targets,** each within 2 seasons:
  ``elite``: enters the 100 most valuable players, from #101 to #1,000 now;
  ``step_up``: value rank at least 3 times better, ending in the top 500;
  ``performance``: both of the next two seasons at the 70th ability
  percentile or better, and their average 20 points above now.
  The first two are the market's view (resale); the third is the football
  (Bardakcı at 29: Champions League and Euro 2024, value rank still #1,137).
  ``fall_back`` (decision 1): an elite candidate (#101 to #1,000) whose
  value rank at the end of the 2 seasons is at least twice as bad as now
  (#300 to #600 or worse), the mirror of the step up; risk tables and
  warning flags in ``scout.candidates``.
* **Traits** compare the players who reached a target with players of the
  same position group and age band who did not, on percentiles within their
  own league and season, so league strength and age are not the finding.
  Market value is left out of the traits, and every trait table is shown
  with and without the API rating, whose formula hides the measures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scout import candidates
from scout.features.build import value_asof
from scout.fsutil import atomic_write_text
from scout.models.backtest import check_no_leak, expanding_folds, fold_season
from scout.models.estimators import DEFAULT_PARAMS
from scout.models.moves import auc, average_precision, precision_at
from scout.models.targets import CATEGORICAL, NOT_FEATURES

log = logging.getLogger(__name__)

KEYS = ["league_key", "season", "source_player_id"]
HORIZON_SEASONS = 2
ELITE_RANK = 100
STEP_UP_RANK = 500
STEP_UP_FACTOR = 3.0
ELITE_REACH = 1000  # elite candidates start here; further away it is too rare to learn
FALL_BACK_FACTOR = 2.0  # an elite candidate whose value rank is twice as bad at the end
PERF_RISE = 20.0  # the next two seasons average this many percentiles above now
PERF_LEVEL = 70.0  # and both are at least this good: a rise that holds, not one lucky season
RISER_MIN_SHARE = 0.5  # decision 3: risers play at least half of team minutes now
SHRINK_MINUTES = 900  # an ability score from 900 minutes is half data, half position average
MIN_MOVES = 50  # league strength from fewer moves is pulled towards the prior
LABEL_TOLERANCE = pd.Timedelta(days=31)  # a value from 12 June 2026 stands for 30 June 2026
BIG_FIVE = ["epl", "la_liga", "serie_a", "bundesliga", "ligue_1"]

POSITION_GROUPS = {
    "Goalkeeper": "goalkeeper",
    "Centre-Back": "centre_back",
    "Left-Back": "full_back",
    "Right-Back": "full_back",
    "Defensive Midfield": "defensive_mid",
    "Central Midfield": "central_mid",
    "Attacking Midfield": "attacking_mid",
    "Second Striker": "attacking_mid",
    "Left Winger": "winger",
    "Right Winger": "winger",
    "Left Midfield": "winger",
    "Right Midfield": "winger",
    "Centre-Forward": "centre_forward",
}
AGE_BANDS = [0, 21, 24, 27, 30, 50]
AGE_LABELS = ["under 21", "21 to 23", "24 to 26", "27 to 29", "30 and over"]

# Measures the trait report compares. Value columns are never traits.
TRAIT_STATS = [
    "rating",
    "minutes_share",
    "goals_per90",
    "assists_per90",
    "shots_total_per90",
    "passes_key_per90",
    "passes_total_per90",
    "pass_accuracy",
    "dribbles_success_per90",
    "dribble_success_rate",
    "duels_won_per90",
    "duel_win_rate",
    "tackles_total_per90",
    "interceptions_per90",
    "blocks_per90",
    "fouls_drawn_per90",
    "saves_per90",
    "share_goals",
    "share_assists",
    "share_shots",
    "share_key_passes",
    "share_dribbles",
    "club_cup_minutes_12m",
    "intl_minutes_12m",
    "adj_rating",
]

# Version 2 candidates: the per 90 and rate measures among the traits. Pass
# accuracy is left out: the API has it only from 2020.
COMPOSITE_CANDIDATES = [
    s for s in TRAIT_STATS if s.endswith("_per90") or s in ("duel_win_rate", "dribble_success_rate")
]
COMPOSITE_MIN_GAP = 5.0  # percentile points above same age peers to enter the composite
SHRINK_FIT_MINUTES = 270  # rows used to fit the prior strength: three full matches or more
MIN_PRIOR_ROWS = 100
STRENGTH_SOURCES = ["transfer", "elo", "transfer_elo", "v2"]
DEFAULT_STRENGTH = "v2"  # decision 8: more stable, elite and step up AUC as good (MODELS.md)


# League strength and ability


def league_strength(
    features: pd.DataFrame, team_column: str = "team_gd_per_match"
) -> tuple[pd.DataFrame, float]:
    """Each league's rating effect against the Premier League (positive: a
    player rates higher there, so the league is weaker), the moves behind
    it, and the rating effect of one goal of team goal difference per match.

    Fit on consecutive seasons of the same player with 900+ minutes in both:
    rating change = league effect (new) - league effect (old) + team effect
    x team strength change + drift + age term, by least squares.
    """
    pairs = _consecutive_pairs(features, [team_column])
    leagues = sorted(set(pairs["league_key_a"]) | set(pairs["league_key_b"]))
    others = [lg for lg in leagues if lg != "epl"]
    col = {lg: i for i, lg in enumerate(others)}
    x = np.zeros((len(pairs), len(others) + 3))
    for i, (la, lb) in enumerate(zip(pairs["league_key_a"], pairs["league_key_b"], strict=True)):
        if lb in col:
            x[i, col[lb]] += 1
        if la in col:
            x[i, col[la]] -= 1
    x[:, -3] = pairs[f"{team_column}_b"] - pairs[f"{team_column}_a"]
    x[:, -2] = 1.0
    x[:, -1] = pairs["age_a"] - 25
    y = (pairs["rating_b"] - pairs["rating_a"]).to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    moved = pairs[pairs["league_key_a"] != pairs["league_key_b"]]
    moves = pd.concat([moved["league_key_a"], moved["league_key_b"]]).value_counts()
    out = pd.DataFrame({"league_key": ["epl", *others], "raw_effect": [0.0, *coef[: len(others)]]})
    out["moves"] = out["league_key"].map(moves).fillna(0).astype(int)
    small = out[~out["league_key"].isin(BIG_FIVE) & (out["moves"] >= MIN_MOVES)]
    prior = float(small["raw_effect"].median()) if len(small) else 0.0
    weight = out["moves"] / (out["moves"] + MIN_MOVES)
    out["effect"] = np.where(
        out["league_key"] == "epl", 0.0, weight * out["raw_effect"] + (1 - weight) * prior
    )
    return out.sort_values("effect").reset_index(drop=True), float(coef[-3])


def position_group(features: pd.DataFrame) -> pd.Series:
    return features["sub_position"].map(POSITION_GROUPS)


def _consecutive_pairs(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The same player in consecutive seasons, 900+ minutes in both (one row
    per player and season: the league with most minutes)."""
    r = features[(features["tier"] == "match") & (features["minutes"] >= 900)]
    r = r[r["rating"].notna() & r[columns].notna().all(axis=1)]
    r = r.sort_values("minutes").drop_duplicates(["source_player_id", "season"], keep="last")
    a = r[["source_player_id", "season", "league_key", "rating", "age", *columns]].copy()
    a["season"] = a["season"].astype(int)
    for column in ("rating", "age", *columns):
        a[column] = a[column].astype(float)
    return a.merge(
        a.assign(season=a["season"] - 1), on=["source_player_id", "season"], suffixes=("_a", "_b")
    ).dropna()


def elo_terms(features: pd.DataFrame, teams: pd.DataFrame, leagues: pd.DataFrame) -> pd.DataFrame:
    """Per row, in Elo points / 100: ``elo_league`` (league mean club rating
    that season minus the Premier League's) and ``elo_team`` (the club's
    season rating minus its league's). A club missing from the Elo table
    counts as its league's average."""
    keys = ["league_key", "season"]
    lg = leagues[[*keys, "elo"]].copy()
    epl = lg[lg["league_key"] == "epl"].set_index("season")["elo"]
    lg["elo_league"] = (lg["elo"] - lg["season"].map(epl)) / 100
    tm = teams[[*keys, "team_id", "elo"]].rename(columns={"elo": "team_elo"})
    rows = features[[*keys, "team_id"]].copy()
    rows["season"] = rows["season"].astype(int)
    rows["team_id"] = rows["team_id"].astype("Int64")
    tm = tm.assign(season=tm["season"].astype(int), team_id=tm["team_id"].astype("Int64"))
    lg = lg.assign(season=lg["season"].astype(int))
    rows = rows.merge(lg[[*keys, "elo", "elo_league"]], on=keys, how="left")
    rows = rows.merge(tm, on=[*keys, "team_id"], how="left")
    out = pd.DataFrame(index=features.index)
    out["elo_league"] = rows["elo_league"].to_numpy()
    out["elo_team"] = ((rows["team_elo"] - rows["elo"]) / 100).fillna(0).to_numpy()
    out.loc[out["elo_league"].isna(), "elo_team"] = np.nan
    return out


def elo_league_strength(
    features: pd.DataFrame,
    terms: pd.DataFrame,
    team_column: str = "elo_team",
    movers_only: bool = True,
) -> tuple[pd.DataFrame, float, float]:
    """Rating effect per 100 Elo of league strength (``league_coef``,
    negative: a stronger league rates its players lower) and of team strength
    within the league (``team_coef``), fit on the same player pairs as
    ``league_strength``: rating change = league_coef x league Elo change +
    team_coef x team Elo change + drift + age term. Returns the per league
    and season effect against the Premier League, in rating units.

    With ``movers_only`` (the default) the league term counts only for
    players who changed leagues, as in ``league_strength``: a player who
    stays sees his league's Elo drift against the Premier League from season
    to season, but his rating, measured against the same opponents, does
    not follow, and those pairs pulled the coefficient to half its size
    (reports/elo.md)."""
    data = pd.concat([features.drop(columns=terms.columns, errors="ignore"), terms], axis=1)
    pairs = _consecutive_pairs(data, ["elo_league", team_column])
    league_change = pairs["elo_league_b"] - pairs["elo_league_a"]
    if movers_only:
        league_change = league_change.where(pairs["league_key_a"] != pairs["league_key_b"], 0.0)
    x = np.column_stack(
        [
            league_change,
            pairs[f"{team_column}_b"] - pairs[f"{team_column}_a"],
            np.ones(len(pairs)),
            pairs["age_a"] - 25,
        ]
    )
    y = (pairs["rating_b"] - pairs["rating_a"]).to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    league_coef, team_coef = float(coef[0]), float(coef[1])
    table = (
        data[["league_key", "season", "elo_league"]]
        .dropna()
        .drop_duplicates(["league_key", "season"])
        .assign(season=lambda d: d["season"].astype(int))
    )
    table["effect"] = league_coef * table["elo_league"]
    table = table.sort_values(["season", "effect"]).reset_index(drop=True)
    return table, league_coef, team_coef


def ability(features: pd.DataFrame, strength: pd.DataFrame, team_effect: float) -> pd.DataFrame:
    """adj_rating and ability_pct per row with the transfer based league
    strength (match tier, rated rows only)."""
    effect = features["league_key"].map(strength.set_index("league_key")["effect"])
    team = team_effect * features["team_gd_per_match"].astype(float).fillna(0)
    return ability_from(features, effect.astype(float), team)


def ability_elo(
    features: pd.DataFrame, terms: pd.DataFrame, league_coef: float, team_coef: float
) -> pd.DataFrame:
    """adj_rating and ability_pct with the Elo based league and team terms."""
    return ability_from(features, league_coef * terms["elo_league"], team_coef * terms["elo_team"])


def ability_from(features: pd.DataFrame, effect: pd.Series, team: pd.Series) -> pd.DataFrame:
    """Rating minus the league term and the team term, shrunk by minutes, as a
    percentile within position group and season."""
    out = pd.DataFrame(index=features.index)
    raw = features["rating"].astype(float) - effect - team
    group = position_group(features)
    minutes = features["minutes"].astype(float).fillna(0)
    usable = raw.notna() & group.notna() & (minutes > 0)
    mean = raw[usable].groupby([group[usable], features.loc[usable, "season"]]).transform("mean")
    shrunk = (minutes[usable] * raw[usable] + SHRINK_MINUTES * mean) / (
        minutes[usable] + SHRINK_MINUTES
    )
    out["adj_rating"] = shrunk.reindex(features.index)
    regular = features["regular"].astype(bool) & usable
    pct = (
        out.loc[regular, "adj_rating"]
        .groupby([group[regular], features.loc[regular, "season"]])
        .rank(pct=True)
        .mul(100)
    )
    out["ability_pct"] = pct.reindex(features.index)
    out["league_effect"] = effect.astype(float)
    return out


def ability_transfer_elo(features: pd.DataFrame, terms: pd.DataFrame) -> tuple:
    """Transfer based league effect, Elo team term, both fitted together on
    the rating. Returns (scores, strength, team_coef)."""
    strength, team_coef = league_strength(
        pd.concat([features.drop(columns=terms.columns, errors="ignore"), terms], axis=1),
        team_column="elo_team",
    )
    effect = features["league_key"].map(strength.set_index("league_key")["effect"]).astype(float)
    return ability_from(features, effect, team_coef * terms["elo_team"]), strength, team_coef


# Ability version 2: per 90 composite


def composite_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """The candidate measures on match tier rows. The API leaves a count
    empty rather than writing 0 (39% of regulars have no goals entry), so a
    missing per 90 count reads as 0 where the league season records that
    count for anyone; a league season without it stays missing."""
    match = (features["tier"] == "match") & (features["minutes"].astype(float).fillna(0) > 0)
    out = features[COMPOSITE_CANDIDATES].astype(float).where(match)
    keys = [features["league_key"], features["season"]]
    for stat in COMPOSITE_CANDIDATES:
        if not stat.endswith("_per90"):
            continue  # a rate without attempts is unknown, not 0
        recorded = out[stat].notna().groupby(keys).transform("any")
        out.loc[match & recorded & out[stat].isna(), stat] = 0.0
    return out


def composite_weights(
    features: pd.DataFrame, inputs: pd.DataFrame, step_up: pd.Series
) -> pd.DataFrame:
    """Per position group, the measures on which players who made the step up
    stood at least ``COMPOSITE_MIN_GAP`` percentiles above same age peers
    (``trait_table`` on league percentiles), weighted by that gap."""
    frame = pd.concat(
        [features[["league_key", "season", "sub_position", "regular"]], inputs], axis=1
    )
    data = league_percentiles(frame, COMPOSITE_CANDIDATES).add_prefix("pct_")
    data["group"] = position_group(features)
    data["age_band"] = pd.cut(features["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    data["step_up"] = step_up.where(features["tier"] == "match")  # rows with the measures
    table = trait_table(data, "step_up", top=len(COMPOSITE_CANDIDATES), with_rating=False)
    if table.empty:
        raise ValueError("too few step ups to choose the version 2 measures")
    table = table[table["gap"] >= COMPOSITE_MIN_GAP].copy()
    table["weight"] = table["gap"] / table.groupby("position")["gap"].transform("sum")
    return table.reset_index(drop=True)


def composite(features: pd.DataFrame, inputs: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Weighted mean of the measures, each standardised within position group
    and season against that season's regulars. Needs at least half of the
    group's weight present."""
    group = position_group(features)
    season = features["season"].astype(int)
    regular = features["regular"].astype(bool) & group.notna()
    total = pd.Series(0.0, index=features.index)
    weight = pd.Series(0.0, index=features.index)
    for stat in weights["measure"].unique():
        values = inputs[stat]
        ref = values[regular].groupby([group[regular], season[regular]]).agg(["mean", "std"])
        at = ref.reindex(pd.MultiIndex.from_arrays([group, season]))
        z = (values - at["mean"].to_numpy()) / at["std"].replace(0, np.nan).to_numpy()
        w = group.map(weights[weights["measure"] == stat].set_index("position")["weight"])
        w = w.fillna(0.0).where(z.notna(), 0.0)
        total += (w * z.fillna(0)).astype(float)
        weight += w.astype(float)
    return (total / weight).where(weight >= 0.5)


def prior_minutes(score: pd.Series, minutes: pd.Series, keys: list[pd.Series]) -> float:
    """Empirical Bayes prior strength, in minutes: a season's score scatters
    around the position mean with variance tau^2 + sigma^2 / minutes (true
    spread plus sampling noise). Least squares of the squared deviation on
    1 / minutes gives both; ``sigma^2 / tau^2`` is the number of minutes at
    which a score is half data, half position mean."""
    use = score.notna() & (minutes >= SHRINK_FIT_MINUTES)
    dev2 = (score - score[use].groupby([k[use] for k in keys]).transform("mean")) ** 2
    x = np.column_stack([np.ones(int(use.sum())), 1.0 / minutes[use].to_numpy()])
    (tau2, sigma2), *_ = np.linalg.lstsq(x, dev2[use].to_numpy(), rcond=None)
    if tau2 <= 0:
        raise ValueError("no true spread left after the sampling noise; cannot shrink")
    return float(max(sigma2, 0.0) / tau2)


def two_seasons(features: pd.DataFrame, score: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Each row's score pooled with the player's previous season (all his
    league rows that season), weighted by minutes; returns (score, minutes).
    The previous season ended before this row's feature date."""
    minutes = features["minutes"].astype(float).fillna(0)
    has = score.notna() & (minutes > 0)
    frame = pd.DataFrame(
        {
            "pid": features["source_player_id"].astype(int),
            "season": features["season"].astype(int),
            "m": minutes.where(has, 0.0),
            "sm": (score * minutes).where(has, 0.0),
        }
    )
    prev = frame.groupby(["pid", "season"])[["m", "sm"]].sum()
    prev.index = prev.index.set_levels(prev.index.levels[1] + 1, level=1)
    at = prev.reindex(pd.MultiIndex.from_frame(frame[["pid", "season"]])).fillna(0.0)
    m_prev, sm_prev = at["m"].to_numpy(), at["sm"].to_numpy()
    total = frame["m"] + m_prev
    pooled = (frame["sm"] + sm_prev) / total.where(total > 0)
    return pooled.where(has), total.where(has)


def ability_v2(
    features: pd.DataFrame,
    terms: pd.DataFrame,
    weights: pd.DataFrame,
    pool: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """Version 2: the per 90 composite, less the transfer league effect and
    the Elo team term (both refitted in composite units on the same player
    pairs as the rating version), pooled over two seasons by minutes, shrunk
    towards the position mean with the fitted prior strength, as a
    percentile within position group and season. Returns (scores, league
    strength, team coefficient, prior minutes per position group)."""
    inputs = composite_inputs(features)
    base = composite(features, inputs, weights)
    on_base = features.drop(columns=terms.columns, errors="ignore").assign(rating=base)
    strength, team_coef = league_strength(
        pd.concat([on_base, terms], axis=1), team_column="elo_team"
    )
    effect = features["league_key"].map(strength.set_index("league_key")["effect"]).astype(float)
    adjusted = base - effect - team_coef * terms["elo_team"]
    group = position_group(features)
    season = features["season"].astype(int)
    minutes = features["minutes"].astype(float).fillna(0)
    prior = {}
    for g in sorted(group.dropna().unique()):
        rows = (group == g) & adjusted.notna()
        if rows.sum() >= MIN_PRIOR_ROWS:  # a group without measures has no score
            prior[g] = prior_minutes(adjusted[rows], minutes[rows], [season[rows]])
    score, used = two_seasons(features, adjusted) if pool else (adjusted, minutes)
    usable = score.notna() & group.notna() & (used > 0)
    mean = adjusted[usable].groupby([group[usable], season[usable]]).transform("mean")
    k = group[usable].map(prior)
    shrunk = mean + used[usable] / (used[usable] + k) * (score[usable] - mean)
    out = pd.DataFrame(index=features.index)
    out["adj_rating"] = shrunk.reindex(features.index)  # the adjusted base, here the composite
    regular = features["regular"].astype(bool) & usable
    out["ability_pct"] = (
        out.loc[regular, "adj_rating"]
        .groupby([group[regular], season[regular]])
        .rank(pct=True)
        .mul(100)
        .reindex(features.index)
    )
    out["league_effect"] = effect
    return out, strength, team_coef, prior


def ability_scores(
    features: pd.DataFrame,
    source: str,
    elo_tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    step_up: pd.Series | None = None,
    pool: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """The ability score of one version: (scores, league strength, team
    coefficient, details). ``v2`` needs the step up labels for its weights."""
    if source == "transfer":
        strength, team_effect = league_strength(features)
        return ability(features, strength, team_effect), strength, team_effect, {}
    if elo_tables is None:
        raise ValueError(f"strength {source!r} needs the Elo tables (run `scout elo` first)")
    terms = elo_terms(features, *elo_tables)
    if source == "elo":
        strength, league_coef, team_coef = elo_league_strength(features, terms)
        return ability_elo(features, terms, league_coef, team_coef), strength, team_coef, {}
    if source == "transfer_elo":
        return (*ability_transfer_elo(features, terms), {})
    if source == "v2":
        if step_up is None:
            raise ValueError("ability v2 needs the step up labels to choose its measures")
        weights = composite_weights(features, composite_inputs(features), step_up)
        scores, strength, team_coef, prior = ability_v2(features, terms, weights, pool=pool)
        return scores, strength, team_coef, {"weights": weights, "prior_minutes": prior}
    raise ValueError(f"unknown strength source {source!r}; one of {STRENGTH_SOURCES}")


def team_shares(features: pd.DataFrame) -> pd.DataFrame:
    """A player's share of his team's goals, assists, shots, key passes and
    successful dribbles in the league season: carrying a weak team shows here
    even when his per 90 numbers look ordinary (Raphinha at Leeds)."""
    out = pd.DataFrame(index=features.index)
    keys = ["league_key", "season", "team_id"]
    for name, column in {
        "share_goals": "goals",
        "share_assists": "assists",
        "share_shots": "shots_total",
        "share_key_passes": "passes_key",
        "share_dribbles": "dribbles_success",
    }.items():
        values = features[column].astype(float)
        total = values.groupby([features[k] for k in keys]).transform("sum")
        out[name] = values / total.where(total > 0)
    return out


def league_percentiles(features: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Percentile of each measure among regular players of the same league,
    season and position group (0 to 100)."""
    group = position_group(features)
    regular = features["regular"].astype(bool) & group.notna()
    out = pd.DataFrame(index=features.index)
    keys = [features.loc[regular, "league_key"], features.loc[regular, "season"], group[regular]]
    for stat in stats:
        values = features.loc[regular, stat].astype(float)
        out[stat] = values.groupby(keys).rank(pct=True).mul(100).reindex(features.index)
    return out


# Targets


def value_ranks(features: pd.DataFrame) -> pd.Series:
    """Rank of each row's market value among one row per player per season."""
    per_player = features.sort_values("minutes").drop_duplicates(
        ["season", "source_player_id"], keep="last"
    )
    rank = per_player.groupby("season")["market_value"].rank(ascending=False, method="first")
    by_player = pd.Series(
        rank.to_numpy(), index=pd.MultiIndex.from_frame(per_player[["season", "source_player_id"]])
    )
    keys = pd.MultiIndex.from_frame(features[["season", "source_player_id"]])
    return pd.Series(by_player.reindex(keys).to_numpy(), index=features.index)


def _rank_of(values: pd.Series, seasons: pd.Series, table: dict[int, np.ndarray]) -> pd.Series:
    """Where a value would rank in a season's value list (1 is the highest)."""
    out = pd.Series(np.nan, index=values.index)
    for season, sorted_desc in table.items():
        mask = (seasons == season) & values.notna()
        if mask.any():
            # number of values strictly above, plus one
            above = len(sorted_desc) - np.searchsorted(
                sorted_desc[::-1], values[mask], side="right"
            )
            out[mask] = above + 1
    return out


def value_table(features: pd.DataFrame) -> dict[int, np.ndarray]:
    """Each season's market values, one per player, highest first."""
    per_season = features.sort_values("minutes").drop_duplicates(
        ["season", "source_player_id"], keep="last"
    )
    return {
        int(s): np.sort(g["market_value"].dropna().to_numpy())[::-1]
        for s, g in per_season.groupby("season")
    }


def rank_after(
    rows: pd.DataFrame, valuations: pd.DataFrame, table: dict, years: int
) -> tuple[pd.Series, pd.Series]:
    """Where each row's value ``years`` after its feature date ranks in that
    season's value list; returns (rank, the date looked at)."""
    when = rows.assign(_t=rows["feature_date"] + pd.DateOffset(years=years))
    found = value_asof(when, valuations, "_t")["market_value_eur"]
    return _rank_of(found, rows["season"].astype(int) + years, table), when["_t"]


def fall_back_label(value_rank: pd.Series, end_rank: pd.Series, reachable: pd.Series) -> pd.Series:
    """1 when an elite candidate's value rank at the end of the window is at
    least ``FALL_BACK_FACTOR`` times worse; missing outside the candidates or
    when the end lies past the data."""
    within_reach = value_rank.between(ELITE_RANK + 1, ELITE_REACH)
    known = reachable & end_rank.notna() & value_rank.notna()
    fell = end_rank >= FALL_BACK_FACTOR * value_rank
    return fell.astype(float).where(known & within_reach)


def fall_back_after(
    features: pd.DataFrame, valuations: pd.DataFrame, data_end: pd.Timestamp, years: int
) -> pd.Series:
    """The fall back label over another window (the three year check)."""
    rows = features[["season", "feature_date", "tm_player_id"]].copy()
    value_rank = value_ranks(features)
    end_rank, when = rank_after(rows, valuations, value_table(features), years)
    return fall_back_label(value_rank, end_rank, when <= data_end + LABEL_TOLERANCE)


def targets(
    features: pd.DataFrame,
    extra: pd.DataFrame,
    valuations: pd.DataFrame,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """elite, step_up, fall_back and performance labels per row, with the
    date each is known (``label_date``: two seasons after the feature date)."""
    out = features[[*KEYS, "feature_date", "tm_player_id", "regular"]].copy()
    out["value_rank"] = value_ranks(features)
    out["label_date"] = out["feature_date"] + pd.DateOffset(years=HORIZON_SEASONS)

    table = value_table(features)
    best = pd.Series(np.inf, index=out.index)
    reachable = pd.Series(True, index=out.index)
    for k in range(1, HORIZON_SEASONS + 1):
        rank, when = rank_after(out, valuations, table, k)
        best = best.where(~(rank < best), rank)
        reachable &= when <= data_end + LABEL_TOLERANCE
    best = best.replace(np.inf, np.nan)
    out["best_rank_2y"] = best
    known = reachable & out["value_rank"].notna() & best.notna()
    # Elite only within reach: from below #1,000 it happened 34 times in 15
    # seasons, too rare to learn; those players have the step up target.
    within_reach = out["value_rank"].between(ELITE_RANK + 1, ELITE_REACH)
    out["elite"] = (best <= ELITE_RANK).astype(float).where(known & within_reach)
    step = (best <= STEP_UP_RANK) & (out["value_rank"] / best >= STEP_UP_FACTOR)
    out["step_up"] = step.astype(float).where(known & (out["value_rank"] > ELITE_RANK))
    out["fall_back"] = fall_back_label(out["value_rank"], rank, reachable)

    # Performance: a rise that holds. Half of all one season rises were gone a
    # season later (the score's season to season correlation is 0.53), so both
    # of the next two seasons must be good and their average well above now.
    # A season not rated as a regular counts as 0: it breaks the rise.
    later, seen = season_scores(features, extra["ability_pct"], range(1, HORIZON_SEASONS + 1))
    later = later.fillna(0.0)
    now = extra["ability_pct"]
    rose = (later.min(axis=1) >= PERF_LEVEL) & (later.mean(axis=1) - now >= PERF_RISE)
    # Never seen again (left our leagues) and seasons too new to have two
    # later ones are unknown.
    complete = out["season"].astype(int) + HORIZON_SEASONS <= int(features["season"].max())
    out["performance"] = rose.astype(float).where(now.notna() & seen & complete)
    out["ability_pct"] = now
    if "ability_pct_1s" in extra:
        share = features.get("minutes_share", pd.Series(np.nan, index=features.index))
        rise = riser_target(features, extra["ability_pct_1s"], share)
        out["riser_base"], out["riser"] = rise["riser_base"], rise["riser"]
    else:
        out["riser_base"] = out["riser"] = np.nan
    return out


def season_scores(
    features: pd.DataFrame, values: pd.Series, offsets
) -> tuple[pd.DataFrame, pd.Series]:
    """Each row's player score ``k`` seasons away for every ``k`` in
    ``offsets`` (his best league row that season; missing when he has no
    score then), and whether he has any row in a later season."""
    pid = features["source_player_id"].astype(int)
    season = features["season"].astype(int)
    best = pd.DataFrame({"pid": pid, "season": season, "v": values}).groupby(["pid", "season"])
    best = best["v"].max()
    present = set(best.index)
    found = best.dropna().to_dict()
    pairs = list(zip(pid.tolist(), season.tolist(), strict=True))
    out = pd.DataFrame(
        {k: [found.get((p, s + k), np.nan) for p, s in pairs] for k in offsets},
        index=features.index,
    )
    seen = pd.Series(
        [any((p, s + k) in present for k in offsets if k > 0) for p, s in pairs],
        index=features.index,
    )
    return out, seen


def riser_target(
    features: pd.DataFrame, one_season: pd.Series, minutes_share: pd.Series
) -> pd.DataFrame:
    """Decision 3, on the one season ability score (version 2 not pooled):
    a riser's next season beats the best of his current and previous one
    season scores by ``PERF_RISE`` percentiles, and the season after stays
    at that level. Only players with at least half of team minutes in the
    current season. Returns ``riser_base`` (the best of the two seasons,
    known at the feature date) and ``riser``."""
    scores, seen = season_scores(features, one_season, [-1, 0, 1, 2])
    base = scores[[-1, 0]].max(axis=1).where(scores[0].notna())
    later = scores[[1, 2]].fillna(0.0)  # a season not rated as a regular breaks the rise
    rose = later.min(axis=1) >= base + PERF_RISE
    complete = features["season"].astype(int) + HORIZON_SEASONS <= int(features["season"].max())
    floor = minutes_share.astype(float) >= RISER_MIN_SHARE
    known = base.notna() & seen & complete & floor
    return pd.DataFrame(
        {"riser_base": base, "riser": rose.astype(float).where(known)}, index=features.index
    )


# Models and backtest

TARGET_NAMES = ["elite", "step_up", "performance"]
# Decision 3's riser replaces the performance list; ``performance`` stays in
# the backtest as the old definition. Version 2 only.
SCOUTING_TARGETS = [*TARGET_NAMES, "riser"]
# Known at the feature date but given to the riser model only, so the other
# targets keep their columns.
RISER_COLUMNS = ["ability_pct_1s", "riser_base"]
LABEL_COLUMNS = [
    "value_rank",
    "best_rank_2y",
    "label_date",
    "fall_back",
    "riser_base",
    *SCOUTING_TARGETS,
]


def classifier(rounds: int = 400):
    params = {
        "objective": "binary",
        "seed": 7,
        "deterministic": True,
        "verbosity": -1,
        **DEFAULT_PARAMS,
    }

    def model(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
        booster = lgb.train(params, lgb.Dataset(train[columns], label=train["y"]), rounds)
        return booster.predict(test[columns])

    return model


def model_columns(data: pd.DataFrame) -> list[str]:
    skip = NOT_FEATURES | set(KEYS) | {"y", "label_date", "best_rank_2y", *SCOUTING_TARGETS}
    skip |= {"regular"}
    skip |= {"fall_back", *RISER_COLUMNS}
    skip |= {c for c in data.columns if c.startswith("pct_")}
    return [c for c in data.columns if c not in skip and data[c].dtype != object]


RANK_BANDS = [0, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, np.inf]
MIN_GROUP = 30


def rank_rate(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Baseline: the success rate in the training seasons of players with a
    similar value rank and age, falling back to the value rank alone and then
    to the overall rate. It knows what the market knows and nothing else."""

    def bands(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rank_band": pd.cut(frame["value_rank"], RANK_BANDS, labels=False).fillna(-1),
                "age_band": pd.cut(frame["age"], AGE_BANDS, right=False, labels=False).fillna(-1),
                "y": frame.get("y", np.nan),
            },
            index=frame.index,
        )

    tr, te = bands(train), bands(test)
    pred = pd.Series(np.nan, index=test.index)
    for keys in (["rank_band", "age_band"], ["rank_band"]):
        stats = tr.groupby(keys)["y"].agg(["mean", "size"])
        stats = stats[stats["size"] >= MIN_GROUP]["mean"]
        found = stats.reindex(pd.MultiIndex.from_frame(te[keys]) if len(keys) > 1 else te[keys[0]])
        pred = pred.fillna(pd.Series(found.to_numpy(), index=test.index))
    return pred.fillna(train["y"].mean()).to_numpy()


def backtest(data: pd.DataFrame, columns: list[str], models: dict | None = None) -> pd.DataFrame:
    """Season by season: each test season scored by models trained on seasons
    at least two earlier, purged of labels observed after the test date.
    One ``score_<model>`` column per model; ``score`` is the LightGBM one."""
    models = models or {"lightgbm": classifier()}
    folds = fold_season(data["feature_date"])
    out = []
    for fold in expanding_folds(folds.tolist(), _Horizon()):
        train = data[folds.isin(fold.train_seasons)]
        test = data[folds == fold.test_season]
        train = train[train["label_date"] <= test["feature_date"].min()]
        check_no_leak(train, test, _Horizon())
        frame = test[[*KEYS, "player_name", "age", "feature_date", "y"]].copy()
        for name, model in models.items():
            frame[f"score_{name}"] = model(train, test, columns)
        out.append(frame)
    out = pd.concat(out, ignore_index=True)
    if "score_lightgbm" in out:
        out["score"] = out["score_lightgbm"]
    return out


class _Horizon:
    horizon_years = HORIZON_SEASONS
    label_date = "label_date"


def evaluate(predictions: pd.DataFrame, k: int = 50, score: str = "score") -> dict:
    per_season = predictions.groupby("season").apply(
        lambda g: precision_at(g["y"], g[score], k), include_groups=False
    )
    return {
        "rows": len(predictions),
        "successes": int(predictions["y"].sum()),
        "base_rate": float(predictions["y"].mean()),
        "auc": auc(predictions["y"], predictions[score]),
        "average_precision": average_precision(predictions["y"], predictions[score]),
        f"precision_top{k}": float(per_season.mean()),
    }


# Traits


RATING_STATS = {"rating", "adj_rating"}


def trait_table(
    data: pd.DataFrame, target: str, top: int = 6, with_rating: bool = True
) -> pd.DataFrame:
    """Per position group: the measures on which players who reached the
    target stood furthest above same age non achievers of the same position,
    as median league percentiles."""
    rows = []
    labelled = data[data[target].notna() & data["group"].notna()]
    for group, g in labelled.groupby("group"):
        hits = g[g[target] == 1]
        if len(hits) < 15:
            continue
        # Age matched comparison: weight the non achievers to the hits' age mix.
        mix = hits["age_band"].value_counts(normalize=True)
        rest = g[g[target] == 0]
        diffs = []
        for stat in TRAIT_STATS:
            if not with_rating and stat in RATING_STATS:
                continue
            column = f"pct_{stat}"
            if column not in g or hits[column].notna().mean() < 0.5:
                continue
            hit_median = hits[column].median()
            base = sum(
                share * rest.loc[rest["age_band"] == band, column].median()
                for band, share in mix.items()
                if (rest["age_band"] == band).any()
            )
            diffs.append((stat, hit_median, base, hit_median - base))
        diffs.sort(key=lambda d: d[3], reverse=True)
        for stat, hit, base, gap in diffs[:top]:
            rows.append(
                {
                    "position": group,
                    "successes": len(hits),
                    "measure": stat,
                    "successes_median_pct": round(hit, 0),
                    "others_same_age_pct": round(base, 0),
                    "gap": round(gap, 0),
                }
            )
    return pd.DataFrame(rows)


# Assembly


def build(
    features: pd.DataFrame,
    valuations: pd.DataFrame,
    data_end: pd.Timestamp,
    strength_source: str = DEFAULT_STRENGTH,
    elo_tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    pool: bool = True,
    weights_until: int | None = None,
):
    """Ability score of the chosen version, targets, traits. Returns (data,
    league strength, team coefficient, details of the version).
    ``weights_until`` chooses the version 2 measures from the step up labels
    of seasons up to that one only (the look ahead check)."""
    step_up = None
    if strength_source == "v2":
        blank = pd.DataFrame({"ability_pct": np.nan}, index=features.index)
        step_up = targets(features, blank, valuations, data_end)["step_up"]
        if weights_until is not None:
            step_up = step_up.where(features["season"].astype(int) <= weights_until)
    scores, strength, team_effect, details = ability_scores(
        features, strength_source, elo_tables, step_up, pool=pool
    )
    extra = pd.concat([scores, team_shares(features)], axis=1)
    if strength_source == "v2":  # the riser is defined on one season scores
        one, *_ = ability_scores(features, "v2", elo_tables, step_up, pool=False)
        extra["ability_pct_1s"] = one["ability_pct"]
    data = pd.concat([features, extra], axis=1)
    pct = league_percentiles(data, TRAIT_STATS).add_prefix("pct_")
    data = pd.concat([data, pct], axis=1)
    labels = targets(features, extra, valuations, data_end)
    data = data.merge(
        labels[[*KEYS, *LABEL_COLUMNS]],
        on=KEYS,
    )
    data["group"] = position_group(data)
    data["age_band"] = pd.cut(data["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    return data, strength, team_effect, details


def load_inputs(settings, strength_source: str = DEFAULT_STRENGTH):
    """Features, linked valuations, the valuation data end and, unless the
    version is transfer only, the Elo tables. Reads local files only."""
    from scout.providers.transfermarkt import TransfermarktValuationProvider
    from scout.tables import read_feature_table

    root = settings.paths.processed_dir / "features"
    features = read_feature_table(root / "player_season_features")
    linked = set(features["tm_player_id"].dropna().astype(int))
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    valuations = tm.valuations(None)
    valuations = valuations[valuations["source_player_id"].isin(linked)]
    elo_tables = None
    if strength_source != "transfer":
        from scout.elo import read_tables

        elo_tables = read_tables(settings.paths.processed_dir)
    return features, valuations, tm.data_end(), elo_tables


def prepare_models(data: pd.DataFrame) -> list[str]:
    for column in CATEGORICAL:
        if column in data:
            data[column] = data[column].astype("category")
    data["group"] = data["group"].astype("category")
    return model_columns(data)


def target_rows(data: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = data[data[target].notna() & data["regular"].astype(bool)]
    return rows.assign(y=rows[target]).reset_index(drop=True)


def run_scouting(settings, strength_source: str = DEFAULT_STRENGTH) -> Path:
    features, valuations, data_end, elo_tables = load_inputs(settings, strength_source)
    data, strength, team_effect, details = build(
        features, valuations, data_end, strength_source, elo_tables
    )
    columns = prepare_models(data)
    log.info("scouting: %d rows, %d model columns", len(data), len(columns))

    results, scored = {}, {}
    models = {"baseline": rank_rate, "lightgbm": classifier()}
    for target in SCOUTING_TARGETS:
        if not data[target].notna().any():
            continue  # the riser needs version 2
        rows = target_rows(data, target)
        use = columns + RISER_COLUMNS if target == "riser" else columns
        predictions = backtest(rows, use, models)
        results[target] = evaluate(predictions)
        results[f"{target} (baseline)"] = evaluate(predictions, score="score_baseline")
        scored[target] = predictions
        log.info("%s: %s", target, {k: round(v, 3) for k, v in results[target].items()})
        newest = data[
            (data["season"] == data["season"].max()) & data["regular"].astype(bool)
        ].copy()
        data.loc[newest.index, f"score_{target}"] = classifier()(rows, newest, use)

    data = data.merge(
        features[KEYS].assign(fall_back_3y=fall_back_after(features, valuations, data_end, 3)),
        on=KEYS,
        how="left",
    )
    profiled = candidate_profiles(data, scored["elite"])
    data = data.join(profiled)
    out_dir = settings.paths.processed_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        *KEYS,
        "player_name",
        "team_name",
        "group",
        "age",
        "value_rank",
        "adj_rating",
        "ability_pct",
        "ability_pct_1s",
        "riser_base",
        *SCOUTING_TARGETS,
        "fall_back",
        *profiled.columns,
    ]
    data[keep + [c for c in data.columns if c.startswith("score_")]].to_parquet(
        out_dir / "scouting.parquet", index=False
    )
    path = write_report(
        data,
        strength,
        team_effect,
        results,
        scored,
        settings.paths.reports_dir,
        strength_source,
        details,
    )
    log.info("Scouting report: %s", path)
    path = candidates.write_report(
        data, scored["elite"], data.loc[profiled.index], settings.paths.reports_dir
    )
    log.info("Candidates report: %s", path)
    return path


def candidate_profiles(data: pd.DataFrame, elite_predictions: pd.DataFrame) -> pd.DataFrame:
    """Elite chance, fall back risk and flags for the newest season's elite
    candidates (decision 2)."""
    tables = candidates.risk_tables(elite_predictions, candidates.fall_back_rows(data))
    current = data[
        (data["season"] == data["season"].max())
        & data["regular"].astype(bool)
        & data["value_rank"].between(ELITE_RANK + 1, ELITE_REACH)
        & data["score_elite"].notna()
    ]
    return candidates.profiles(current, "score_elite", tables)


def _lists(data: pd.DataFrame, size: int = 25) -> dict[str, pd.DataFrame]:
    newest = data[(data["season"] == data["season"].max()) & data["regular"].astype(bool)].copy()
    newest["value_m"] = (newest["market_value"] / 1e6).round(1)
    show = [
        "player_name",
        "league_key",
        "team_name",
        "group",
        "age",
        "value_m",
        "value_rank",
        "ability_pct",
    ]
    elite = newest[newest["value_rank"].between(ELITE_RANK + 1, ELITE_REACH)]
    elite = elite.sort_values("score_elite", ascending=False)
    step = newest[newest["value_rank"] > STEP_UP_RANK].sort_values("score_step_up", ascending=False)
    bargains = newest[(newest["ability_pct"] >= 85) & (newest["value_rank"] > 1000)]
    bargains = bargains.sort_values(["ability_pct", "value_rank"], ascending=[False, False])
    risers = newest[newest["minutes_share"] >= RISER_MIN_SHARE] if "score_riser" in newest else None
    rounding = {
        "age": 1,
        "ability_pct": 0,
        "score_elite": 3,
        "score_step_up": 3,
        "score_performance": 3,
        "score_riser": 3,
        "ability_pct_1s": 0,
        "riser_base": 0,
    }

    def fmt(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
        return frame[show + extra].head(size).round(rounding)

    lists = {
        "Investment: likely to enter the top 100 (now #101 to #1,000)": fmt(
            elite, ["score_elite", "score_step_up"]
        ),
        "Investment: likely big step up (now outside the top 500)": fmt(
            step, ["score_step_up", "score_elite"]
        ),
        "Bargains: ability in the top 15% of the position, value outside the top 1,000": fmt(
            bargains, ["score_performance"]
        ),
    }
    if risers is not None:
        title = (
            "Performance risers: next season 20+ percentiles above the best of the last two, "
            "and holding (decision 3)"
        )
        risers = risers.sort_values("score_riser", ascending=False)
        lists[title] = fmt(risers, ["ability_pct_1s", "riser_base", "score_riser"])
    return lists


def season_precision(predictions: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """Per test season: players, base rate, top ``k`` precision of the model
    and of the baseline."""
    rows = {}
    for season, g in predictions.groupby("season"):
        rows[season] = {
            "players": len(g),
            "base_rate": g["y"].mean(),
            f"top{k}": precision_at(g["y"], g["score"], k),
            f"top{k}_baseline": precision_at(g["y"], g["score_baseline"], k),
        }
    out = pd.DataFrame(rows).T
    out.loc["mean"] = out.mean()
    out.loc["mean", "players"] = out["players"].iloc[:-1].sum()
    return out


def riser_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat({"old": season_precision(old), "new": season_precision(new)}, axis=1).round(3)
    both.index = both.index.astype(str)
    both.columns = [f"{a} {b}" for a, b in both.columns]
    return both


TEAM_NOTES = {
    "transfer": "{:+.3f} rating per goal of goal difference per match",
    "elo": "{:+.3f} rating per 100 Elo above the league average",
    "transfer_elo": "{:+.3f} rating per 100 Elo above the league average",
    "v2": "{:+.3f} composite units per 100 Elo above the league average",
}
VERSION_NOTES = {
    "transfer": "version 1: match rating, transfer league effect, team goal difference",
    "elo": "match rating, Elo league and Elo team terms",
    "transfer_elo": "match rating, transfer league effect, Elo team term",
    "v2": "version 2: per 90 composite, transfer league effect, Elo team term, two seasons",
}


def write_report(
    data,
    strength,
    team_effect,
    results,
    scored,
    reports: Path,
    strength_source=DEFAULT_STRENGTH,
    details=None,
) -> Path:
    if strength_source == "elo":
        strength = strength[strength["season"] == strength["season"].max()]
    team_note = "Team strength: " + TEAM_NOTES[strength_source].format(team_effect) + "."
    parts = [
        "# Scouting report",
        "",
        "Decisions are in the module docstring of `src/scout/scouting.py` and in MODELS.md.",
        f"Ability score: {VERSION_NOTES[strength_source]} (`--strength {strength_source}`).",
        "",
        "## League strength (effect against the Premier League)",
        "",
        f"A positive effect means a player of the same level scores higher there. {team_note}",
        "",
        strength.round(3).to_markdown(index=False),
        "",
    ]
    if details and "weights" in details:
        prior = pd.Series(details["prior_minutes"], name="prior_minutes").round(0)
        parts += [
            "## Version 2 composite: measures and weights per position group",
            "",
            "From the step up trait comparison (percentiles above same age peers).",
            "",
            details["weights"].round(3).to_markdown(index=False),
            "",
            "Prior strength (minutes at which a score is half data, half position mean):",
            "",
            prior.to_frame().to_markdown(),
            "",
        ]
    parts += [
        "## Backtest of the three targets (2 seasons ahead)",
        "",
        pd.DataFrame(results).T.round(3).to_markdown(),
        "",
    ]
    if "riser" in scored:
        parts += [
            "## Performance risers: old and new definition (decision 3)",
            "",
            "Old: both next two seasons at the 70th percentile of the two season score",
            "and their average 20 above now (all regular players). New: on the one",
            "season score, the next season 20 above the best of this and last season",
            "and the season after still there, for players with half of team minutes.",
            "Top 50 precision of each season's list against the season's base rate.",
            "",
            riser_comparison(scored["performance"], scored["riser"]).to_markdown(),
            "",
        ]
    for target in [t for t in SCOUTING_TARGETS if t in scored]:
        name = target.replace("_", " ")
        parts += [
            f"## Traits: {name}",
            "",
            "With the API rating (its formula mixes all the stats):",
            "",
            trait_table(data, target).to_markdown(index=False),
            "",
            "Without it, the measures underneath:",
            "",
            trait_table(data, target, with_rating=False).to_markdown(index=False),
            "",
        ]
        top = scored[target].sort_values("score", ascending=False).groupby("season").head(25)
        hits = top[top["y"] == 1][["season", "player_name", "age"]]
        parts += [
            f"Caught in their season's top 25 ({len(hits)}):",
            "",
            ", ".join(
                f"{n} ({int(s)})" for n, s in zip(hits["player_name"], hits["season"], strict=True)
            ),
            "",
        ]
    for title, table in _lists(data).items():
        parts += [f"## {title}", "", table.to_markdown(index=False), ""]
    path = reports / "scouting.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
