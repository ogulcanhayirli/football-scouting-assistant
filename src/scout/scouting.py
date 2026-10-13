"""Scouting views on top of the feature tables: league strength, an ability
score, three career targets, the traits behind them, and two shortlists.

Decisions, each argued in SCOUTING.md:

* **League strength** comes from players who changed leagues: the change in
  their match rating when they move, with the change in team strength and
  age held fixed, measured against the Premier League. Leagues with few
  moves are pulled towards the average of the smaller leagues.
* **Ability score:** the match rating with league strength and team strength
  taken out, shrunk towards the position average for players with few
  minutes, as a percentile within the position group and season.
* **Targets,** each within 2 seasons:
  ``elite``: enters the 100 most valuable players (from outside them);
  ``step_up``: value rank at least 3 times better, ending in the top 500;
  ``performance``: ability percentile up by 20 points or more, to 75 or more.
  The first two are the market's view (resale); the third is the football
  (Bardakcı at 29: Champions League and Euro 2024, value rank still #1,137).
* **Traits** compare the players who reached a target with players of the
  same position group and age band who did not, on percentiles within their
  own league and season, so league strength and age are not the finding.
  Market value is left out of the traits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scout.features.build import value_asof
from scout.fsutil import atomic_write_text
from scout.models.backtest import expanding_folds, fold_season
from scout.models.estimators import DEFAULT_PARAMS
from scout.models.moves import auc, average_precision, precision_at
from scout.models.targets import CATEGORICAL, NOT_FEATURES

log = logging.getLogger(__name__)

KEYS = ["league_key", "season", "source_player_id"]
HORIZON_SEASONS = 2
ELITE_RANK = 100
STEP_UP_RANK = 500
STEP_UP_FACTOR = 3.0
PERF_RISE = 20.0
PERF_LEVEL = 75.0
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


# League strength and ability


def league_strength(features: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Each league's rating effect against the Premier League (positive: a
    player rates higher there, so the league is weaker), the moves behind
    it, and the rating effect of one goal of team goal difference per match.

    Fit on consecutive seasons of the same player with 900+ minutes in both:
    rating change = league effect (new) - league effect (old) + team effect
    x team strength change + drift + age term, by least squares.
    """
    r = features[(features["tier"] == "match") & (features["minutes"] >= 900)]
    r = r[r["rating"].notna() & r["team_gd_per_match"].notna()]
    r = r.sort_values("minutes").drop_duplicates(["source_player_id", "season"], keep="last")
    a = r[["source_player_id", "season", "league_key", "rating", "team_gd_per_match", "age"]].copy()
    a["season"] = a["season"].astype(int)
    for column in ("rating", "team_gd_per_match", "age"):
        a[column] = a[column].astype(float)
    pairs = a.merge(
        a.assign(season=a["season"] - 1), on=["source_player_id", "season"], suffixes=("_a", "_b")
    ).dropna()
    leagues = sorted(set(pairs["league_key_a"]) | set(pairs["league_key_b"]))
    others = [lg for lg in leagues if lg != "epl"]
    col = {lg: i for i, lg in enumerate(others)}
    x = np.zeros((len(pairs), len(others) + 3))
    for i, (la, lb) in enumerate(zip(pairs["league_key_a"], pairs["league_key_b"], strict=True)):
        if lb in col:
            x[i, col[lb]] += 1
        if la in col:
            x[i, col[la]] -= 1
    x[:, -3] = pairs["team_gd_per_match_b"] - pairs["team_gd_per_match_a"]
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


def ability(features: pd.DataFrame, strength: pd.DataFrame, team_effect: float) -> pd.DataFrame:
    """adj_rating and ability_pct per row (match tier, rated rows only)."""
    out = pd.DataFrame(index=features.index)
    effect = features["league_key"].map(strength.set_index("league_key")["effect"])
    raw = (
        features["rating"].astype(float)
        - effect.astype(float)
        - team_effect * features["team_gd_per_match"].astype(float).fillna(0)
    )
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


def targets(
    features: pd.DataFrame,
    extra: pd.DataFrame,
    valuations: pd.DataFrame,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """elite, step_up and performance labels per row, with the date each is
    known (``label_date``: two seasons after the feature date)."""
    out = features[[*KEYS, "feature_date", "tm_player_id", "regular"]].copy()
    out["value_rank"] = value_ranks(features)
    out["label_date"] = out["feature_date"] + pd.DateOffset(years=HORIZON_SEASONS)

    per_season = features.sort_values("minutes").drop_duplicates(
        ["season", "source_player_id"], keep="last"
    )
    table = {
        int(s): np.sort(g["market_value"].dropna().to_numpy())[::-1]
        for s, g in per_season.groupby("season")
    }
    best = pd.Series(np.inf, index=out.index)
    reachable = pd.Series(True, index=out.index)
    for k in range(1, HORIZON_SEASONS + 1):
        when = out.assign(_t=out["feature_date"] + pd.DateOffset(years=k))
        found = value_asof(when, valuations, "_t")["market_value_eur"]
        rank = _rank_of(found, out["season"].astype(int) + k, table)
        best = best.where(~(rank < best), rank)
        reachable &= when["_t"] <= data_end + LABEL_TOLERANCE
    best = best.replace(np.inf, np.nan)
    out["best_rank_2y"] = best
    candidate = out["value_rank"] > ELITE_RANK
    known = reachable & out["value_rank"].notna() & best.notna()
    out["elite"] = (best <= ELITE_RANK).astype(float).where(known & candidate)
    step = (best <= STEP_UP_RANK) & (out["value_rank"] / best >= STEP_UP_FACTOR)
    out["step_up"] = step.astype(float).where(known & candidate)

    # Performance: the best ability percentile in the next two seasons.
    pct = (
        pd.DataFrame(
            {
                "pid": features["source_player_id"].astype(int),
                "season": features["season"].astype(int),
                "pct": extra["ability_pct"],
            }
        )
        .groupby(["pid", "season"])["pct"]
        .max()
    )
    present = set(pct.index)
    pct = pct.dropna().to_dict()
    pid = out["source_player_id"].astype(int).tolist()
    season = out["season"].astype(int).tolist()
    future = pd.Series(np.nan, index=out.index)
    seen = pd.Series(False, index=out.index)
    for k in range(1, HORIZON_SEASONS + 1):
        later = pd.Series(
            [pct.get((p, s + k), np.nan) for p, s in zip(pid, season, strict=True)], index=out.index
        )
        seen |= pd.Series(
            [(p, s + k) in present for p, s in zip(pid, season, strict=True)], index=out.index
        )
        future = pd.concat([future, later], axis=1).max(axis=1)
    now = extra["ability_pct"]
    rose = (future - now >= PERF_RISE) & (future >= PERF_LEVEL)
    # Seen again but never rated as a regular counts as no rise; never seen
    # again (left our leagues) and seasons too new to have two later ones are unknown.
    complete = out["season"].astype(int) + HORIZON_SEASONS <= int(features["season"].max())
    out["performance"] = rose.astype(float).where(now.notna() & seen & complete)
    out["ability_pct"] = now
    return out


# Models and backtest

TARGET_NAMES = ["elite", "step_up", "performance"]


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
    skip = NOT_FEATURES | set(KEYS) | {"y", "label_date", "best_rank_2y", *TARGET_NAMES, "regular"}
    skip |= {c for c in data.columns if c.startswith("pct_")}
    return [c for c in data.columns if c not in skip and data[c].dtype != object]


def backtest(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    folds = fold_season(data["feature_date"])
    model = classifier()
    out = []
    for fold in expanding_folds(folds.tolist(), _Horizon()):
        train = data[folds.isin(fold.train_seasons)]
        test = data[folds == fold.test_season]
        train = train[train["label_date"] <= test["feature_date"].min()]
        frame = test[[*KEYS, "player_name", "age", "y"]].copy()
        frame["score"] = model(train, test, columns)
        out.append(frame)
    return pd.concat(out, ignore_index=True)


class _Horizon:
    horizon_years = HORIZON_SEASONS


def evaluate(predictions: pd.DataFrame, k: int = 50) -> dict:
    per_season = predictions.groupby("season").apply(
        lambda g: precision_at(g["y"], g["score"], k), include_groups=False
    )
    return {
        "rows": len(predictions),
        "successes": int(predictions["y"].sum()),
        "base_rate": float(predictions["y"].mean()),
        "auc": auc(predictions["y"], predictions["score"]),
        "average_precision": average_precision(predictions["y"], predictions["score"]),
        f"precision_top{k}": float(per_season.mean()),
    }


# Traits


def trait_table(data: pd.DataFrame, target: str, top: int = 6) -> pd.DataFrame:
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


def build(features: pd.DataFrame, valuations: pd.DataFrame, data_end: pd.Timestamp):
    strength, team_effect = league_strength(features)
    extra = pd.concat([ability(features, strength, team_effect), team_shares(features)], axis=1)
    data = pd.concat([features, extra], axis=1)
    pct = league_percentiles(data, TRAIT_STATS).add_prefix("pct_")
    data = pd.concat([data, pct], axis=1)
    labels = targets(features, extra, valuations, data_end)
    data = data.merge(
        labels[[*KEYS, "value_rank", "best_rank_2y", "label_date", *TARGET_NAMES]], on=KEYS
    )
    data["group"] = position_group(data)
    data["age_band"] = pd.cut(data["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    return data, strength, team_effect


def run_scouting(settings) -> Path:
    from scout.providers.transfermarkt import TransfermarktValuationProvider
    from scout.tables import read_feature_table

    root = settings.paths.processed_dir / "features"
    features = read_feature_table(root / "player_season_features")
    linked = set(features["tm_player_id"].dropna().astype(int))
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    valuations = tm.valuations(None)
    valuations = valuations[valuations["source_player_id"].isin(linked)]
    data_end = pd.Timestamp("2026-06-12")
    data, strength, team_effect = build(features, valuations, data_end)
    for column in CATEGORICAL:
        if column in data:
            data[column] = data[column].astype("category")
    data["group"] = data["group"].astype("category")
    columns = model_columns(data)
    log.info("scouting: %d rows, %d model columns", len(data), len(columns))

    results, scored = {}, {}
    for target in TARGET_NAMES:
        rows = data[data[target].notna() & data["regular"].astype(bool)].assign(
            y=lambda d, t=target: d[t]
        )
        rows = rows.reset_index(drop=True)
        predictions = backtest(rows, columns)
        results[target] = evaluate(predictions)
        scored[target] = predictions
        log.info("%s: %s", target, {k: round(v, 3) for k, v in results[target].items()})
        newest = data[
            (data["season"] == data["season"].max()) & data["regular"].astype(bool)
        ].copy()
        data.loc[newest.index, f"score_{target}"] = classifier()(rows, newest, columns)

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
        *TARGET_NAMES,
    ]
    data[keep + [c for c in data.columns if c.startswith("score_")]].to_parquet(
        out_dir / "scouting.parquet", index=False
    )
    path = write_report(data, strength, team_effect, results, scored, settings.paths.reports_dir)
    log.info("Scouting report: %s", path)
    return path


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
    elite = newest[newest["value_rank"] > ELITE_RANK].sort_values("score_elite", ascending=False)
    step = newest[newest["value_rank"] > STEP_UP_RANK].sort_values("score_step_up", ascending=False)
    bargains = newest[(newest["ability_pct"] >= 85) & (newest["value_rank"] > 1000)]
    bargains = bargains.sort_values(["ability_pct", "value_rank"], ascending=[False, False])
    rising = newest.sort_values("score_performance", ascending=False)
    rounding = {
        "age": 1,
        "ability_pct": 0,
        "score_elite": 3,
        "score_step_up": 3,
        "score_performance": 3,
    }

    def fmt(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
        return frame[show + extra].head(size).round(rounding)

    return {
        "Investment: likely to enter the top 100 (now outside it)": fmt(
            elite, ["score_elite", "score_step_up"]
        ),
        "Investment: likely big step up (now outside the top 500)": fmt(
            step, ["score_step_up", "score_elite"]
        ),
        "Bargains: ability in the top 15% of the position, value outside the top 1,000": fmt(
            bargains, ["score_performance"]
        ),
        "Performance risers: likely to climb 20+ ability percentiles": fmt(
            rising, ["score_performance"]
        ),
    }


def write_report(data, strength, team_effect, results, scored, reports: Path) -> Path:
    parts = [
        "# Scouting report",
        "",
        "See SCOUTING.md for every decision behind these tables.",
        "",
        "## League strength (rating effect against the Premier League)",
        "",
        f"A positive effect means a player of the same level rates higher there. "
        f"Team strength: {team_effect:+.3f} rating per goal of goal difference per match.",
        "",
        strength.round(3).to_markdown(index=False),
        "",
        "## Backtest of the three targets (2 seasons ahead)",
        "",
        pd.DataFrame(results).T.round(3).to_markdown(),
        "",
    ]
    for target in TARGET_NAMES:
        table = trait_table(data, target)
        parts += [f"## Traits: {target.replace('_', ' ')}", "", table.to_markdown(index=False), ""]
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
