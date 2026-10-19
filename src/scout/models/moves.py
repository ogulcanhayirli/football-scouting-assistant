"""The big move: will a player who is not at a top club join one within 24 months?

* **Top club:** the ``TOP_N`` clubs with the highest squad market value at
  a season's feature date, over every league we track (the sum of their
  players' values on that date, each player counted once). The list is
  recomputed every season, so it only uses what was known then.
* **Rows:** regular players whose club at the feature date is not top.
* **Label:** 1 when the API lists a move (transfer, loan or free) to a club
  that was top in the row's season and that the player had never left
  before (so not a loan return or a homecoming), dated after the feature
  date and within ``MONTHS`` months; 0 otherwise. Missing when the window
  ends after the transfer data does.

The backtest is the same expanding window as for value (D16, D18): a label
is treated as observed at the end of its window, so training rows whose
window closes after a test row's feature date are purged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scout.fsutil import atomic_write_text
from scout.models.backtest import expanding_folds, fold_season
from scout.models.estimators import AGE_BANDS, DEFAULT_PARAMS, MIN_GROUP
from scout.models.targets import CATEGORICAL, Target, feature_columns

log = logging.getLogger(__name__)

TOP_N = 20
MONTHS = 24
MOVE_KINDS = {"transfer", "loan", "free", "unknown"}  # a loan return is not a new move
TARGET = Target("big_move_24m", "big_move_24m", 2, "move_label_date", True)
KEYS = ["league_key", "season", "source_player_id"]


def top_clubs(features: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """season, team_id, team_name, squad_value, rank for the top ``n`` clubs."""
    last = features.sort_values("minutes").drop_duplicates(
        ["season", "source_player_id"], keep="last"
    )
    squads = (
        last.groupby(["season", "team_id", "team_name"])["market_value"]
        .sum()
        .rename("squad_value")
        .reset_index()
    )
    squads["rank"] = squads.groupby("season")["squad_value"].rank(ascending=False, method="first")
    return squads[squads["rank"] <= n].reset_index(drop=True)


def move_labels(
    features: pd.DataFrame,
    transfers: pd.DataFrame,
    top: pd.DataFrame,
    data_end: pd.Timestamp,
    months: int = MONTHS,
) -> pd.DataFrame:
    """One row per feature row: the label, when its window closes, and the
    first qualifying move (date and club) for positives."""
    out = features[[*KEYS, "feature_date", "team_id"]].copy()
    out["move_label_date"] = out["feature_date"] + pd.DateOffset(months=months)
    top_keys = set(zip(top["season"].astype(int), top["team_id"].astype(int), strict=True))
    at_top = [
        (int(s), int(t)) in top_keys if pd.notna(t) else False
        for s, t in zip(out["season"], out["team_id"], strict=True)
    ]
    out["at_top_club"] = at_top
    # Or he already moved to one before the feature date, while his row still
    # shows the league club he played for (Mastantuono: River Plate 2025,
    # Real Madrid from August 2025, feature date January 2026).
    out["at_top_club"] |= _joined_top_before(out, transfers, top_keys)

    moves = transfers[transfers["kind"].isin(MOVE_KINDS)].dropna(subset=["team_in_id"])
    moves = moves[["source_player_id", "transfer_date", "team_in_id", "team_in_name"]].copy()
    moves["source_player_id"] = moves["source_player_id"].astype("int64")
    moves["team_in_id"] = moves["team_in_id"].astype("int64")
    pairs = (
        out[["source_player_id", "season", "feature_date", "move_label_date"]]
        .reset_index(names="row")
        .astype({"source_player_id": "int64", "season": "int64"})
        .merge(moves, on="source_player_id")
    )
    inside = (pairs["transfer_date"] > pairs["feature_date"]) & (
        pairs["transfer_date"] <= pairs["move_label_date"]
    )
    pairs = pairs[inside]
    is_top = [(s, t) in top_keys for s, t in zip(pairs["season"], pairs["team_in_id"], strict=True)]
    pairs = pairs[is_top]
    pairs = pairs[~_is_return(pairs, transfers)]
    first = pairs.sort_values("transfer_date").groupby("row").first()

    out["big_move_24m"] = 0.0
    out.loc[first.index, "big_move_24m"] = 1.0
    out["move_date"] = first["transfer_date"].reindex(out.index)
    out["move_to"] = first["team_in_name"].reindex(out.index)
    unknown = out["at_top_club"] | (out["move_label_date"] > data_end)
    out.loc[unknown, "big_move_24m"] = np.nan
    return out.drop(columns=["team_id"])


def _joined_top_before(
    out: pd.DataFrame, transfers: pd.DataFrame, top_keys: set[tuple[int, int]]
) -> pd.Series:
    """True where the player's last move before the feature date took him to a
    club that is top in the row's season."""
    moves = transfers.dropna(subset=["source_player_id", "transfer_date"])[
        ["source_player_id", "transfer_date", "team_in_id"]
    ].copy()
    moves["source_player_id"] = moves["source_player_id"].astype("int64")
    moves["transfer_date"] = moves["transfer_date"].astype("datetime64[ns]")
    left = out[["source_player_id", "season", "feature_date"]].reset_index(names="row").dropna()
    left = left.astype({"source_player_id": "int64", "season": "int64"})
    left["feature_date"] = left["feature_date"].astype("datetime64[ns]")
    last = pd.merge_asof(
        left.sort_values("feature_date"),
        moves.sort_values("transfer_date"),
        left_on="feature_date",
        right_on="transfer_date",
        by="source_player_id",
        direction="backward",
        allow_exact_matches=False,
    )
    joined = [
        pd.notna(t) and (s, int(t)) in top_keys
        for s, t in zip(last["season"], last["team_in_id"], strict=True)
    ]
    return pd.Series(joined, index=last["row"].to_numpy()).reindex(out.index, fill_value=False)


def _is_return(pairs: pd.DataFrame, transfers: pd.DataFrame) -> pd.Series:
    """A move to a club the player had already left: the end of a loan from
    that club (the API often types it "N/A", not "Back from loan") or a return
    to a former club. Neither is the jump this model is about. Without this
    rule a third of the "big moves" were loanees going back to their owner."""
    left = transfers.dropna(subset=["team_out_id"])[
        ["source_player_id", "team_out_id", "transfer_date"]
    ].rename(columns={"team_out_id": "team_in_id", "transfer_date": "left_on"})
    left = left.astype({"source_player_id": "int64", "team_in_id": "int64"})
    keyed = pairs.reset_index(names="pair")[
        ["pair", "source_player_id", "team_in_id", "transfer_date"]
    ]
    hits = keyed.merge(left, on=["source_player_id", "team_in_id"])
    earlier = hits[hits["left_on"] < hits["transfer_date"]]["pair"].unique()
    return pd.Series(pairs.index.isin(earlier), index=pairs.index)


# Models: each returns a score per test row, higher meaning a move is likelier.


def group_rate(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Baseline: the move rate of similar players in the training seasons
    (league, position, age band), falling back to coarser groups."""
    levels = [["league_key", "position", "age_band"], ["league_key", "age_band"], ["league_key"]]
    train = train.assign(age_band=pd.cut(train["age"], AGE_BANDS, right=False, labels=False))
    test = test.assign(age_band=pd.cut(test["age"], AGE_BANDS, right=False, labels=False))
    pred = pd.Series(np.nan, index=test.index)
    for keys in levels:
        stats = train.groupby(keys, observed=True)["y"].agg(["mean", "size"])
        stats = stats[stats["size"] >= MIN_GROUP]["mean"]
        found = (
            test[keys].apply(tuple, axis=1).map(lambda k, s=stats: s.get(k if len(k) > 1 else k[0]))
        )
        pred = pred.fillna(found.astype(float))
    return pred.fillna(train["y"].mean()).to_numpy()


def lightgbm_classifier(rounds: int = 400):
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

    model.params = params
    model.rounds = rounds
    return model


# Metrics


def auc(y: pd.Series, score: pd.Series) -> float:
    """Chance that a random mover is scored above a random non mover."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = score.rank()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y: pd.Series, score: pd.Series) -> float:
    """Mean precision at each mover's position in the ranked list."""
    order = np.argsort(-score.to_numpy(), kind="stable")
    hits = y.to_numpy()[order]
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def precision_at(y: pd.Series, score: pd.Series, k: int) -> float:
    top = score.sort_values(ascending=False).index[:k]
    return float(y.loc[top].mean())


def summarize_moves(predictions: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """Per model: AUC, average precision and the share of movers among the
    top ``k`` of each test season (what a scout reading a shortlist sees),
    next to the base rate."""
    models = [c.removeprefix("score_") for c in predictions.columns if c.startswith("score_")]
    rows = []
    for model in models:
        column = f"score_{model}"
        per_season = predictions.groupby("season").apply(
            lambda g, c=column: precision_at(g["y"], g[c], k), include_groups=False
        )
        rows.append(
            {
                "model": model,
                "rows": len(predictions),
                "movers": int(predictions["y"].sum()),
                "base_rate": float(predictions["y"].mean()),
                "auc": auc(predictions["y"], predictions[column]),
                "average_precision": average_precision(predictions["y"], predictions[column]),
                f"precision_top{k}_per_season": float(per_season.mean()),
            }
        )
    return pd.DataFrame(rows)


# Dataset and backtest


def move_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = features.merge(
        labels[[*KEYS, "big_move_24m", "move_label_date", "move_date", "move_to"]], on=KEYS
    )
    data = data[data["regular"].astype(bool) & data["big_move_24m"].notna()].copy()
    data["y"] = data["big_move_24m"].astype(float)
    columns = feature_columns(features, TARGET)
    for column in CATEGORICAL:
        if column in data:
            data[column] = data[column].astype("category")
    return data.reset_index(drop=True), columns


def run_move_backtest(data: pd.DataFrame, columns: list[str], models: dict) -> pd.DataFrame:
    folds = fold_season(data["feature_date"])
    out = []
    for fold in expanding_folds(folds.tolist(), TARGET):
        train = data[folds.isin(fold.train_seasons)]
        test = data[folds == fold.test_season]
        first_test = test["feature_date"].min()
        train = train[train["move_label_date"] <= first_test]  # purge open windows
        assert (train["feature_date"] < first_test).all()
        frame = test[[*KEYS, "player_name", "team_name", "age", "y", "move_date", "move_to"]].copy()
        frame["train_rows"] = len(train)
        for name, model in models.items():
            frame[f"score_{name}"] = model(train, test, columns)
        out.append(frame)
        log.info(
            "%s: test %d (%d rows, %d movers) on %d rows",
            TARGET.name,
            fold.test_season,
            len(test),
            int(test["y"].sum()),
            len(train),
        )
    return pd.concat(out, ignore_index=True)


def current_scores(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    data: pd.DataFrame,
    columns: list[str],
    model,
) -> pd.DataFrame:
    """Score the newest season's rows, whose windows are still open, with a
    model trained on every labelled row: every regular player not at a top
    club, highest score first."""
    newest = features["season"].max()
    pool = features.merge(labels[[*KEYS, "at_top_club", "move_to", "move_date"]], on=KEYS)
    pool = pool[(pool["season"] == newest) & pool["regular"].astype(bool) & ~pool["at_top_club"]]
    pool = pool.copy()
    for column in CATEGORICAL:
        if column in pool:
            pool[column] = pd.Categorical(pool[column], categories=data[column].cat.categories)
    pool["score"] = model(data, pool, columns)
    # Moves to a top club already listed since the feature date (the window
    # is still open, so these are not labels, just a first check of the list).
    pool["moved_since"] = pool["move_to"].where(
        pool["move_to"].isna(),
        pool["move_to"] + " (" + pool["move_date"].dt.strftime("%Y-%m-%d") + ")",
    )
    show = [
        *KEYS,
        "player_name",
        "team_name",
        "age",
        "position",
        "market_value",
        "score",
        "moved_since",
    ]
    return pool.sort_values("score", ascending=False)[show].reset_index(drop=True)


def write_move_report(
    predictions: pd.DataFrame, top: pd.DataFrame, short: pd.DataFrame, reports: Path
) -> Path:
    overall = summarize_moves(predictions)
    by_league = []
    for league, group in predictions.groupby("league_key"):
        for model in ("group_rate", "lightgbm"):
            by_league.append(
                {
                    "league": league,
                    "model": model,
                    "rows": len(group),
                    "movers": int(group["y"].sum()),
                    "auc": auc(group["y"], group[f"score_{model}"]),
                }
            )
    caught = predictions.sort_values("score_lightgbm", ascending=False).groupby("season").head(50)
    caught = caught[caught["y"] == 1][
        ["season", "player_name", "team_name", "age", "move_to", "move_date"]
    ]
    newest_top = top[top["season"] == top["season"].max()].sort_values("rank")
    short = short.assign(
        market_value=(short["market_value"] / 1e6).round(1), score=short["score"].round(3)
    )
    parts = [
        "# Big move model",
        "",
        f"Will a regular player who is not at a top {TOP_N} club join one within "
        f"{MONTHS} months? Top clubs are the {TOP_N} highest squad values per season. "
        "Expanding window backtest, every test season scored by a model trained on "
        "earlier seasons only.",
        "",
        "## Overall",
        "",
        overall.round(3).to_markdown(index=False),
        "",
        "## By league (AUC)",
        "",
        pd.DataFrame(by_league).round(3).to_markdown(index=False),
        "",
        "## Movers the model had in its top 50 of their season",
        "",
        caught.to_markdown(index=False),
        "",
        f"## Top {TOP_N} clubs, newest season",
        "",
        ", ".join(newest_top["team_name"]),
        "",
        "## Shortlist: newest season, highest scores (value in millions)",
        "",
        short.to_markdown(index=False),
        "",
    ]
    path = reports / "moves.md"
    atomic_write_text(path, "\n".join(parts))
    return path


TRANSFERS_DATA_END = pd.Timestamp("2026-10-11")  # the /transfers download


def run_moves(settings) -> pd.DataFrame:
    """``scout moves``: labels, backtest, report and shortlist."""
    from scout.backfill.transfers import transfers_path
    from scout.tables import read_feature_table

    root = settings.paths.processed_dir / "features"
    features = read_feature_table(root / "player_season_features")
    transfers = pd.read_parquet(transfers_path(settings))
    top = top_clubs(features)
    labels = move_labels(features, transfers, top, TRANSFERS_DATA_END)
    data, columns = move_dataset(features, labels)
    log.info(
        "big move: %d rows, %d movers, %d columns", len(data), int(data["y"].sum()), len(columns)
    )
    models = {"group_rate": group_rate, "lightgbm": lightgbm_classifier()}
    predictions = run_move_backtest(data, columns, models)
    out_dir = settings.paths.processed_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "backtest_big_move_24m.parquet", index=False)
    labels.to_parquet(out_dir / "big_move_labels.parquet", index=False)
    current = current_scores(features, labels, data, columns, models["lightgbm"])
    current.to_parquet(out_dir / "big_move_current.parquet", index=False)
    short = current.drop(columns=KEYS[1:]).head(50)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    path = write_move_report(predictions, top, short, settings.paths.reports_dir)
    log.info("Move report: %s", path)
    print(summarize_moves(predictions).round(3).to_string(index=False))
    return predictions
