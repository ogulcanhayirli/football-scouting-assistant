"""Style vectors, a learned player embedding and similar players (decision
10, prompt 18).

* **Style vector,** one per regular player season from 2015 with match
  stats: the version 2 per 90 inputs (``scout.scouting.composite_inputs``)
  and four on pitch shares, the player's share of his team's shots,
  dribbles, passes and duels divided by his share of team minutes (so
  playing time is not read as style). Each measure is shrunk towards the
  position group and season mean by minutes, with a prior strength fitted
  per measure and position group by empirical Bayes as version 2 does, on
  one season only (pooling two seasons, as the ability score does, would
  share data between the two seasons of the identity test), then
  standardised within position group and season and clipped at 4.
* **Level removed.** Within each position group and season every column is
  regressed on a cubic in the two ability percentiles (two season and one
  season, version 2) and only the residual is kept; the residual is then
  divided by its spread at that ability (the squared residual regressed on
  the same cubic), so neither the mean nor the spread of a measure says how
  good the player is. Similar then means "plays the same way", not "is as
  good". The check (``level_check``): the correlation between a query's
  ability percentile and his ten nearest neighbours'. Cosine similarity on
  these vectors is the baseline.
* **Learned embedding.** A small encoder (one hidden layer, numpy, CPU)
  maps a style vector to a unit vector, trained contrastively (InfoNCE):
  the same player in two consecutive seasons is the positive pair, the
  other players of the same position group and season in the batch are
  the negatives. Settings are chosen on seasons up to 2021 (validated on
  2021 to 2022), the tested model is trained on seasons up to 2022 and
  scored on 2023 to 2025, and the stored model is refit on every season.
  It is adopted only if it beats the baseline on identity retrieval.
* **Search** is one matrix product on unit vectors held in memory; no
  vector database (about 93,000 vectors of 48 numbers).

``scout embed --train`` runs the tests, refits and stores the model
(``stored/embedding.json``) and writes ``reports/embeddings.md``; ``scout
embed`` (the weekly run) embeds every row with the stored model and writes
``data/processed/embeddings/``: ``vectors.parquet``, ``neighbours.parquet``
(the ten nearest players of every current player) and ``dictionary.csv``.
Nothing here feeds a model of the lists.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from unidecode import unidecode

from scout.fsutil import atomic_write_text
from scout.scouting import (
    COMPOSITE_CANDIDATES,
    KEYS,
    composite_inputs,
    position_group,
    prior_minutes,
)

log = logging.getLogger(__name__)

FIRST_SEASON = 2015
SHARES = {  # on pitch share of the team's count: share of the count / share of minutes
    "share_shots": "shots_total",
    "share_dribbles": "dribbles_attempts",
    "share_passes": "passes_total",
    "share_duels": "duels_total",
}
STYLE_COLUMNS = [*COMPOSITE_CANDIDATES, *SHARES]
CLIP = 4.0  # standardised values beyond 4 (an outfielder's emergency saves) are cut there
LEVEL_DEGREE = 3
MIN_CELL = 50  # a position group and season smaller than this is only centred

# The learned model
TRAIN_UNTIL = 2022
TEST_SEASONS = (2023, 2024, 2025)
VALIDATE_UNTIL = 2021  # settings are chosen on seasons up to 2021, validated on 2021 to 2022
DIMENSIONS = (32, 48, 64)
TEMPERATURES = (0.05, 0.1, 0.2)
HIDDEN = 128
EPOCHS = 30
BATCH = 256
MIN_BATCH = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 18

NEIGHBOURS = 10
HIT_AT = 5  # identity retrieval: the other season among the five nearest
REPLACE_AT = 10  # the replacement check: the signing among the ten nearest
REPLACE_DAYS = 60
VECTOR_MAX_AGE_DAYS = 400  # a move uses a season vector at most this old
AGE_WINDOW = 1  # "looks like X at Y": today's players aged Y - 1 to Y + 1

STORED = "embedding.json"
OUT = "embeddings"

META = [
    *KEYS,
    "player_name",
    "feature_date",
    "age",
    "team_id",
    "team_name",
    "sub_position",
    "group",
    "minutes",
    "market_value",
    "value_rank",
    "ability_pct",
    "ability_pct_1s",
]


# Style vectors


def style_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Per row: the version 2 per 90 inputs and the on pitch shares. A count
    the player has no entry for reads as 0 when his team's total is known,
    as ``composite_inputs`` does for the per 90 counts."""
    out = composite_inputs(features)
    keys = [features[k] for k in ("league_key", "season", "team_id")]
    minutes_share = features["minutes_share"].astype(float)
    for name, column in SHARES.items():
        values = features[column].astype(float)
        total = values.groupby(keys).transform("sum")
        share = values.fillna(0.0) / total.where(total > 0)
        out[name] = share / minutes_share.where(minutes_share > 0)
    return out


def vector_rows(features: pd.DataFrame, ability: pd.DataFrame) -> pd.Series:
    """Rows that get a vector: regular (450+ minutes), match stats, a position
    group, an ability percentile, from 2015."""
    return (
        (features["tier"] == "match")
        & features["regular"].astype(bool)
        & position_group(features).notna()
        & (features["season"].astype(int) >= FIRST_SEASON)
        & ability["ability_pct"].notna()
    )


def main_rows(meta: pd.DataFrame) -> pd.Series:
    """One row per player and season: the league row with most minutes."""
    order = meta.sort_values(["minutes", "league_key"], ascending=[False, True])
    first = ~order.duplicated(["source_player_id", "season"])
    return first.reindex(meta.index)


def fit_priors(
    inputs: pd.DataFrame, minutes: pd.Series, group: pd.Series, season: pd.Series
) -> dict[str, dict[str, float | None]]:
    """Prior minutes per position group and measure (``prior_minutes``, the
    version 2 empirical Bayes fit); None when the measure has no true spread
    in the group, so it is shrunk all the way to the mean."""
    priors: dict[str, dict[str, float | None]] = {}
    for g in sorted(group.dropna().unique()):
        priors[g] = {}
        for column in inputs.columns:
            rows = (group == g) & inputs[column].notna()
            try:
                k = prior_minutes(inputs.loc[rows, column], minutes[rows], [season[rows]])
            except (ValueError, np.linalg.LinAlgError):
                k = None
            priors[g][column] = k
    return priors


def shrink_standardise(
    inputs: pd.DataFrame,
    minutes: pd.Series,
    group: pd.Series,
    season: pd.Series,
    priors: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    """Each measure shrunk towards its position group and season mean by
    minutes (weight minutes / (minutes + prior)), then standardised within
    position group and season and clipped; a missing measure reads as the
    mean (0)."""
    keys = [group, season]
    out = pd.DataFrame(0.0, index=inputs.index, columns=inputs.columns)
    for column in inputs.columns:
        values = inputs[column].astype(float)
        mean = values.groupby(keys).transform("mean")
        prior = group.map({g: p.get(column) for g, p in priors.items()}).astype(float)
        weight = (minutes / (minutes + prior)).fillna(0.0)  # no prior: all the way to the mean
        shrunk = mean + weight * (values - mean)
        centre = shrunk.groupby(keys).transform("mean")
        spread = shrunk.groupby(keys).transform("std")
        z = (shrunk - centre) / spread.where(spread > 1e-9)
        out[column] = z.clip(-CLIP, CLIP).fillna(0.0)
    return out


def level_basis(ability: pd.DataFrame) -> np.ndarray:
    """A constant and a cubic in each ability percentile (two season, one season)."""
    a = ability["ability_pct"].astype(float) / 100
    b = ability["ability_pct_1s"].astype(float).fillna(ability["ability_pct"].astype(float)) / 100
    terms = [np.ones(len(a))]
    for x in (a.to_numpy(), b.to_numpy()):
        terms += [x**d for d in range(1, LEVEL_DEGREE + 1)]
    return np.column_stack(terms)


def remove_level(
    values: np.ndarray, ability: pd.DataFrame, group: pd.Series, season: pd.Series
) -> np.ndarray:
    """Within position group and season: the residual of every column on the
    level basis, divided by its spread at that ability (the squared residual
    on the same basis) and put back on the column's overall spread."""
    basis = level_basis(ability)
    out = np.zeros_like(values, dtype=float)
    cells = pd.Series(np.arange(len(values))).groupby([group.to_numpy(), season.to_numpy()]).indices
    for idx in cells.values():
        y = values[idx]
        if len(idx) < MIN_CELL:
            out[idx] = y - y.mean(axis=0)
            continue
        a = basis[idx]
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        r = y - a @ coef
        total = (r**2).mean(axis=0)
        coef2, *_ = np.linalg.lstsq(a, r**2, rcond=None)
        local = np.maximum(a @ coef2, 0.1 * total)  # a fitted spread never below a tenth
        scaled = r * np.sqrt(np.divide(total, local, out=np.zeros_like(local), where=local > 0))
        out[idx] = scaled
    return out


@dataclass
class StyleVectors:
    """Vectors and their rows: ``meta`` (keys, name, age, club, group,
    ability, ``main``) and ``style`` (level removed, rows in the same order)."""

    meta: pd.DataFrame
    style: np.ndarray
    columns: list[str]
    priors: dict = field(default_factory=dict)


def style_vectors(
    features: pd.DataFrame,
    ability: pd.DataFrame,
    priors: dict | None = None,
    fit_until: int | None = None,
    keep_level: bool = False,
) -> StyleVectors:
    """The level free style vector of every vector row. ``ability`` holds
    ``ability_pct`` and ``ability_pct_1s`` aligned with ``features``. The
    priors are fitted on seasons up to ``fit_until`` unless given. With
    ``keep_level`` the level stays in (the check's "before")."""
    inputs = style_inputs(features)
    rows = vector_rows(features, ability)
    f = features[rows]
    group = position_group(f)
    season = f["season"].astype(int)
    minutes = f["minutes"].astype(float)
    x = inputs[rows]
    if priors is None:
        use = season <= fit_until if fit_until is not None else pd.Series(True, index=f.index)
        priors = fit_priors(x[use], minutes[use], group[use], season[use])
    z = shrink_standardise(x, minutes, group, season, priors)
    style = z.to_numpy() if keep_level else remove_level(z.to_numpy(), ability[rows], group, season)
    meta = pd.concat([f, ability.loc[rows, ["ability_pct", "ability_pct_1s"]]], axis=1)
    meta = meta.assign(group=group, season=season)
    meta = meta[[c for c in META if c in meta]].reset_index(drop=True)
    meta["main"] = main_rows(meta)
    return StyleVectors(meta, style, list(x.columns), priors)


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


# The learned embedding


@dataclass
class Encoder:
    """Style vector -> hidden layer (ReLU) -> unit vector."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.w2.shape[1])

    def embed(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        return unit(hidden @ self.w2 + self.b2)

    def params(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def to_dict(self) -> dict:
        return {k: getattr(self, k).round(7).tolist() for k in ("w1", "b1", "w2", "b2")}

    @classmethod
    def from_dict(cls, data: dict) -> Encoder:
        return cls(*(np.asarray(data[k], dtype=float) for k in ("w1", "b1", "w2", "b2")))


def init_encoder(inputs: int, dim: int, hidden: int, rng: np.random.Generator) -> Encoder:
    return Encoder(
        rng.normal(0.0, np.sqrt(2.0 / inputs), (inputs, hidden)),
        np.zeros(hidden),
        rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, dim)),
        np.zeros(dim),
    )


def info_nce(
    encoder: Encoder, xa: np.ndarray, xb: np.ndarray, temperature: float
) -> tuple[float, list[np.ndarray]]:
    """Symmetric InfoNCE of a batch of pairs (row i of ``xa`` and of ``xb``
    are the same player; every other row is a negative) and its gradients
    for ``encoder.params()``."""
    n = len(xa)
    x = np.vstack([xa, xb])
    pre = x @ encoder.w1 + encoder.b1
    hidden = np.maximum(pre, 0.0)
    z = hidden @ encoder.w2 + encoder.b2
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    u = z / norm
    ua, ub = u[:n], u[n:]
    logits = ua @ ub.T / temperature
    rows = logits - logits.max(axis=1, keepdims=True)
    p_rows = np.exp(rows)
    p_rows /= p_rows.sum(axis=1, keepdims=True)
    cols = logits - logits.max(axis=0, keepdims=True)
    p_cols = np.exp(cols)
    p_cols /= p_cols.sum(axis=0, keepdims=True)
    diag = np.arange(n)
    loss = -0.5 * (np.log(p_rows[diag, diag]).mean() + np.log(p_cols[diag, diag]).mean())
    eye = np.eye(n)
    d_logits = 0.5 * ((p_rows - eye) + (p_cols - eye)) / n
    du = np.vstack([d_logits @ ub, d_logits.T @ ua]) / temperature
    dz = (du - u * (u * du).sum(axis=1, keepdims=True)) / norm
    d_w2 = hidden.T @ dz
    d_b2 = dz.sum(axis=0)
    d_pre = (dz @ encoder.w2.T) * (pre > 0)
    d_w1 = x.T @ d_pre
    d_b1 = d_pre.sum(axis=0)
    return float(loss), [d_w1, d_b1, d_w2, d_b2]


def season_pairs(meta: pd.DataFrame) -> pd.DataFrame:
    """The same player in consecutive seasons (main rows): positions ``a``
    (season t) and ``b`` (season t + 1) in ``meta``, with the group and t."""
    main = meta[meta["main"]]
    left = pd.DataFrame(
        {"pid": main["source_player_id"].astype(int), "season": main["season"], "a": main.index}
    )
    right = pd.DataFrame(
        {"pid": main["source_player_id"].astype(int), "season": main["season"] - 1, "b": main.index}
    )
    pairs = left.merge(right, on=["pid", "season"])
    pairs["group"] = meta.loc[pairs["a"], "group"].to_numpy()
    return pairs.reset_index(drop=True)


def train_encoder(
    x: np.ndarray,
    pairs: pd.DataFrame,
    dim: int,
    temperature: float,
    epochs: int = EPOCHS,
    hidden: int = HIDDEN,
    seed: int = SEED,
) -> tuple[Encoder, list[float]]:
    """Adam with weight decay on batches drawn within one position group and
    season (so the negatives are players of the same group and season).
    Returns the encoder and the mean loss of each epoch."""
    rng = np.random.default_rng(seed)
    encoder = init_encoder(x.shape[1], dim, hidden, rng)
    params = encoder.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    cells = list(pairs.groupby(["group", "season"]).indices.values())
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    step, history = 0, []
    for _ in range(epochs):
        batches = []
        for idx in cells:
            order = rng.permutation(idx)
            batches += [c for c in np.array_split(order, max(1, len(order) // BATCH))]
        batches = [c for c in batches if len(c) >= MIN_BATCH]
        losses = []
        for i in rng.permutation(len(batches)):
            chunk = batches[i]
            loss, grads = info_nce(encoder, x[a[chunk]], x[b[chunk]], temperature)
            step += 1
            for p, g, mi, vi in zip(params, grads, m, v, strict=True):
                mi *= beta1
                mi += (1 - beta1) * g
                vi *= beta2
                vi += (1 - beta2) * g * g
                m_hat = mi / (1 - beta1**step)
                v_hat = vi / (1 - beta2**step)
                p -= LEARNING_RATE * (m_hat / (np.sqrt(v_hat) + eps) + WEIGHT_DECAY * p)
            losses.append(loss)
        history.append(float(np.mean(losses)))
    return encoder, history


# The tests


def identity_retrieval(
    vectors: np.ndarray, meta: pd.DataFrame, seasons, k: int = HIT_AT
) -> pd.DataFrame:
    """For every player with main rows in two consecutive ``seasons``, hide
    one season and search it with the other: the rank of his hidden row
    among every main row of that season and position group, both ways.
    ``hit``: within the ``k`` nearest."""
    seasons = set(seasons)
    pairs = season_pairs(meta)
    pairs = pairs[pairs["season"].isin(seasons) & (pairs["season"] + 1).isin(seasons)]
    u = unit(vectors)
    main = meta[meta["main"]]
    out = []
    for query_col, target_col in (("b", "a"), ("a", "b")):
        target_season = meta.loc[pairs[target_col], "season"].to_numpy()
        for (g, s), rows in pairs.groupby([pairs["group"], target_season]).groups.items():
            chunk = pairs.loc[rows]
            pool = main.index[(main["group"] == g) & (main["season"] == s)].to_numpy()
            where = {p: i for i, p in enumerate(pool)}
            sims = u[chunk[query_col].to_numpy()] @ u[pool].T
            true = sims[np.arange(len(chunk)), [where[t] for t in chunk[target_col]]]
            rank = 1 + (sims > true[:, None]).sum(axis=1)
            out.append(
                pd.DataFrame(
                    {
                        "group": g,
                        "query_season": meta.loc[chunk[query_col], "season"].to_numpy(),
                        "target_season": s,
                        "rank": rank,
                        "pool": len(pool),
                    }
                )
            )
    result = pd.concat(out, ignore_index=True)
    result["hit"] = result["rank"] <= k
    result["random"] = np.minimum(k, result["pool"]) / result["pool"]
    return result


def retrieval_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hit rate per position group and overall for each method, with the
    random rate and the number of searches."""
    first = next(iter(results.values()))
    table = pd.DataFrame(
        {
            "searches": first.groupby("group").size(),
            "random": first.groupby("group")["random"].mean(),
        }
    )
    for name, res in results.items():
        table[name] = res.groupby("group")["hit"].mean()
    overall = {"searches": len(first), "random": first["random"].mean()}
    overall.update({name: res["hit"].mean() for name, res in results.items()})
    table.loc["overall"] = pd.Series(overall)
    table["searches"] = table["searches"].astype(int)
    return table


def level_check(
    vectors: np.ndarray, meta: pd.DataFrame, seasons, k: int = NEIGHBOURS
) -> dict[str, float]:
    """Correlation between a query's ability percentile and that of each of
    his ``k`` nearest neighbours (main rows, same position group and season),
    and the neighbours' mean ability for queries in the bottom and top
    tenth."""
    u = unit(vectors)
    main = meta[meta["main"] & meta["season"].isin(set(seasons))]
    q_all, n_all = [], []
    for _, idx in main.groupby(["group", "season"]).groups.items():
        rows = np.asarray(idx)
        sims = u[rows] @ u[rows].T
        np.fill_diagonal(sims, -np.inf)
        top = np.argsort(-sims, axis=1)[:, :k]
        ability = meta.loc[rows, "ability_pct"].to_numpy(float)
        q_all.append(np.repeat(ability, top.shape[1]))
        n_all.append(ability[top].ravel())
    q, n = np.concatenate(q_all), np.concatenate(n_all)
    return {
        "pairs": len(q),
        "pearson": float(np.corrcoef(q, n)[0, 1]),
        "spearman": float(pd.Series(q).corr(pd.Series(n), method="spearman")),
        "bottom_tenth": float(n[q <= 10].mean()),
        "top_tenth": float(n[q > 90].mean()),
    }


def vector_at(meta: pd.DataFrame, moves: pd.DataFrame, date: str) -> pd.Series:
    """Position in ``meta`` of each move's player vector: his latest main row
    with a feature date on or before the move, at most
    ``VECTOR_MAX_AGE_DAYS`` old; -1 when there is none."""
    main = meta[meta["main"]].copy()
    main["pos"] = main.index
    main["pid"] = main["source_player_id"].astype(int)
    main = main.sort_values("feature_date")
    left = moves[["pid", date]].copy()
    left["_order"] = np.arange(len(left))
    left[date] = pd.to_datetime(left[date]).astype("datetime64[ns]")
    main["feature_date"] = pd.to_datetime(main["feature_date"]).astype("datetime64[ns]")
    found = pd.merge_asof(
        left.sort_values(date),
        main[["pid", "feature_date", "pos"]],
        left_on=date,
        right_on="feature_date",
        by="pid",
        direction="backward",
        tolerance=pd.Timedelta(days=VECTOR_MAX_AGE_DAYS),
    ).sort_values("_order")
    return pd.Series(found["pos"].fillna(-1).astype(int).to_numpy(), index=moves.index)


def replacement_pairs(transfers: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """A club sells a player (a paid transfer out) and signs another of the
    same position group within ``REPLACE_DAYS`` before or after (any arrival
    but a loan return). Each side is described by his latest season vector
    before his own move."""
    moves = transfers.dropna(subset=["source_player_id", "transfer_date"]).drop_duplicates(
        ["source_player_id", "transfer_date", "team_out_id", "team_in_id"]
    )
    moves = moves.assign(pid=moves["source_player_id"].astype(int))
    sales = moves[moves["kind"] == "transfer"].dropna(subset=["team_out_id"])
    arrivals = moves[moves["kind"] != "loan_return"].dropna(subset=["team_in_id"])
    sales = sales.assign(club=sales["team_out_id"].astype(int))
    arrivals = arrivals.assign(club=arrivals["team_in_id"].astype(int))
    sales = sales.assign(out_pos=vector_at(meta, sales, "transfer_date"))
    arrivals = arrivals.assign(in_pos=vector_at(meta, arrivals, "transfer_date"))
    sales, arrivals = sales[sales["out_pos"] >= 0], arrivals[arrivals["in_pos"] >= 0]
    pairs = sales[["club", "pid", "transfer_date", "out_pos", "team_out_name"]].merge(
        arrivals[["club", "pid", "transfer_date", "in_pos"]], on="club", suffixes=("_out", "_in")
    )
    gap = (pairs["transfer_date_in"] - pairs["transfer_date_out"]).dt.days.abs()
    pairs = pairs[(gap <= REPLACE_DAYS) & (pairs["pid_out"] != pairs["pid_in"])]
    same = (
        meta.loc[pairs["out_pos"], "group"].to_numpy()
        == meta.loc[pairs["in_pos"], "group"].to_numpy()
    )
    pairs = pairs[same].reset_index(drop=True)
    pairs["group"] = meta.loc[pairs["in_pos"], "group"].to_numpy()
    pairs["season"] = meta.loc[pairs["in_pos"], "season"].to_numpy()
    return pairs


def replacement_ranks(vectors: np.ndarray, meta: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Rank of the signing among every main row of his season and position
    group (the departed player left out) by similarity to the departed
    player; ``hit`` within ``REPLACE_AT``, ``random`` the chance of a random
    player of that group and season, ``percentile`` 100 for the nearest."""
    u = unit(vectors)
    main = meta[meta["main"]]
    ranks = np.zeros(len(pairs), dtype=int)
    pools = np.zeros(len(pairs), dtype=int)
    cells = pairs.groupby([pairs["group"].to_numpy(), pairs["season"].to_numpy()]).indices
    for (g, s), at in cells.items():
        chunk = pairs.iloc[at]
        pool = main.index[(main["group"] == g) & (main["season"] == s)].to_numpy()
        where = {p: i for i, p in enumerate(pool)}
        sims = u[chunk["out_pos"].to_numpy()] @ u[pool].T
        own = np.array([where.get(p, -1) for p in chunk["out_pos"]])
        sims[np.flatnonzero(own >= 0), own[own >= 0]] = -np.inf
        true = sims[np.arange(len(chunk)), [where[p] for p in chunk["in_pos"]]]
        ranks[at] = 1 + (sims > true[:, None]).sum(axis=1)
        pools[at] = len(pool) - (own >= 0)
    out = pairs[["group", "season"]].copy()
    out["rank"], out["pool"] = ranks, pools
    out["hit"] = out["rank"] <= REPLACE_AT
    out["random"] = np.minimum(REPLACE_AT, out["pool"]) / out["pool"]
    out["percentile"] = 100 * (1 - (out["rank"] - 1) / out["pool"])
    return out


# Queries


def fold(text) -> str:
    return unidecode(str(text or "")).lower().strip()


@dataclass
class SimilarIndex:
    """Every vector row with its unit vector, searched in memory. Today's
    players are the main rows of the newest season."""

    meta: pd.DataFrame
    vectors: np.ndarray

    def __post_init__(self):
        self.vectors = unit(np.asarray(self.vectors, dtype=np.float32))
        self.newest = int(self.meta["season"].max())
        self._names = self.meta["player_name"].map(fold)

    def find(self, player) -> int:
        """A player id from an id or a name (accents ignored): an exact name
        first, else every name containing the text, else the surname with the
        first name's initial (API Football writes "K. Yıldız" for Kenan
        Yıldız); among several players the one ranked highest by value in any
        season."""
        if isinstance(player, (int, np.integer)) or str(player).isdigit():
            pid = int(player)
            if not (self.meta["source_player_id"].astype(int) == pid).any():
                raise KeyError(f"no player {pid}")
            return pid
        text = fold(player)
        hits = self.meta[self._names == text]
        if hits.empty:
            hits = self.meta[self._names.str.contains(text, regex=False)]
        words = text.replace(".", " ").split()
        if hits.empty and len(words) > 1:
            surname = self._names.str.contains(words[-1], regex=False)
            hits = self.meta[surname & self._names.str.startswith(words[0][0])]
        if hits.empty:
            raise KeyError(f"no player named {player!r}")
        best = hits.groupby("source_player_id")["value_rank"].min().fillna(np.inf)
        best = best.sort_values(kind="stable")
        if len(best) > 1:
            log.info("%r matches %d players; taking %s", player, len(best), best.index[0])
        return int(best.index[0])

    def rows_of(self, pid: int) -> pd.DataFrame:
        rows = self.meta[(self.meta["source_player_id"].astype(int) == pid) & self.meta["main"]]
        return rows.sort_values("season")

    def _search(self, query: int, pool: pd.Index, k: int) -> pd.DataFrame:
        sims = self.vectors[pool.to_numpy()] @ self.vectors[query]
        top = np.argsort(-sims, kind="stable")[:k]
        out = self.meta.loc[pool[top]].copy()
        out.insert(0, "rank", np.arange(1, len(top) + 1))
        out["similarity"] = sims[top]
        return out.reset_index(drop=True)

    def today(self) -> pd.DataFrame:
        return self.meta[self.meta["main"] & (self.meta["season"] == self.newest)]

    def similar_now(
        self,
        player,
        k: int = NEIGHBOURS,
        age: tuple[float | None, float | None] = (None, None),
        leagues: list[str] | None = None,
        ability: tuple[float | None, float | None] = (None, None),
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Today's players of the same position group nearest to the player's
        latest season, optionally only those of an age range, some leagues or
        an ability percentile range. Returns (the query row, the neighbours)."""
        pid = self.find(player)
        query = self.rows_of(pid).iloc[-1]
        pool = self.today()
        keep = (pool["group"] == query["group"]) & (pool["source_player_id"] != pid)
        years = np.floor(pool["age"].astype(float))  # whole years, as the page shows them
        if age[0] is not None:
            keep &= years >= age[0]
        if age[1] is not None:
            keep &= years <= age[1]
        if leagues:
            keep &= pool["league_key"].isin(leagues)
        if ability[0] is not None:
            keep &= pool["ability_pct"] >= ability[0]
        if ability[1] is not None:
            keep &= pool["ability_pct"] <= ability[1]
        return query, self._search(query.name, pool.index[keep.to_numpy()], k)

    def looks_like(
        self, player, age: int, k: int = NEIGHBOURS, window: int = AGE_WINDOW
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Today's players aged ``age`` plus or minus ``window``, same position
        group, nearest to the player's season at that age (age at the feature
        date; the nearest age when he has no season at it)."""
        pid = self.find(player)
        rows = self.rows_of(pid)
        if rows.empty:
            raise KeyError(f"player {pid} has no season with a vector")
        whole = np.floor(rows["age"].astype(float))
        query = rows.loc[(whole - age).abs().sort_values(kind="stable").index[0]]
        pool = self.today()
        years = np.floor(pool["age"].astype(float))
        pool = pool[
            (pool["group"] == query["group"])
            & (pool["source_player_id"] != pid)
            & years.between(age - window, age + window)
        ]
        return query, self._search(query.name, pool.index, k)


def current_neighbours(index: SimilarIndex, k: int = NEIGHBOURS) -> pd.DataFrame:
    """For every player of today (main rows of the newest season), his ``k``
    nearest players of today in the same position group."""
    today = index.today()
    out = []
    for _, rows in today.groupby("group").groups.items():
        rows = pd.Index(rows)
        u = index.vectors[rows.to_numpy()]
        sims = u @ u.T
        np.fill_diagonal(sims, -np.inf)
        top = np.argsort(-sims, axis=1, kind="stable")[:, : min(k, len(rows) - 1)]
        q = today.loc[rows]
        for r in range(top.shape[1]):
            nb = today.loc[rows[top[:, r]]]
            out.append(
                pd.DataFrame(
                    {
                        "season": q["season"].to_numpy(),
                        "player_id": q["source_player_id"].astype(int).to_numpy(),
                        "player": q["player_name"].to_numpy(),
                        "league_key": q["league_key"].to_numpy(),
                        "group": q["group"].to_numpy(),
                        "rank": r + 1,
                        "neighbour_id": nb["source_player_id"].astype(int).to_numpy(),
                        "neighbour": nb["player_name"].to_numpy(),
                        "neighbour_age": nb["age"].astype(float).to_numpy(),
                        "neighbour_club": nb["team_name"].to_numpy(),
                        "neighbour_league_key": nb["league_key"].to_numpy(),
                        "neighbour_ability_pct": nb["ability_pct"].astype(float).to_numpy(),
                        "similarity": sims[np.arange(len(rows)), top[:, r]],
                    }
                )
            )
    table = pd.concat(out, ignore_index=True)
    return table.sort_values(["player_id", "rank"]).reset_index(drop=True)


# Stored model


@dataclass
class StoredEmbedding:
    """What the weekly run needs to embed a season without training: the
    style columns, the shrinkage priors, the method adopted and, when it is
    the learned one, the encoder."""

    columns: list[str]
    priors: dict
    method: str  # "learned" or "baseline"
    encoder: Encoder | None
    meta: dict

    def vectors(self, style: np.ndarray) -> np.ndarray:
        if self.method == "learned":
            return self.encoder.embed(style)
        return unit(style)

    def to_json(self) -> str:
        data = {
            "columns": self.columns,
            "priors": self.priors,
            "method": self.method,
            "encoder": self.encoder.to_dict() if self.encoder is not None else None,
            **self.meta,
        }
        return json.dumps(data, indent=1, default=_native) + "\n"

    @classmethod
    def from_json(cls, text: str) -> StoredEmbedding:
        data = json.loads(text)
        encoder = Encoder.from_dict(data.pop("encoder")) if data.get("encoder") else None
        data.pop("encoder", None)
        return cls(data.pop("columns"), data.pop("priors"), data.pop("method"), encoder, data)


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def load_stored(store: Path) -> StoredEmbedding:
    path = Path(store) / STORED
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: run `scout embed --train` once and sync it up (docs/DEPLOY.md)"
        )
    return StoredEmbedding.from_json(path.read_text())


# Outputs


DICTIONARY = {
    "vectors": {
        "league_key": "League of the row.",
        "season": "Season (the year it starts).",
        "source_player_id": "API Football player id.",
        "player_name": "Player name as API Football writes it.",
        "feature_date": "The date the row describes the player at.",
        "age": "Age at the feature date.",
        "team_id": "API Football club id that season.",
        "team_name": "Club that season.",
        "sub_position": "Transfermarkt position.",
        "group": "Position group the vector is standardised and searched in.",
        "minutes": "League minutes that season (450 or more: regular players only).",
        "market_value": "Transfermarkt value at the feature date, euros.",
        "value_rank": "Rank of that value among one row per player that season.",
        "ability_pct": "Ability score version 2 percentile within position group and season.",
        "ability_pct_1s": "The one season ability percentile (version 2, not pooled).",
        "main": "True on the league row with most minutes of the player's season; searches "
        "use these rows only.",
        "style_*": "Style vector: the measure shrunk by minutes, standardised within position "
        "group and season, with the level (both ability percentiles) taken out; 0 is the "
        "position mean at the player's level.",
        "emb_*": "The learned embedding (unit length); present when the learned model is adopted.",
    },
    "neighbours": {
        "season": "Season of today's rows (the newest).",
        "player_id": "API Football id of the player searched for.",
        "player": "His name.",
        "league_key": "His league.",
        "group": "His position group; neighbours come from the same group.",
        "rank": "1 is the most similar.",
        "neighbour_id": "API Football id of the similar player.",
        "neighbour": "His name.",
        "neighbour_age": "His age at the feature date.",
        "neighbour_club": "His club that season.",
        "neighbour_league_key": "His league.",
        "neighbour_ability_pct": "His ability percentile; similar style says nothing about "
        "level, so this varies.",
        "similarity": "Cosine similarity of the two vectors, 1 at most.",
    },
}


def dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [(t, c, m) for t, cols in DICTIONARY.items() for c, m in cols.items()],
        columns=["table", "column", "meaning"],
    )


def vectors_frame(sv: StyleVectors, stored: StoredEmbedding) -> pd.DataFrame:
    frame = sv.meta.copy()
    style = pd.DataFrame(sv.style, columns=[f"style_{c}" for c in sv.columns])
    parts = [frame, style.astype("float32")]
    if stored.method == "learned":
        emb = stored.vectors(sv.style)
        parts.append(
            pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])]).astype("float32")
        )
    return pd.concat(parts, axis=1)


def write_outputs(processed: Path, sv: StyleVectors, stored: StoredEmbedding) -> dict[str, Path]:
    out = Path(processed) / OUT
    out.mkdir(parents=True, exist_ok=True)
    frame = vectors_frame(sv, stored)
    paths = {"vectors": out / "vectors.parquet", "neighbours": out / "neighbours.parquet"}
    frame.to_parquet(paths["vectors"], index=False)
    index = SimilarIndex(sv.meta, stored.vectors(sv.style))
    current_neighbours(index).to_parquet(paths["neighbours"], index=False)
    paths["dictionary"] = out / "dictionary.csv"
    atomic_write_text(paths["dictionary"], dictionary_frame().to_csv(index=False))
    return paths


def read_index(processed: Path) -> SimilarIndex:
    """The search index from ``vectors.parquet`` (``scout similar``, the page)."""
    frame = pd.read_parquet(Path(processed) / OUT / "vectors.parquet")
    emb = [c for c in frame.columns if c.startswith("emb_")]
    cols = emb or [c for c in frame.columns if c.startswith("style_")]
    meta = frame[[c for c in frame.columns if not c.startswith(("emb_", "style_"))]]
    return SimilarIndex(meta, frame[cols].to_numpy())


# Commands


def load_inputs(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and the ability percentiles of ``scouting.parquet`` aligned
    with them (``scout scouting`` or ``scout score`` writes it)."""
    from scout.tables import read_feature_table

    processed = settings.paths.processed_dir
    features = read_feature_table(processed / "features" / "player_season_features")
    scouting = pd.read_parquet(
        processed / "models" / "scouting.parquet",
        columns=[*KEYS, "ability_pct", "ability_pct_1s", "value_rank"],
    )
    scouting = scouting.drop_duplicates(KEYS)
    joined = features[KEYS].merge(scouting, on=KEYS, how="left")
    joined.index = features.index
    features = features.assign(value_rank=joined["value_rank"])
    return features, joined[["ability_pct", "ability_pct_1s"]]


def embed_stored(settings) -> dict[str, Path]:
    """``scout embed``: every row embedded with the stored model; the
    vectors and today's neighbours written. Nothing is trained."""
    from scout.models.store import store_dir

    stored = load_stored(store_dir(settings.paths.processed_dir))
    features, ability = load_inputs(settings)
    sv = style_vectors(features, ability, priors=stored.priors)
    if sv.columns != stored.columns:
        raise ValueError(
            f"style columns changed since the stored model: {stored.columns} against "
            f"{sv.columns}; run `scout embed --train`"
        )
    paths = write_outputs(settings.paths.processed_dir, sv, stored)
    log.info("embed: %d vectors (%s), neighbours of today's players", len(sv.meta), stored.method)
    return paths


# Training, the tests and the report (scout embed --train)

# The owner's five example queries (prompt 18), by API Football id.
EXAMPLES = [
    ("Rodri at 21", 44, 21),
    ("Lamine Yamal now", 386828, None),
    ("Kenan Yıldız now", 339883, None),
    ("Jude Bellingham at 18", 129718, 18),
    ("Mike Maignan now", 22221, None),
]


def choose_settings(sv: StyleVectors) -> tuple[int, float, pd.DataFrame]:
    """Dimension and temperature by identity retrieval on the validation
    seasons (trained on seasons up to ``VALIDATE_UNTIL``, searched between it
    and the next): no test season is looked at."""
    pairs = season_pairs(sv.meta)
    train = pairs[pairs["season"] + 1 <= VALIDATE_UNTIL]
    seasons = (VALIDATE_UNTIL, VALIDATE_UNTIL + 1)
    rows = [
        {
            "model": "baseline",
            "dim": sv.style.shape[1],
            "temperature": np.nan,
            "hit": identity_retrieval(sv.style, sv.meta, seasons)["hit"].mean(),
        }
    ]
    for dim in DIMENSIONS:
        for temperature in TEMPERATURES:
            encoder, _ = train_encoder(sv.style, train, dim, temperature)
            hit = identity_retrieval(encoder.embed(sv.style), sv.meta, seasons)["hit"].mean()
            rows.append({"model": "learned", "dim": dim, "temperature": temperature, "hit": hit})
            log.info("embed: validation dim %d, temperature %.2f: %.4f", dim, temperature, hit)
    grid = pd.DataFrame(rows)
    best = grid[grid["model"] == "learned"].sort_values("hit", ascending=False, kind="stable")
    return int(best["dim"].iloc[0]), float(best["temperature"].iloc[0]), grid


def replacement_summary(ranks: dict[str, pd.DataFrame], first: int | None = None) -> pd.DataFrame:
    rows = []
    for name, r in ranks.items():
        if first is not None:
            r = r[r["season"] >= first]
        rows.append(
            {
                "model": name,
                "pairs": len(r),
                "hit": r["hit"].mean(),
                "random": r["random"].mean(),
                "median_percentile": r["percentile"].median(),
            }
        )
    return pd.DataFrame(rows)


def search_timing(index: SimilarIndex, repeats: int = 200) -> dict[str, float]:
    """Milliseconds for one search over every vector, and for one filtered
    "similar now" query (median of ``repeats``)."""
    rng = np.random.default_rng(SEED)
    queries = rng.integers(0, len(index.vectors), repeats)
    whole = []
    for q in queries:
        start = time.perf_counter()
        sims = index.vectors @ index.vectors[q]
        np.argpartition(-sims, NEIGHBOURS)[:NEIGHBOURS]
        whole.append(time.perf_counter() - start)
    today = index.today()["source_player_id"].astype(int).to_numpy()
    query = []
    for pid in rng.choice(today, min(repeats, 50), replace=False):
        start = time.perf_counter()
        index.similar_now(int(pid), age=(18, 30))
        query.append(time.perf_counter() - start)
    return {
        "vectors": len(index.vectors),
        "dim": int(index.vectors.shape[1]),
        "all_ms": 1000 * float(np.median(whole)),
        "query_ms": 1000 * float(np.median(query)),
    }


def run_examples(
    index: SimilarIndex, leagues: dict[str, str]
) -> list[tuple[str, pd.Series, pd.DataFrame]]:
    out = []
    for label, pid, age in EXAMPLES:
        try:
            if age is None:
                query, found = index.similar_now(pid)
            else:
                query, found = index.looks_like(pid, age)
        except KeyError as exc:
            log.warning("embed: example %s skipped: %s", label, exc)
            continue
        found = found.assign(league=found["league_key"].map(leagues).fillna(found["league_key"]))
        out.append((label, query, found))
    return out


def train_embeddings(settings) -> Path:
    """``scout embed --train``: the level check, identity retrieval and the
    replacement check on held out seasons, the adoption decision, the
    stored model refit on every season, the outputs and the report."""
    from scout.lists import league_names
    from scout.models.store import store_dir

    processed = settings.paths.processed_dir
    features, ability = load_inputs(settings)
    transfers = pd.read_parquet(processed / "transfers" / "transfers.parquet")

    test = style_vectors(features, ability, fit_until=TRAIN_UNTIL)
    kept = style_vectors(features, ability, priors=test.priors, keep_level=True)
    dim, temperature, grid = choose_settings(test)
    pairs = season_pairs(test.meta)
    train = pairs[pairs["season"] + 1 <= TRAIN_UNTIL]
    encoder, _ = train_encoder(test.style, train, dim, temperature)
    learned = encoder.embed(test.style)
    identity = {
        "baseline": identity_retrieval(test.style, test.meta, TEST_SEASONS),
        "learned": identity_retrieval(learned, test.meta, TEST_SEASONS),
    }
    table = retrieval_table(identity)
    adopted = bool(table.loc["overall", "learned"] > table.loc["overall", "baseline"])
    level = {
        "level kept": level_check(kept.style, kept.meta, TEST_SEASONS),
        "baseline (level removed)": level_check(test.style, test.meta, TEST_SEASONS),
        "learned": level_check(learned, test.meta, TEST_SEASONS),
    }
    moves = replacement_pairs(transfers, test.meta)
    replace = {
        "baseline": replacement_ranks(test.style, test.meta, moves),
        "learned": replacement_ranks(learned, test.meta, moves),
    }

    sv = style_vectors(features, ability)
    all_pairs = season_pairs(sv.meta)
    final = None
    if adopted:
        final, _ = train_encoder(sv.style, all_pairs, dim, temperature)
    newest = int(sv.meta["season"].max())
    stored = StoredEmbedding(
        sv.columns,
        sv.priors,
        "learned" if adopted else "baseline",
        final,
        {
            "adopted": adopted,
            "dim": dim,
            "temperature": temperature,
            "hidden": HIDDEN,
            "epochs": EPOCHS,
            "trained_seasons": [FIRST_SEASON, newest],
            "trained_pairs": len(all_pairs) if adopted else 0,
            "tested": {
                "train_until": TRAIN_UNTIL,
                "test_seasons": list(TEST_SEASONS),
                "hit_at": HIT_AT,
                "baseline": float(table.loc["overall", "baseline"]),
                "learned": float(table.loc["overall", "learned"]),
                "train_pairs": len(train),
            },
            "created": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        },
    )
    store = store_dir(processed)
    store.mkdir(parents=True, exist_ok=True)
    atomic_write_text(store / STORED, stored.to_json())
    write_outputs(processed, sv, stored)
    index = SimilarIndex(sv.meta, stored.vectors(sv.style))
    leagues = league_names(settings)
    path = write_report(
        settings.paths.reports_dir,
        stored=stored,
        grid=grid,
        table=table,
        level=level,
        replace=replace,
        examples=run_examples(index, leagues),
        timing=search_timing(index),
        counts={"vectors": len(sv.meta), "main": int(sv.meta["main"].sum())},
        names=leagues,
    )
    log.info(
        "embed: learned %.4f against baseline %.4f (top %d, %s); %s adopted",
        table.loc["overall", "learned"],
        table.loc["overall", "baseline"],
        HIT_AT,
        ", ".join(map(str, TEST_SEASONS)),
        stored.method,
    )
    return path


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _example_lines(
    label: str, query: pd.Series, found: pd.DataFrame, names: dict[str, str]
) -> list[str]:
    league = names.get(query["league_key"], query["league_key"])
    lines = [
        f"### {label}",
        "",
        f"Query: {query['player_name']}, {query['season']}/{str(query['season'] + 1)[-2:]}, "
        f"age {int(np.floor(query['age']))}, {query['team_name']} ({league}), "
        f"{query['group'].replace('_', ' ')}, ability percentile {query['ability_pct']:.0f}.",
        "",
        "| # | Player | Age | Club | League | Ability | Similarity |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in found.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.player_name} | {int(np.floor(row.age))} | {row.team_name} | "
            f"{row.league} | {row.ability_pct:.0f} | {row.similarity:.2f} |"
        )
    return [*lines, ""]


def write_report(
    reports_dir: Path,
    *,
    stored: StoredEmbedding,
    grid: pd.DataFrame,
    table: pd.DataFrame,
    level: dict,
    replace: dict,
    examples: list,
    timing: dict,
    counts: dict,
    names: dict[str, str],
) -> Path:
    first, last = TEST_SEASONS[0], TEST_SEASONS[-1]
    learned_wins = stored.meta["adopted"]
    lines = [
        "# Similar players (decision 10)",
        "",
        "Written by `scout embed --train` (`src/scout/embeddings.py`). Style vectors for "
        f"{counts['vectors']:,} regular player seasons from {FIRST_SEASON} with match stats "
        f"({counts['main']:,} after one row per player and season), {len(stored.columns)} "
        "measures each: the version 2 per 90 inputs and the on pitch shares of the team's "
        "shots, dribbles, passes and duels.",
        "",
        "## The level taken out",
        "",
        "Within each position group and season, every measure is regressed on a cubic in "
        "both ability percentiles (two season and one season); the residual is kept and "
        "divided by its spread at that ability. The check: for every player of "
        f"{first} to {last}, the ability percentile of his ten nearest players of the same "
        "position group and season against his own (0 means unrelated).",
        "",
        "| Vectors | Correlation | Neighbours' mean ability, bottom tenth | top tenth |",
        "|---|---|---|---|",
    ]
    for name, c in level.items():
        lines.append(
            f"| {name} | {c['pearson']:.3f} | {c['bottom_tenth']:.1f} | {c['top_tenth']:.1f} |"
        )
    lines += [
        "",
        f"Over {next(iter(level.values()))['pairs']:,} query and neighbour pairs.",
        "",
        f"## Identity retrieval, {first} to {last}",
        "",
        f"Trained on seasons up to {TRAIN_UNTIL} ({stored.meta['tested']['train_pairs']:,} "
        "pairs). For every player with regular seasons in two consecutive test seasons, one "
        "season is hidden and searched with the other among every player of that position "
        f"group and season, both ways: how often he is among the {HIT_AT} nearest. Random is "
        f"{HIT_AT} over the pool.",
        "",
        "| Position group | Searches | Random | Baseline (cosine) | Learned |",
        "|---|---|---|---|---|",
    ]
    for g, row in table.iterrows():
        name = "**overall**" if g == "overall" else g.replace("_", " ")
        lines.append(
            f"| {name} | {int(row['searches']):,} | {_pct(row['random'], 2)} | "
            f"{_pct(row['baseline'])} | {_pct(row['learned'])} |"
        )
    verdict = (
        "**The learned model is adopted**: it beats the baseline."
        if learned_wins
        else "**The baseline stays**: the learned model does not beat it."
    )
    lines += [
        "",
        f"{verdict} The stored model is refit on every season ({FIRST_SEASON} to "
        f"{stored.meta['trained_seasons'][1]}, {stored.meta['trained_pairs']:,} pairs) "
        f"with the settings chosen on validation: {stored.meta['dim']} dimensions, "
        f"temperature {stored.meta['temperature']}, {HIDDEN} hidden units, {EPOCHS} epochs.",
        "",
        f"Settings, validation only (trained to {VALIDATE_UNTIL}, searched between "
        f"{VALIDATE_UNTIL} and {VALIDATE_UNTIL + 1}):",
        "",
        "| Model | Dimensions | Temperature | Top 5 |",
        "|---|---|---|---|",
    ]
    for row in grid.itertuples(index=False):
        temp = "" if pd.isna(row.temperature) else f"{row.temperature}"
        lines.append(f"| {row.model} | {row.dim} | {temp} | {_pct(row.hit)} |")
    lines += [
        "",
        "## Replacement check",
        "",
        "A club sells a player (a paid transfer) and signs another of the same position "
        f"group within {REPLACE_DAYS} days before or after (any arrival but a loan return); "
        "each player is described by his latest season before his move. How often is the "
        f"signing among the {REPLACE_AT} nearest players of the departed one (in the "
        "signing's season and position group), against a random player of that group; and "
        "the signing's median percentile of similarity (50 at random). The transfers are "
        "those of the clubs of the original 17 leagues.",
        "",
        "| Seasons | Model | Pairs | Top 10 | Random | Median percentile |",
        "|---|---|---|---|---|---|",
    ]
    for label, start in ((f"{first} to {last}", first), (f"{FIRST_SEASON} to {last}", None)):
        for row in replacement_summary(replace, start).itertuples(index=False):
            lines.append(
                f"| {label} | {row.model} | {row.pairs:,} | {_pct(row.hit, 2)} | "
                f"{_pct(row.random, 2)} | {row.median_percentile:.1f} |"
            )
    lines += [
        "",
        "## Examples",
        "",
        '"Now" searches today\'s players (the newest season) of the same position group; '
        f"\"at Y\" takes the player's season at age Y and searches today's players aged Y "
        f"minus {AGE_WINDOW} to Y plus {AGE_WINDOW}. Ability is the percentile of the "
        "neighbour's own season.",
        "",
    ]
    for label, query, found in examples:
        lines += _example_lines(label, query, found, names)
    lines += [
        "## Speed",
        "",
        f"No vector database: the {timing['vectors']:,} vectors of {timing['dim']} numbers are "
        "held in memory; one search over all of them takes "
        f'{timing["all_ms"]:.1f} ms and a filtered "similar now" query '
        f"{timing['query_ms']:.1f} ms (numpy, one core).",
        "",
    ]
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "embeddings.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def similar_text(
    index: SimilarIndex,
    player,
    names: dict[str, str],
    *,
    at_age: int | None = None,
    k: int = NEIGHBOURS,
    age: tuple = (None, None),
    leagues: list[str] | None = None,
    ability: tuple = (None, None),
) -> str:
    """``scout similar``: the query and its neighbours as text; ``names``
    maps league keys to league names."""
    if at_age is None:
        query, found = index.similar_now(player, k=k, age=age, leagues=leagues, ability=ability)
        title = f"Playing like {query['player_name']} now"
    else:
        query, found = index.looks_like(player, at_age, k=k)
        title = (
            f"Today's players aged {at_age - AGE_WINDOW} to {at_age + AGE_WINDOW} who look "
            f"like {query['player_name']} at {at_age}"
        )
    found = found.assign(league=found["league_key"].map(names).fillna(found["league_key"]))
    lines = _example_lines(title, query, found, names)
    return "\n".join(line.removeprefix("### ") for line in lines)
