"""The measured greats (prompt 18b): style vectors for great players of the
past from StatsBomb open data, set against today's players.

**Data.** StatsBomb's open La Liga seasons 2004/05 to 2020/21 (Barcelona's
matches; 2015/16 in full) and the Premier League 2003/04 (Arsenal's), parsed
into our per match player table (``providers/statsbomb``), aggregated into
season rows as the API Football matches are (``match_tier_features``).

**Peers.** A StatsBomb season is standardised within position group and
season against the players of those matches: everyone with at least
``PEER_MINUTES`` minutes, which before 2015/16 means Barcelona (or Arsenal)
and the opponents of their matches, most of them seen twice a season or
less. A thin peer group would make the regulars' standardised values
explode (shrunk low minute peers have a small spread), so the spread is
put on a regular's scale: the true spread tau of a measure is estimated
from the peers by the method of moments (each peer's squared deviation over
1 + k / minutes, with k the stored prior minutes), and a regular's shrunk
deviation is divided by tau times the square root of the mean shrinkage
weight our own regulars get. That is what our standardisation divides by on
a full league season.

**Level.** An ability percentile is computed inside the same peer group (the
stored version 2 weights over the shrunk measures, pooled over two seasons
by minutes, no league or team term) and the level is taken out with the
curves our own data gives (each standardised measure against the two
ability percentiles, a cubic per position group over 2015 to 2025, and its
spread at that level), because 30 to 60 peers per group and season are too
few to fit a cubic.

**Validation** (``greats_check``) decides which measures are trusted: a
style measure whose match counts correlate below 0.7 with API Football's is
left out (it reads as the position mean). The cross source identity check
embeds the StatsBomb seasons of 2015/16 to 2020/21 with the stored model and
asks whether each lands next to the same player's API Football season.

**The panel.** From 2015 the greats' seasons are our own vectors; before it,
StatsBomb's. Ages are taken in the middle of the season (1 January; 1 July
for calendar year leagues), so "Messi at 19" is 2006/07; today's players
are aged the same way. Nothing here feeds a model of the lists.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scout.embeddings import (
    CLIP,
    FIRST_SEASON,
    HIT_AT,
    StoredEmbedding,
    level_basis,
    load_inputs,
    load_stored,
    style_inputs,
    style_vectors,
    unit,
    vector_rows,
)
from scout.features.build import REGULAR_MINUTES, feature_dates, match_tier_features
from scout.fsutil import atomic_write_text
from scout.scouting import COMPOSITE_CANDIDATES, KEYS, position_group, two_seasons

log = logging.getLogger(__name__)

OUT = "statsbomb"  # processed/statsbomb/: the match table and the season vectors
GREATS_OUT = "greats"  # processed/greats/: the panel's table and the logo
PEER_MINUTES = 90
THIN_MINUTES = REGULAR_MINUTES
# Who a StatsBomb season before 2015/16 is measured against.
PEER_GROUPS = {
    "la_liga": "Barcelona and its opponents",
    "epl": "Arsenal and its opponents",
}  # a great's season below this is shown, flagged thin
TOP = 10
LABEL = (
    "Style similarity from box score stats is coarse; the same player is found in the top "
    "five about one time in five; treat lists as shortlists."
)
ATTRIBUTION = (
    "Seasons before 2015/16 use StatsBomb open data (github.com/statsbomb/open-data), "
    "non commercial use with attribution."
)


@dataclass(frozen=True)
class Great:
    name: str
    born: str | None = None  # needed for StatsBomb seasons (the data has no birth dates)
    statsbomb_id: int | None = None
    api_football_id: int | None = None


GREATS = [
    Great("Lionel Messi", "1987-06-24", 5503, 154),
    Great("Xavi", "1980-01-25", 20131),
    Great("Andrés Iniesta", "1984-05-11", 5216, 33243),
    Great("Sergio Busquets", "1988-07-16", 5203, 144),
    Great("Carles Puyol", "1978-04-13", 20125),
    Great("Gerard Piqué", "1987-02-02", 5213, 136),
    Great("Dani Alves", "1983-05-06", 4324, 256),
    Great("Thierry Henry", "1977-08-17", 15516),
    Great("Patrick Vieira", "1976-06-23", 15515),
    Great("Dennis Bergkamp", "1969-05-10", 15042),
    Great("Robert Pirès", "1973-10-29", 19312),
    Great("Cristiano Ronaldo", api_football_id=874),
    Great("Luka Modrić", api_football_id=754),
    Great("Sergio Ramos", api_football_id=738),
    Great("Robert Lewandowski", api_football_id=521),
    Great("Toni Kroos", api_football_id=752),
    Great("Neymar", api_football_id=276),
    Great("Luis Suárez", api_football_id=157),
    Great("Kevin De Bruyne", api_football_id=629),
    Great("N'Golo Kanté", api_football_id=2290),
    Great("Virgil van Dijk", api_football_id=290),
    Great("Gianluigi Buffon", api_football_id=254),
    Great("Giorgio Chiellini", api_football_id=856),
]
# The report's three examples: (great, age).
# Xavi's season at 25 is thin (437 minutes in the 17 Barcelona matches StatsBomb
# holds for 2005/06), so his next season is shown too.
EXAMPLES = [("Lionel Messi", 19), ("Xavi", 25), ("Thierry Henry", 26), ("Xavi", 26)]


# The match table


def match_dir(processed: Path) -> Path:
    return Path(processed) / OUT / "player_match_stats"


def write_match_table(settings) -> pd.DataFrame:
    """``scout statsbomb build``: every cached match parsed, checked by the
    player_match_stats contract and written per league and season."""
    from scout.contracts.schemas import PLAYER_MATCH_STATS, validate
    from scout.providers.statsbomb.adapter import read_match_table
    from scout.storage import LocalRawStore

    frame = read_match_table(LocalRawStore(settings.paths.raw_dir), settings.statsbomb)
    validate(PLAYER_MATCH_STATS, frame)
    root = match_dir(settings.paths.processed_dir)
    for (league, season), part in frame.groupby(["league_key", "season"]):
        path = root / f"league={league}" / f"season={season}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
    log.info(
        "statsbomb: %d player match rows in %d matches", len(frame), frame["fixture_id"].nunique()
    )
    return frame


def read_matches(processed: Path) -> pd.DataFrame:
    root = match_dir(processed)
    parts = sorted(root.glob("league=*/season=*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{root} is empty: run `scout statsbomb build`")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# Season rows


def season_rows(match: pd.DataFrame) -> pd.DataFrame:
    """One row per league, season and player, built as the API Football
    match rows are, with the season's position (the Transfermarkt label of
    the StatsBomb position he played most minutes in) and his names."""
    keys = ["league_key", "season"]
    dates = feature_dates(match[keys].drop_duplicates(), match)
    rows = match_tier_features(match, dates).merge(dates, on=keys)
    rows["tier"] = "match"
    rows["regular"] = rows["minutes"] >= REGULAR_MINUTES
    played = match[match["minutes"].fillna(0) > 0]
    pos = (
        played.groupby([*KEYS, "sub_position"])["minutes"]
        .sum()
        .reset_index()
        .sort_values(["minutes", "sub_position"], ascending=[False, True])
        .drop_duplicates(KEYS)
    )
    names = played.sort_values("kickoff").groupby(KEYS)[["player_name", "player_nickname"]].last()
    rows = rows.merge(pos[[*KEYS, "sub_position"]], on=KEYS, how="left")
    rows = rows.merge(names.reset_index(), on=KEYS, how="left")
    return rows.reset_index(drop=True)


def display_name(rows: pd.DataFrame) -> pd.Series:
    return rows["player_nickname"].where(rows["player_nickname"].notna(), rows["player_name"])


# Peer standardisation


def reference_weights(features: pd.DataFrame, ability: pd.DataFrame, priors: dict) -> dict:
    """Per position group and measure, the mean shrinkage weight m / (m + k)
    of our own regular vector rows: the spread a regular's shrunk value has
    is tau times its square root."""
    rows = vector_rows(features, ability)
    minutes = features.loc[rows, "minutes"].astype(float)
    group = position_group(features[rows])
    out: dict = {}
    for g, p in priors.items():
        m = minutes[group == g]
        out[g] = {c: (float((m / (m + k)).mean()) if k is not None else None) for c, k in p.items()}
    return out


def peer_z(
    inputs: pd.DataFrame,
    minutes: pd.Series,
    group: pd.Series,
    season: pd.Series,
    priors: dict,
    ref_w: dict,
    peers: pd.Series,
) -> pd.DataFrame:
    """Each measure shrunk towards the peers' minutes weighted mean and put on
    a regular's scale (module docstring); clipped at ``CLIP``, missing reads
    as the mean (0)."""
    keys = [group, season]
    out = pd.DataFrame(0.0, index=inputs.index, columns=inputs.columns)
    m = minutes.astype(float)
    for column in inputs.columns:
        x = inputs[column].astype(float)
        k = group.map({g: p.get(column) for g, p in priors.items()}).astype(float)
        rw = group.map({g: p.get(column) for g, p in ref_w.items()}).astype(float)
        use = peers & x.notna() & k.notna() & (m > 0)
        weight = m.where(use)
        mu = (x * weight).groupby(keys).transform("sum") / weight.groupby(keys).transform("sum")
        term = ((x - mu) ** 2 / (1 + k / m)).where(use)
        tau2 = term.groupby(keys).transform("mean")
        shrink = m / (m + k)
        z = shrink * (x - mu) / np.sqrt(tau2 * rw).where(tau2 * rw > 1e-12)
        out[column] = z.clip(-CLIP, CLIP).fillna(0.0)
    return out


def peer_ability(
    z: pd.DataFrame, rows: pd.DataFrame, weights: pd.DataFrame, peers: pd.Series, skip=()
) -> pd.DataFrame:
    """Ability inside the peer group: the stored version 2 weights over the
    shrunk standardised measures (measures in ``skip`` left out and the rest
    reweighted), pooled with the player's previous season by minutes, as a
    percentile within position group and season among the peers."""
    group = position_group(rows)
    season = rows["season"].astype(int)
    total = pd.Series(0.0, index=rows.index)
    wsum = pd.Series(0.0, index=rows.index)
    for measure in weights["measure"].unique():
        if measure in skip or measure not in z:
            continue
        w = group.map(weights[weights["measure"] == measure].set_index("position")["weight"])
        w = w.fillna(0.0).astype(float)
        total += w * z[measure]
        wsum += w
    composite = (total / wsum).where(wsum > 0)
    pooled, _ = two_seasons(rows, composite)
    use = peers & group.notna() & composite.notna()
    out = pd.DataFrame(index=rows.index)
    for name, score in (("ability_pct", pooled), ("ability_pct_1s", composite)):
        pct = score[use].groupby([group[use], season[use]]).rank(pct=True).mul(100)
        # A row below the peer floor is placed in its peers' distribution.
        other = group.notna() & score.notna() & ~use
        for (g, s), idx in score[other].groupby([group[other], season[other]]).groups.items():
            ref = np.sort(score[use & (group == g) & (season == s)].to_numpy())
            if len(ref):
                found = np.searchsorted(ref, score[idx].to_numpy(), side="right")
                pct = pd.concat([pct, pd.Series(100 * found / len(ref), index=idx)])
        out[name] = pct.reindex(rows.index)
    return out


@dataclass
class LevelCurves:
    """Per position group: the cubic of every standardised measure on the
    two ability percentiles, the cubic of its squared residual and the mean
    squared residual (``remove_level`` fitted on our pooled seasons)."""

    coef: dict
    coef2: dict
    total: dict

    def apply(self, z: np.ndarray, ability: pd.DataFrame, group: pd.Series) -> np.ndarray:
        basis = level_basis(ability)
        out = np.zeros_like(z, dtype=float)
        for g in self.coef:
            idx = np.flatnonzero((group == g).to_numpy())
            if not len(idx):
                continue
            a = basis[idx]
            r = z[idx] - a @ self.coef[g]
            local = np.maximum(a @ self.coef2[g], 0.1 * self.total[g])
            out[idx] = r * np.sqrt(
                np.divide(self.total[g], local, out=np.zeros_like(local), where=local > 0)
            )
        return out


def level_curves(z: np.ndarray, ability: pd.DataFrame, group: pd.Series) -> LevelCurves:
    basis = level_basis(ability)
    coef, coef2, total = {}, {}, {}
    for g in sorted(group.dropna().unique()):
        idx = np.flatnonzero((group == g).to_numpy())
        a, y = basis[idx], z[idx]
        c, *_ = np.linalg.lstsq(a, y, rcond=None)
        r = y - a @ c
        c2, *_ = np.linalg.lstsq(a, r**2, rcond=None)
        coef[g], coef2[g], total[g] = c, c2, (r**2).mean(axis=0)
    return LevelCurves(coef, coef2, total)


@dataclass
class OwnData:
    """What the StatsBomb vectors borrow from our own data: the stored model,
    the stored ability weights, the regulars' shrinkage weights and the
    level curves; plus our vectors (the search pool and the API greats) and
    the feature table (the season level check)."""

    stored: StoredEmbedding
    weights: pd.DataFrame
    ref_w: dict
    curves: LevelCurves
    vectors: pd.DataFrame
    features: pd.DataFrame


def own_data(settings) -> OwnData:
    from scout.models.store import load_ability, store_dir

    processed = settings.paths.processed_dir
    stored = load_stored(store_dir(processed))
    fit = load_ability(store_dir(processed))
    features, ability = load_inputs(settings)
    kept = style_vectors(features, ability, priors=stored.priors, keep_level=True)
    curves = level_curves(
        kept.style, kept.meta[["ability_pct", "ability_pct_1s"]], kept.meta["group"]
    )
    vectors = pd.read_parquet(processed / "embeddings" / "vectors.parquet")
    ref_w = reference_weights(features, ability, stored.priors)
    return OwnData(stored, fit.weights, ref_w, curves, vectors, features)


def statsbomb_vectors(
    rows: pd.DataFrame,
    own: OwnData,
    excluded=(),
    sub_position: pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Peer standardised, level free StatsBomb vectors and their embedding
    for every row with minutes; the peers are the rows with ``PEER_MINUTES``
    or more. ``excluded`` style measures
    read as the position mean; ``sub_position`` overrides the season's
    position (the cross source check puts a linked player in his API
    Football position group). Returns (meta, embedding)."""
    if sub_position is not None:
        rows = rows.assign(
            sub_position=sub_position.reindex(rows.index).fillna(rows["sub_position"])
        )
    rows = rows[rows["minutes"].astype(float) > 0].reset_index(drop=True)
    inputs = style_inputs(rows)[own.stored.columns]
    for column in excluded:
        inputs[column] = np.nan
    group = position_group(rows)
    season = rows["season"].astype(int)
    minutes = rows["minutes"].astype(float)
    peers = group.notna() & (minutes >= PEER_MINUTES)
    z = peer_z(inputs, minutes, group, season, own.stored.priors, own.ref_w, peers)
    ability = peer_ability(z, rows, own.weights, peers, skip=excluded)
    keep = group.notna() & ability["ability_pct"].notna()
    style = own.curves.apply(z.to_numpy(), ability, group)
    meta = rows.assign(
        group=group, ability_pct=ability["ability_pct"], ability_pct_1s=ability["ability_pct_1s"]
    )
    meta["player"] = display_name(meta)
    emb = own.stored.vectors(style)
    return meta[keep].reset_index(drop=True), emb[keep.to_numpy()]


# The cross source identity check


def identity_ranks(
    sb_meta: pd.DataFrame, sb_emb: np.ndarray, vectors: pd.DataFrame, ids: pd.Series
) -> pd.DataFrame:
    """For every StatsBomb regular season linked to an API Football player:
    the rank of his own API Football season (same season) among every API
    Football main row of that season and his position group, by similarity
    to his StatsBomb vector. ``hit`` within ``HIT_AT``."""
    emb_cols = [c for c in vectors.columns if c.startswith("emb_")]
    main = vectors[vectors["main"]].reset_index(drop=True)
    u = unit(main[emb_cols].to_numpy(dtype=float))
    q = sb_meta.assign(af_player=sb_meta["source_player_id"].map(ids))
    q = q[q["regular"].astype(bool) & q["af_player"].notna()]
    out = []
    for i, row in q.iterrows():
        own = main.index[
            (main["source_player_id"] == int(row["af_player"])) & (main["season"] == row["season"])
        ]
        if not len(own):
            continue
        g = main.loc[own[0], "group"]
        pool = main.index[(main["group"] == g) & (main["season"] == row["season"])].to_numpy()
        sims = u[pool] @ sb_emb[i]
        true = float(u[own[0]] @ sb_emb[i])
        rank = int(1 + (sims > true).sum())
        out.append(
            {
                "season": int(row["season"]),
                "player": row["player"],
                "team": row["team_name"],
                "group": g,
                "minutes": float(row["minutes"]),
                "rank": rank,
                "pool": len(pool),
                "similarity": true,
            }
        )
    result = pd.DataFrame(out)
    if len(result):
        result["hit"] = result["rank"] <= HIT_AT
        result["random"] = np.minimum(HIT_AT, result["pool"]) / result["pool"]
    return result


# Ages and the greats' rows


def mid_season(season: pd.Series, calendar: pd.Series) -> pd.Series:
    """1 January of the season's second year; 1 July for calendar year leagues."""
    year = season.astype(int)
    days = np.where(calendar, year.astype(str) + "-07-01", (year + 1).astype(str) + "-01-01")
    return pd.Series(pd.to_datetime(days), index=season.index)


def mid_season_age(meta: pd.DataFrame, calendar_leagues) -> pd.Series:
    """Age in the middle of the season from the age at the feature date."""
    mid = mid_season(meta["season"], meta["league_key"].isin(set(calendar_leagues)))
    lag = (pd.to_datetime(meta["feature_date"]) - mid).dt.days / 365.25
    return meta["age"].astype(float) - lag


def great_rows(
    sb_meta: pd.DataFrame, sb_emb: np.ndarray, own: OwnData, calendar_leagues
) -> pd.DataFrame:
    """Every great's seasons: StatsBomb before 2015/16 (also thin ones,
    flagged), our own main rows from it; one row per great and season with
    the age in the middle of the season and the embedding."""
    emb_cols = [c for c in own.vectors.columns if c.startswith("emb_")]
    parts = []
    for great in GREATS:
        if great.statsbomb_id is not None:
            at = sb_meta.index[
                (sb_meta["source_player_id"] == great.statsbomb_id)
                & (sb_meta["season"] < FIRST_SEASON)
            ]
            if len(at):
                m = sb_meta.loc[at]
                mid = mid_season(m["season"], pd.Series(False, index=m.index))
                part = pd.DataFrame(
                    {
                        "great": great.name,
                        "source": "statsbomb",
                        "season": m["season"].astype(int).to_numpy(),
                        "age": ((mid - pd.Timestamp(great.born)).dt.days / 365.25).to_numpy(),
                        "team": m["team_name"].to_numpy(),
                        "league_key": m["league_key"].to_numpy(),
                        "group": m["group"].to_numpy(),
                        "minutes": m["minutes"].astype(float).to_numpy(),
                        "ability_pct": m["ability_pct"].astype(float).to_numpy(),
                        "peers": m["league_key"].map(PEER_GROUPS).to_numpy(),
                    }
                )
                parts.append(pd.concat([part, pd.DataFrame(sb_emb[at], columns=emb_cols)], axis=1))
        if great.api_football_id is not None:
            v = own.vectors[
                (own.vectors["source_player_id"] == great.api_football_id) & own.vectors["main"]
            ]
            if len(v):
                part = pd.DataFrame(
                    {
                        "great": great.name,
                        "source": "api_football",
                        "season": v["season"].astype(int).to_numpy(),
                        "age": mid_season_age(v, calendar_leagues).to_numpy(),
                        "team": v["team_name"].to_numpy(),
                        "league_key": v["league_key"].to_numpy(),
                        "group": v["group"].to_numpy(),
                        "minutes": v["minutes"].astype(float).to_numpy(),
                        "ability_pct": v["ability_pct"].astype(float).to_numpy(),
                        "peers": "every player of his league",
                    }
                )
                parts.append(pd.concat([part, v[emb_cols].reset_index(drop=True)], axis=1))
    table = pd.concat(parts, ignore_index=True)
    table["age_years"] = np.floor(table["age"]).astype(int)
    table["thin"] = table["minutes"] < THIN_MINUTES
    # One row per great and age: the season with most minutes at that age.
    table = table.sort_values(["great", "age_years", "minutes"], ascending=[True, True, False])
    table = table.drop_duplicates(["great", "age_years"]).sort_values(["great", "season"])
    table[emb_cols] = table[emb_cols].astype("float32")
    return table.reset_index(drop=True)


def today_pool(own: OwnData, calendar_leagues) -> pd.DataFrame:
    """Today's players (main rows of the newest season) with their age in the
    middle of the season."""
    v = own.vectors[own.vectors["main"]]
    today = v[v["season"] == v["season"].max()].copy()
    today["age_mid"] = mid_season_age(today, calendar_leagues)
    return today.reset_index(drop=True)


def nearest_to_great(
    greats: pd.DataFrame,
    today: pd.DataFrame,
    name: str,
    age: int,
    k: int = TOP,
    only: set | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Today's players of that age (in the middle of the season) and the
    great's position group, nearest to the great at that age; with ``only``,
    among those player ids (the elite pool)."""
    rows = greats[(greats["great"] == name) & (greats["age_years"] == age)]
    if rows.empty:
        raise KeyError(f"no season of {name} at {age}")
    query = rows.iloc[0]
    emb_cols = [c for c in greats.columns if c.startswith("emb_")]
    pool = today[(np.floor(today["age_mid"]) == age) & (today["group"] == query["group"])]
    if only is not None:
        pool = pool[pool["source_player_id"].astype(int).isin(only)]
    sims = (
        unit(pool[emb_cols].to_numpy(dtype=float))
        @ unit(query[emb_cols].to_numpy(dtype=float)[None, :])[0]
    )
    order = np.argsort(-sims, kind="stable")[:k]
    found = pool.iloc[order].copy()
    found.insert(0, "rank", np.arange(1, len(order) + 1))
    found["similarity"] = sims[order]
    return query, found.reset_index(drop=True)


# The command


def run_greats(settings) -> Path:
    """``scout greats``: the mapping validation, the peer group vectors, the
    cross source identity check, the greats' table and reports/greats.md."""
    from scout import greats_check as check
    from scout.lists import league_names

    processed = settings.paths.processed_dir
    calendar = [lg.key for lg in settings.leagues if lg.calendar_year]
    sb = read_matches(processed)
    af = pd.concat(
        [
            pd.read_parquet(
                processed / "player_match_stats" / "league=la_liga" / f"season={s}.parquet"
            )
            for s in check.CHECK_SEASONS
        ],
        ignore_index=True,
    )
    sb_check = sb[(sb["league_key"] == "la_liga") & sb["season"].isin(list(check.CHECK_SEASONS))]
    matches = check.link_matches(sb_check, af)
    links = check.link_players(sb_check, af, matches)
    pairs = check.paired_rows(sb_check, af, links)
    table = check.measure_table(pairs)
    by_season = check.season_measures(pairs)
    excluded = check.excluded_measures(table)
    ids = check.player_ids(links)
    log.info(
        "greats: %d of %d matches linked; excluded %s",
        len(matches),
        sb_check["fixture_id"].nunique(),
        sorted(excluded),
    )

    own = own_data(settings)
    rows = season_rows(sb)
    sb_meta, sb_emb = statsbomb_vectors(rows, own, excluded=list(excluded))

    # The check: linked players in their API Football position group.
    af_pos = own.vectors.drop_duplicates("source_player_id").set_index("source_player_id")[
        "sub_position"
    ]
    linked_pos = rows["source_player_id"].map(ids).map(af_pos)
    linked_pos = linked_pos.where(rows["season"].isin(list(check.CHECK_SEASONS)))
    chk_meta, chk_emb = statsbomb_vectors(
        rows, own, excluded=list(excluded), sub_position=linked_pos
    )
    identity = identity_ranks(chk_meta, chk_emb, own.vectors, ids)
    every_meta, every_emb = statsbomb_vectors(rows, own, sub_position=linked_pos)
    identity_all = identity_ranks(every_meta, every_emb, own.vectors, ids)
    peer_bias = thin_peer_check(rows, own, list(excluded), ids, sb)

    season_cmp = check.season_table(rows, own.features, ids, COMPOSITE_CANDIDATES)

    greats = great_rows(sb_meta, sb_emb, own, calendar)
    out = processed / GREATS_OUT
    out.mkdir(parents=True, exist_ok=True)
    greats.to_parquet(out / "greats.parquet", index=False)
    logo = settings.paths.raw_dir / "statsbomb" / "files" / "statsbomb-logo.png"
    if logo.is_file():
        shutil.copyfile(logo, out / "statsbomb-logo.png")
    atomic_write_text(
        out / "greats.json",
        json.dumps(
            {
                "label": LABEL,
                "attribution": ATTRIBUTION,
                "excluded": sorted(excluded),
                "identity_hit": float(identity["hit"].mean()) if len(identity) else None,
                "created": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
            },
            indent=1,
        )
        + "\n",
    )
    sb_meta.drop(
        columns=[c for c in sb_meta.columns if c.endswith("_per90")], errors="ignore"
    ).assign(
        **{f"emb_{i}": sb_emb[:, i].astype("float32") for i in range(sb_emb.shape[1])}
    ).to_parquet(processed / OUT / "vectors.parquet", index=False)

    today = today_pool(own, calendar)
    names = league_names(settings)
    chances = elite_chances(processed)
    pool = set(chances.index[chances["elite_chance_pct"].notna()].astype(int))
    examples = []
    for name, age in EXAMPLES:
        try:
            query, found = nearest_to_great(greats, today, name, age, k=5)
            _, in_pool = nearest_to_great(greats, today, name, age, k=5, only=pool)
        except KeyError as exc:
            log.warning("greats: example %s at %d skipped: %s", name, age, exc)
            continue
        examples.append((name, age, query, found, in_pool))
    path = write_report(
        settings.paths.reports_dir,
        download=download_summary(settings.paths.raw_dir),
        sb=sb,
        rows=rows,
        matches=matches,
        pairs=pairs,
        table=table,
        by_season=by_season,
        excluded=excluded,
        season_cmp=season_cmp,
        identity=identity,
        identity_all=identity_all,
        peer_bias=peer_bias,
        greats=greats,
        examples=examples,
        names=names,
        chances=chances,
    )
    log.info(
        "greats: %d great seasons of %d greats; %s", len(greats), greats["great"].nunique(), path
    )
    return path


def thin_peer_check(rows, own, excluded, ids, sb) -> dict:
    """2015/16 is complete in StatsBomb: Barcelona's players standardised
    against the whole league and against Barcelona and its opponents only
    (what every season before it has). The cosine between the two vectors
    of the same player season, and the identity check on each."""
    season = 2015
    barca = sb[
        (sb["league_key"] == "la_liga")
        & (sb["season"] == season)
        & (sb["team_name"] == "Barcelona")
    ]
    in_barca = sb["fixture_id"].isin(barca["fixture_id"].unique())
    thin = season_rows(sb[(sb["league_key"] == "la_liga") & (sb["season"] == season) & in_barca])
    full = rows[(rows["league_key"] == "la_liga") & (rows["season"] == season)].reset_index(
        drop=True
    )
    pos_full = (
        full["source_player_id"]
        .map(ids)
        .map(
            own.vectors.drop_duplicates("source_player_id").set_index("source_player_id")[
                "sub_position"
            ]
        )
    )
    pos_thin = (
        thin["source_player_id"]
        .map(ids)
        .map(
            own.vectors.drop_duplicates("source_player_id").set_index("source_player_id")[
                "sub_position"
            ]
        )
    )
    fm, fe = statsbomb_vectors(full, own, excluded, sub_position=pos_full)
    tm, te = statsbomb_vectors(thin, own, excluded, sub_position=pos_thin)
    team = barca["team_id"].iloc[0]
    fb = fm.index[(fm["team_id"] == team) & fm["regular"].astype(bool)]
    tb = tm.index[(tm["team_id"] == team) & tm["regular"].astype(bool)]
    joined = pd.DataFrame({"pid": fm.loc[fb, "source_player_id"].to_numpy(), "f": fb}).merge(
        pd.DataFrame({"pid": tm.loc[tb, "source_player_id"].to_numpy(), "t": tb}), on="pid"
    )
    cos = np.array([float(fe[f] @ te[t]) for f, t in zip(joined["f"], joined["t"], strict=True)])
    full_id = identity_ranks(fm.loc[fb].reset_index(drop=True), fe[fb], own.vectors, ids)
    thin_id = identity_ranks(tm.loc[tb].reset_index(drop=True), te[tb], own.vectors, ids)
    return {
        "players": len(joined),
        "cosine_median": float(np.median(cos)) if len(cos) else np.nan,
        "cosine_min": float(cos.min()) if len(cos) else np.nan,
        "peers_full": int((full["minutes"] >= PEER_MINUTES).sum()),
        "peers_thin": int((thin["minutes"] >= PEER_MINUTES).sum()),
        "full_hit": float(full_id["hit"].mean()) if len(full_id) else np.nan,
        "thin_hit": float(thin_id["hit"].mean()) if len(thin_id) else np.nan,
        "full_rank": float(full_id["rank"].median()) if len(full_id) else np.nan,
        "thin_rank": float(thin_id["rank"].median()) if len(thin_id) else np.nan,
        "searched": len(full_id),
    }


def download_summary(raw_dir: Path) -> dict:
    from scout.providers.statsbomb.client import body_bytes

    files, size = body_bytes(raw_dir)
    return {"files": files, "bytes": size}


# The report


def elite_chances(processed: Path) -> pd.DataFrame:
    """player_id -> the calibrated elite chance of the candidates table
    (blank outside the elite pool)."""
    path = Path(processed) / "lists" / "candidates.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["elite_chance_pct", "elite_chance_players"])
    cand = pd.read_parquet(path, columns=["player_id", "elite_chance_pct", "elite_chance_players"])
    return cand.drop_duplicates("player_id").set_index("player_id")


def _pct(x: float, digits: int = 1) -> str:
    return "" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def _r(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


def _season(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


def _chance(pct, players) -> str:
    if pd.isna(pct):
        return "outside the elite pool"
    return ("under 5%" if pct < 5 else f"{int(pct)}%") + f" (of {int(players):,})"


def write_report(reports_dir: Path, **k) -> Path:
    from scout.greats_check import MIN_CORRELATION, STYLE_COUNTS
    from scout.providers.statsbomb.adapter import MAPPING

    sb, table, identity = k["sb"], k["table"], k["identity"]
    names = k["names"]
    per = sb.groupby(["league_key", "season"]).agg(
        matches=("fixture_id", "nunique"), played=("minutes", lambda m: int((m > 0).sum()))
    )
    lines = [
        "# The measured greats (prompt 18b)",
        "",
        "Written by `scout greats` (`src/scout/greats.py`, `src/scout/greats_check.py`). "
        "StatsBomb open data (github.com/statsbomb/open-data), used for non commercial "
        "analysis under the StatsBomb Public Data User Agreement, credited with the StatsBomb "
        "logo wherever results are published. The data itself is never published.",
        "",
        "## Data",
        "",
        f"Cached under `data/raw/statsbomb`: {k['download']['files']:,} files, "
        f"{k['download']['bytes'] / 1e9:.2f} GB ({k['download']['bytes']:,} bytes; matches, "
        "lineups and events, plus the user agreement and the logo). "
        f"{sb['fixture_id'].nunique():,} matches, {len(sb):,} player match rows, "
        f"{int((sb['minutes'].fillna(0) > 0).sum()):,} with minutes; the contract "
        "(`PLAYER_MATCH_STATS`) passes.",
        "",
        "| League | Season | Matches | Player matches played |",
        "|---|---|---|---|",
    ]
    for (league, season), r in per.iterrows():
        name = names.get(league, league)
        lines.append(f"| {name} | {_season(season)} | {r['matches']} | {r['played']:,} |")
    lines += [
        "",
        "## The mapping",
        "",
        "| Column | StatsBomb event | What is counted |",
        "|---|---|---|",
    ]
    for column, (event, what) in MAPPING.items():
        lines.append(f"| `{column}` | {event} | {what} |")
    m = k["matches"]
    check = sb[(sb["league_key"] == "la_liga") & sb["season"].between(2015, 2020)]
    lines += [
        "",
        "## Validation: the same player and match in both sources",
        "",
        f"La Liga 2015/16 to 2020/21: {len(m):,} of "
        f"{check['fixture_id'].nunique():,} "
        f"StatsBomb matches found in API Football (same day, score and clubs), "
        f"{len(k['pairs']):,} player matches linked (shirt number and name) where both "
        "sources give minutes. API Football's blank counts are read as 0 where its league "
        "season records them for anyone, as the style inputs read them. Mean difference is "
        "StatsBomb minus API Football, per match.",
        "",
        "| Measure | Player matches | Correlation | Mean StatsBomb | Mean API Football "
        "| Difference |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples(index=False):
        flag = " (below 0.7)" if r.correlation < MIN_CORRELATION else ""
        lines.append(
            f"| `{r.measure}` | {r.rows:,} | {_r(r.correlation)}{flag} | {r.mean_statsbomb:.2f} | "
            f"{r.mean_api_football:.2f} | {r.mean_difference:+.2f} |"
        )
    by = k["by_season"]
    lines += [
        "",
        "By season, the measures whose agreement moves:",
        "",
        "| Season | Passes r | diff | Completed passes r | diff | Tackles r "
        "| Interceptions r | Blocks r |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in by.itertuples(index=False):
        lines.append(
            f"| {_season(r.season)} | {_r(r.passes_r)} | {r.passes_diff:+.1f} | "
            f"{_r(r.completed_passes_r)} | {r.completed_passes_diff:+.1f} | {_r(r.tackles_r)} | "
            f"{_r(r.interceptions_r)} | {_r(r.blocks_r)} |"
        )
    excluded = k["excluded"]
    kept = [s for s in STYLE_COUNTS if s not in excluded]
    lines += [
        "",
        "Passes: API Football's pass count is the completed passes until 2019/20 (StatsBomb's "
        "completed passes match it within a pass a match) and every pass from 2020/21; the "
        "StatsBomb count is every pass but throw ins, as API Football counts from 2020/21 and "
        "as today's players are counted. Both correlate far above 0.7, and each season is "
        "standardised on its own, so the vectors keep it. Tackles and blocks exist in API "
        "Football's La Liga only from 2017/18; its 2017/18 tackles disagree with StatsBomb "
        "(see the table) while 2018/19 to 2020/21 agree at about 0.8.",
        "",
        f"**Left out of the StatsBomb vectors** (counts below {MIN_CORRELATION}): "
        + (
            "; ".join(f"`{s}` ({', '.join(c)})" for s, c in excluded.items())
            if excluded
            else "none"
        )
        + f". They read as the position mean. Kept: {len(kept)} of {len(STYLE_COUNTS)}.",
        "",
        "Season level, the per 90 measures of the same player and season (450 minutes or more "
        "in both):",
        "",
        "| Measure | Player seasons | Correlation |",
        "|---|---|---|",
    ]
    for r in k["season_cmp"].itertuples(index=False):
        lines.append(f"| `{r.measure}` | {r.player_seasons:,} | {_r(r.correlation)} |")
    lines += [
        "",
        "## Cross source identity check",
        "",
        "Every StatsBomb season of 2015/16 to 2020/21 with 450 minutes or more whose player is "
        "linked to API Football is embedded with the stored model (peer group "
        "standardisation as for the old seasons, the player in his API Football position "
        "group) and searched among every API Football player of that season and position "
        f"group: how often his own API Football season is among the {HIT_AT} nearest. The "
        "same player's next season was found 19.9% of the time within our own data "
        "(reports/embeddings.md).",
        "",
        "| Seasons | Peer group | Searches | Top 5 | Random | Median rank | Median pool |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, res in (("with the exclusions", identity), ("every measure", k["identity_all"])):
        for name, part in (
            ("2015/16", res[res["season"] == 2015]),
            ("2016/17 to 2020/21", res[res["season"] > 2015]),
            ("all", res),
        ):
            if part.empty:
                continue
            peers = (
                "whole league"
                if name == "2015/16"
                else ("Barcelona and opponents" if name != "all" else "both")
            )
            lines.append(
                f"| {name}, {label} | {peers} | {len(part):,} | {_pct(part['hit'].mean())} | "
                f"{_pct(part['random'].mean(), 2)} | {part['rank'].median():.0f} | "
                f"{part['pool'].median():.0f} |"
            )
    if len(identity):
        lines += [
            "",
            "By season (with the exclusions):",
            "",
            "| Season | Searches | Top 5 | Median rank |",
            "|---|---|---|---|",
        ]
        for season, part in identity.groupby("season"):
            lines.append(
                f"| {_season(season)} | {len(part):,} | {_pct(part['hit'].mean())} | "
                f"{part['rank'].median():.0f} |"
            )
        barca = identity[identity["team"] == "Barcelona"]
        lines += [
            "",
            f"Barcelona's own players: {len(barca):,} searches, top 5 {_pct(barca['hit'].mean())}, "
            f"median rank {barca['rank'].median():.0f}.",
        ]
    pb = k["peer_bias"]
    lines += [
        "",
        "## The peer group",
        "",
        "Before 2015/16 a StatsBomb season holds Barcelona's league matches only (Arsenal's for "
        "the Premier League 2003/04), so the peer group a season is standardised against is "
        "**Barcelona and its opponents only**: thinner (opponents are seen in one or two "
        "matches) and biased (every opponent's numbers are those of a team playing "
        "Barcelona: less of the ball, more defending). The ability percentile that takes the "
        "level out is computed inside the same peer group. Cross era comparisons are rough: "
        "our league and team adjustment starts in 2010 and is not applied here at all.",
        "",
        f"2015/16 is complete in StatsBomb, which measures the cost: Barcelona's "
        f"{pb['players']} regulars standardised against the whole league "
        f"({pb['peers_full']:,} players with {PEER_MINUTES}+ minutes) and against Barcelona "
        f"and its opponents only ({pb['peers_thin']:,}). The median cosine between the two "
        f"vectors of the same player season is {pb['cosine_median']:.2f} (lowest "
        f"{pb['cosine_min']:.2f}); the identity check finds his API Football season in the "
        f"top 5 {_pct(pb['full_hit'])} of the time with the whole league and "
        f"{_pct(pb['thin_hit'])} with the thin peer group (median rank {pb['full_rank']:.0f} "
        f"against {pb['thin_rank']:.0f}, {pb['searched']} searches).",
        "",
        "## The greats",
        "",
        "From 2015/16 a great's seasons are our own vectors (API Football, every league); "
        "before it, StatsBomb's. Ages are in the middle of the season (1 January; 1 July for "
        "calendar year leagues), so Messi at 19 is 2006/07, and today's players are aged the "
        f"same way. A season under {THIN_MINUTES} minutes is kept but marked thin: its vector "
        "is mostly the position mean.",
        "",
        "| Great | Ages | StatsBomb seasons | Own seasons | Thin |",
        "|---|---|---|---|---|",
    ]
    for great, g in k["greats"].groupby("great", sort=False):
        lines.append(
            f"| {great} | {g['age_years'].min()} to {g['age_years'].max()} | "
            f"{int((g['source'] == 'statsbomb').sum())} | "
            f"{int((g['source'] == 'api_football').sum())} | "
            f"{int(g['thin'].sum())} |"
        )
    lines += ["", "## Examples", "", f"_{LABEL}_", ""]
    chances = k.get("chances")

    def nearest_table(found: pd.DataFrame) -> list[str]:
        out = [
            "| # | Player | Age | Club | League | Ability | Similarity | Elite chance |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in found.itertuples(index=False):
            pid = int(r.source_player_id)
            if chances is not None and pid in chances.index:
                chance = _chance(
                    chances.at[pid, "elite_chance_pct"], chances.at[pid, "elite_chance_players"]
                )
            else:
                chance = "not in the candidates table"
            out.append(
                f"| {r.rank} | {r.player_name} | {int(np.floor(r.age_mid))} | {r.team_name} | "
                f"{names.get(r.league_key, r.league_key)} | {r.ability_pct:.0f} | "
                f"{r.similarity:.2f} | {chance} |"
            )
        return out

    for name, age, query, found, in_pool in k["examples"]:
        source = "StatsBomb" if query["source"] == "statsbomb" else "API Football"
        thin = " **Thin season**: mostly the position mean." if query["thin"] else ""
        lines += [
            f"### {name} at {age}",
            "",
            f"{_season(int(query['season']))} at {query['team']} "
            f"({names.get(query['league_key'], query['league_key'])}), "
            f"{query['group'].replace('_', ' ')}, {int(query['minutes']):,} minutes, {source}; "
            f"ability {query['ability_pct']:.0f} percentile within its peer group.{thin}",
            "",
            f"Today's {query['group'].replace('_', ' ')}s aged {age} (in the middle of the "
            "season), nearest five; ability is the percentile of their own season:",
            "",
            *nearest_table(found),
            "",
            "The nearest five in the elite pool (ranked #101 to #1,000 by value, 23 or under), "
            "where an elite chance exists:",
            "",
            *(nearest_table(in_pool) if len(in_pool) else ["None of that age and position."]),
            "",
        ]
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "greats.md"
    atomic_write_text(path, "\n".join(lines))
    return path
