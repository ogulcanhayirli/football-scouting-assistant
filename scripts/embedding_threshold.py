"""Can the similar players index go below 450 league minutes? (prompt 20f)

The style vectors exist for regular player seasons only (450 league minutes
or more), because the level is taken out with the two ability percentiles,
which version 2 gives regulars only. This check extends both percentiles to
the seasons from ``--minutes`` up to 449 (each such season's ability score,
version 2 with the stored fit, ranked against the regulars of its position
group and season) and reruns the identity retrieval test of decision 10 on
both populations, with the stored embedding's settings (dimension,
temperature, epochs), trained on seasons up to 2022 and tested on 2023 to
2025:

* overall: the hit rate (the same player's other season among his five
  nearest) of each population as a whole;
* regulars: the pairs where both seasons have 450 minutes or more, so the
  old queries are judged against the larger pools the new seasons make;
* added: the pairs where a season has fewer than 450 minutes.

The owner's rule (prompt 20f): lower the threshold only if the test does not
drop. Writes nothing but ``reports/embedding_threshold.md``.

    uv run python scripts/embedding_threshold.py --minutes 270
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from scout.config import load_settings
from scout.embeddings import (
    TEST_SEASONS,
    TRAIN_UNTIL,
    identity_retrieval,
    load_inputs,
    season_pairs,
    style_vectors,
    train_encoder,
)
from scout.features.build import REGULAR_MINUTES
from scout.fsutil import atomic_write_text
from scout.models.store import load_ability, store_dir
from scout.scouting import ability_scores, position_group

log = logging.getLogger("embedding_threshold")


def extended_percentiles(features: pd.DataFrame, score: pd.Series, floor: int) -> pd.Series:
    """The percentile of every season from ``floor`` minutes: a regular's as
    version 2 ranks it, anyone else's against the regulars of his position
    group and season."""
    group = position_group(features)
    season = features["season"].astype(int)
    minutes = features["minutes"].astype(float)
    regular = features["regular"].astype(bool) & score.notna() & group.notna()
    out = pd.Series(np.nan, index=features.index)
    out[regular] = score[regular].groupby([group[regular], season[regular]]).rank(pct=True).mul(100)
    extra = ~regular & (minutes >= floor) & score.notna() & group.notna()
    for (g, s), rows in features[extra].groupby([group[extra], season[extra]]).groups.items():
        ref = np.sort(score[regular & (group == g) & (season == s)].to_numpy())
        if len(ref):
            out[rows] = np.searchsorted(ref, score[rows].to_numpy(), side="right") / len(ref) * 100
    return out


def run(features, ability, floor, dim, temperature, epochs) -> dict:
    f = features.assign(regular=features["minutes"].astype(float) >= floor)
    test = style_vectors(f, ability, fit_until=TRAIN_UNTIL)
    pairs = season_pairs(test.meta)
    train = pairs[pairs["season"] + 1 <= TRAIN_UNTIL]
    encoder, _ = train_encoder(test.style, train, dim, temperature, epochs=epochs)
    learned = encoder.embed(test.style)
    minutes = test.meta["minutes"].astype(float).to_numpy()
    out = {"vectors": len(test.meta), "train_pairs": len(train)}
    for name, vectors in (("baseline", test.style), ("learned", learned)):
        result = identity_retrieval_both(vectors, test.meta, minutes)
        out[name] = {
            "overall": float(result["hit"].mean()),
            "pairs": len(result),
            "regulars": float(result.loc[result["regular_pair"], "hit"].mean()),
            "regular_pairs": int(result["regular_pair"].sum()),
            "added": float(result.loc[~result["regular_pair"], "hit"].mean())
            if (~result["regular_pair"]).any()
            else None,
            "added_pairs": int((~result["regular_pair"]).sum()),
        }
    return out


def identity_retrieval_both(vectors, meta, minutes) -> pd.DataFrame:
    """``identity_retrieval`` with each pair marked: both seasons regulars?"""
    seasons = set(TEST_SEASONS)
    pairs = season_pairs(meta)
    pairs = pairs[pairs["season"].isin(seasons) & (pairs["season"] + 1).isin(seasons)]
    both = (minutes[pairs["a"]] >= REGULAR_MINUTES) & (minutes[pairs["b"]] >= REGULAR_MINUTES)
    result = identity_retrieval(vectors, meta, TEST_SEASONS)
    # identity_retrieval lists the pairs grouped, both ways; rebuild the same order
    order = []
    for target_col in ("a", "b"):  # the order identity_retrieval searches in
        target_season = meta.loc[pairs[target_col], "season"].to_numpy()
        for _, rows in pairs.groupby([pairs["group"], target_season]).groups.items():
            order.append(both[pairs.index.get_indexer(rows)])
    result["regular_pair"] = np.concatenate(order)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backfill.ultra.yaml")
    parser.add_argument("--minutes", type=int, default=270)
    args = parser.parse_args()
    settings = load_settings(args.config)
    processed = settings.paths.processed_dir
    stored = pd.read_json(store_dir(processed) / "embedding.json", typ="series")
    dim, temperature, epochs = (
        int(stored["dim"]),
        float(stored["temperature"]),
        int(stored["epochs"]),
    )

    from scout.elo import read_tables

    features, ability = load_inputs(settings)
    fit = load_ability(store_dir(processed))
    elo = read_tables(processed)
    two, *_ = ability_scores(features, "v2", elo, None, pool=True, fit=fit)
    one, *_ = ability_scores(features, "v2", elo, None, pool=False, fit=fit)
    extended = pd.DataFrame(
        {
            "ability_pct": extended_percentiles(features, two["adj_rating"], args.minutes),
            "ability_pct_1s": extended_percentiles(features, one["adj_rating"], args.minutes),
        },
        index=features.index,
    )
    regular = features["regular"].astype(bool) & ability["ability_pct"].notna()
    agree = np.allclose(
        extended.loc[regular, "ability_pct"], ability.loc[regular, "ability_pct"], atol=1e-6
    )
    log.info("regulars' percentiles rebuilt as stored: %s", agree)

    results = {
        REGULAR_MINUTES: run(features, ability, REGULAR_MINUTES, dim, temperature, epochs),
        args.minutes: run(features, extended, args.minutes, dim, temperature, epochs),
    }
    newest = features[features["season"].astype(int) == max(TEST_SEASONS)]
    with_minutes = int((newest["minutes"].astype(float) > 0).sum())
    lines = [
        "# Similar players below 450 minutes? (prompt 20f)",
        "",
        f"`scripts/embedding_threshold.py --minutes {args.minutes}`: the identity retrieval "
        f"test (top 5, trained to {TRAIN_UNTIL}, tested on {TEST_SEASONS[0]} to "
        f"{TEST_SEASONS[-1]}, dimension {dim}, temperature {temperature}, {epochs} epochs) with "
        f"vectors from {REGULAR_MINUTES} league minutes (as stored) and from {args.minutes}. "
        f"The regulars' percentiles rebuilt with the stored fit match the stored ones: {agree}.",
        "",
        "| From | Vectors | Method | Overall | Regular pairs | Added pairs |",
        "|---|---|---|---|---|---|",
    ]
    for floor, r in results.items():
        for method in ("baseline", "learned"):
            m = r[method]
            added = "" if m["added"] is None else f"{100 * m['added']:.1f}% of {m['added_pairs']:,}"
            lines.append(
                f"| {floor} | {r['vectors']:,} | {method} | {100 * m['overall']:.1f}% of "
                f"{m['pairs']:,} | {100 * m['regulars']:.1f}% of {m['regular_pairs']:,} | "
                f"{added} |"
            )
    lines += ["", f"League rows of {max(TEST_SEASONS)} with a minute: {with_minutes:,}.", ""]
    path = settings.paths.reports_dir / "embedding_threshold.md"
    atomic_write_text(path, "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
