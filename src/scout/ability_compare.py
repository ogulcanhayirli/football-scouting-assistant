"""Ability score versions side by side (decision 8, ``scout ability``,
``reports/ability.md``).

* ``transfer``: version 1, the match rating less the transfer league effect
  and team goal difference.
* ``transfer_elo``: the same rating base, transfer league effect, Elo team
  term (the most stable mix in ``reports/elo.md``).
* ``v2``: the per 90 composite with the same two terms, pooled over two
  seasons and shrunk by minutes.

Stability is measured as in ``reports/elo.md``: the season to season
correlation of the ability percentile, one row per player and season,
regular in both, scored by every version. It is run on the original 17
leagues (the pairs of ``reports/elo.md``) and on all 37. Version 2 pools two
seasons, so this season's score and next season's share a season; that
overlap raises the correlation by itself. Two checks without it are shown:
version 2 from one season only, and the correlation two seasons apart
(seasons t-1 and t against t+1 and t+2).

The backtest runs the scouting targets with each version's columns; the
baseline is the success rate by value rank and age. Version 2 chooses its
measures from the step up outcomes, so a look ahead check repeats it with
measures chosen from seasons up to ``EARLY_WEIGHTS`` only and compares test
seasons from ``LATE_TEST`` on, whose models could have known those labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scout.elo_compare import _long_stability, stability
from scout.fsutil import atomic_write_text
from scout.scouting import (
    TARGET_NAMES,
    ability_scores,
    backtest,
    build,
    classifier,
    evaluate,
    load_inputs,
    prepare_models,
    rank_rate,
    target_rows,
    targets,
)

log = logging.getLogger(__name__)

VERSIONS = ["transfer", "transfer_elo", "v2"]
EARLY_WEIGHTS = 2017  # step up labels of these seasons were all known by mid 2020
LATE_TEST = 2020
STABILITY_GAIN = 0.0  # version 2 must be more stable than version 1, by any margin
AUC_TOLERANCE = 0.005


def original_leagues(settings) -> list[str]:
    """The 17 leagues of the first build: the config less the expansion file."""
    from scout.config import load_settings

    expansion = load_settings(Path("configs/backfill.expansion.yaml"))
    added = {lg.key for lg in expansion.leagues}
    return [lg.key for lg in settings.leagues if lg.key not in added]


def stability_tables(features, valuations, data_end, elo_tables) -> dict:
    blank = pd.DataFrame({"ability_pct": np.nan}, index=features.index)
    step_up = targets(features, blank, valuations, data_end)["step_up"]
    scores = {v: ability_scores(features, v, elo_tables, step_up)[0] for v in VERSIONS}
    scores["v2_one_season"] = ability_scores(features, "v2", elo_tables, step_up, pool=False)[0]
    out = {}
    for lag in (1, 2):
        stable, ci = stability(features, scores, lag=lag)
        out[lag] = (_long_stability(stable), ci)
    return out


def run_backtests(data: pd.DataFrame, names=TARGET_NAMES, baseline: bool = True) -> tuple:
    columns = prepare_models(data)
    models = {"lightgbm": classifier()}
    if baseline:
        models = {"baseline": rank_rate, **models}
    predictions = {}
    for target in names:
        predictions[target] = backtest(target_rows(data, target), columns, models)
    return predictions


def summary(predictions: dict, version: str, score: str = "score", since: int | None = None):
    rows = []
    for target, p in predictions.items():
        if since is not None:
            p = p[p["season"].astype(int) >= since]
        rows.append({"target": target, "version": version, **evaluate(p, score=score)})
    return rows


def decide(stable: pd.DataFrame, results: pd.DataFrame) -> tuple[str, list[str]]:
    """The owner's rule: version 2 replaces version 1 if it is more stable and
    its elite and step up AUC are within ``AUC_TOLERANCE`` of version 1 or
    better; otherwise the most stable version that passes."""
    allp = stable[stable["pairs"] == "all pairs"].set_index("version")["pearson"]
    auc = results.pivot(index="version", columns="target", values="auc")
    notes = []
    passes = {}
    for v in VERSIONS:
        floor = auc.loc["transfer"] - AUC_TOLERANCE
        ok = all(auc.loc[v, t] >= floor[t] for t in ("elite", "step_up"))
        passes[v] = ok
        notes.append(
            f"{v}: stability {allp[v]:.3f}, elite AUC {auc.loc[v, 'elite']:.3f}, "
            f"step up AUC {auc.loc[v, 'step_up']:.3f}, AUC rule {'passed' if ok else 'failed'}"
        )
    if passes["v2"] and allp["v2"] > allp["transfer"] + STABILITY_GAIN:
        return "v2", notes
    candidates = [v for v in VERSIONS if passes[v]]
    return max(candidates, key=lambda v: allp[v]), notes


def run_ability_comparison(settings) -> Path:
    features, valuations, data_end, elo_tables = load_inputs(settings, "v2")
    original = features["league_key"].isin(original_leagues(settings))
    stab = {
        "17": stability_tables(features[original], valuations, data_end, elo_tables),
        "37": stability_tables(features, valuations, data_end, elo_tables),
    }
    log.info("ability: stability done")

    rows, late, details = [], [], {}
    for version in VERSIONS:
        data, strength, team, det = build(features, valuations, data_end, version, elo_tables)
        details[version] = (strength, team, det)
        predictions = run_backtests(data)
        rows += summary(predictions, version)
        late += summary({t: predictions[t] for t in ("elite", "step_up")}, version, since=LATE_TEST)
        if version == "transfer":
            rows += summary(predictions, "baseline", score="score_baseline")
        log.info("ability: backtest %s done", version)
    data, *_ = build(features, valuations, data_end, "v2", elo_tables, weights_until=EARLY_WEIGHTS)
    early = run_backtests(data, ("elite", "step_up"), baseline=False)
    late += summary(early, f"v2, measures from seasons to {EARLY_WEIGHTS}", since=LATE_TEST)

    results = pd.DataFrame(rows)
    chosen, notes = decide(stab["37"][1][0], results[results["version"] != "baseline"])
    path = write_report(
        settings.paths.reports_dir, stab, results, pd.DataFrame(late), details, chosen, notes
    )
    log.info("Ability report: %s (chosen: %s)", path, chosen)
    return path


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].round(3).to_markdown(index=False)


def write_report(reports: Path, stab, results, late, details, chosen, notes) -> Path:
    shown = ["target", "version", "rows", "successes", "base_rate", "auc", "average_precision"]
    shown.append("precision_top50")
    market = results[results["target"].isin(["elite", "step_up"])]
    performance = results[results["target"] == "performance"]
    _, _, v2 = details["v2"]
    prior = pd.Series(v2["prior_minutes"], name="prior_minutes").round(0).to_frame()
    parts = [
        "# Ability score versions (decision 8)",
        "",
        "Built by `scout ability` from local files, with no API request.",
        "",
        "* `transfer`: version 1 (match rating, transfer league effect, team goal difference).",
        "* `transfer_elo`: match rating, transfer league effect, Elo team term "
        "(mean pre match club rating over the season's league matches, against the league's).",
        "* `v2`: per 90 composite, transfer league effect and Elo team term refitted in "
        "composite units, last two seasons pooled by minutes, shrunk by empirical Bayes.",
        "* `v2_one_season`: version 2 without the pooling (stability check only).",
        "",
        "## Stability, original 17 leagues (the pairs of reports/elo.md)",
        "",
        stab["17"][1][0].to_markdown(index=False),
        "",
        stab["17"][1][1].round(3).to_markdown(index=False),
        "",
        "## Stability, all 37 leagues",
        "",
        stab["37"][1][0].to_markdown(index=False),
        "",
        stab["37"][1][1].round(3).to_markdown(index=False),
        "",
        "Version 2 pools two seasons, so a score and the next season's share one "
        "season, which raises the correlation by itself. Without the overlap:",
        "",
        "Two seasons apart (season t against t+2), all 37 leagues:",
        "",
        stab["37"][2][0].to_markdown(index=False),
        "",
        stab["37"][2][1].round(3).to_markdown(index=False),
        "",
        "## Backtest, all 37 leagues (2 seasons ahead): elite and step up",
        "",
        "Baseline: the success rate of players with a similar value rank and age in the "
        "training seasons.",
        "",
        _table(market, shown),
        "",
        "## Performance target (not like for like)",
        "",
        "The performance label is built from each version's own ability percentile, so "
        "each row below predicts a different target.",
        "",
        _table(performance, shown),
        "",
        f"## Look ahead check: test seasons {LATE_TEST} on",
        "",
        "Version 2 chooses its measures from step up outcomes of every season. Here it is "
        f"repeated with measures chosen from seasons up to {EARLY_WEIGHTS} only, whose "
        f"labels were known before the {LATE_TEST} test season.",
        "",
        _table(late, shown),
        "",
        "## Decision",
        "",
        "Rule (owner): version 2 replaces version 1 only if it is more stable and its elite "
        f"and step up AUC are within {AUC_TOLERANCE} of version 1 or better; otherwise the "
        "most stable version that passes. Stability: all pairs, Pearson, 37 leagues.",
        "",
        *[f"* {n}" for n in notes],
        "",
        f"Chosen: **{chosen}**.",
        "",
        "## Version 2 measures and weights per position group",
        "",
        v2["weights"].round(3).to_markdown(index=False),
        "",
        "Prior strength (minutes at which one season's score is half data, half position mean):",
        "",
        prior.to_markdown(),
        "",
        "League effects in composite units (positive: a player scores higher there):",
        "",
        details["v2"][0].round(3).to_markdown(index=False),
        "",
        f"Team term: {details['v2'][1]:+.3f} composite units per 100 Elo above the league "
        f"average ({details['transfer_elo'][1]:+.3f} rating per 100 Elo on the rating base).",
        "",
    ]
    path = reports / "ability.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
