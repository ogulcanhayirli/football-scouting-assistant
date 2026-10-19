"""Team Elo against the transfer based league strength, and the ability score
built on each (``scout elo``, ``reports/elo.md``).

Three questions:

* Does the Elo predict results better than knowing only who plays at home?
  Scored on matches after the fit window, so the settings never saw them.
* Do the two league strength estimates agree, and where not?
* Is the ability score more stable from one season to the next with the Elo
  terms, on the same players? Stability is the year to year correlation of
  the ability percentile for players rated as regulars in both seasons with
  both versions; league movers are shown apart, since league strength
  matters most for them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scout.elo import FIT_END, build_elo, write_tables
from scout.fsutil import atomic_write_text
from scout.scouting import (
    ability,
    ability_elo,
    ability_from,
    elo_league_strength,
    elo_terms,
    league_strength,
)

log = logging.getLogger(__name__)

SEASONS = range(2015, 2026)  # the match tier rows the ability score is built on
BOOTSTRAP = 500


def prediction_table(rated: pd.DataFrame) -> pd.DataFrame:
    """Brier score of the Elo after the fit window, next to the baseline that
    knows only the home side: the home result rate of the fit window."""
    fit = rated[rated["kickoff"] < FIT_END]
    test = rated[rated["kickoff"] >= FIT_END]
    home_rate = float(fit["result_home"].mean())
    rows = []
    for name, part in {
        "all": test,
        "domestic": test[~test["cross_league"]],
        "cross league (cups)": test[test["cross_league"]],
    }.items():
        rows.append(
            {
                "matches": name,
                "count": len(part),
                "elo_brier": ((part["result_home"] - part["expected_home"]) ** 2).mean(),
                "home_rate_brier": ((part["result_home"] - home_rate) ** 2).mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["improvement"] = 1 - out["elo_brier"] / out["home_rate_brier"]
    return out


def strength_comparison(
    transfer: pd.DataFrame, elo_strength: pd.DataFrame, leagues: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Per league: transfer effect (fixed over seasons) and the Elo league
    strength (mean over 2015 to 2025, Elo points against the Premier League),
    the Elo effect in rating units, and each league's rank under both."""
    elo = elo_strength[elo_strength["season"].isin(SEASONS)]
    per = elo.groupby("league_key").agg(
        elo_vs_epl=("elo_league", "mean"), elo_effect=("effect", "mean"), seasons=("season", "size")
    )
    per["elo_vs_epl"] *= 100
    raw = leagues[leagues["season"].isin(SEASONS)].groupby("league_key")["elo"].mean()
    table = transfer.set_index("league_key")[["effect", "moves"]].rename(
        columns={"effect": "transfer_effect"}
    )
    table = table.join(per, how="inner").join(raw.rename("mean_club_elo"))
    table["transfer_rank"] = table["transfer_effect"].rank(method="min").astype(int)
    table["elo_rank"] = table["elo_effect"].rank(method="min").astype(int)
    table["rank_change"] = table["transfer_rank"] - table["elo_rank"]  # + : stronger under Elo
    table["effect_gap"] = table["elo_effect"] - table["transfer_effect"]
    stats = {
        "leagues": len(table),
        "pearson": float(table["transfer_effect"].corr(table["elo_vs_epl"])),
        "spearman": float(table["transfer_effect"].corr(table["elo_vs_epl"], method="spearman")),
        "pearson_effects": float(table["transfer_effect"].corr(table["elo_effect"])),
    }
    table = table.reset_index().sort_values("transfer_effect")
    return table, stats


def _per_player_season(features: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    regular = features["regular"].astype(bool) & (features["tier"] == "match")
    frame = features.loc[regular, ["source_player_id", "season", "league_key", "minutes"]].copy()
    for name, s in scores.items():
        frame[name] = s.loc[regular, "ability_pct"]
    frame = frame.dropna(subset=list(scores))
    frame["season"] = frame["season"].astype(int)
    return frame.sort_values("minutes").drop_duplicates(["source_player_id", "season"], keep="last")


def stability(
    features: pd.DataFrame, scores: dict[str, pd.DataFrame], lag: int = 1
) -> tuple[pd.DataFrame, dict]:
    """Season to season correlation of the ability percentile per version, on
    the same pairs: one row per player and season (the league with most
    minutes), regular in both seasons, scored by every version. ``lag`` 2
    pairs a season with the one after next."""
    a = _per_player_season(features, scores)
    pairs = a.merge(
        a.assign(season=a["season"] - lag),
        on=["source_player_id", "season"],
        suffixes=("_now", "_next"),
    )
    pairs["moved"] = pairs["league_key_now"] != pairs["league_key_next"]
    rows = []
    for subset, part in {
        "all pairs": pairs,
        "same league": pairs[~pairs["moved"]],
        "changed league": pairs[pairs["moved"]],
    }.items():
        row = {"pairs": subset, "count": len(part)}
        for name in scores:
            now, nxt = part[f"{name}_now"], part[f"{name}_next"]
            row[f"{name}_pearson"] = now.corr(nxt)
            row[f"{name}_spearman"] = now.corr(nxt, method="spearman")
            row[f"{name}_mean_abs_change"] = (nxt - now).abs().mean()
        rows.append(row)
    # Paired bootstrap over players for the gap in correlation, all pairs.
    rng = np.random.default_rng(7)
    players = pairs["source_player_id"].unique()
    by_player = pairs.groupby("source_player_id").indices
    names = list(scores)
    gaps = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(players, len(players))
        idx = np.concatenate([by_player[p] for p in pick])
        b = pairs.iloc[idx]
        c = [b[f"{n}_now"].corr(b[f"{n}_next"]) for n in names]
        gaps.append([x - c[0] for x in c[1:]])
    gaps = np.array(gaps)
    full = [pairs[f"{n}_now"].corr(pairs[f"{n}_next"]) for n in names]
    ci = pd.DataFrame(
        {
            "version": names[1:],
            "pearson_minus_transfer": [x - full[0] for x in full[1:]],
            "ci_low": np.quantile(gaps, 0.025, axis=0),
            "ci_high": np.quantile(gaps, 0.975, axis=0),
        }
    )
    return pd.DataFrame(rows), ci


def _long_stability(stable: pd.DataFrame) -> pd.DataFrame:
    """One row per version and subset: pearson, spearman, mean absolute change."""
    rows = []
    versions = [c[: -len("_pearson")] for c in stable.columns if c.endswith("_pearson")]
    for _, row in stable.iterrows():
        for v in versions:
            rows.append(
                {
                    "pairs": row["pairs"],
                    "count": int(row["count"]),
                    "version": v,
                    "pearson": round(row[f"{v}_pearson"], 3),
                    "spearman": round(row[f"{v}_spearman"], 3),
                    "mean_abs_change": round(row[f"{v}_mean_abs_change"], 1),
                }
            )
    return pd.DataFrame(rows)


def season_strength_table(elo_strength: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    e = elo_strength[elo_strength["season"].isin(SEASONS) & elo_strength["league_key"].isin(keys)]
    return (e.pivot(index="league_key", columns="season", values="elo_league") * 100).round(0)


def run_elo_comparison(settings) -> Path:
    from scout.tables import read_feature_table

    params, fit_table, rated, teams, leagues = build_elo(settings)
    write_tables(settings.paths.processed_dir / "elo", rated, teams, leagues)

    features = read_feature_table(
        settings.paths.processed_dir / "features" / "player_season_features"
    )
    transfer, team_gd = league_strength(features)
    terms = elo_terms(features, teams, leagues)
    elo_strength, league_coef, team_coef = elo_league_strength(features, terms)
    # The two mixed versions say which half of the change does what.
    gd = features["team_gd_per_match"].astype(float)
    _, mixed_league, mixed_gd = elo_league_strength(
        features, terms.assign(team_gd=gd), team_column="team_gd"
    )
    mixed_strength, mixed_team = league_strength(
        pd.concat([features, terms], axis=1), team_column="elo_team"
    )
    effect = features["league_key"].map(mixed_strength.set_index("league_key")["effect"])
    _, all_league, all_team = elo_league_strength(features, terms, movers_only=False)
    scores = {
        "transfer": ability(features, transfer, team_gd),
        "elo": ability_elo(features, terms, league_coef, team_coef),
        "elo_all_pairs_fit": ability_elo(features, terms, all_league, all_team),
        "elo_league_gd_team": ability_from(
            features, mixed_league * terms["elo_league"], mixed_gd * gd.fillna(0)
        ),
        "transfer_league_elo_team": ability_from(
            features, effect.astype(float), mixed_team * terms["elo_team"]
        ),
    }
    comparison, corr = strength_comparison(transfer, elo_strength, leagues)
    stable, ci = stability(features, scores)
    coverage = float(terms.loc[features["tier"] == "match", "elo_league"].notna().mean())
    path = write_report(
        settings.paths.reports_dir,
        params=params,
        fit_table=fit_table,
        prediction=prediction_table(rated),
        comparison=comparison,
        corr=corr,
        seasons=season_strength_table(elo_strength, list(comparison["league_key"])),
        stable=stable,
        ci=ci,
        coefs=(league_coef, team_coef, team_gd),
        counts={
            "matches": len(rated),
            "clubs": int(teams["team_id"].nunique()),
            "coverage": coverage,
        },
    )
    log.info("Elo report: %s", path)
    return path


def write_report(reports: Path, **r) -> Path:
    p = r["params"]
    league_coef, team_coef, team_gd = r["coefs"]
    stable = _long_stability(r["stable"])
    ci = r["ci"].set_index("version")
    elo_ci = ci.loc["elo"]
    verdict = (
        "improved"
        if elo_ci["ci_low"] > 0
        else ("got worse" if elo_ci["ci_high"] < 0 else "did not change measurably")
    )
    movers = (
        r["comparison"]
        .reindex(r["comparison"]["rank_change"].abs().sort_values(ascending=False).index)
        .head(6)
    )
    parts = [
        "# Team Elo and league strength",
        "",
        "Built by `scout elo` from the cached `/fixtures` responses, with no API request. "
        "ClubElo is not used: its API went offline in September 2026.",
        "",
        f"* {r['counts']['matches']:,} club matches from 2010 (domestic leagues and the five "
        f"club cups); {r['counts']['clubs']:,} clubs have at least one league season.",
        f"* Fitted on matches before {FIT_END.date()} only: home advantage "
        f"**{p.home_advantage:.0f}** Elo points, K **{p.k:.0f}**, league link "
        f"**{p.spread:.2f}** (a cup result moves the club's league mates by this share of the "
        "club's own change).",
        f"* {r['counts']['coverage']:.1%} of match tier feature rows get an Elo league term.",
        "",
        "## Does the Elo predict results? (matches after the fit window)",
        "",
        "Baseline: every match gets the home result rate of the fit window.",
        "",
        r["prediction"].round(4).to_markdown(index=False),
        "",
        "## League strength: Elo against transfers",
        "",
        f"Pearson correlation of the transfer effect with the Elo gap to the Premier League: "
        f"**{r['corr']['pearson']:.2f}**, Spearman **{r['corr']['spearman']:.2f}** "
        f"({r['corr']['leagues']} leagues; negative is agreement, since a positive transfer "
        "effect means a weaker league). In rating units the two effects correlate "
        f"{r['corr']['pearson_effects']:.2f}.",
        "",
        f"Elo terms in rating units, fitted on the same player pairs (league term from "
        f"league movers): {league_coef:+.3f} rating "
        f"per 100 Elo of league strength, {team_coef:+.3f} per 100 Elo of club strength above "
        f"the league average (the transfer version uses {team_gd:+.3f} per goal of goal "
        "difference per match).",
        "",
        "Rank 1 is the strongest league; `rank_change` above zero means stronger under Elo. "
        "Elo columns are averages over 2015 to 2025.",
        "",
        r["comparison"].round(3).to_markdown(index=False),
        "",
        "Leagues that move most between the two estimates:",
        "",
        movers[["league_key", "transfer_rank", "elo_rank", "rank_change", "moves", "elo_vs_epl"]]
        .round(0)
        .to_markdown(index=False),
        "",
        "## Elo league strength by season (Elo points against the Premier League)",
        "",
        r["seasons"].to_markdown(),
        "",
        "## Ability score stability, same players",
        "",
        "Season to season correlation of the ability percentile, for players who were "
        "regulars in both seasons and are scored by both versions.",
        "",
        "* `transfer`: the current score (transfer league effect, team goal difference).",
        "* `elo`: Elo league term and Elo team term (the candidate; `--strength elo`).",
        "* `elo_all_pairs_fit`: the same, but the league coefficient also fitted on players "
        "who stayed in their league (the first version; their league's Elo drifts while "
        "their rating does not, which halves the coefficient).",
        "* `elo_league_gd_team`: Elo league term, team goal difference (refitted).",
        "* `transfer_league_elo_team`: transfer league effect, Elo team term (refitted).",
        "",
        stable.to_markdown(index=False),
        "",
        "Pearson against the current score on all pairs, with a 95% bootstrap interval "
        "over players:",
        "",
        r["ci"].round(3).to_markdown(index=False),
        "",
        f"Stability with the Elo terms **{verdict}** ({elo_ci['pearson_minus_transfer']:+.3f}).",
        "",
        "## Fit grid (best 10 of the search, matches 2012-07 to 2015-06)",
        "",
        r["fit_table"].head(10).round(5).to_markdown(index=False)
        if r["fit_table"] is not None
        else "(settings given, not fitted)",
        "",
    ]
    path = reports / "elo.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
