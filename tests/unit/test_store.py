"""Stored models: a saved booster scores exactly as the trained one, and the
stored ability fit reproduces the fitted score."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from scout.models.store import (
    AbilityFit,
    load_ability,
    load_booster,
    save_ability,
    save_booster,
    stored_folds,
)
from scout.scouting import COMPOSITE_CANDIDATES, ability_v2


def _rows(rng, n=600, leagues=("epl", "la_liga", "serie_a")):
    frame = pd.DataFrame(
        {
            "league_key": pd.Categorical(rng.choice(leagues, n), categories=sorted(leagues)),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    frame["y"] = (frame["x"] + (frame["league_key"] == "epl") + rng.normal(0, 0.5, n) > 0.5) * 1.0
    return frame


def test_a_stored_booster_predicts_exactly_like_the_trained_one(tmp_path):
    rng = np.random.default_rng(0)
    train = _rows(rng)
    columns = ["league_key", "x", "z"]
    params = {"objective": "binary", "verbosity": -1, "seed": 7, "deterministic": True}
    booster = lgb.train(params, lgb.Dataset(train[columns], label=train["y"]), 30)
    save_booster(booster, tmp_path, "elite", columns, model="elite", trained_rows=len(train))
    stored = load_booster(tmp_path, "elite")
    assert stored.columns == columns
    assert stored.meta["trained_rows"] == len(train)
    test = _rows(np.random.default_rng(1), n=50)
    np.testing.assert_array_equal(stored.predict(test), booster.predict(test[columns]))
    # Categories are mapped back by value: the newest rows may list them in
    # another order, or bring one the model never saw.
    shuffled = test.assign(
        league_key=pd.Categorical(
            test["league_key"].astype(str), categories=["serie_a", "liga_mx", "la_liga", "epl"]
        )
    )
    np.testing.assert_array_equal(stored.predict(shuffled), booster.predict(test[columns]))


def test_a_missing_model_names_the_command_that_trains_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="Retraining"):
        load_booster(tmp_path, "riser")


def test_stored_folds_reads_the_fold_from_the_name(tmp_path):
    for name in ("fair_value_fold_2024.txt", "fair_value_fold_2025.txt", "elite.txt"):
        (tmp_path / name).write_text("x")
    assert stored_folds(tmp_path) == [2024, 2025]


def _ability_rows():
    rng = np.random.default_rng(4)
    shift = {"epl": 0.0, "super_lig": 0.3, "eredivisie": 0.5}
    leagues = list(shift)
    level = rng.normal(0, 1, 600)
    rows = []
    for season in range(2015, 2020):
        for p in range(600):
            league = leagues[(p // 3 + (season if p % 3 == 0 else 0)) % 3]
            minutes = rng.uniform(500, 3000)
            rows.append(
                {
                    "league_key": league,
                    "season": season,
                    "source_player_id": p + 1,
                    "tier": "match",
                    "minutes": minutes,
                    "age": 25.0,
                    "regular": True,
                    "sub_position": "Centre-Back" if p % 2 else "Centre-Forward",
                    "goals_per90": level[p] + shift[league] + rng.normal(0, np.sqrt(600 / minutes)),
                }
            )
    features = pd.DataFrame(rows)
    for c in COMPOSITE_CANDIDATES:
        if c != "goals_per90":
            features[c] = np.nan
    terms = pd.DataFrame({"elo_league": 0.0, "elo_team": 0.0}, index=features.index)
    weights = pd.DataFrame(
        {
            "position": ["centre_back", "centre_forward"],
            "measure": ["goals_per90", "goals_per90"],
            "gap": [10.0, 12.0],
            "weight": [1.0, 1.0],
        }
    )
    return features, terms, weights


def test_the_stored_ability_fit_gives_the_fitted_score(tmp_path):
    features, terms, weights = _ability_rows()
    fitted, strength, team_coef, prior = ability_v2(features, terms, weights)
    save_ability(AbilityFit(weights, strength, team_coef, prior), tmp_path)
    fit = load_ability(tmp_path)
    pd.testing.assert_frame_equal(fit.weights, weights)
    again, _, team2, prior2 = ability_v2(features, terms, fit.weights, fit=fit)
    pd.testing.assert_frame_equal(again, fitted)
    assert team2 == team_coef
    assert prior2 == prior
    # A new season is scored on the stored scale, not refitted with it.
    later = features[features["season"] == 2019].assign(season=2020)
    later["goals_per90"] += 5.0  # would move every fitted coefficient
    both = pd.concat([features, later], ignore_index=True)
    terms_both = pd.DataFrame({"elo_league": 0.0, "elo_team": 0.0}, index=both.index)
    _, strength3, team3, prior3 = ability_v2(both, terms_both, fit.weights, fit=fit)
    assert team3 == team_coef
    assert prior3 == prior
    pd.testing.assert_frame_equal(strength3, fit.strength)


def test_replace_season_keeps_each_players_place():
    from scout.models.run import replace_season

    old = pd.DataFrame(
        {
            "league_key": ["epl", "epl", "j1", "epl"],
            "season": [2024, 2025, 2025, 2025],
            "source_player_id": [1, 1, 2, 3],
            "pred_lightgbm": [1.0, 2.0, 3.0, 4.0],
        }
    )
    new = pd.DataFrame(
        {
            "league_key": ["epl", "epl", "j1"],
            "season": [2025, 2025, 2025],
            "source_player_id": [4, 1, 2],
            "pred_lightgbm": [9.0, 2.5, 3.5],
        }
    )
    out = replace_season(old, new, 2025)
    assert out["source_player_id"].tolist() == [1, 1, 2, 4]  # player 3 gone, 4 new and last
    assert out["pred_lightgbm"].tolist() == [1.0, 2.5, 3.5, 9.0]
    same = replace_season(old, old[old["season"] == 2025], 2025)
    pd.testing.assert_frame_equal(same, old)
