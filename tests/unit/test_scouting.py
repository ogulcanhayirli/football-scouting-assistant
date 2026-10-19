import numpy as np
import pandas as pd
import pytest

from scout.scouting import (
    COMPOSITE_CANDIDATES,
    DEFAULT_STRENGTH,
    STRENGTH_SOURCES,
    _rank_of,
    ability,
    ability_elo,
    ability_v2,
    composite_inputs,
    elo_league_strength,
    elo_terms,
    league_strength,
    prior_minutes,
    rank_rate,
    targets,
    team_shares,
    trait_table,
    two_seasons,
)


def _season_rows(rng, n_players=400, seasons=range(2015, 2021)):
    """Players with a fixed true level; ratings are level + league effect +
    team effect + noise. Some players switch leagues every season."""
    effects = {"epl": 0.0, "super_lig": 0.15, "eredivisie": 0.25}
    leagues = list(effects)
    level = rng.normal(7.0, 0.2, n_players)
    rows = []
    for season in seasons:
        for p in range(n_players):
            league = leagues[(p + (season if p % 3 == 0 else 0)) % 3]
            gd = rng.normal(0, 0.5)
            rows.append(
                {
                    "league_key": league,
                    "season": season,
                    "source_player_id": p + 1,
                    "tier": "match",
                    "minutes": 2000.0,
                    "rating": level[p] + effects[league] + 0.08 * gd + rng.normal(0, 0.03),
                    "team_gd_per_match": gd,
                    "age": 25.0,
                    "regular": True,
                    "sub_position": "Centre-Back",
                    "team_id": p % 20,
                }
            )
    return pd.DataFrame(rows), effects


def test_league_strength_recovers_the_true_effects():
    features, effects = _season_rows(np.random.default_rng(0))
    strength, team_effect = league_strength(features)
    by = strength.set_index("league_key")["raw_effect"]
    for league, effect in effects.items():
        assert by[league] == pytest.approx(effect, abs=0.02)
    assert team_effect == pytest.approx(0.08, abs=0.02)


def test_ability_removes_league_and_team_and_shrinks_small_samples():
    features, _ = _season_rows(np.random.default_rng(1), n_players=60)
    strength, team_effect = league_strength(features)
    out = ability(features, strength, team_effect)
    # The same player rates the same after adjustment wherever he plays.
    spread = out.groupby(features["source_player_id"])["adj_rating"].std()
    assert spread.median() < 0.05
    assert out["ability_pct"].between(0, 100).all()
    few = features.assign(minutes=90.0)
    shrunk = ability(few, strength, team_effect)["adj_rating"]
    assert shrunk.std() < out["adj_rating"].std() / 3  # 90 minutes say little


def _elo_rows(rng, n_players=400, seasons=range(2015, 2021)):
    """Ratings fall 0.05 per 100 Elo of league strength and rise 0.03 per 100
    Elo of club strength; each league's Elo also drifts from season to
    season, which the ratings of players who stay do not follow."""
    level = {"epl": 0.0, "super_lig": -2.5, "eredivisie": -3.0}  # Elo / 100 against the EPL
    leagues = list(level)
    base = rng.normal(7.0, 0.2, n_players)
    drift = {(lg, s): rng.normal(0, 0.8) if lg != "epl" else 0.0 for lg in leagues for s in seasons}
    rows, terms = [], []
    for season in seasons:
        for p in range(n_players):
            league = leagues[(p + (season if p % 3 == 0 else 0)) % 3]
            team = rng.normal(0, 1.5)
            rows.append(
                {
                    "league_key": league,
                    "season": season,
                    "source_player_id": p + 1,
                    "tier": "match",
                    "minutes": 2000.0,
                    "rating": base[p] - 0.05 * level[league] + 0.03 * team + rng.normal(0, 0.03),
                    "age": 25.0,
                    "regular": True,
                    "sub_position": "Centre-Back",
                    "team_id": p % 20,
                }
            )
            terms.append({"elo_league": level[league] + drift[(league, season)], "elo_team": team})
    return pd.DataFrame(rows), pd.DataFrame(terms)


def test_elo_league_strength_uses_league_movers():
    features, terms = _elo_rows(np.random.default_rng(2))
    table, league_coef, team_coef = elo_league_strength(features, terms)
    assert league_coef == pytest.approx(-0.05, abs=0.01)
    assert team_coef == pytest.approx(0.03, abs=0.005)
    # Counting stayers too: their league's Elo drifts, their rating does not.
    _, attenuated, _ = elo_league_strength(features, terms, movers_only=False)
    assert abs(attenuated) < abs(league_coef) * 0.8
    assert set(table.columns) >= {"league_key", "season", "effect"}
    out = ability_elo(features, terms, league_coef, team_coef)
    assert out["ability_pct"].between(0, 100).all()


def test_elo_terms_are_against_the_premier_league_and_the_league_average():
    features = pd.DataFrame(
        {
            "league_key": ["epl", "super_lig", "super_lig", "ligue_1"],
            "season": pd.array([2020, 2020, 2020, 2020], dtype="Int64"),
            "team_id": pd.array([1, 2, 9, 3], dtype="Int64"),
        }
    )
    teams = pd.DataFrame(
        {
            "league_key": ["epl", "super_lig"],
            "season": [2020, 2020],
            "team_id": [1, 2],
            "elo": [1900.0, 1600.0],
        }
    )
    leagues = pd.DataFrame(
        {"league_key": ["epl", "super_lig"], "season": [2020, 2020], "elo": [1800.0, 1500.0]}
    )
    out = elo_terms(features, teams, leagues)
    assert out["elo_league"].tolist()[:3] == [0.0, -3.0, -3.0]
    assert out["elo_team"].tolist()[:3] == [1.0, 1.0, 0.0]  # club 9 unrated: league average
    assert out.iloc[3].isna().all()  # league without Elo


def test_team_shares_sum_to_one_per_team():
    frame = pd.DataFrame(
        {
            "league_key": ["epl"] * 3,
            "season": [2020] * 3,
            "team_id": [1, 1, 2],
            "goals": [3.0, 1.0, 0.0],
            "assists": [1.0, 1.0, 0.0],
            "shots_total": [10.0, 30.0, 5.0],
            "passes_key": [5.0, 5.0, 1.0],
            "dribbles_success": [4.0, 0.0, 2.0],
        }
    )
    out = team_shares(frame)
    assert out.loc[:1, "share_goals"].tolist() == [0.75, 0.25]
    assert np.isnan(out.loc[2, "share_goals"])  # the team scored nothing
    assert out.loc[2, "share_shots"] == 1


def test_rank_of_counts_the_values_above():
    table = {2020: np.array([100.0, 50.0, 50.0, 10.0])}
    ranks = _rank_of(pd.Series([60.0, 50.0, 5.0, np.nan]), pd.Series([2020] * 4), table)
    assert ranks.tolist()[:3] == [2, 2, 5]
    assert np.isnan(ranks.iloc[3])


def _target_inputs():
    """One season of 1,200 valued players, then two later seasons."""
    rows = []
    for season in (2018, 2019, 2020):
        for p in range(1, 1201):
            rows.append(
                {
                    "league_key": "epl",
                    "season": season,
                    "source_player_id": p,
                    "tm_player_id": p,
                    "feature_date": pd.Timestamp(f"{season + 1}-06-30"),
                    "regular": True,
                    "minutes": 2000.0,
                    "market_value": float(10_000 - p),  # rank = player id
                }
            )
    features = pd.DataFrame(rows)
    vals = features[["tm_player_id", "feature_date", "market_value"]].rename(
        columns={
            "tm_player_id": "source_player_id",
            "feature_date": "valuation_date",
            "market_value": "market_value_eur",
        }
    )
    return features, vals


def test_elite_and_step_up_labels():
    features, vals = _target_inputs()
    # Player 300 jumps to the top in 2019, player 1100 climbs to #200 in 2020.
    for pid, season, value in ((300, 2019, 1e9), (1100, 2020, 9_850.0)):
        vals.loc[
            (vals["source_player_id"] == pid)
            & (vals["valuation_date"] == pd.Timestamp(f"{season + 1}-06-30")),
            "market_value_eur",
        ] = value
    extra = pd.DataFrame({"ability_pct": np.nan}, index=features.index)
    out = targets(features, extra, vals, data_end=pd.Timestamp("2021-06-30"))
    first = out[out["season"] == 2018].set_index("source_player_id")
    assert first.loc[300, "elite"] == 1
    assert first.loc[300, "step_up"] == 1  # #300 to #1
    assert first.loc[500, "elite"] == 0
    assert np.isnan(first.loc[1100, "elite"])  # too far away for elite
    assert first.loc[1100, "step_up"] == 1  # #1100 to #151 or so
    assert np.isnan(first.loc[50, "elite"])  # already in the top 100
    assert out.loc[out["season"] == 2019, "elite"].isna().all()  # window not over


def test_performance_needs_a_rise_that_holds():
    features, vals = _target_inputs()
    ids = features["source_player_id"]
    season = features["season"]
    pct = pd.Series(50.0, index=features.index)
    # Player 1 rises and stays; player 2 has one good season; player 3 is good already.
    pct[(ids == 1) & (season > 2018)] = 85.0
    pct[(ids == 2) & (season == 2019)] = 95.0
    pct[ids == 3] = 90.0
    out = targets(features, pd.DataFrame({"ability_pct": pct}), vals, pd.Timestamp("2021-06-30"))
    first = out[out["season"] == 2018].set_index("source_player_id")
    assert first.loc[1, "performance"] == 1
    assert first.loc[2, "performance"] == 0
    assert first.loc[3, "performance"] == 0


def test_trait_table_compares_with_peers_of_the_same_age():
    rng = np.random.default_rng(2)
    n = 400
    data = pd.DataFrame(
        {
            "group": "winger",
            "age_band": rng.choice(["under 21", "24 to 26"], n),
            "step_up": (np.arange(n) < 60).astype(float),
        }
    )
    data["pct_share_dribbles"] = np.where(data["step_up"] == 1, 80.0, 50.0)
    data["pct_rating"] = np.where(data["step_up"] == 1, 90.0, 50.0)
    with_rating = trait_table(data, "step_up")
    without = trait_table(data, "step_up", with_rating=False)
    assert with_rating.iloc[0]["measure"] == "rating"
    assert without.iloc[0]["measure"] == "share_dribbles"
    assert without.iloc[0]["gap"] == 30


def test_composite_inputs_read_a_missing_count_as_zero_only_where_recorded():
    features = pd.DataFrame(
        {
            "league_key": ["epl", "epl", "mls", "epl"],
            "season": [2020, 2020, 2020, 2020],
            "tier": ["match", "match", "match", "season"],
            "minutes": [900.0, 900.0, 900.0, 900.0],
            **{c: np.nan for c in COMPOSITE_CANDIDATES},
        }
    )
    features.loc[0, "goals_per90"] = 0.5
    features.loc[0, "duel_win_rate"] = 0.6
    out = composite_inputs(features)
    assert out.loc[1, "goals_per90"] == 0  # the API leaves zero goals empty
    assert np.isnan(out.loc[1, "duel_win_rate"])  # a rate without attempts is unknown
    assert np.isnan(out.loc[2, "goals_per90"])  # not recorded in that league season
    assert out.loc[3].isna().all()  # season totals tier has no per 90 measures


def test_prior_minutes_recovers_noise_over_true_spread():
    rng = np.random.default_rng(3)
    n = 20_000
    minutes = pd.Series(rng.uniform(270, 3000, n))
    tau, sigma2 = 1.0, 600.0  # true spread 1, noise variance 600 / minutes
    score = pd.Series(rng.normal(0, tau, n) + rng.normal(0, np.sqrt(sigma2 / minutes)))
    k = prior_minutes(score, minutes, [pd.Series(2020, index=score.index)])
    assert k == pytest.approx(sigma2 / tau**2, rel=0.15)


def test_two_seasons_pools_the_previous_season_by_minutes():
    features = pd.DataFrame(
        {
            "source_player_id": [1, 1, 1, 2],
            "season": [2019, 2020, 2020, 2020],
            "minutes": [1000.0, 500.0, 500.0, 900.0],
        }
    )
    score = pd.Series([1.0, 4.0, 2.0, 3.0])
    pooled, used = two_seasons(features, score)
    assert pooled.tolist() == [1.0, 2.0, pytest.approx(4 / 3), 3.0]
    assert used.tolist() == [1000.0, 1500.0, 1500.0, 900.0]


def test_ability_v2_removes_the_league_effect_from_the_composite():
    """Per 90 numbers look better in weaker leagues; movers are spread over
    the leagues each season (in lockstep, standardising each season would
    absorb part of the league effect)."""
    rng = np.random.default_rng(4)
    shift = {"epl": 0.0, "super_lig": 0.3, "eredivisie": 0.5}
    leagues = list(shift)
    level = rng.normal(0, 1, 2000)
    rows = []
    for season in range(2015, 2021):
        for p in range(2000):
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
                    "sub_position": "Centre-Back",
                    # sampling noise falls with minutes, so the prior is positive
                    "goals_per90": level[p] + shift[league] + rng.normal(0, np.sqrt(600 / minutes)),
                }
            )
    features = pd.DataFrame(rows)
    for c in COMPOSITE_CANDIDATES:
        if c != "goals_per90":
            features[c] = np.nan
    terms = pd.DataFrame({"elo_league": 0.0, "elo_team": 0.0}, index=features.index)
    weights = pd.DataFrame(
        {"position": ["centre_back"], "measure": ["goals_per90"], "weight": [1.0]}
    )
    out, strength, _, prior = ability_v2(features, terms, weights)
    by = strength.set_index("league_key")["raw_effect"]
    sd = features.groupby("season")["goals_per90"].std().mean()
    assert by["eredivisie"] == pytest.approx(0.5 / sd, abs=0.05)  # in standardised units
    assert by["super_lig"] == pytest.approx(0.3 / sd, abs=0.05)
    assert out["ability_pct"].between(0, 100).all()
    assert prior["centre_back"] > 0
    movers = features["source_player_id"] % 3 == 1
    spread = out[movers].groupby(features["source_player_id"])["ability_pct"].std()
    assert spread.median() < 10  # a mover ranks alike wherever he plays


def test_rank_rate_baseline_uses_value_rank_and_age():
    train = pd.DataFrame(
        {
            "value_rank": [120.0] * 40 + [900.0] * 40,
            "age": [22.0] * 80,
            "y": [1.0] * 20 + [0.0] * 20 + [0.0] * 40,
        }
    )
    test = pd.DataFrame({"value_rank": [130.0, 950.0], "age": [22.0, 22.0], "y": [0.0, 0.0]})
    assert rank_rate(train, test, []).tolist() == [0.5, 0.0]


def test_cli_strength_choices_match_the_module():
    from scout.cli import SCOUTING_STRENGTHS

    assert SCOUTING_STRENGTHS == STRENGTH_SOURCES
    assert DEFAULT_STRENGTH in STRENGTH_SOURCES
