import warnings

import numpy as np
import pandas as pd
import pytest

from scout.models.backtest import (
    LeakError,
    check_no_leak,
    expanding_folds,
    fold_season,
    metrics,
    purge,
    run_backtest,
    summarize,
)
from scout.models.estimators import group_median, make_lightgbm
from scout.models.targets import TARGETS, VALUE_COLUMNS, Target, build_dataset

FAIR = TARGETS["fair_value"]
# A label observed two seasons later, like the big move and scouting targets.
TWO_YEARS = Target("two_years", "log_growth_24m", 2, "value_24m_date", True)


def test_folds_leave_a_gap_as_long_as_the_label_horizon():
    seasons = list(range(2010, 2020))
    fair = {f.test_season: f.train_seasons for f in expanding_folds(seasons, FAIR)}
    growth = {f.test_season: f.train_seasons for f in expanding_folds(seasons, TWO_YEARS)}
    assert fair[2013] == [2010, 2011, 2012]  # first fold: three seasons of history
    assert min(fair) == 2013
    assert growth[2015] == [2010, 2011, 2012, 2013]  # 2014's labels are not known yet
    assert all(max(train) <= test - 2 for test, train in growth.items())


def _rows(season, feature_date, label_date=None, n=3):
    return pd.DataFrame(
        {
            "season": season,
            "feature_date": pd.Timestamp(feature_date),
            "value_24m_date": pd.Timestamp(label_date) if label_date else pd.NaT,
        },
        index=range(n),
    )


def test_purge_drops_labels_observed_after_the_test_date_and_the_check_agrees():
    # Season 2019 ran to July 2020; its 24 month labels can land after 30 June 2022.
    train = pd.concat(
        [_rows(2018, "2019-06-30", "2021-06-01"), _rows(2019, "2020-07-27", "2022-07-20")],
        ignore_index=True,
    )
    test = _rows(2021, "2022-06-30")
    with pytest.raises(LeakError, match="3 training labels"):
        check_no_leak(train, test, TWO_YEARS)
    kept = purge(train, test, TWO_YEARS)
    assert kept["season"].tolist() == [2018] * 3
    check_no_leak(kept, test, TWO_YEARS)


def test_leak_check_rejects_training_rows_from_the_test_season():
    with pytest.raises(LeakError, match="feature date"):
        check_no_leak(_rows(2021, "2022-06-30"), _rows(2021, "2022-06-30"), FAIR)


def _dataset(seed=0, seasons=range(2010, 2020), per_season=120):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        n = per_season
        skill = rng.normal(size=n)
        age = rng.uniform(18, 34, size=n)
        rows.append(
            pd.DataFrame(
                {
                    "league_key": pd.Categorical(rng.choice(["epl", "super_lig"], n)),
                    "season": season,
                    "source_player_id": np.arange(n) + season * 1000,
                    "player_name": "P",
                    "tier": pd.Categorical(["match"] * n),
                    "position": pd.Categorical(rng.choice(["Defender", "Attacker"], n)),
                    "feature_date": pd.Timestamp(f"{season + 1}-06-30"),
                    "age": age,
                    "skill": skill,
                    "y": 15 + 1.5 * skill - 0.05 * (age - 25) ** 2 + rng.normal(0, 0.2, n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_backtest_predicts_every_later_season_from_the_past_only():
    data = _dataset()
    models = {"group_median": group_median, "lightgbm": make_lightgbm(rounds=150)}
    predictions = run_backtest(data, ["league_key", "position", "age", "skill"], FAIR, models)
    assert sorted(predictions["season"].unique()) == list(range(2013, 2020))
    assert predictions.groupby("season")["train_rows"].first().is_monotonic_increasing
    overall = summarize(predictions).set_index("model")
    # The signal is in "skill", which only the tree model sees.
    assert overall.loc["lightgbm", "mae_log"] < overall.loc["group_median", "mae_log"]
    assert overall.loc["lightgbm", "spearman"] > 0.9


def test_folds_follow_the_feature_date_so_january_rows_sit_with_the_season_before():
    dates = pd.Series(
        pd.to_datetime(["2024-06-30", "2020-07-27", "2025-01-01", "2024-12-10", "2025-06-30"])
    )
    assert fold_season(dates).tolist() == [2023, 2019, 2023, 2023, 2024]


def test_two_year_backtest_with_a_calendar_year_league_purges_only_its_own_rows():
    data = _dataset(seasons=range(2010, 2020))
    data["value_24m_date"] = data["feature_date"] + pd.DateOffset(months=24)
    # A calendar year league: its season 2016 ends in December, feature date January 2017.
    brazil = _dataset(seed=1, seasons=range(2010, 2020), per_season=30)
    brazil["league_key"] = pd.Categorical(["brasileirao"] * len(brazil))
    brazil["feature_date"] = pd.to_datetime((brazil["season"] + 1).astype(str) + "-01-01")
    brazil["value_24m_date"] = brazil["feature_date"] + pd.DateOffset(months=24)
    both = pd.concat([data, brazil], ignore_index=True)
    both["league_key"] = both["league_key"].astype("category")
    predictions = run_backtest(both, ["age", "skill"], TWO_YEARS, {"group_median": group_median})
    european = predictions[predictions["league_key"] != "brasileirao"]
    assert sorted(european["season"].unique()) == list(range(2013, 2020))
    # Test 2016 (dated June 2017): all European seasons up to 2014 (5 x 120) and
    # the calendar seasons up to 2015 (6 x 30), less 2015, whose labels (January
    # 2018) come after June 2017. With folds by season number the European 2014
    # rows (labels June 2017) would have been purged against the January rows.
    assert european.loc[european["season"] == 2016, "train_rows"].iloc[0] == 600 + 150


def test_group_median_falls_back_to_coarser_groups():
    train = pd.DataFrame(
        {
            "league_key": ["epl"] * 20 + ["super_lig"] * 3,
            "position": ["Defender"] * 20 + ["Attacker"] * 3,
            "age": [24.0] * 23,
            "y": [10.0] * 20 + [99.0] * 3,
        }
    )
    test = pd.DataFrame(
        {
            "league_key": ["epl", "super_lig"],
            "position": ["Defender", "Attacker"],
            "age": [24.0, 24.0],
        }
    )
    # Super Lig has only 3 training rows: too few for any group, so the overall median.
    assert group_median(train, test, []).tolist() == [10.0, 10.0]


def test_build_dataset_keeps_value_features_out_of_fair_value():
    features = pd.DataFrame(
        {
            "league_key": ["epl", "epl", "epl"],
            "season": pd.array([2020, 2020, 2020], dtype="Int64"),
            "source_player_id": pd.array([1, 2, 3], dtype="Int64"),
            "regular": [True, True, False],
            "feature_date": pd.to_datetime(["2021-06-30"] * 3),
            "player_name": ["a", "b", "c"],
            "minutes": [900.0, 1000.0, 100.0],
            **{c: [1.0, 2.0, 3.0] for c in VALUE_COLUMNS},
        }
    )
    labels = features[["league_key", "season", "source_player_id"]].assign(
        value_now=[1e6, np.nan, 3e6],
        log_growth_24m=[0.5, 0.1, 0.2],
        value_24m_date=pd.Timestamp("2023-06-01"),
    )
    fair, fair_cols = build_dataset(features, labels, FAIR)
    assert fair["source_player_id"].tolist() == [1]  # 2 has no value, 3 is not regular
    assert fair.loc[0, "y"] == pytest.approx(np.log(1e6))
    assert not set(VALUE_COLUMNS) & set(fair_cols)
    assert "player_name" not in fair_cols
    two, two_cols = build_dataset(features, labels, TWO_YEARS)
    assert two["source_player_id"].tolist() == [1, 2]
    assert set(VALUE_COLUMNS) <= set(two_cols)  # a later label may use the value now


def test_metrics_are_relative_and_constant_predictions_have_no_rank():
    y = pd.Series([0.0, 1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = metrics(y, pd.Series(np.zeros(len(y))))
    assert np.isnan(m["spearman"])
    assert m["mae_log"] == pytest.approx(1.5)
    exact = metrics(y, y + np.log(1.1))
    assert exact["median_ape"] == pytest.approx(0.1)
    assert exact["spearman"] == pytest.approx(1.0)
