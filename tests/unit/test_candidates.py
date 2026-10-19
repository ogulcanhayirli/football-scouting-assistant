import numpy as np
import pandas as pd
import pytest

from scout.candidates import (
    MIN_BUCKET,
    ability_decision,
    chance_table,
    count_bucket,
    flag_effects,
    merge_low,
    rate_table,
    rounded,
    score_deciles,
    show_rate,
)
from scout.flags import ABILITY_MIN_RATIO, FLAGS, describe, warning_flags
from scout.scouting import elite_pool, fall_back_label


def test_fall_back_needs_a_rank_twice_as_bad_and_a_young_candidate():
    value_rank = pd.Series([300.0, 300.0, 50.0, 1500.0, 300.0, 300.0, 300.0])
    end_rank = pd.Series([600.0, 599.0, 500.0, 5000.0, np.nan, 900.0, 900.0])
    reachable = pd.Series([True] * 7)
    age = pd.Series([22.0, 22.0, 22.0, 22.0, 22.0, 23.99, 24.0])
    out = fall_back_label(value_rank, end_rank, reachable, age)
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0
    assert np.isnan(out.iloc[2])  # already in the top 100
    assert np.isnan(out.iloc[3])  # not an elite candidate
    assert np.isnan(out.iloc[4])  # no value at the end
    assert out.iloc[5] == 1  # still 23 at the feature date
    assert np.isnan(out.iloc[6])  # 24: outside the elite pool
    assert fall_back_label(value_rank, end_rank, ~reachable, age).isna().all()


def test_elite_pool_is_rank_101_to_1000_and_23_or_under():
    rank = pd.Series([100.0, 101.0, 1000.0, 1001.0, 500.0, 500.0])
    age = pd.Series([20.0, 20.0, 23.5, 20.0, 24.0, np.nan])
    assert elite_pool(rank, age).tolist() == [False, True, True, False, False, False]


def test_warning_flags_read_the_thresholds_and_ignore_missing():
    data = pd.DataFrame(
        {
            "minutes_share": [0.46, 0.47, np.nan],
            "injury_matches_missed_12m": [5.0, 4.0, np.nan],
            "value_change_1y": [0.99, 1.0, np.nan],
        }
    )
    flags = warning_flags(data)
    assert list(flags.columns) == ["low_minutes", "injured", "value_falling"]
    assert flags.iloc[0].all()
    assert not flags.iloc[1].any()
    assert not flags.iloc[2].any()  # unknown raises no flag
    text = describe(data)
    assert text["flag_count"].tolist() == [3, 0, 0]
    assert "46% of team minutes" in text.loc[0, "flags"]
    assert text.loc[2, "flags"] == "none"


def test_the_ability_flag_follows_the_owners_rule():
    y = pd.Series([1.0, 0.0] * 50 + [1.0, 0.0, 0.0] * 50)
    ability = pd.Series([40.0] * 100 + [80.0] * 150)  # 50% with the flag, 33% without
    ratio, passes = ability_decision(y, ability)
    assert ratio == pytest.approx(1.5)
    assert passes == (ratio >= ABILITY_MIN_RATIO)
    assert "low_ability" not in FLAGS  # 1.26 on the elite pool: left out of the count


def test_buckets_round_to_five_and_hide_small_groups():
    assert count_bucket(pd.Series([0, 1, 2, 3, 4])).tolist() == ["0", "1", "2", "3+", "3+"]
    assert rounded(0.374) == 35
    assert rounded(0.376) == 40
    assert rounded(0.02) == 0
    assert show_rate(0.02, 100) == "under 5% (n=100)"
    assert show_rate(0.0, 100) == "0% (n=100)"
    assert show_rate(0.5, MIN_BUCKET - 1).startswith("fewer than")
    y = pd.Series([1.0] * 10 + [0.0] * 30 + [1.0] * 5)
    bucket = pd.Series(["0"] * 40 + ["1"] * 5)
    season = pd.Series([2020] * 20 + [2021] * 25)
    table = rate_table(y, bucket, season)
    pooled = table[table["season"] == "all"].set_index("bucket")
    assert pooled.loc["0", "players"] == 40
    assert pooled.loc["0", "pct"] == 25
    assert np.isnan(pooled.loc["1", "rate"])  # 5 players
    assert table[table["season"] != "all"]["rate"].isna().all()  # no season has 30 in a bucket


def test_flag_effects_compare_with_and_without():
    y = pd.Series([1, 1, 0, 0, 0, 1, 0, 0], dtype=float)
    flags = pd.DataFrame({"low_minutes": [True, True, True, True, False, False, False, False]})
    row = flag_effects(y, flags).iloc[0]
    assert row["rate_with"] == 0.5
    assert row["rate_without"] == 0.25
    assert row["ratio"] == 2.0


def test_score_deciles_are_within_season():
    score = pd.Series(list(range(20)) + list(range(100, 120)), dtype=float)
    season = pd.Series([2020] * 20 + [2021] * 20)
    d = score_deciles(score, season)
    assert d.tolist() == [d for d in range(1, 11) for _ in range(2)] * 2


def test_low_deciles_merge_until_each_bucket_holds_the_floor():
    full = pd.Series(40, index=range(1, 11))
    assert set(merge_low(full).values()) == {str(d) for d in range(1, 11)}
    thin = pd.Series([10, 10, 15, 40, 40, 40, 40, 40, 40, 40], index=range(1, 11))
    mapping = merge_low(thin)
    assert mapping[1] == mapping[2] == mapping[3] == "1 to 3"
    assert mapping[4] == "4"
    short_top = pd.Series([40] * 9 + [5], index=range(1, 11))
    assert merge_low(short_top)[10] == merge_low(short_top)[9] == "9 to 10"


def test_chance_table_merges_small_deciles_and_looks_up_rates():
    season = pd.Series([2020] * 200)
    score = pd.Series(np.arange(200), dtype=float)
    y = (score >= 180).astype(float)  # the top decile always makes it
    chance = chance_table(pd.DataFrame({"season": season, "score": score, "y": y}))
    assert chance.mapping[1] == "1 to 2"  # 20 players per decile
    assert chance.pooled.loc["9 to 10", "rate"] == 0.5
    found = chance.lookup(pd.Series([1, 10]))
    assert found["players"].tolist() == [40, 40]
    assert found["pct"].tolist() == [0, 50]
