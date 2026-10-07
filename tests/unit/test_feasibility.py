import pandas as pd

from scout.feasibility import (
    STAT_FIELDS,
    completeness_by_season,
    conflicting_matches,
    estimate_requests,
    feature_date,
    field_completeness,
    first_usable_season,
    label_coverage,
    summarize_value_vs_fee,
    valuation_horizon_coverage,
    value_vs_fee,
)


def test_feature_date_is_season_end():
    assert str(feature_date(2022).date()) == "2023-06-30"


def test_field_completeness_by_position():
    stats = pd.DataFrame(
        {
            "league_key": ["epl"] * 3,
            "position": ["Defender", "Defender", "Attacker"],
            "minutes": [900.0, 0.0, 400.0],
        }
    )
    for col in STAT_FIELDS:
        if col != "minutes":
            stats[col] = None
    stats["tackles_total"] = [5.0, 1.0, None]
    table = field_completeness(stats).set_index("position")
    assert table.loc["Defender", "rows"] == 1  # zero minute row excluded
    assert table.loc["Defender", "tackles_total"] == 1.0
    assert table.loc["Attacker", "tackles_total"] == 0.0


def test_valuation_horizon_coverage_respects_data_end():
    apps = pd.DataFrame(
        {
            "league_key": ["epl", "epl", "epl"],
            "season": [2021, 2021, 2024],
            "source_player_id": [1, 2, 1],
        }
    )
    vals = pd.DataFrame(
        {
            "source_player_id": [1, 1, 1, 2],
            "valuation_date": pd.to_datetime(
                ["2022-05-01", "2024-06-01", "2025-06-01", "2020-01-01"]
            ),
        }
    )
    end = pd.Timestamp("2026-06-12")  # last valuation in the transfermarkt snapshot
    table = valuation_horizon_coverage(apps, vals, data_end=end).set_index("season")
    assert table.loc[2021, "players"] == 2
    assert table.loc[2021, "value_at_feature_date"] == 1  # player 2's value is too old
    assert table.loc[2021, "value_at_+24m"] == 1
    assert table.loc[2021, "value_at_+36m"] == 1  # 2025-06-30 target, value from 2025-06-01
    assert pd.isna(table.loc[2024, "value_at_+24m"])  # 2027 lies past the data


def test_estimate_requests_uses_mean_for_unmeasured_seasons():
    table = estimate_requests(
        pages_per_season={("epl", 2023): 40, ("epl", 2022): 44},
        seasons_per_league={"epl": [2020, 2021, 2022, 2023]},
        fixtures_per_season={"epl": 380},
    ).iloc[0]
    assert table["season_stats_requests"] == 40 + 44 + 42 + 42
    assert table["match_stats_requests"] == 4 * 380 + 4


def _stats(rows):
    cols = ["league_key", "season", "source_player_id", "minutes", "rating"]
    return pd.DataFrame(rows, columns=cols)


def test_completeness_by_season_counts_regular_players_only():
    stats = _stats(
        [
            ("epl", 2014, 1, 900, None),
            ("epl", 2014, 2, 900, 7.0),
            ("epl", 2014, 3, 100, None),  # below 450 minutes, ignored
            ("epl", 2015, 1, 900, 6.8),
        ]
    )
    table = completeness_by_season(stats, fields=["minutes", "rating"])
    assert table["players"].tolist() == [2, 1]
    assert table["rating"].tolist() == [0.5, 1.0]


def test_first_usable_season_needs_every_later_season():
    table = pd.DataFrame(
        {
            "league_key": ["epl"] * 4,
            "season": [2014, 2015, 2016, 2017],
            "players": [1] * 4,
            "rating": [0.95, 0.2, 0.95, 0.97],  # 2015 breaks the run
            "shots_total": [0.0, 0.0, 0.0, 0.5],  # never usable
        }
    )
    row = first_usable_season(table).iloc[0]
    assert row["rating"] == 2016
    assert row["shots_total"] is None


def test_label_coverage_counts_each_player_once():
    stats = _stats([("epl", 2022, 1, 900, 7.0), ("epl", 2022, 2, 900, 7.0)])
    matches = pd.DataFrame(
        {
            "af_player_id": [1, 1, 2],  # player 1 matched in both leagues
            "tm_player_id": [11, 11, pd.NA],
            "reason": ["matched", "matched", "low_score"],
        }
    )
    vals = pd.DataFrame(
        {
            "source_player_id": [11],
            "valuation_date": [pd.Timestamp("2023-03-01")],
            "market_value_eur": [5e6],
        }
    )
    row = label_coverage(stats, matches, vals).iloc[0]
    assert (row["regular_players"], row["matched"], row["valued_at_feature_date"]) == (2, 1, 1)
    assert row["label_rate"] == 0.5


def test_conflicting_matches_flags_two_targets_for_one_player():
    matches = pd.DataFrame(
        {"af_player_id": [1, 1, 2, 2], "tm_player_id": [11, 12, 21, 21], "reason": ["matched"] * 4}
    )
    assert conflicting_matches(matches)["af_player_id"].unique().tolist() == [1]


def test_value_vs_fee_uses_the_last_value_before_the_transfer():
    transfers = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 4],
            "transfer_date": pd.to_datetime(
                ["2020-08-01", "2020-08-01", "2020-08-01", "2027-01-01"]
            ),
            "from_club_id": [100, 200, 100, 100],
            "to_club_id": [200, 100, 300, 200],
            "transfer_fee_eur": [20e6, 0.0, 4e6, 9e6],  # 0 is a free move; 2027 is past the end
        }
    )
    vals = pd.DataFrame(
        {
            "source_player_id": [1, 1, 1, 3],
            "valuation_date": pd.to_datetime(
                ["2019-12-01", "2020-06-01", "2020-09-01", "2018-01-01"]
            ),
            "market_value_eur": [5e6, 10e6, 30e6, 4e6],  # player 3's value is too old
        }
    )
    clubs = pd.DataFrame({"club_id": [200, 300], "league_key": ["epl", "super_lig"]})
    pairs = value_vs_fee(transfers, vals, clubs, data_end=pd.Timestamp("2026-06-12"))
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["league_key"], row["direction"]) == ("epl", "in")
    assert row["market_value_eur"] == 10e6  # not the later 30m
    assert row["fee_to_value"] == 2.0


def test_summarize_value_vs_fee():
    pairs = pd.DataFrame(
        {
            "league_key": ["epl"] * 4,
            "transfer_date": pd.to_datetime(
                ["2015-07-01", "2016-07-01", "2017-07-01", "2018-07-01"]
            ),
            "market_value_eur": [1e6, 2e6, 3e6, 4e6],
            "transfer_fee_eur": [1e6, 3e6, 5e6, 20e6],
        }
    )
    pairs["fee_to_value"] = pairs["transfer_fee_eur"] / pairs["market_value_eur"]
    row = summarize_value_vs_fee(pairs).iloc[0]
    assert row["spearman"] == 1.0
    assert row["within_2x"] == 0.75
    assert row["first_transfer"] == "2015-07-01"
