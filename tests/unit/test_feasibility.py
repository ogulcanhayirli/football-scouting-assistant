import pandas as pd

from scout.feasibility import (
    STAT_FIELDS,
    estimate_requests,
    feature_date,
    field_completeness,
    valuation_horizon_coverage,
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
