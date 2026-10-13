import numpy as np
import pandas as pd
import pytest

from scout.models.moves import (
    auc,
    average_precision,
    move_labels,
    precision_at,
    run_move_backtest,
    top_clubs,
)


def feature_rows():
    """Season 2022 (feature date 30 June 2023): club 1 is rich, clubs 2 and 3 are not."""
    return pd.DataFrame(
        {
            "league_key": ["epl"] * 5,
            "season": pd.array([2022] * 5, dtype="Int64"),
            "source_player_id": pd.array([10, 11, 12, 13, 14], dtype="Int64"),
            "feature_date": pd.to_datetime(["2023-06-30"] * 5),
            "team_id": pd.array([1, 2, 2, 3, 2], dtype="Int64"),
            "team_name": ["Rich", "Mid", "Mid", "Small", "Mid"],
            "minutes": [3000.0, 2500.0, 2000.0, 1000.0, 900.0],
            "market_value": [90e6, 10e6, 5e6, 1e6, 3e6],
        }
    )


def transfer(player, date, out_id, in_id, kind="transfer", in_name="Rich"):
    return {
        "source_player_id": player,
        "transfer_date": pd.Timestamp(date),
        "team_out_id": out_id,
        "team_in_id": in_id,
        "team_in_name": in_name,
        "kind": kind,
    }


def test_top_clubs_rank_squads_by_value():
    top = top_clubs(feature_rows(), n=1)
    assert top["team_id"].tolist() == [1]


def test_labels_count_new_moves_to_top_clubs_inside_the_window_only():
    features = feature_rows()
    transfers = pd.DataFrame(
        [
            transfer(11, "2024-01-15", 2, 1),  # a real jump: 1
            transfer(12, "2025-08-01", 2, 1),  # after the 24 month window: 0
            transfer(13, "2022-07-01", 1, 3, kind="loan", in_name="Small"),  # loaned out by Rich
            transfer(13, "2023-07-01", 3, 1, kind="unknown"),  # back to the owner: not a jump
            transfer(14, "2023-03-01", 2, 1),  # joined Rich before the date: already top
        ]
    )
    top = top_clubs(features, n=1)
    labels = move_labels(features, transfers, top, data_end=pd.Timestamp("2026-10-11"))
    by = labels.set_index("source_player_id")
    assert np.isnan(by.loc[10, "big_move_24m"])  # already at a top club
    assert by.loc[11, "big_move_24m"] == 1
    assert by.loc[11, "move_to"] == "Rich"
    assert by.loc[12, "big_move_24m"] == 0
    assert by.loc[13, "big_move_24m"] == 0
    assert np.isnan(by.loc[14, "big_move_24m"])
    assert by.loc[14, "at_top_club"]
    assert by.loc[11, "move_label_date"] == pd.Timestamp("2025-06-30")


def test_labels_are_missing_when_the_window_ends_after_the_data():
    features = feature_rows()
    labels = move_labels(
        features,
        pd.DataFrame([transfer(11, "2024-01-15", 2, 1)]),
        top_clubs(features, n=1),
        data_end=pd.Timestamp("2025-01-01"),
    )
    assert labels["big_move_24m"].isna().all()


def test_ranking_metrics():
    y = pd.Series([1, 0, 1, 0, 0])
    perfect = pd.Series([0.9, 0.1, 0.8, 0.2, 0.3])
    assert auc(y, perfect) == 1
    assert auc(y, -perfect) == 0
    assert average_precision(y, perfect) == 1
    assert precision_at(y, perfect, 2) == 1
    assert precision_at(y, pd.Series([0.1, 0.9, 0.2, 0.8, 0.3]), 2) == 0
    assert average_precision(y, pd.Series([0.1, 0.9, 0.8, 0.2, 0.3])) == pytest.approx(
        (1 / 2 + 2 / 5) / 2
    )


def test_move_backtest_purges_rows_whose_window_is_still_open():
    rng = np.random.default_rng(0)
    frames = []
    for season in range(2010, 2018):
        n = 50
        frames.append(
            pd.DataFrame(
                {
                    "league_key": "epl",
                    "season": season,
                    "source_player_id": np.arange(n) + season * 1000,
                    "player_name": "P",
                    "team_name": "T",
                    "age": rng.uniform(18, 30, n),
                    "feature_date": pd.Timestamp(f"{season + 1}-06-30"),
                    "y": (rng.uniform(size=n) < 0.2).astype(float),
                    "move_date": pd.NaT,
                    "move_to": None,
                }
            )
        )
    data = pd.concat(frames, ignore_index=True)
    data["move_label_date"] = data["feature_date"] + pd.DateOffset(months=24)

    def mean_rate(train, test, columns):
        return np.full(len(test), train["y"].mean())

    out = run_move_backtest(data, ["age"], {"rate": mean_rate})
    # Test 2014 (June 2015): seasons up to 2012 (windows close by June 2015).
    assert out.loc[out["season"] == 2014, "train_rows"].iloc[0] == 3 * 50
