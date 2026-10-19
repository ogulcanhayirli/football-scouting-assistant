import json

import numpy as np
import pandas as pd

from scout.bargains import backtest, current_lists, eligible, outcomes, warning_flags
from scout.providers.transfermarkt import TransfermarktValuationProvider


def _rows(n=40, season=2020, group="centre_back", start_id=1, age=25.0, ratio=None):
    """Eligible players by default: regular, ability 80, two thirds of minutes."""
    ids = np.arange(start_id, start_id + n)
    ratio = np.linspace(0.2, 2.0, n) if ratio is None else ratio
    return pd.DataFrame(
        {
            "league_key": "epl",
            "season": season,
            "source_player_id": ids,
            "player_name": [f"p{i}" for i in ids],
            "team_id": 1,
            "feature_date": pd.Timestamp(f"{season + 1}-06-30"),
            "value_date": pd.Timestamp(f"{season + 1}-06-01"),
            "regular": True,
            "minutes": 2000.0,
            "minutes_share": 0.66,
            "age": age,
            "group": group,
            "ability_pct": 80.0,
            "value": 1e6 * ratio,
            "predicted": 1e6,
            "prev_minutes": 2000.0,
            "injury_matches_missed_12m": 0.0,
            "value_change_1y": 1.0,
            "form_minutes_share": 0.7,
            "on_loan": 0.0,
        }
    )


def test_eligible_filters_and_ranks_the_ratio_within_position_group():
    backs = _rows()
    keepers = _rows(group="goalkeeper", start_id=100, ratio=np.linspace(0.05, 0.5, 40))
    data = pd.concat([backs, keepers], ignore_index=True)
    data.loc[0, "ability_pct"] = 69.0  # below the ability floor
    data.loc[1, "minutes_share"] = 0.4  # below half of team minutes
    data.loc[2, "age"] = 31.0
    data.loc[3, "age"] = 34.0  # too old for either list
    data.loc[4, "age"] = 33.9  # 33 is still listed
    rows = eligible(data)
    assert not rows["source_player_id"].isin([1, 2, 4]).any()
    # keepers are cheaper in absolute terms, yet each group has its own 0 to 100 scale
    cheapest = rows.sort_values("ratio_pct").groupby("group").head(1)
    assert set(cheapest["group"]) == {"centre_back", "goalkeeper"}
    assert rows.groupby("group")["ratio_pct"].max().round(6).eq(100).all()
    assert rows.loc[rows["source_player_id"].isin([3, 5]), "list"].eq("experienced").all()
    assert (rows.loc[~rows["source_player_id"].isin([3, 5]), "age"] < 30).all()


def test_eligible_keeps_one_row_per_player_and_season():
    data = _rows(n=3)
    extra = data.iloc[[0]].assign(league_key="championship", minutes=500.0)
    rows = eligible(pd.concat([data, extra], ignore_index=True))
    assert len(rows) == 3
    assert rows.loc[rows["source_player_id"] == 1, "league_key"].item() == "epl"


def test_warning_flags_are_the_shared_core_flags_plus_two_tags():
    data = _rows(n=2)
    data.loc[0, ["injury_matches_missed_12m", "value_change_1y", "on_loan"]] = [12.0, 0.5, 1.0]
    data.loc[0, "value_date"] = pd.Timestamp("2020-06-01")
    flags = warning_flags(data)
    assert flags["flag_count"].tolist() == [2, 0]  # tags are never counted
    text = flags.loc[0, "flags"]
    assert "12 matches missed injured" in text
    assert "value -50% on a year before" in text
    assert flags.loc[1, "flags"] == "none"
    assert flags["on_loan"].tolist() == [True, False]
    assert flags["stale_value"].tolist() == [True, False]
    assert flags.loc[0, "stale_days"] == 394


def test_outcomes_find_a_move_to_a_stronger_club_and_value_growth():
    features = pd.DataFrame(
        {
            "league_key": ["epl"] * 5,
            "season": [2018, 2019, 2018, 2019, 2020],
            "source_player_id": [1, 1, 2, 2, 2],
            "team_id": [10, 20, 10, 10, 30],
        }
    )
    teams = pd.DataFrame(
        {
            "league_key": "epl",
            "season": [2018, 2019, 2020],
            "team_id": [10, 20, 30],
            "elo": [1600.0, 1700.0, 1500.0],
        }
    )
    teams = pd.concat(
        [
            teams,
            pd.DataFrame({"league_key": "epl", "season": [2019], "team_id": [10], "elo": [1600.0]}),
        ]
    )
    labels = features[["league_key", "season", "source_player_id"]].assign(
        log_growth_24m=[0.3, np.nan, -0.1, np.nan, np.nan]
    )
    out = outcomes(features, labels, teams).set_index(["source_player_id", "season"])
    assert out.loc[(1, 2018), "stronger_club"] == 1.0  # 1600 to 1700, another club
    assert out.loc[(2, 2018), "stronger_club"] == 0.0  # same club, then a weaker one
    assert np.isnan(out.loc[(1, 2019), "stronger_club"])  # 2021 not in the data
    assert out.loc[(1, 2018), "gained_value"] == 1.0
    assert out.loc[(2, 2018), "gained_value"] == 0.0
    assert np.isnan(out.loc[(1, 2019), "gained_value"])


def test_backtest_compares_the_top_with_players_of_the_same_bands():
    data = _rows(n=60)
    data["gained_value"] = (data["value"] < 0.8e6).astype(float)  # cheap ones grow
    data["stronger_club"] = 0.0
    rows = eligible(data)
    result = backtest(rows, k=10).set_index(["list", "outcome", "season"])
    row = result.loc[("prime", "gained_value", "2020")]
    assert row["players"] == 10
    assert row["rate"] == 1.0
    # 50 others, of which the cheapest 7 or so grew
    others = rows.sort_values("ratio_pct").iloc[10:]
    assert row["matched_rate"] == others["gained_value"].mean()
    assert row["z"] > 2
    assert ("prime", "gained_value", "pooled") in result.index


def test_contract_expiry_is_shown_only_after_the_snapshot_date():
    rows = eligible(_rows(n=3))
    profiles = pd.DataFrame(
        {
            "af_player_id": [1, 2, 3],
            "contract_expiry": pd.to_datetime(["2026-06-30", "2028-06-30", None]),
        }
    )
    lists = current_lists(rows, profiles, pd.Timestamp("2026-09-05"))
    shown = lists["prime"].set_index("source_player_id")["contract_expiry"]
    assert pd.isna(shown[1])
    assert shown[2] == pd.Timestamp("2028-06-30")
    assert pd.isna(shown[3])
    assert lists["prime"]["list_rank"].tolist() == [1, 2, 3]
    assert lists["experienced"].empty


def test_snapshot_date_comes_from_the_players_sidecar(tmp_path):
    provider = TransfermarktValuationProvider(tmp_path)
    assert provider.snapshot_date() is None
    meta = {"headers": {"Last-Modified": "Sat, 05 Sep 2026 09:12:22 GMT"}}
    (tmp_path / "players.csv.gz.meta.json").write_text(json.dumps(meta))
    assert provider.snapshot_date() == pd.Timestamp("2026-09-05")
