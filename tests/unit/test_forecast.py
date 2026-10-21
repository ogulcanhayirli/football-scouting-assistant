import json

import numpy as np
import pandas as pd
import pytest

from scout.forecast import freeze, read_folder, score, write_folder

KEYS = ["league_key", "season", "source_player_id"]


def _scouting():
    """Season 2020 to 2022. Forty filler clubs, club 99 the richest; the five
    forecast players play for small clubs."""
    rows = []
    for season in (2020, 2021, 2022):
        for team in range(1, 41):
            rows.append(
                {
                    "source_player_id": 1000 + team,
                    "team_id": 99 if team == 1 else team,
                    "market_value": 1e9 if team == 1 else 1e7 + team,
                    "value_rank": 5000.0,
                    "season": season,
                }
            )
    later = {  # player: (rank 2020, rank 2021, rank 2022)
        1: (150, 120, 80),  # elite
        2: (900, 250, 400),  # step up
        3: (200, 300, 500),  # falls back
        4: (300, None, None),  # no later rows: unknown
        5: (1500, 1400, 1300),  # joins club 99
    }
    for pid, ranks in later.items():
        for season, rank in zip((2020, 2021, 2022), ranks, strict=True):
            if rank is None:
                continue
            rows.append(
                {
                    "source_player_id": pid,
                    "team_id": 500 + pid,
                    "market_value": 1e8 / rank,
                    "value_rank": float(rank),
                    "season": season,
                }
            )
    out = pd.DataFrame(rows)
    out["league_key"] = "epl"
    out["team_name"] = "club " + out["team_id"].astype(str)
    out["minutes"] = 2000.0
    out["feature_date"] = pd.to_datetime(out["season"].astype(str) + "-06-30") + pd.DateOffset(
        years=1
    )
    out["value_date"] = out["feature_date"] - pd.Timedelta(days=20)
    out["regular"] = True
    return out


def _transfers():
    return pd.DataFrame(
        {
            "source_player_id": [5],
            "transfer_date": pd.to_datetime(["2022-01-15"]),
            "team_out_id": [505.0],
            "team_in_id": [99.0],
            "team_in_name": ["club 99"],
            "kind": ["transfer"],
        }
    )


def _forecast():
    cand = pd.DataFrame(
        {
            "season": 2020,
            "player_id": [1, 2, 3, 4, 5],
            "value_rank": [150, 900, 200, 300, 1500],
            "elite_chance_pct": [40, None, 40, 0, None],
            "fall_back_risk_pct": [15, None, 15, 15, None],
            "step_up_chance_pct": [5, 20, 5, 5, 0],
            "big_move_chance_pct": [10, 10, 10, 10, 25],
        }
    )
    barg = pd.DataFrame(
        {
            "season": 2020,
            "list": "prime",
            "list_rank": [1, 2, 60],
            "player_id": [1, 3, 2],
            "value_eur": [1e8 / 150, 1e8 / 200, 1e8 / 900],
        }
    )
    meta = {"season": 2020, "created": "2021-07-01", "feature_dates": {"2021-06-30": ["epl"]}}
    return meta, cand, barg


def test_a_written_folder_reads_back_and_scores_each_promise(tmp_path):
    meta, cand, barg = _forecast()
    folder = write_folder(tmp_path / "f", cand, barg, pd.DataFrame({"column": ["x"]}), meta)
    meta, cand, barg = read_folder(folder)
    result = score(meta, cand, barg, _scouting(), _transfers(), pd.Timestamp("2023-06-30"))
    assert result["seen"] == 2
    out = result["candidates"].set_index("player_id")
    assert out.loc[1, "elite"] == 1.0
    assert out.loc[2, "step_up"] == 1.0
    assert out.loc[3, "fall_back"] == 1.0
    assert out.loc[1, "fall_back"] == 0.0
    assert np.isnan(out.loc[4, "elite"])  # no later row: unknown, never a miss
    assert out.loc[5, "big_move"] == 1.0
    assert out.loc[1, "big_move"] == 0.0
    elite = result["tables"]["elite"].set_index("bucket")
    assert elite.loc["40%", "players"] == 2
    assert elite.loc["40%", "actual"] == 0.5
    assert elite.loc["under 5%", "known"] == 0
    assert elite.loc["all (base rate)", "actual"] == 0.5
    move = result["tables"]["big_move"].set_index("bucket")
    assert move.loc["25%", "actual"] == 1.0
    # a player with no later row still has a known move outcome: no transfer, no move
    assert move.loc["all (base rate)", "actual"] == pytest.approx(1 / 5)
    bargains = result["tables"]["bargains"]
    bargains = bargains[bargains["list"] == "prime"].set_index("rows")
    assert bargains.loc["top 50", "value_rose"] == 0.5  # player 1 rose, player 3 fell
    assert bargains.loc["all (base rate)", "players"] == 3


def test_an_early_score_is_interim_and_uses_nothing_after_the_as_of_date():
    meta, cand, barg = _forecast()
    one = score(meta, cand, barg, _scouting(), _transfers(), pd.Timestamp("2022-06-30"))
    assert one["seen"] == 1
    out = one["candidates"].set_index("player_id")
    assert out.loc[1, "elite"] == 0.0  # #120 after one season
    assert out.loc[2, "step_up"] == 1.0  # #250 already three times better
    assert out.loc[5, "big_move"] == 1.0  # January 2022
    none = score(meta, cand, barg, _scouting(), _transfers(), pd.Timestamp("2021-12-31"))
    assert none["seen"] == 0
    assert none["candidates"][["elite", "step_up", "fall_back"]].isna().all().all()
    assert none["candidates"]["big_move"].fillna(0).eq(0).all()  # the move came later


def test_freeze_copies_the_lists_and_never_overwrites(tmp_path):
    lists = tmp_path / "lists"
    lists.mkdir()
    _, cand, barg = _forecast()
    cand.assign(season=2020).to_csv(lists / "candidates.csv", index=False)
    barg.to_csv(lists / "bargains.csv", index=False)
    pd.DataFrame({"table": ["candidates"], "column": ["x"], "meaning": ["y"]}).to_csv(
        lists / "dictionary.csv", index=False
    )
    folder = freeze(lists, _scouting(), {"epl": "Premier League"}, tmp_path / "2021-07-01")
    assert (folder / "candidates.csv").read_bytes() == (lists / "candidates.csv").read_bytes()
    meta = json.loads((folder / "forecast.json").read_text())
    assert meta["season"] == 2020
    assert meta["feature_dates"] == {"2021-06-30": ["Premier League"]}
    assert meta["valuation_end"] == "2026-06-12"
    text = (folder / "README.md").read_text()
    assert "2021-06-30 for Premier League" in text
    assert "elite_chance_pct" in text
    with pytest.raises(FileExistsError):
        freeze(lists, _scouting(), {}, folder)
