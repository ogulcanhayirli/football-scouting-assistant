import numpy as np
import pandas as pd
import pytest

from scout.lists import (
    bargains_table,
    base_columns,
    candidates_table,
    chance_tables,
    dictionary,
    rising_ids,
)

KEYS = ["league_key", "season", "source_player_id"]


def _scouting(rng, seasons=range(2015, 2026), n=2400):
    """n players per season, value rank = player id, ages 18 to 33."""
    rows = []
    for season in seasons:
        ids = np.arange(1, n + 1)
        rows.append(
            pd.DataFrame(
                {
                    "league_key": "epl",
                    "season": season,
                    "source_player_id": ids,
                    "player_name": [f"p{i}" for i in ids],
                    "team_name": "club",
                    "sub_position": "Centre-Forward",
                    "position": "Attack",
                    "regular": True,
                    "minutes": 2000.0,
                    "age": 18.0 + ids % 16,
                    "value_rank": ids.astype(float),
                    "market_value": 1e8 / ids,
                    "minutes_share": rng.uniform(0.2, 1.0, n),
                    "injury_matches_missed_12m": rng.integers(0, 10, n).astype(float),
                    "value_change_1y": rng.uniform(0.5, 1.5, n),
                    "ability_pct": rng.uniform(0, 100, n),
                    "ability_pct_1s": rng.uniform(0, 100, n),
                    "fall_back": rng.integers(0, 2, n).astype(float),
                    "score_elite": rng.random(n),
                    "score_step_up": rng.random(n),
                    "score_riser": rng.random(n),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    pool = out["value_rank"].between(101, 1000) & (out["age"] < 24)
    out["fall_back"] = out["fall_back"].where(pool & (out["season"] < 2024))
    return out


def _backtest(scouting, score="score"):
    past = scouting[scouting["season"] < 2024]
    return pd.DataFrame(
        {
            **{k: past[k] for k in KEYS},
            "age": past["age"],
            "value_rank": past["value_rank"],
            "y": (past["score_elite"] > 0.8).astype(float),
            score: past["score_elite"],
            "score_baseline": past["score_step_up"],
        }
    )


def _inputs(seed=5):
    rng = np.random.default_rng(seed)
    s = _scouting(rng)
    newest = s[s["season"] == 2025]
    return {
        "scouting": s,
        "elite": _backtest(s),
        "step_up": _backtest(s),
        "riser": _backtest(s),
        "big_move": _backtest(s, "score_lightgbm"),
        "big_move_now": newest[KEYS].assign(score=rng.random(len(newest))),
        "fair_value": newest[KEYS].assign(pred_lightgbm=np.log(2e6)),
    }


def test_candidates_cover_ranks_101_to_2000_with_chances_only_where_measured():
    inputs = _inputs()
    out = candidates_table(inputs, chance_tables(inputs), {"epl": "Premier League"})
    assert list(out.columns) == base_columns("candidates")
    assert out["player_id"].is_unique
    ranked = out["value_rank"].between(101, 2000)
    assert ranked.sum() == 1900
    assert out.loc[~ranked, "rising"].all()  # only risers valued below #2,000 are added
    out = out[ranked]
    pool = (out["value_rank"] <= 1000) & (out["age"] < 24)
    assert out.loc[pool, "elite_chance_pct"].notna().all()
    assert out.loc[~pool, "elite_chance_pct"].isna().all()  # blank outside the elite pool
    assert out.loc[~pool, "fall_back_risk_pct"].isna().all()
    assert out.loc[pool, "fall_back_risk_players"].ge(30).all()
    assert out["step_up_chance_pct"].notna().all()
    assert out["big_move_chance_pct"].notna().all()
    assert (out["league"] == "Premier League").all()
    assert (out["fair_value_eur"] == 2e6).all()
    # sorted by elite decile, the pool first; here the chances rise with it
    assert out["elite_chance_pct"].iloc[: int(pool.sum())].notna().all()
    assert out["elite_decile"].dropna().is_monotonic_decreasing
    shown = out["elite_chance_pct"].dropna()
    assert shown.is_monotonic_decreasing
    assert out["rising"].sum() <= 50


def test_the_order_follows_the_decile_when_a_shown_rate_dips():
    inputs = _inputs()
    tables = chance_tables(inputs)
    before = candidates_table(inputs, tables, {"epl": "Premier League"})
    rates = tables["elite"].table
    rates.loc[(rates["season"] == "all") & (rates["bucket"] == "1"), "rate"] = 0.99
    after = candidates_table(inputs, tables, {"epl": "Premier League"})
    assert after["player_id"].tolist() == before["player_id"].tolist()


def test_rising_is_the_riser_models_top_50_among_half_of_team_minutes():
    s = _inputs()["scouting"]
    ids = rising_ids(s)
    newest = s[s["season"] == 2025].set_index("source_player_id")
    assert len(ids) == 50
    assert (newest.loc[list(ids), "minutes_share"] >= 0.5).all()
    cut = newest.loc[list(ids), "score_riser"].min()
    pool = newest[newest["minutes_share"] >= 0.5]
    assert (pool["score_riser"] > cut).sum() == 49


def test_bargains_table_and_dictionary_agree():
    frame = pd.DataFrame(
        {
            "league_key": "epl",
            "season": 2025,
            "source_player_id": [1, 2],
            "list_rank": [1, 2],
            "player_name": ["a", "b"],
            "age": [25.0, 31.0],
            "sub_position": [None, "Left-Back"],
            "position": ["Defender", "Defender"],
            "team_name": "club",
            "ability_pct": [80.0, 90.0],
            "minutes_share": [0.6, 0.7],
            "value": [1e6, 2e6],
            "predicted": [2e6, 3e6],
            "ratio": [0.5, 0.67],
            "ratio_pct": [1.0, 2.0],
            "contract_expiry": pd.to_datetime(["2028-06-30", None]),
            "flag_count": [0, 1],
            "flags": ["none", "value lower than a year before (value -20% on a year before)"],
            "explanation": ["No stat stands out.", "Plays 70% of team minutes, top 5% of x."],
            "on_loan": [False, True],
            "stale_value": [False, False],
            "value_date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
        }
    )
    out = bargains_table({"prime": frame.iloc[[0]], "experienced": frame.iloc[[1]]}, {})
    assert list(out.columns) == base_columns("bargains")
    assert out["below_fair_value"].tolist() == [True, True]  # prompt 20g
    assert out["list"].tolist() == ["prime", "experienced"]
    assert out["position"].tolist() == ["Defender", "Left-Back"]
    words = dictionary()
    assert set(words["table"]) == {"candidates", "bargains", "players"}
    assert set(dictionary(present=False)["table"]) == {"candidates", "bargains"}
    assert words["meaning"].str.len().gt(20).all()  # one plain sentence each
    assert not words["meaning"].str.contains("\n").any()


def test_risers_valued_below_2000_are_added_with_the_chances_blank():
    inputs = _inputs()
    s = inputs["scouting"]
    newest = s["season"] == 2025
    # make the top riser scores belong to players ranked below #2,000
    low = newest & (s["value_rank"] > 2000) & (s["minutes_share"] >= 0.5)
    s.loc[low, "score_riser"] = 2.0 + s.loc[low, "score_riser"]
    out = candidates_table(inputs, chance_tables(inputs), {})
    added = out[out["value_rank"] > 2000]
    assert len(added) == min(50, int(low.sum()))
    assert added["rising"].all()
    for column in (
        "elite_chance_pct",
        "step_up_chance_pct",
        "big_move_chance_pct",
        "fall_back_risk_pct",
        "step_up_decile",
        "big_move_decile",
    ):
        assert added[column].isna().all(), column
    assert added["flags"].notna().all()
    assert added["explanation"].str.len().gt(0).all()
    # the deciles of the ranked rows ignore the added ones
    ranked = out[out["value_rank"] <= 2000]
    assert ranked["step_up_decile"].value_counts().min() >= 189
    assert out.index[out["value_rank"] > 2000].min() > out.index[out["value_rank"] <= 2000].max()


def test_a_bargain_is_priced_below_fair_value_and_the_contract_holds_it():
    from pandera.errors import SchemaErrors

    from scout.contracts.schemas import BARGAINS_TABLE, validate

    ratio = pd.Series([0.4, 0.9995, 1.0, 1.36])
    frame = pd.DataFrame(
        {
            "season": 2025,
            "list": "prime",
            "list_rank": [1, 2, 3, 4],
            "player_id": pd.array([1, 2, 3, 4], dtype="Int64"),
            "player": ["a", "b", "c", "Kylian Mbappé"],
            "age": [22.0, 23.0, 24.0, 27.0],
            "value_eur": [1e6, 2e6, 3e6, 180e6],
            "fair_value_eur": [2.5e6, 2e6, 3e6, 132e6],
            "value_to_fair": ratio.round(3),
            "value_to_fair_pct": [1.0, 20.0, 30.0, 48.8],
            "below_fair_value": ratio.round(3) < 1,
        }
    )
    # 0.9995 shows as 1.0 (100% of fair): at fair value is not below it.
    assert frame["below_fair_value"].tolist() == [True, False, False, False]
    validate(BARGAINS_TABLE, frame)
    with pytest.raises(SchemaErrors):
        validate(BARGAINS_TABLE, frame.assign(below_fair_value=True))
    with pytest.raises(SchemaErrors):
        validate(BARGAINS_TABLE, frame.assign(list_rank=1))
