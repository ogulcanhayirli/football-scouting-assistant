"""The players table (prompt 20f): every player of the season, with why he
is on no list."""

import json

import numpy as np
import pandas as pd
import pytest

from scout.chat.data import parse_players
from scout.contracts.schemas import PLAYERS_TABLE, validate
from scout.lists import dictionary
from scout.players import COLUMNS, DICTIONARY, compact, players_table

KEYS = ["league_key", "season", "source_player_id"]
LEAGUES = {"epl": "Premier League", "serie_a": "Serie A"}


def _row(pid, name, **kw):
    minutes = kw.pop("minutes", 2000.0)
    row = {
        "league_key": kw.pop("league", "epl"),
        "season": 2025,
        "source_player_id": pid,
        "player_name": name,
        "team_name": kw.pop("club", "Club A"),
        "group": kw.pop("group", "centre_forward"),
        "age": kw.pop("age", 24.0),
        "feature_date": pd.Timestamp("2026-06-30"),
        "value_date": pd.Timestamp("2026-06-01"),
        "regular": minutes >= 450,
        "minutes": minutes,
        "team_id": 1,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "market_value": kw.pop("value", 10e6),
        "value_rank": kw.pop("rank", 500.0),
        "minutes_share": kw.pop("share", 0.8),
        "injury_matches_missed_12m": kw.pop("injured", 0.0),
        "value_change_1y": kw.pop("change", 1.1),
        "adj_rating": kw.pop("score", 0.1),
        "ability_pct": kw.pop("ability", 80.0),
    }
    assert not kw, kw
    return row


def _inputs():
    rows = [
        _row(1, "Star", rank=2.0, value=2e8, age=26.0, ability=97.0),
        _row(2, "Rodri", rank=98.0, value=5e7, age=30.0, share=0.44, ability=99.0),
        _row(3, "Young", rank=305.0, age=20.0),
        _row(4, "Bench", minutes=200.0, rank=np.nan, value=np.nan, ability=np.nan, score=np.nan),
        _row(5, "Veteran", rank=5000.0, value=5e5, age=35.0, ability=50.0),
        # one player in two leagues: the row with most minutes describes him
        _row(6, "Mover", league="epl", minutes=500.0, rank=700.0),
        _row(6, "Mover", league="serie_a", minutes=1500.0, rank=700.0, club="Club B"),
        _row(7, None, minutes=30.0, rank=np.nan, value=np.nan, ability=np.nan, score=np.nan),
    ]
    scouting = pd.DataFrame(rows)
    for c in ("ability_pct", "adj_rating"):  # non regulars carry no ability
        scouting[c] = scouting[c].where(scouting["regular"])
    features = scouting[KEYS].assign(nationality="Spain")
    regular = scouting[scouting["regular"] & scouting["market_value"].notna()]
    fair = regular[KEYS].assign(pred_lightgbm=np.log(regular["market_value"] / 1.25))
    cand = pd.DataFrame(
        {
            "player_id": pd.array([3, 6], dtype="Int64"),
            "elite_decile": pd.array([10, pd.NA], dtype="Int64"),
            "elite_chance_pct": pd.array([30, pd.NA], dtype="Int64"),
            "elite_chance_players": pd.array([142, pd.NA], dtype="Int64"),
            "big_move_decile": pd.array([7, pd.NA], dtype="Int64"),
            "big_move_chance_pct": pd.array([10, pd.NA], dtype="Int64"),
            "big_move_chance_players": pd.array([390, pd.NA], dtype="Int64"),
        }
    )
    barg = pd.DataFrame({"player_id": pd.array([1, 3], dtype="Int64")})
    reports = {"available": True, "rows": {"3": ["report", "Text."], "1": ["template", "Rule."]}}
    return scouting, features, fair, cand, barg, reports


@pytest.fixture
def table():
    return players_table(*_inputs()[:5], LEAGUES, _inputs()[5])


def _by_name(table):
    return table.set_index("player")


def test_one_row_per_player_with_a_minute_and_nameless_rows_left_out(table):
    assert list(table.columns) == COLUMNS
    assert sorted(table["player_id"]) == [1, 2, 3, 4, 5, 6]
    mover = _by_name(table).loc["Mover"]
    assert (mover["league"], mover["club"], mover["minutes"]) == ("Serie A", "Club B", 1500)
    # the most valuable first, players without a value last
    assert table["player"].tolist()[:2] == ["Star", "Rodri"]
    assert table["player"].iloc[-1] == "Bench"
    validate(PLAYERS_TABLE, table)


def test_the_reasons_name_every_rule_the_player_fails(table):
    t = _by_name(table)
    assert pd.isna(t.loc["Young", "why_not_candidate"])
    assert pd.isna(t.loc["Young", "why_not_bargain"])
    assert t.loc["Star", "why_not_candidate"] == (
        "not a candidate: already among the 100 most valuable"
    )
    assert pd.isna(t.loc["Star", "why_not_bargain"])  # on the bargains list
    assert t.loc["Rodri", "why_not_bargain"] == (
        "not a bargain: under half of his team's league minutes"
    )
    assert t.loc["Bench", "why_not_candidate"] == (
        "not a candidate: fewer than 450 league minutes this season, no value on record and "
        "not one of the riser model's top 50"
    )
    assert t.loc["Bench", "why_not_bargain"] == (
        "not a bargain: fewer than 450 league minutes this season and no value on record"
    )
    assert t.loc["Veteran", "why_not_candidate"] == (
        "not a candidate: valued below #2,000 and not one of the riser model's top 50"
    )
    assert t.loc["Veteran", "why_not_bargain"] == (
        "not a bargain: ability below the 70th percentile of his position and 34 or older"
    )


def test_blank_chances_say_why_and_given_ones_are_copied(table):
    t = _by_name(table)
    assert t.loc["Young", "elite_chance_pct"] == 30
    assert t.loc["Young", "big_move_chance_players"] == 390
    assert pd.isna(t.loc["Young", "elite_chance_why_blank"])
    assert t.loc["Rodri", "elite_chance_why_blank"] == (
        "older than 23 and already among the 100 most valuable"
    )
    assert t.loc["Veteran", "big_move_chance_why_blank"] == "valued below #2,000"
    # a candidate ranked #101 to #2,000 without a big move chance is at a top 20 club
    assert t.loc["Mover", "big_move_chance_why_blank"] == "at a top 20 club"
    assert t.loc["Mover", "elite_chance_why_blank"] == "older than 23"


def test_ability_and_fair_value_for_regulars_only(table):
    t = _by_name(table)
    assert pd.isna(t.loc["Bench", "ability_pct"])
    assert pd.isna(t.loc["Bench", "fair_value_eur"])
    assert t.loc["Star", "value_to_fair"] == 1.25
    assert t.loc["Star", "nationality"] == "Spain"


def test_report_available_follows_the_pages_reports(table):
    t = _by_name(table)
    assert bool(t.loc["Young", "report_available"])
    assert not bool(t.loc["Star", "report_available"])  # the template is no report


def test_the_contract_rejects_a_list_player_without_a_reason(table):
    broken = table.copy()
    broken.loc[broken["player"] == "Rodri", "why_not_candidate"] = None
    with pytest.raises(Exception, match="why_not_candidate must explain not in_candidates"):
        validate(PLAYERS_TABLE, broken)
    leak = table.copy()
    leak.loc[0, "value_date"] = pd.Timestamp("2026-07-02")  # after the feature date
    with pytest.raises(Exception, match="value_date after data_date"):
        validate(PLAYERS_TABLE, leak)


def test_a_list_player_missing_from_the_season_is_an_error():
    scouting, features, fair, cand, barg, reports = _inputs()
    barg = pd.DataFrame({"player_id": pd.array([1, 99], dtype="Int64")})
    with pytest.raises(ValueError, match="list players have no row"):
        players_table(scouting, features, fair, cand, barg, LEAGUES, reports)


def test_compact_json_round_trips_through_the_chats_reader(table):
    d = json.loads(json.dumps(compact(table), allow_nan=False))
    assert d["season"] == 2025
    assert "season" not in d["columns"]
    rows = parse_players(d)
    assert len(rows) == len(table)
    rodri = next(o for o in rows if o["player"] == "Rodri")
    assert rodri["club"] == "Club A"
    assert rodri["why_not_bargain"] == "not a bargain: under half of his team's league minutes"
    assert rodri["value_date"] == "2026-06-01"
    assert isinstance(rodri["minutes"], int)
    bench = next(o for o in rows if o["player"] == "Bench")
    assert bench["value_eur"] is None
    assert bench["ability_pct"] is None


def test_every_column_has_a_sentence_in_the_dictionary():
    words = dictionary()
    assert set(words.loc[words["table"] == "players", "column"]) == set(DICTIONARY) == set(COLUMNS)
