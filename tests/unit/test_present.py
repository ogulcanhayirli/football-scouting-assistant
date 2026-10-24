"""Prompt 16: snapshot cache keys, squads and current club, the season so far
block, the present columns of the two tables and the window forecast."""

import json
import logging

import httpx
import numpy as np
import pandas as pd
import pandera.errors
import pytest
import respx

from scout.backfill.current import resolve_current_club
from scout.contracts import CURRENT_CLUB, PLAYER_MATCH_STATS, SQUADS, validate
from scout.features.build import COUNT_STATS
from scout.providers.api_football import ApiFootballClient
from scout.providers.api_football.current import parse_squad, squad_rows_to_frame
from scout.providers.api_football.matches import (
    match_rows_to_frame,
    parse_fixture_players,
    placeholder_blocks,
)
from scout.ratelimit import QuotaGuard
from scout.season_so_far import season_so_far, thin_leagues
from scout.storage import CacheKey, LocalRawStore, RawMeta
from scout.window import WINDOW_END, WINDOW_START, freeze_window, top_elo_clubs, window_hits

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-ratelimit-requests-remaining": "90", "x-ratelimit-remaining": "9"}
SQUAD = {
    "errors": [],
    "response": [
        {
            "team": {"id": 645, "name": "Galatasaray"},
            "players": [
                {"id": 1, "name": "A", "age": 25, "number": 4, "position": "Defender"},
                {"id": 2, "name": "B", "age": 2025, "number": 190, "position": "Coach"},
                {"id": 0, "name": "no id", "age": 20, "number": 9, "position": "Attacker"},
                {"id": 1, "name": "A", "age": 25, "number": 4, "position": "Defender"},
            ],
        }
    ],
}


@pytest.fixture
def client(tmp_path):
    guard = QuotaGuard(daily_limit=100, per_minute_limit=10, state_path=tmp_path / "q.json")
    c = ApiFootballClient(
        api_key="k",
        store=LocalRawStore(tmp_path / "raw"),
        guard=guard,
        base_url=BASE,
        sleep=lambda _: None,
    )
    yield c
    c.close()


# Snapshot cache keys


@respx.mock
def test_snapshot_is_cached_per_day_and_never_sent(client):
    route = respx.get(f"{BASE}/players/squads").mock(
        return_value=httpx.Response(200, json=SQUAD, headers=HEADERS)
    )
    first = client.get("players/squads", {"team": 645}, snapshot="2026-10-24")
    again = client.get("players/squads", {"team": 645}, snapshot="2026-10-24")
    later = client.get("players/squads", {"team": 645}, snapshot="2026-12-15")
    assert route.call_count == 2  # one per snapshot day
    assert again.from_cache
    assert not first.from_cache
    assert not later.from_cache
    assert all("as_of" not in str(call.request.url) for call in route.calls)
    assert str(first.key.relative_path()) == (
        "api_football/players/squads/as_of=2026-10-24/team=645.json"
    )
    meta = client.store.get_meta(first.key)
    assert meta.params == {"team": "645"}


def test_snapshot_parameter_is_reserved(client):
    with pytest.raises(ValueError, match="reserved"):
        client.get("transfers", {"team": 1, "as_of": "2026-10-24"})


def test_keys_skip_sub_endpoints(tmp_path):
    store = LocalRawStore(tmp_path)
    meta = RawMeta(url="u", params={}, fetched_at="t", status_code=200, headers={})
    page = CacheKey.of("api_football", "players", {"league": 39, "season": 2026, "page": 1})
    squad = CacheKey.of("api_football", "players/squads", {"team": 1, "as_of": "2026-10-24"})
    store.put(page, b"{}", meta)
    store.put(squad, b"{}", meta)
    assert store.keys("api_football", "players") == [page]
    assert store.keys("api_football", "players/squads") == [squad]


# Squads and the current club


def test_squad_source_errors_are_repaired_and_logged(caplog):
    with caplog.at_level(logging.WARNING):
        frame = squad_rows_to_frame(parse_squad(SQUAD, "2026-10-24"))
    assert frame["source_player_id"].tolist() == [1, 2]
    b = frame[frame["source_player_id"] == 2].iloc[0]
    assert pd.isna(b["age"])
    assert pd.isna(b["shirt_number"])
    assert pd.isna(b["squad_position"])
    assert "impossible" in caplog.text
    validate(SQUADS, frame)


def test_squads_contract_rejects_the_unrepaired_values():
    frame = squad_rows_to_frame(parse_squad(SQUAD, "2026-10-24"))
    frame.loc[frame["source_player_id"] == 2, "age"] = 2025
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(SQUADS, frame)


def _squads(rows):
    frame = pd.DataFrame(rows, columns=["team_id", "team_name", "source_player_id"])
    frame["player_name"] = "p" + frame["source_player_id"].astype(str)
    frame["squad_position"] = "Defender"
    frame["shirt_number"] = pd.array([5] * len(frame), dtype="Int64")
    frame["as_of"] = pd.Timestamp("2026-10-24")
    frame["source"] = "api_football"
    return frame.astype({"team_id": "Int64", "source_player_id": "Int64"})


def _transfers(rows):
    frame = pd.DataFrame(
        rows, columns=["source_player_id", "transfer_date", "team_in_id", "team_in_name", "kind"]
    )
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
    frame["player_name"] = "p" + frame["source_player_id"].astype(str)
    return frame.astype({"source_player_id": "Int64", "team_in_id": "Int64"})


LEAGUES = pd.DataFrame(
    {
        "team_id": pd.array([157, 645], dtype="Int64"),
        "team_name": ["Bayern München", "Galatasaray"],
        "league_key": ["bundesliga", "super_lig"],
        "league_name": ["Bundesliga", "Süper Lig"],
    }
)


def test_current_club_picks_the_latest_move_then_our_leagues():
    squads = _squads(
        [
            (157, "Bayern München", 2195),  # Boey: listed by both after his loan ended
            (645, "Galatasaray", 2195),
            (999, "Reserves", 7),  # a club outside our leagues
            (645, "Galatasaray", 7),
        ]
    )
    moves = _transfers(
        [
            (2195, "2026-02-04", 645, "Galatasaray", "loan"),
            (2195, "2026-06-29", 157, "Bayern München", "loan_return"),
            (15912, "2026-09-01", 398, "FC Nordsjaelland", "free"),  # in no squad we hold
            (55, "2026-05-01", 398, "FC Nordsjaelland", "transfer"),  # before the forecast
        ]
    )
    out = validate(CURRENT_CLUB, resolve_current_club(squads, moves, LEAGUES))
    by = out.set_index("source_player_id")
    assert by.loc[2195, "current_club"] == "Bayern München"
    assert by.loc[2195, "current_league"] == "Bundesliga"
    assert by.loc[2195, "squads_listing"] == 2
    assert by.loc[7, "current_club"] == "Galatasaray"
    assert by.loc[15912, "current_club"] == "FC Nordsjaelland"
    assert by.loc[15912, "club_source"] == "transfer"
    assert pd.isna(by.loc[15912, "current_league"])
    assert 55 not in by.index


# Match adapter repairs


def _fixture(home=1, away=2, fid=10):
    return {
        "fixture": {"id": fid, "date": "2026-09-01T18:00:00+00:00"},
        "league": {"season": 2026, "round": "1"},
        "teams": {"home": {"id": home, "name": "Home"}, "away": {"id": away, "name": "Away"}},
        "goals": {"home": 1, "away": 0},
    }


def _player(pid, minutes=90, passes=30, duels=8, goals=0):
    return {
        "player": {"id": pid, "name": f"p{pid}"},
        "statistics": [
            {
                "games": {"minutes": minutes, "rating": "7.0", "position": "M"},
                "passes": {"total": passes},
                "duels": {"total": duels, "won": min(duels, 3)},
                "goals": {"total": goals},
            }
        ],
    }


def test_players_under_an_unknown_team_id_are_given_the_other_side(epl, caplog):
    payload = {
        "response": [
            {"team": {"id": 1, "name": "Home"}, "players": [_player(11)]},
            {"team": {"id": 22722, "name": None}, "players": [_player(21)]},
        ]
    }
    with caplog.at_level(logging.WARNING):
        rows = parse_fixture_players(_fixture(), payload, epl)
    away = next(r for r in rows if r["source_player_id"] == 21)
    assert away["team_id"] == 2
    assert away["team_name"] == "Away"
    assert away["opponent_id"] == 1
    assert "unknown team 22722" in caplog.text


class _League:
    key = "ukrainian_premier_league"
    api_football_id = 333


def _rows(players, fid=10):
    payload = {"response": [{"team": {"id": 1, "name": "Home"}, "players": players}]}
    return parse_fixture_players(_fixture(fid=fid), payload, _League())


def test_placeholder_zeros_become_missing_and_the_contract_rejects_them(caplog):
    zeros = [_player(i, passes=0, duels=0, goals=1 if i == 1 else 0) for i in range(1, 12)]
    real = [_player(i) for i in range(101, 112)]
    raw = pd.DataFrame(_rows(zeros) + _rows(real, fid=11))
    raw["opponent_id"] = 2
    with caplog.at_level(logging.WARNING):
        frame = match_rows_to_frame(raw.to_dict("records"))
    zero = frame[frame["fixture_id"] == 10]
    assert zero["passes_total"].isna().all()
    assert zero["rating"].isna().all()
    assert zero["minutes"].eq(90).all()
    assert zero["goals"].sum() == 1  # the real counts stay
    assert frame[frame["fixture_id"] == 11]["passes_total"].eq(30).all()
    assert "placeholder zeros" in caplog.text
    assert not placeholder_blocks(frame).any()
    unrepaired = frame.copy()
    unrepaired.loc[unrepaired["fixture_id"] == 10, ["passes_total", "duels_total"]] = 0
    assert placeholder_blocks(unrepaired).sum() == 11
    with pytest.raises(pandera.errors.SchemaErrors, match="placeholder"):
        validate(PLAYER_MATCH_STATS, unrepaired)


# Season so far


def _season(league="epl", minutes=(90, 90, 90, 90, 20), passes=30.0, teams=(1, 2)):
    """Five matches between two clubs; player 1 of club 1 plays the given
    minutes and scores in the first match."""
    rows = []
    for n, m in enumerate(minutes):
        kick = pd.Timestamp("2026-08-20", tz="UTC") + pd.Timedelta(days=7 * n)
        for team, opp in (teams, teams[::-1]):
            for pid in range(team * 100, team * 100 + 11):
                mine = pid == 100
                rows.append(
                    {
                        "league_key": league,
                        "season": 2026,
                        "fixture_id": 1000 * team + n if team == teams[0] else 1000 * opp + n,
                        "kickoff": kick,
                        "team_id": team,
                        "team_name": f"club {team}",
                        "opponent_id": opp,
                        "team_goals": 1 if team == teams[0] else 0,
                        "opponent_goals": 0 if team == teams[0] else 1,
                        "source_player_id": pid,
                        "minutes": float(m if mine else 90),
                        "substitute": False,
                        "goals": 1.0 if mine and n == 0 else 0.0,
                        "assists": 0.0,
                        "passes_total": passes,
                        "rating": 7.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=[*frame.columns, *(c for c in COUNT_STATS if c not in frame)])
    for column in [*COUNT_STATS, "minutes", "rating"]:
        frame[column] = frame[column].astype("Float64")
    return frame


def test_season_so_far_minutes_share_rates_and_flags():
    matches = _season()
    positions = pd.DataFrame(
        {"source_player_id": [100, 101], "sub_position": ["Centre-Forward", "Centre-Back"]}
    )
    value = pd.DataFrame({"source_player_id": [100], "value_change_1y": [0.8]})
    injuries = pd.DataFrame(
        {
            "league_key": "epl",
            "fixture_id": range(5),
            "fixture_date": pd.date_range("2026-03-01", periods=5, freq="7D", tz="UTC"),
            "source_player_id": 100,
            "status": "Missing Fixture",
        }
    )
    out = season_so_far(matches, positions, injuries, value, pull_day=pd.Timestamp("2026-10-24"))
    me = out.set_index("source_player_id").loc[100]
    assert me["so_far_minutes"] == 380
    assert me["so_far_matches"] == 5
    assert me["so_far_minutes_share"] == round(380 / 450, 2)
    assert me["so_far_goals_per90"] == round(90 / 380, 2)
    assert pd.isna(me["so_far_ability_pct"])  # no stored fit given
    assert me["so_far_flag_count"] == 2  # injured (5 matches) and value falling
    assert "value lower" in me["so_far_flags"]
    assert "injured" in me["so_far_flags"]
    cb = out.set_index("source_player_id").loc[101]
    assert pd.isna(cb["so_far_goals_per90"])  # not shown for centre backs
    assert cb["so_far_flag_count"] == 0  # covered league, no injury, no value change known


def test_season_so_far_blanks_rates_under_270_minutes():
    out = season_so_far(
        _season(minutes=(90, 90, 0, 0, 0)),
        pd.DataFrame({"source_player_id": [100], "sub_position": ["Centre-Forward"]}),
        pd.DataFrame(),
        pd.DataFrame(columns=["source_player_id", "value_change_1y"]),
    )
    me = out.set_index("source_player_id").loc[100]
    assert me["so_far_minutes"] == 180
    assert pd.isna(me["so_far_goals_per90"])
    assert np.isnan(me["so_far_ability_pct"])


def test_thin_leagues_have_detailed_stats_in_under_half_the_matches():
    thin = _season("ukrainian_premier_league", passes=np.nan, teams=(3, 4))
    full = _season()
    assert thin_leagues(pd.concat([thin, full], ignore_index=True)) == ["ukrainian_premier_league"]


# The present columns of the tables


def test_present_frame_marks_moves_by_club_id():
    from scout.present import present_frame

    players = pd.DataFrame(
        {
            "player_id": pd.array([2195, 15912, 9], dtype="Int64"),
            "forecast_team_id": pd.array([645, 398, 10], dtype="Int64"),
        }
    )
    current = pd.DataFrame(
        {
            "source_player_id": pd.array([2195, 15912], dtype="Int64"),
            "current_club_id": pd.array([157, 398], dtype="Int64"),
            "current_club": ["Bayern München", "FC Nordsjaelland"],
            "current_league": ["Bundesliga", "Danish Superliga"],
        }
    )
    so_far = pd.DataFrame(
        {
            "source_player_id": pd.array([15912], dtype="Int64"),
            "so_far_date": [pd.Timestamp("2026-10-24")],
            "so_far_league_key": ["danish_superliga"],
            "so_far_matches": [5],
            "so_far_minutes": [450.0],
            "so_far_flag_count": [2],
        }
    )
    out = present_frame(players, current, so_far, {"danish_superliga": "Danish Superliga"})
    assert out["moved_since_forecast"].tolist() == [True, False, pd.NA]
    assert out["current_club"].tolist()[:2] == ["Bayern München", "FC Nordsjaelland"]
    assert out["so_far_minutes"].tolist() == [0.0, 450.0, 0.0]
    assert out["so_far_league"].tolist()[1] == "Danish Superliga"
    assert out["so_far_date"].notna().all()


def test_present_frame_without_files_keeps_the_shape():
    from scout.lists import PRESENT_COLUMNS
    from scout.present import present_frame

    players = pd.DataFrame({"player_id": pd.array([1], dtype="Int64"), "forecast_team_id": [5]})
    out = present_frame(players, None, None, {})
    assert list(out.columns) == PRESENT_COLUMNS
    assert out.isna().all(axis=None)


def test_dictionary_places_the_present_after_the_league():
    from scout.lists import DICTIONARY, PRESENT_COLUMNS, base_columns, dictionary

    for table in ("candidates", "bargains"):
        columns = list(DICTIONARY[table])
        at = columns.index("league")
        assert columns[at + 1 : at + 4] == [
            "current_club",
            "current_league",
            "moved_since_forecast",
        ]
        assert columns[-1] == "so_far_flags"
        assert "club" not in columns
        assert not set(base_columns(table)) & set(PRESENT_COLUMNS)
    assert not set(dictionary(present=False)["column"]) & set(PRESENT_COLUMNS)


# The window forecast


def test_top_elo_clubs_take_the_latest_rating_of_recent_league_clubs():
    elo = pd.DataFrame(
        {
            "kind": ["league", "league", "league", "cup"],
            "league_key": ["epl", "epl", "epl", "champions_league"],
            "kickoff": pd.to_datetime(
                ["2026-05-01", "2026-05-20", "2024-05-01", "2026-05-25"], utc=True
            ),
            "home_id": [1, 2, 3, 1],
            "home_name": ["A", "B", "C", "A"],
            "away_id": [2, 1, 4, 9],
            "away_name": ["B", "A", "D", "Z"],
            "home_post": [1700.0, 1650.0, 1900.0, 9999.0],
            "away_post": [1600.0, 1720.0, 1500.0, 1.0],
        }
    )
    out = top_elo_clubs(elo, {"epl"}, pd.Timestamp("2026-10-24"))
    assert out["team_id"].tolist() == [1, 2]  # clubs 3 and 4 have not played for a year
    assert out["elo"].tolist() == [1720.0, 1650.0]
    assert out["by_elo_rank"].tolist() == [1, 2]


def test_window_hits_count_only_new_moves_to_a_top_club_inside_the_window():
    pool = pd.DataFrame(
        {
            "player_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "club_id_at_freeze": pd.array([10, 10, 10, 10, 50, 10], dtype="Int64"),
        }
    )
    moves = pd.DataFrame(
        {
            "source_player_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "transfer_date": [
                WINDOW_START,  # joins
                WINDOW_END + pd.Timedelta(days=1),  # too late
                WINDOW_START,  # a loan return is not a new move
                WINDOW_START,  # not a top club
                WINDOW_START,  # already at that club
                WINDOW_END,  # last day counts
            ],
            "team_in_id": pd.array([50, 50, 50, 77, 50, 51], dtype="Int64"),
            "kind": ["transfer", "transfer", "loan_return", "transfer", "loan", "loan"],
        }
    )
    assert window_hits(pool, moves, {50, 51}).tolist() == [1, 0, 0, 0, 0, 1]


def test_freeze_window_refuses_to_overwrite(tmp_path):
    folder = tmp_path / "window-2027-01"
    folder.mkdir()
    (folder / "window.json").write_text(json.dumps({"frozen": "2026-10-24"}))
    with pytest.raises(FileExistsError, match="refreeze"):
        freeze_window(None, folder, "2026-12-15")
