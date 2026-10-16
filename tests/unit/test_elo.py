import numpy as np
import pandas as pd
import pytest

from scout.elo import (
    EloParams,
    brier,
    expected,
    fit,
    goal_weight,
    league_seasons,
    match_long,
    parse_fixtures,
    rating_asof,
    run_elo,
    team_seasons,
)


def _fixture(fid, league_id, country, status="FT", home=(1, 2), full=(2, 1), goals=None):
    goals = goals or full
    return {
        "fixture": {"id": fid, "date": "2020-01-01T15:00:00+00:00", "status": {"short": status}},
        "league": {"id": league_id, "name": "L", "country": country, "season": 2019},
        "teams": {"home": {"id": home[0], "name": "A"}, "away": {"id": home[1], "name": "B"}},
        "goals": {"home": goals[0], "away": goals[1]},
        "score": {"fulltime": {"home": full[0], "away": full[1]}},
    }


def test_parse_keeps_club_matches_played_to_the_end():
    payload = {
        "response": [
            _fixture(1, 39, "England"),
            _fixture(2, 39, "England", status="AWD"),  # awarded at the desk
            _fixture(3, 39, "England", status="NS"),
            _fixture(4, 2, "World", status="AET", full=(1, 1), goals=(2, 1)),
            _fixture(5, 1, "World"),  # World Cup: national teams
            _fixture(6, 14, "World"),  # UEFA Youth League
        ]
    }
    rows = parse_fixtures(payload, {39: "epl"})
    assert [r["fixture_id"] for r in rows] == [1, 4]
    assert rows[0]["league_key"] == "epl"
    assert rows[0]["kind"] == "league"
    # extra time is not counted: the 90 minute score was a draw
    assert (rows[1]["home_goals"], rows[1]["away_goals"], rows[1]["kind"]) == (1, 1, "cup")


def test_goal_weight_and_expected():
    assert goal_weight([0, 1, 2, 3, 5]).tolist() == [1.0, 1.0, 1.5, 14 / 8, 16 / 8]
    assert expected(0) == pytest.approx(0.5)
    assert expected(400) == pytest.approx(10 / 11)


def _matches(rows):
    frame = pd.DataFrame(
        rows,
        columns=[
            "fixture_id",
            "kickoff",
            "league_key",
            "season",
            "kind",
            "home_id",
            "away_id",
            "home_goals",
            "away_goals",
        ],
    )
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame["home_name"] = frame["home_id"].astype(str)
    frame["away_name"] = frame["away_id"].astype(str)
    return frame


def test_ratings_are_point_in_time():
    base = _matches(
        [
            (1, "2020-01-01", "a", 2019, "league", 1, 2, 3, 0),
            (2, "2020-01-08", "a", 2019, "league", 2, 1, 1, 1),
        ]
    )
    later = pd.concat(
        [base, _matches([(3, "2020-02-01", "a", 2019, "league", 1, 2, 0, 5)])], ignore_index=True
    )
    p = EloParams(70, 20, 0.0)
    a, b = run_elo(base, p), run_elo(later, p)
    # a later result never changes an earlier match's pre match ratings
    pd.testing.assert_frame_equal(
        a[["home_pre", "away_pre"]], b.iloc[:2][["home_pre", "away_pre"]].reset_index(drop=True)
    )
    assert a.loc[0, "home_pre"] == a.loc[0, "away_pre"] == 1500
    assert a.loc[1, "away_pre"] > 1500  # club 1 won the first match
    long = match_long(b)
    assert rating_asof(long, 1, "2020-01-08") == pytest.approx(b.loc[0, "home_post"])
    assert np.isnan(rating_asof(long, 1, "2020-01-01"))


def test_cross_league_result_moves_the_whole_league():
    m = _matches(
        [
            (1, "2020-01-01", "a", 2019, "league", 1, 2, 1, 1),
            (2, "2020-01-01", "b", 2019, "league", 3, 4, 1, 1),
            (3, "2020-02-01", "cup", 2019, "cup", 1, 3, 3, 0),
            (4, "2020-03-01", "a", 2019, "league", 2, 1, 0, 0),
        ]
    )
    rated = run_elo(m, EloParams(0, 20, 1.0))
    cup = rated.loc[2]
    gain = cup["home_post"] - cup["home_pre"]
    assert gain > 0
    assert bool(cup["cross_league"])
    # club 2 did not play the cup match but its league did
    assert rated.loc[3, "home_pre"] == pytest.approx(rated.loc[0, "away_post"] + gain)
    no_link = run_elo(m, EloParams(0, 20, 0.0))
    assert no_link.loc[3, "home_pre"] == pytest.approx(no_link.loc[0, "away_post"])


def _simulated(n_teams=16, seasons=6, home=80.0, seed=3):
    rng = np.random.default_rng(seed)
    strength = rng.normal(1500, 120, n_teams)
    rows, fid = [], 0
    day = pd.Timestamp("2010-08-01")
    for season in range(seasons):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                fid += 1
                p = expected(strength[h] + home - strength[a])
                u = rng.random()
                # draws a quarter of the time; wins so that the expected score is p
                win = np.clip((p - 0.125) / 0.75, 0, 1)
                hg, ag = (1, 1) if u < 0.25 else ((2, 0) if rng.random() < win else (0, 2))
                rows.append(
                    (
                        fid,
                        day + pd.Timedelta(days=fid),
                        "a",
                        2010 + season,
                        "league",
                        h + 1,
                        a + 1,
                        hg,
                        ag,
                    )
                )
    return _matches(rows)


def test_fit_finds_the_home_advantage():
    matches = _simulated()
    grid = {"home_advantage": [0.0, 80.0, 200.0], "k": [20.0], "spread": [0.0]}
    params, table = fit(matches, grid)
    assert params.home_advantage == 80.0
    assert set(table.columns) >= {"objective", "brier_all", "brier_cross"}


def test_team_and_league_seasons():
    rated = run_elo(_simulated(n_teams=4, seasons=2), EloParams(70, 20, 0.0))
    teams = team_seasons(match_long(rated))
    assert len(teams) == 8
    assert (teams["matches"] == 6).all()
    leagues = league_seasons(teams)
    assert leagues["clubs"].tolist() == [4, 4]
    # ratings are zero sum inside a closed league
    assert leagues["elo_end"].iloc[0] == pytest.approx(1500)
    assert brier(rated, rated["kickoff"].min(), rated["kickoff"].max())["matches"] > 0
