"""StatsBomb open data: the cached download and the events to stats mapping (prompt 18b)."""

import json

import httpx
import pandas as pd
import pytest

from scout.config import StatsBombCompetition, StatsBombConfig
from scout.contracts import PLAYER_MATCH_STATS, validate
from scout.providers.api_football.matches import MATCH_NUMERIC_STATS
from scout.providers.statsbomb.adapter import (
    MAPPING,
    NO_EQUIVALENT,
    parse_match,
    rows_to_frame,
    spell_minutes,
)
from scout.providers.statsbomb.client import StatsBombClient, download, season_ids
from scout.storage import CacheKey, LocalRawStore

MATCH = {
    "match_id": 100,
    "match_date": "2006-10-21",
    "kick_off": "21:00:00.000",
    "match_week": 8,
    "home_team": {"home_team_id": 217, "home_team_name": "Barcelona"},
    "away_team": {"away_team_id": 220, "away_team_name": "Real Madrid"},
    "home_score": 1,
    "away_score": 0,
}


def spell(position, start="00:00", end=None, start_period=1, end_period=None):
    return {
        "position": position,
        "from": start,
        "to": end,
        "from_period": start_period,
        "to_period": end_period,
    }


def lineup(team_id, name, players):
    return {
        "team_id": team_id,
        "team_name": name,
        "lineup": [
            {
                "player_id": pid,
                "player_name": f"Player {pid}",
                "player_nickname": nick,
                "jersey_number": pid % 99,
                "cards": cards,
                "positions": spells,
            }
            for pid, nick, spells, cards in players
        ],
    }


def event(kind, pid, period=1, minute=10, **detail):
    e = {
        "id": f"e{pid}{kind}{minute}{period}{len(detail)}",
        "type": {"name": kind},
        "period": period,
        "minute": minute,
        "player": {"id": pid, "name": f"Player {pid}"},
    }
    e.update(detail)
    return e


def _match():
    lineups = [
        lineup(
            217,
            "Barcelona",
            [
                (1, "Lionel Messi", [spell("Right Wing")], []),
                (2, None, [spell("Center Forward", end="60:00", end_period=2)], []),
                (3, None, [spell("Center Forward", start="60:00", start_period=2)], []),
                (4, None, [], []),  # on the bench all match
            ],
        ),
        lineup(
            220,
            "Real Madrid",
            [
                (5, None, [spell("Goalkeeper")], [{"card_type": "Yellow Card"}]),
                (
                    6,
                    None,
                    [spell("Center Back", end="46:30", end_period=1)],
                    [{"card_type": "Yellow Card"}, {"card_type": "Second Yellow"}],
                ),
            ],
        ),
    ]
    shot_id = "shot1"
    events = [
        {
            "type": {"name": "Starting XI"},
            "period": 1,
            "tactics": {"lineup": [{"player": {"id": i}} for i in (1, 2, 5, 6)]},
        },
        event("Pass", 2, pass_={}) | {"pass": {"goal_assist": True}},
        event("Pass", 2) | {"pass": {"type": {"name": "Throw-in"}}},
        event("Pass", 1) | {"pass": {"outcome": {"name": "Incomplete"}, "shot_assist": True}},
        event("Shot", 1) | {"id": shot_id, "shot": {"outcome": {"name": "Goal"}}},
        event("Shot", 1, period=5)
        | {"shot": {"outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}},
        event("Dribble", 1) | {"dribble": {"outcome": {"name": "Complete"}}},
        event("Dribble", 1) | {"dribble": {"outcome": {"name": "Incomplete"}}},
        event("Dribbled Past", 6),
        event("Duel", 6) | {"duel": {"type": {"name": "Tackle"}, "outcome": {"name": "Won"}}},
        event("Duel", 6) | {"duel": {"type": {"name": "Aerial Lost"}}},
        event("Clearance", 6) | {"clearance": {"aerial_won": True}},
        event("Block", 6) | {"related_events": [shot_id]},
        event("Block", 6) | {"related_events": ["a pass"]},
        event("Interception", 6) | {"interception": {"outcome": {"name": "Lost In Play"}}},
        event("Foul Committed", 6) | {"foul_committed": {"penalty": True}},
        event("Foul Won", 1) | {"foul_won": {"penalty": True}},
        event("Goal Keeper", 5) | {"goalkeeper": {"type": {"name": "Shot Saved"}}},
        event("Goal Keeper", 5) | {"goalkeeper": {"type": {"name": "Goal Conceded"}}},
        event("Shot", 3, period=2, minute=80) | {"shot": {"outcome": {"name": "Saved"}}},
    ]
    return lineups, events


@pytest.fixture
def frame():
    lineups, events = _match()
    return rows_to_frame(parse_match(MATCH, lineups, events, "la_liga", 2006)).set_index(
        "source_player_id"
    )


def test_minutes_read_the_regulation_clock():
    assert spell_minutes(spell("Right Wing"), 2) == 90
    # off at 60:00 in the second half, on for the last 30 minutes
    assert spell_minutes(spell("x", end="60:00", end_period=2), 2) == 60
    assert spell_minutes(spell("x", start="60:00", start_period=2), 2) == 30
    # sent off in first half stoppage time: the half is capped at 45
    assert spell_minutes(spell("x", end="46:30", end_period=1), 2) == 45
    assert spell_minutes(spell("x"), 4) == 120  # extra time played


def test_counts_follow_the_mapping(frame):
    messi = frame.loc[1]
    assert messi["minutes"] == 90
    assert messi["player_nickname"] == "Lionel Messi"
    assert (messi["goals"], messi["shots_total"], messi["shots_on"]) == (1, 1, 1)  # no shoot out
    assert messi["penalty_scored"] == 0  # the shoot out penalty does not count
    assert (messi["passes_total"], messi["passes_accurate"], messi["passes_key"]) == (1, 0, 1)
    assert (messi["dribbles_attempts"], messi["dribbles_success"]) == (2, 1)
    assert messi["fouls_drawn"] == 1
    assert messi["penalty_won"] == 1
    # duels: two dribbles (one won) and a foul won
    assert (messi["duels_total"], messi["duels_won"]) == (3, 2)
    assert messi["position"] == "F"
    assert messi["sub_position"] == "Right Winger"
    two = frame.loc[2]
    assert (two["passes_total"], two["assists"], two["passes_key"]) == (
        1,
        1,
        1,
    )  # throw in left out
    assert two["minutes"] == 60
    assert not two["substitute"]
    assert frame.loc[3, "minutes"] == 30
    assert frame.loc[3, "substitute"]
    defender = frame.loc[6]
    assert defender["minutes"] == 45
    assert defender["blocks"] == 1  # only the shot block
    assert defender["tackles_total"] == 1
    assert defender["interceptions"] == 1
    assert defender["dribbled_past"] == 1
    # tackle won, aerial lost, aerial won, dribbled past, foul committed
    assert (defender["duels_total"], defender["duels_won"]) == (5, 2)
    assert defender["penalty_committed"] == 1
    assert (defender["cards_yellow"], defender["cards_red"]) == (1, 1)
    assert pd.isna(defender["goals_conceded"])  # goalkeepers only
    keeper = frame.loc[5]
    assert (keeper["saves"], keeper["goals_conceded"]) == (1, 1)


def test_no_equivalent_is_missing_never_zero(frame):
    for column in NO_EQUIVALENT:
        if column in frame:
            assert frame[column].isna().all(), column
    bench = frame.loc[4]
    assert bench["minutes"] == 0
    assert bench[["goals", "passes_total", "duels_total"]].isna().all()  # did not play


def test_every_count_is_mapped_and_the_contract_passes(frame):
    assert set(MATCH_NUMERIC_STATS) - {"minutes"} <= set(MAPPING)
    table = frame.reset_index()
    validate(PLAYER_MATCH_STATS, table)
    assert (table["source"] == "statsbomb").all()
    assert str(table["kickoff"].dtype) == "datetime64[ns, UTC]"
    assert table["kickoff"].iloc[0] == pd.Timestamp("2006-10-21 19:00", tz="UTC")  # Madrid time


# The download


COMPETITIONS = [
    {
        "competition_id": 11,
        "season_id": 39,
        "competition_name": "La Liga",
        "season_name": "2006/2007",
    },
    {
        "competition_id": 2,
        "season_id": 44,
        "competition_name": "Premier League",
        "season_name": "2003/2004",
    },
]
CONFIG = StatsBombConfig(
    competitions=[
        StatsBombCompetition(league_key="la_liga", competition="La Liga", seasons=[2006, 2007]),
    ]
)


def test_season_ids_skip_seasons_the_repository_lacks(caplog):
    found = season_ids(COMPETITIONS, CONFIG)
    assert [(s.league_key, s.season, s.competition_id, s.season_id) for s in found] == [
        ("la_liga", 2006, 11, 39)
    ]
    assert any("2007/2008" in m for m in caplog.messages)


def test_download_caches_every_file_once_and_never_a_refusal(tmp_path):
    lineups, events = _match()
    bodies = {
        "/data/competitions.json": COMPETITIONS,
        "/data/matches/11/39.json": [MATCH],
        "/data/lineups/100.json": lineups,
        "/data/events/100.json": events,
        "/LICENSE.pdf": None,
        "/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png": None,
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        seen.append(path)
        if path not in bodies:
            return httpx.Response(404)
        body = bodies[path]
        return httpx.Response(200, content=b"binary" if body is None else json.dumps(body).encode())

    store = LocalRawStore(tmp_path)
    transport = httpx.MockTransport(handler)
    config = CONFIG.model_copy(update={"base_url": "https://sb.test"})
    first = download(store, tmp_path, config, workers=2, transport=transport)
    again = download(store, tmp_path, config, workers=2, transport=transport)
    assert first["matches"] == 1
    assert first["requests"] == 6
    assert again["requests"] == 0  # everything came from the cache
    key = CacheKey.of("statsbomb", "events", {"match_id": 100})
    assert json.loads(store.get(key)) == events
    assert store.get_meta(key).url.endswith("/data/events/100.json")
    assert (tmp_path / "statsbomb" / "files" / "statsbomb-logo.png").read_bytes() == b"binary"
    with (
        StatsBombClient(
            store=store, base_url="https://sb.test", transport=transport, max_retries=0
        ) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.get("events", {"match_id": 999})
    assert not store.exists(CacheKey.of("statsbomb", "events", {"match_id": 999}))
