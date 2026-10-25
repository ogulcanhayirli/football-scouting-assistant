"""Garbled names from API Football, repaired in the adapter (rule 7, prompt 18b)."""

import logging

import pandas as pd
import pandera.errors
import pytest

from scout.config import LeagueConfig
from scout.contracts import PLAYER_MATCH_STATS, PLAYER_SEASON_STATS, validate
from scout.providers.api_football.adapter import parse_players_page, rows_to_frame
from scout.providers.api_football.matches import match_rows_to_frame, parse_fixture_players
from scout.providers.api_football.names import clean_name, garbled


def mangle(name: str, latin1: bool = False, keep: str = "") -> str:
    """The name as a UTF-8 response read as Windows-1252 (Latin-1 for the
    bytes Windows-1252 leaves unassigned, or throughout with ``latin1``),
    letters in ``keep`` left right: what the cached responses hold."""
    out = []
    for c in name:
        if ord(c) < 128 or c in keep:
            out.append(c)
            continue
        for b in c.encode("utf-8"):
            try:
                out.append(chr(b) if latin1 else bytes([b]).decode("cp1252"))
            except UnicodeDecodeError:
                out.append(chr(b))
    return "".join(out)


GARBLED = {
    mangle("O. Højlund"): "O. Højlund",
    mangle("V. Ríos"): "V. Ríos",
    mangle("A. Avdić"): "A. Avdić",
    mangle("M. Vlahović", latin1=True): "M. Vlahović",
    mangle("O. Çakıroğlu", keep="ı"): "O. Çakıroğlu",  # the ı arrived right
    mangle("Marko Šćepović"): "Marko Šćepović",
    mangle("Łukasz Poręba"): "Łukasz Poręba",
    mangle("Robin Heußer"): "Robin Heußer",
    mangle("Álvaro Killane"): "Álvaro Killane",  # Á's second byte is unassigned in cp1252
    "M. O&apos;Riley": "M. O'Riley",
    "Jo&#227;o Marcelo": "João Marcelo",
}
CLEAN = [
    "Øyvind Ødegaard",
    "São Paulo",
    "Ångström",
    "Kenan Yıldız",
    "Ñíguez",
    "Brighton & Hove",
    "Sulzeer Jeremiah ''Sol' Campbell",
    "Đorđe Petrović",
]


@pytest.mark.parametrize(("raw", "fixed"), GARBLED.items())
def test_garbled_names_are_decoded(raw, fixed):
    assert clean_name(raw) == fixed
    assert clean_name(fixed) == fixed  # a repaired name stays as it is


@pytest.mark.parametrize("name", CLEAN)
def test_real_names_are_never_touched(name):
    assert clean_name(name) == name


def test_missing_values_pass_through():
    assert clean_name(None) is None
    assert pd.isna(clean_name(pd.NA))
    flags = garbled(pd.Series([mangle("O. Højlund"), "Ødegaard", None]))
    assert flags.tolist() == [True, False, False]


LEAGUE = LeagueConfig(key="epl", name="Premier League", api_football_id=39, transfermarkt_id="GB1")


def _fixture():
    return {
        "fixture": {"id": 7, "date": "2024-08-17T14:00:00+00:00", "status": {"short": "FT"}},
        "league": {"id": 39, "season": 2024, "round": "Regular Season - 1"},
        "teams": {"home": {"id": 1, "name": "Home"}, "away": {"id": 2, "name": "Away"}},
        "goals": {"home": 1, "away": 0},
    }


def _player(pid, name):
    stats = {"games": {"minutes": 90, "position": "F", "number": pid}, "goals": {"total": 1}}
    return {"player": {"id": pid, "name": name}, "statistics": [stats]}


def test_the_match_adapter_repairs_names_and_the_contract_rejects_the_raw_ones(caplog):
    payload = {
        "response": [
            {"team": {"id": 1, "name": "Home"}, "players": [_player(10, mangle("O. Højlund"))]},
            {"team": {"id": 2, "name": "Away"}, "players": [_player(20, "M. O&apos;Riley")]},
        ]
    }
    rows = parse_fixture_players(_fixture(), payload, LEAGUE)
    with caplog.at_level(logging.WARNING):
        frame = match_rows_to_frame(rows)
    assert sorted(frame["player_name"]) == ["M. O'Riley", "O. Højlund"]
    assert any("garbled" in m for m in caplog.messages)
    validate(PLAYER_MATCH_STATS, frame)
    raw = frame.assign(player_name=[mangle("O. Højlund"), "M. O&apos;Riley"])
    with pytest.raises(pandera.errors.SchemaErrors, match="names_repaired"):
        validate(PLAYER_MATCH_STATS, raw)


def test_the_season_adapter_repairs_first_and_last_names():
    payload = {
        "response": [
            {
                "player": {
                    "id": 5,
                    "name": mangle("M. Kanté"),
                    "firstname": mangle("Massiré"),
                    "lastname": mangle("Kanté"),
                    "birth": {"date": "1999-01-01"},
                },
                "statistics": [
                    {
                        "team": {"id": 3, "name": "Club"},
                        "league": {"id": 39, "season": 2024},
                        "games": {"minutes": 900, "position": "Midfielder"},
                    }
                ],
            }
        ]
    }
    frame = rows_to_frame(parse_players_page(payload, LEAGUE, 2024))
    row = frame.iloc[0]
    assert (row["player_name"], row["firstname"], row["lastname"]) == (
        "M. Kanté",
        "Massiré",
        "Kanté",
    )
    validate(PLAYER_SEASON_STATS, frame)
