"""Sidelined, coach and event adapters and their contracts, on synthetic payloads."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pandera.errors
import pytest

from scout.contracts import COACH_SPELLS, FIXTURE_EVENTS, SIDELINED, validate
from scout.providers.api_football.extras import (
    chunks,
    coach_rows_to_frame,
    event_rows_to_frame,
    parse_coaches,
    parse_fixture_events,
    parse_sidelined,
    sidelined_rows_to_frame,
)

TODAY = date(2026, 10, 17)


def test_chunks_are_sorted_unique_and_fixed():
    assert chunks([5, 3, 3, 0, -1, None, 41], size=2) == [[3, 5], [41]]
    assert chunks(range(1, 46)) == chunks(reversed(range(1, 46)))
    assert [len(c) for c in chunks(range(1, 46))] == [20, 20, 5]


def sidelined_payload(*players):
    return {"response": [{"id": pid, "sidelined": periods} for pid, periods in players]}


def test_sidelined_parses_and_repairs(caplog):
    payload = sidelined_payload(
        (
            7,
            [
                {"type": "Hamstring", "start": "2018-12-19", "end": "2019-02-23"},
                {"type": "Hamstring", "start": "2018-12-19", "end": "2019-02-23"},  # repeated
                {"type": "Knee Injury", "start": "2026-09-01", "end": "2027-03-01"},  # running
                {"type": "Knock", "start": "2020-05-10", "end": "2020-05-01"},  # backwards
                {"type": "Suspended", "start": "2031-01-01", "end": "2031-01-05"},  # future
            ],
        ),
        (0, [{"type": "Knock", "start": "2020-01-01", "end": "2020-01-02"}]),  # no id
    )
    with caplog.at_level(logging.WARNING):
        frame = sidelined_rows_to_frame(parse_sidelined(payload), TODAY)
    assert frame["type"].tolist() == ["Hamstring", "Knee Injury"]
    assert frame["end_date"].max() == pd.Timestamp("2027-03-01")
    assert any("before start" in m for m in caplog.messages)
    assert any("impossible" in m for m in caplog.messages)
    validate(SIDELINED, frame)


def test_sidelined_contract_rejects_end_before_start():
    frame = sidelined_rows_to_frame(
        parse_sidelined(
            sidelined_payload((7, [{"type": "K", "start": "2020-05-01", "end": "2020-05-10"}]))
        ),
        TODAY,
    )
    frame.loc[0, "end_date"] = pd.Timestamp("2020-04-01")
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(SIDELINED, frame)


def coach(cid, name, career):
    return {
        "id": cid,
        "name": name,
        "birth": {"date": "1970-02-02"},
        "nationality": "Netherlands",
        "career": [
            {"team": {"id": tid, "name": f"Team {tid}"}, "start": start, "end": end}
            for tid, start, end in career
        ],
    }


def test_coaches_keep_every_spell_once():
    one = {
        "response": [
            coach(
                1, "A Coach", [(33, "2022-06-01", "2024-10-01"), (194, "2018-01-01", "2022-05-01")]
            )
        ]
    }
    two = {
        "response": [
            coach(
                1, "A Coach", [(33, "2022-06-01", "2024-10-01"), (194, "2018-01-01", "2022-05-01")]
            ),
            coach(2, "B Coach", [(194, "2024-07-01", None)]),
        ]
    }
    frame = coach_rows_to_frame(parse_coaches(one) + parse_coaches(two), TODAY)
    assert len(frame) == 3
    assert frame.loc[frame.coach_id == 2, "end_date"].isna().all()  # still in charge
    validate(COACH_SPELLS, frame)


def test_coach_contract_rejects_end_before_start():
    frame = coach_rows_to_frame(
        parse_coaches({"response": [coach(1, "A", [(33, "2022-06-01", "2024-10-01")])]}), TODAY
    )
    frame.loc[0, "end_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(COACH_SPELLS, frame)


def match(events, fid=99):
    return {
        "fixture": {"id": fid, "date": "2023-08-11T19:00:00+00:00"},
        "league": {"id": 39, "season": 2023},
        "events": events,
    }


def event(minute, kind, pid=10, assist=None, extra=None, detail="Normal Goal"):
    return {
        "time": {"elapsed": minute, "extra": extra},
        "team": {"id": 50},
        "player": {"id": pid, "name": f"P {pid}"},
        "assist": {"id": assist, "name": None},
        "type": kind,
        "detail": detail,
        "comments": None,
    }


def test_events_parse_and_repair(caplog):
    payload = match(
        [
            event(4, "Goal", assist=44),
            event(23, "subst", pid=629, assist=2291, detail="Substitution 1"),
            event(90, "Card", pid=0, extra=3, detail="Yellow Card"),  # unknown player
            event(400, "Goal"),  # impossible minute
        ]
    )
    with caplog.at_level(logging.WARNING):
        frame = event_rows_to_frame(parse_fixture_events(payload))
    assert frame["event_index"].tolist() == [0, 1, 2, 3]
    assert pd.isna(frame.loc[2, "source_player_id"])
    assert pd.isna(frame.loc[3, "elapsed"])
    assert frame.loc[1, "assist_id"] == 2291
    assert any("impossible" in m for m in caplog.messages)
    validate(FIXTURE_EVENTS, frame)


def test_event_contract_rejects_impossible_minute():
    frame = event_rows_to_frame(parse_fixture_events(match([event(4, "Goal")])))
    frame.loc[0, "elapsed"] = 400
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(FIXTURE_EVENTS, frame)


def test_match_without_events_gives_no_rows():
    assert event_rows_to_frame(parse_fixture_events(match([]))).empty
