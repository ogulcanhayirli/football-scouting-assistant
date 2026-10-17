"""Adapters for sidelined periods, coaches and match events (owner, 17 October).

* ``/sidelined?players=a-b-...`` (up to 20 ids): every injury or suspension
  period the API knows for each player, with start and end dates. The list is
  a snapshot of today, so a feature may only use periods that started on or
  before its feature date, and must not read an end date after it.
* ``/coachs?team=id``: the coaches the API links to a team, each with a career
  of team spells (dates are given to the month). The careers of all coaches
  together give the coach of a team on a date.
* ``/fixtures?ids=a-b-...`` (up to 20 ids): the full match, from which only
  ``events`` (goals, cards, substitutions, VAR decisions) is read here.

Requests are batched in fixed chunks of sorted ids, so a rerun asks for the
same chunks and finds them in the cache.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

from scout.providers.api_football.client import SOURCE, ApiFootballClient

log = logging.getLogger(__name__)

BATCH = 20
EARLIEST = pd.Timestamp("1950-01-01")


def chunks(ids: Iterable[int], size: int = BATCH) -> list[list[int]]:
    unique = sorted({int(i) for i in ids if pd.notna(i) and int(i) > 0})
    return [unique[i : i + size] for i in range(0, len(unique), size)]


def joined(ids: list[int]) -> str:
    return "-".join(str(i) for i in ids)


class ApiFootballExtrasProvider:
    source = SOURCE

    def __init__(self, client: ApiFootballClient) -> None:
        self.client = client

    def sidelined(self, player_ids: list[int]) -> dict[str, Any]:
        return self.client.get("sidelined", {"players": joined(player_ids)}).payload

    def coaches(self, team_id: int) -> dict[str, Any]:
        return self.client.get("coachs", {"team": team_id}).payload

    def fixtures(self, fixture_ids: list[int]) -> dict[str, Any]:
        return self.client.get("fixtures", {"ids": joined(fixture_ids)}).payload


def _date(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _impossible_dates(frame: pd.DataFrame, columns: list[str], what: str, today: date) -> None:
    """Dates before 1950 or after ``today`` (the fetch day) become missing."""
    limit = pd.Timestamp(today)
    for column in columns:
        bad = frame[column].notna() & ((frame[column] < EARLIEST) | (frame[column] > limit))
        for idx in frame.index[bad]:
            log.warning(
                "%s: %s %s is impossible; stored as missing", what, column, frame.at[idx, column]
            )
        frame.loc[bad, column] = pd.NaT


# Sidelined

SIDELINED_COLUMNS = ["source", "source_player_id", "type", "start_date", "end_date"]


def parse_sidelined(payload: dict[str, Any]) -> list[dict]:
    rows = []
    for item in payload.get("response") or []:
        for period in item.get("sidelined") or []:
            rows.append(
                {
                    "source": SOURCE,
                    "source_player_id": item.get("id"),
                    "type": period.get("type"),
                    "start_date": period.get("start"),
                    "end_date": period.get("end"),
                }
            )
    return rows


def sidelined_rows_to_frame(rows: list[dict], today: date) -> pd.DataFrame:
    """End dates may lie after ``today`` (a long injury still running); start dates
    may not. An end before its start becomes missing, as either could be wrong."""
    frame = pd.DataFrame(rows, columns=SIDELINED_COLUMNS)
    frame["source_player_id"] = pd.to_numeric(frame["source_player_id"], errors="coerce").astype(
        "Int64"
    )
    frame["start_date"] = _date(frame["start_date"]).astype("datetime64[ns]")
    frame["end_date"] = _date(frame["end_date"]).astype("datetime64[ns]")
    _impossible_dates(frame, ["start_date"], "sidelined", today)
    too_early = frame["end_date"].notna() & (frame["end_date"] < EARLIEST)
    frame.loc[too_early, "end_date"] = pd.NaT
    backwards = (frame["end_date"] < frame["start_date"]).fillna(False)
    for idx in frame.index[backwards]:
        log.warning(
            "Sidelined player %s: end %s before start %s; both missing",
            frame.at[idx, "source_player_id"],
            frame.at[idx, "end_date"].date(),
            frame.at[idx, "start_date"].date(),
        )
    frame.loc[backwards, ["start_date", "end_date"]] = pd.NaT
    frame = frame[frame["source_player_id"].notna() & (frame["source_player_id"] > 0)]
    frame = frame[frame["start_date"].notna()]
    dup = frame.duplicated(["source_player_id", "type", "start_date", "end_date"])
    if dup.any():
        log.info("Dropping %d repeated sidelined rows", int(dup.sum()))
    return frame[~dup].sort_values(["source_player_id", "start_date"]).reset_index(drop=True)


# Coaches

COACH_COLUMNS = [
    "source",
    "coach_id",
    "coach_name",
    "birth_date",
    "nationality",
    "team_id",
    "team_name",
    "start_date",
    "end_date",
]


def parse_coaches(payload: dict[str, Any]) -> list[dict]:
    """One row per career spell of every coach in the response, for any team."""
    rows = []
    for coach in payload.get("response") or []:
        for spell in coach.get("career") or []:
            team = spell.get("team") or {}
            rows.append(
                {
                    "source": SOURCE,
                    "coach_id": coach.get("id"),
                    "coach_name": coach.get("name"),
                    "birth_date": (coach.get("birth") or {}).get("date"),
                    "nationality": coach.get("nationality"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "start_date": spell.get("start"),
                    "end_date": spell.get("end"),
                }
            )
    return rows


def coach_rows_to_frame(rows: list[dict], today: date) -> pd.DataFrame:
    """Coach spells. A missing end means the spell is still running. The same
    spell arrives once per team the coach worked for, so repeats are dropped."""
    frame = pd.DataFrame(rows, columns=COACH_COLUMNS)
    for column in ["coach_id", "team_id"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["birth_date", "start_date", "end_date"]:
        frame[column] = _date(frame[column]).astype("datetime64[ns]")
    _impossible_dates(frame, ["birth_date", "start_date", "end_date"], "coach", today)
    backwards = (frame["end_date"] < frame["start_date"]).fillna(False)
    for idx in frame.index[backwards]:
        log.warning(
            "Coach %s at team %s: end %s before start %s; both missing",
            frame.at[idx, "coach_id"],
            frame.at[idx, "team_id"],
            frame.at[idx, "end_date"].date(),
            frame.at[idx, "start_date"].date(),
        )
    frame.loc[backwards, ["start_date", "end_date"]] = pd.NaT
    frame = frame[frame["coach_id"].notna() & (frame["coach_id"] > 0)]
    frame = frame[frame["team_id"].notna() & (frame["team_id"] > 0)]
    frame = frame.drop_duplicates(["coach_id", "team_id", "start_date", "end_date"])
    return frame.sort_values(["team_id", "start_date", "coach_id"]).reset_index(drop=True)


# Events

EVENT_COLUMNS = [
    "source",
    "fixture_id",
    "kickoff",
    "league_id",
    "season",
    "event_index",
    "elapsed",
    "extra",
    "team_id",
    "source_player_id",
    "player_name",
    "assist_id",
    "assist_name",
    "type",
    "detail",
    "comments",
]
MAX_ELAPSED = 130  # 120 minutes of extra time plus stoppage; the API caps added time in ``extra``


def parse_fixture_events(fixture: dict[str, Any]) -> list[dict]:
    """Events of one match in the order the API gives them. For substitutions the
    API puts one player in ``player`` and the other in ``assist``."""
    info = fixture.get("fixture") or {}
    league = fixture.get("league") or {}
    rows = []
    for i, event in enumerate(fixture.get("events") or []):
        time = event.get("time") or {}
        player = event.get("player") or {}
        assist = event.get("assist") or {}
        rows.append(
            {
                "source": SOURCE,
                "fixture_id": info.get("id"),
                "kickoff": info.get("date"),
                "league_id": league.get("id"),
                "season": league.get("season"),
                "event_index": i,
                "elapsed": time.get("elapsed"),
                "extra": time.get("extra"),
                "team_id": (event.get("team") or {}).get("id"),
                "source_player_id": player.get("id"),
                "player_name": player.get("name"),
                "assist_id": assist.get("id"),
                "assist_name": assist.get("name"),
                "type": event.get("type"),
                "detail": event.get("detail"),
                "comments": event.get("comments"),
            }
        )
    return rows


def event_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """A player or assist id of 0 or less means unknown and becomes missing; a
    minute outside 0 to ``MAX_ELAPSED`` or negative added time becomes missing."""
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for column in [
        "fixture_id",
        "league_id",
        "season",
        "event_index",
        "elapsed",
        "extra",
        "team_id",
        "source_player_id",
        "assist_id",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce").dt.as_unit("ns")
    for column in ["source_player_id", "assist_id"]:
        frame.loc[(frame[column] <= 0).fillna(False), column] = pd.NA
    bad_minute = ((frame["elapsed"] < 0) | (frame["elapsed"] > MAX_ELAPSED)).fillna(False)
    bad_minute |= (frame["extra"] < 0).fillna(False)
    for idx in frame.index[bad_minute]:
        log.warning(
            "Event %s in fixture %s at minute %s+%s is impossible; minute stored as missing",
            frame.at[idx, "type"],
            frame.at[idx, "fixture_id"],
            frame.at[idx, "elapsed"],
            frame.at[idx, "extra"],
        )
    frame.loc[bad_minute, ["elapsed", "extra"]] = pd.NA
    return frame.reset_index(drop=True)
