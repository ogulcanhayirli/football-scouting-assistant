"""StatsBomb events and lineups to the canonical player_match_stats table
(prompt 18b).

One row per player who appears in a match's lineups, bench included, with
the columns of the API Football match table (``MATCH_COLUMNS``), so the
same contract and the same season aggregation read both. Each count is the
number of one StatsBomb event type by the player, sometimes narrowed by its
outcome or a flag; ``MAPPING`` lists every one, and MODELS.md prints the
same table. Measures without an equivalent in the events (the match
rating, offsides, the captain) are missing, never zero.

Minutes come from the lineups: each position spell runs from its ``from`` to
its ``to`` time (the end of the match when blank), read on a regulation
clock, the first half capped at 45 and the second at 90 (extra time at 105
and 120), so stoppage time is not counted, as API Football counts it.
Periods 5 (a penalty shoot out) never count.

Ids are StatsBomb's own (players, teams, matches): the table lives in its
own folder (``processed/statsbomb/``) and is never mixed with API Football
rows, whose ids are another space.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import pandas as pd

from scout.providers.api_football.matches import (
    MATCH_COLUMNS,
    MATCH_NUMERIC_STATS,
)

log = logging.getLogger(__name__)

SOURCE = "statsbomb"
LEAGUE_IDS = {"la_liga": 140, "epl": 39}  # the API Football ids, for reference only
TIMEZONES = {"la_liga": "Europe/Madrid", "epl": "Europe/London"}
PERIOD_CLOCK = {1: (0, 45), 2: (45, 90), 3: (90, 105), 4: (105, 120)}
PLAYED_PERIODS = set(PERIOD_CLOCK)

ON_TARGET = {"Goal", "Saved", "Saved To Post", "Saved to Post"}
TACKLE_WON = {"Won", "Success", "Success In Play", "Success Out"}
FIFTY_WON = {"Won", "Success To Team"}
SAVES = {
    "Shot Saved",
    "Shot Saved to Post",
    "Shot Saved To Post",
    "Penalty Saved",
    "Penalty Saved to Post",
    "Penalty Saved To Post",
    "Save",
}
PENALTY_SAVED = {"Penalty Saved", "Penalty Saved to Post", "Penalty Saved To Post"}
CONCEDED = {"Goal Conceded", "Penalty Conceded"}
AERIAL_KEYS = ("pass", "shot", "clearance", "miscontrol")
# Passes counted: every pass but throw ins, which API Football leaves out of
# its pass count (the validation of prompt 18b, MODELS.md).
PASS_EXCLUDED = {"Throw-in"}
# Duels include fouls (a foul won is a duel won, a foul committed one lost),
# as API Football's count does (the validation, MODELS.md).
DUELS_WITH_FOULS = True

# Column: (StatsBomb source, event type, what is counted). MODELS.md has the same table.
MAPPING: dict[str, tuple[str, str]] = {
    "minutes": (
        "Lineups: positions",
        "time between each spell's from and to (blank: the final whistle), regulation clock "
        "(first half capped at 45, second at 90, extra time at 105 and 120)",
    ),
    "goals": ("Shot", "outcome Goal, periods 1 to 4; own goals (Own Goal Against) not counted"),
    "assists": ("Pass", "pass.goal_assist"),
    "shots_total": ("Shot", "every outcome, periods 1 to 4"),
    "shots_on": ("Shot", "outcome Goal, Saved or Saved To Post"),
    "passes_total": ("Pass", "every pass but throw ins (pass.type Throw-in)"),
    "passes_key": ("Pass", "pass.shot_assist or pass.goal_assist"),
    "passes_accurate": ("Pass", "no pass.outcome (complete), throw ins left out"),
    "tackles_total": ("Duel", "duel.type Tackle, every outcome"),
    "blocks": ("Block", "blocks of a shot (the Block's related event is a Shot)"),
    "interceptions": ("Interception", "every outcome"),
    "duels_total": (
        "Duel, 50/50, Dribble, Dribbled Past, aerial_won, Foul Won, Foul Committed",
        "tackles (Duel Tackle), aerials lost (Duel Aerial Lost), aerials won (aerial_won on "
        "a Pass, Shot, Clearance or Miscontrol), 50/50s, dribbles, times dribbled past, fouls "
        "won and committed",
    ),
    "duels_won": (
        "Duel, 50/50, Dribble, aerial_won, Foul Won",
        "tackles with outcome Won or Success (In Play, Out), aerials won, 50/50s Won or "
        "Success To Team, dribbles Complete, fouls won",
    ),
    "dribbles_attempts": ("Dribble", "every outcome"),
    "dribbles_success": ("Dribble", "outcome Complete"),
    "dribbled_past": ("Dribbled Past", "every event"),
    "fouls_drawn": ("Foul Won", "every event"),
    "fouls_committed": ("Foul Committed", "every event"),
    "cards_yellow": ("Lineups: cards", "Yellow Card"),
    "cards_red": ("Lineups: cards", "Red Card or Second Yellow"),
    "saves": (
        "Goal Keeper",
        "goalkeeper.type Shot Saved, Shot Saved to Post, Penalty Saved (to Post), Save",
    ),
    "goals_conceded": (
        "Goal Keeper",
        "goalkeeper.type Goal Conceded or Penalty Conceded; goalkeepers only",
    ),
    "penalty_won": ("Foul Won", "foul_won.penalty"),
    "penalty_committed": ("Foul Committed", "foul_committed.penalty"),
    "penalty_scored": ("Shot", "shot.type Penalty, outcome Goal, periods 1 to 4"),
    "penalty_missed": ("Shot", "shot.type Penalty, any other outcome"),
    "penalty_saved": ("Goal Keeper", "goalkeeper.type Penalty Saved (to Post)"),
    "rating": ("none", "no equivalent: missing"),
    "offsides": ("none", "no equivalent (the player caught offside is rarely recorded): missing"),
    "captain": ("none", "no equivalent: missing"),
}
NO_EQUIVALENT = [c for c, (src, _) in MAPPING.items() if src == "none"]

# StatsBomb positions: the one letter code of the match table and the
# Transfermarkt position the season's position group is read from.
POSITIONS: dict[str, tuple[str, str]] = {
    "Goalkeeper": ("G", "Goalkeeper"),
    "Right Back": ("D", "Right-Back"),
    "Right Wing Back": ("D", "Right-Back"),
    "Left Back": ("D", "Left-Back"),
    "Left Wing Back": ("D", "Left-Back"),
    "Right Center Back": ("D", "Centre-Back"),
    "Center Back": ("D", "Centre-Back"),
    "Left Center Back": ("D", "Centre-Back"),
    "Right Defensive Midfield": ("M", "Defensive Midfield"),
    "Center Defensive Midfield": ("M", "Defensive Midfield"),
    "Left Defensive Midfield": ("M", "Defensive Midfield"),
    "Right Center Midfield": ("M", "Central Midfield"),
    "Center Midfield": ("M", "Central Midfield"),
    "Left Center Midfield": ("M", "Central Midfield"),
    "Right Attacking Midfield": ("M", "Attacking Midfield"),
    "Center Attacking Midfield": ("M", "Attacking Midfield"),
    "Left Attacking Midfield": ("M", "Attacking Midfield"),
    "Right Midfield": ("M", "Right Midfield"),
    "Left Midfield": ("M", "Left Midfield"),
    "Right Wing": ("F", "Right Winger"),
    "Left Wing": ("F", "Left Winger"),
    "Secondary Striker": ("F", "Second Striker"),
    "Right Center Forward": ("F", "Centre-Forward"),
    "Striker": ("F", "Centre-Forward"),
    "Center Forward": ("F", "Centre-Forward"),
    "Left Center Forward": ("F", "Centre-Forward"),
}
EXTRA_COLUMNS = ["player_nickname", "sb_position", "sub_position"]
COLUMNS = [*MATCH_COLUMNS, *EXTRA_COLUMNS]


def clock(text: str | None) -> float | None:
    """'59:21' as minutes (59.35)."""
    if not text:
        return None
    minutes, _, seconds = str(text).partition(":")
    return int(minutes) + int(seconds or 0) / 60


def regulation(minute: float, period: int) -> float:
    lo, hi = PERIOD_CLOCK[period]
    return min(max(minute, lo), hi)


def spell_minutes(spell: dict, last_period: int) -> float:
    start_period = int(spell.get("from_period") or 1)
    end_period = spell.get("to_period")
    start = regulation(clock(spell.get("from")) or 0.0, min(start_period, last_period))
    if spell.get("to") is None or end_period is None:
        end = PERIOD_CLOCK[last_period][1]
    else:
        end = regulation(clock(spell["to"]), min(int(end_period), last_period))
    return max(end - start, 0.0)


def _event_counts(events: list[dict]) -> dict[int, Counter]:
    """Per player id, the counts behind every mapped column."""
    counts: dict[int, Counter] = {}
    shots = {e["id"] for e in events if (e.get("type") or {}).get("name") == "Shot"}

    def add(pid: int, column: str, n: int = 1) -> None:
        counts.setdefault(pid, Counter())[column] += n

    for e in events:
        if int(e.get("period") or 0) not in PLAYED_PERIODS or "player" not in e:
            continue
        pid = int(e["player"]["id"])
        kind = (e.get("type") or {}).get("name")
        if any((e.get(k) or {}).get("aerial_won") for k in AERIAL_KEYS):
            add(pid, "duels_total")
            add(pid, "duels_won")
        if kind == "Shot":
            shot = e.get("shot") or {}
            outcome = (shot.get("outcome") or {}).get("name")
            penalty = (shot.get("type") or {}).get("name") == "Penalty"
            add(pid, "shots_total")
            if outcome in ON_TARGET:
                add(pid, "shots_on")
            if outcome == "Goal":
                add(pid, "goals")
            if penalty:
                add(pid, "penalty_scored" if outcome == "Goal" else "penalty_missed")
        elif kind == "Pass":
            p = e.get("pass") or {}
            if (p.get("type") or {}).get("name") in PASS_EXCLUDED:
                continue
            add(pid, "passes_total")
            if p.get("outcome") is None:
                add(pid, "passes_accurate")
            if p.get("goal_assist"):
                add(pid, "assists")
            if p.get("goal_assist") or p.get("shot_assist"):
                add(pid, "passes_key")
        elif kind == "Duel":
            duel = e.get("duel") or {}
            dtype = (duel.get("type") or {}).get("name")
            add(pid, "duels_total")
            if dtype == "Tackle":
                add(pid, "tackles_total")
                if (duel.get("outcome") or {}).get("name") in TACKLE_WON:
                    add(pid, "duels_won")
        elif kind == "50/50":
            add(pid, "duels_total")
            if ((e.get("50_50") or {}).get("outcome") or {}).get("name") in FIFTY_WON:
                add(pid, "duels_won")
        elif kind == "Dribble":
            add(pid, "dribbles_attempts")
            add(pid, "duels_total")
            if ((e.get("dribble") or {}).get("outcome") or {}).get("name") == "Complete":
                add(pid, "dribbles_success")
                add(pid, "duels_won")
        elif kind == "Dribbled Past":
            add(pid, "dribbled_past")
            add(pid, "duels_total")
        elif kind == "Interception":
            add(pid, "interceptions")
        elif kind == "Block" and shots.intersection(e.get("related_events") or []):
            add(pid, "blocks")  # API Football counts blocked shots only (the validation)
        elif kind == "Foul Won":
            add(pid, "fouls_drawn")
            if DUELS_WITH_FOULS:
                add(pid, "duels_total")
                add(pid, "duels_won")
            if (e.get("foul_won") or {}).get("penalty"):
                add(pid, "penalty_won")
        elif kind == "Foul Committed":
            add(pid, "fouls_committed")
            if DUELS_WITH_FOULS:
                add(pid, "duels_total")
            if (e.get("foul_committed") or {}).get("penalty"):
                add(pid, "penalty_committed")
        elif kind == "Goal Keeper":
            gk = ((e.get("goalkeeper") or {}).get("type") or {}).get("name")
            if gk in SAVES:
                add(pid, "saves")
            if gk in PENALTY_SAVED:
                add(pid, "penalty_saved")
            if gk in CONCEDED:
                add(pid, "goals_conceded")
    return counts


def _kickoff(match: dict, league_key: str) -> pd.Timestamp:
    local = f"{match['match_date']} {(match.get('kick_off') or '12:00:00')[:8]}"
    return pd.Timestamp(local).tz_localize(TIMEZONES.get(league_key, "UTC")).tz_convert("UTC")


def parse_match(
    match: dict[str, Any],
    lineups: list[dict[str, Any]],
    events: list[dict[str, Any]],
    league_key: str,
    season: int,
) -> list[dict]:
    """Rows for one match: one per player in either lineup, bench included."""
    played = {int(e.get("period") or 0) for e in events} & PLAYED_PERIODS
    last_period = max(played) if played else 2
    if last_period == 3:
        last_period = 4  # extra time is played in full
    counts = _event_counts(events)
    home = match["home_team"]["home_team_id"]
    names = {
        match["home_team"]["home_team_id"]: match["home_team"]["home_team_name"],
        match["away_team"]["away_team_id"]: match["away_team"]["away_team_name"],
    }
    goals = {
        match["home_team"]["home_team_id"]: match.get("home_score"),
        match["away_team"]["away_team_id"]: match.get("away_score"),
    }
    starters = {
        int(p["player"]["id"])
        for e in events
        if (e.get("type") or {}).get("name") == "Starting XI"
        for p in (e.get("tactics") or {}).get("lineup") or []
    }
    with_events = {int(e["player"]["id"]) for e in events if "player" in e}
    kickoff = _kickoff(match, league_key)
    rows = []
    for team in lineups:
        team_id = int(team["team_id"])
        opponent = next((t for t in names if t != team_id), None)
        for p in team.get("lineup") or []:
            pid = int(p["player_id"])
            spells = p.get("positions") or []
            if spells:
                minutes: float | None = round(sum(spell_minutes(s, last_period) for s in spells))
            elif pid in with_events:
                log.warning(
                    "statsbomb match %s: %s has events but no positions; minutes missing",
                    match["match_id"],
                    p.get("player_name"),
                )
                minutes = None
            else:
                minutes = 0.0
            by_position: Counter = Counter()
            for s in spells:
                by_position[s.get("position")] += spell_minutes(s, last_period)
            main = by_position.most_common(1)[0][0] if by_position else None
            letter, label = POSITIONS.get(main, (None, None))
            row: dict[str, Any] = {
                "source": SOURCE,
                "fixture_id": int(match["match_id"]),
                "kickoff": kickoff,
                "league_key": league_key,
                "league_id": LEAGUE_IDS.get(league_key),
                "season": season,
                "round": f"Regular Season - {match['match_week']}"
                if match.get("match_week")
                else None,
                "team_id": team_id,
                "team_name": team.get("team_name"),
                "opponent_id": opponent,
                "opponent_name": names.get(opponent),
                "is_home": team_id == home,
                "team_goals": goals.get(team_id),
                "opponent_goals": goals.get(opponent),
                "source_player_id": pid,
                "player_name": p.get("player_name"),
                "player_nickname": p.get("player_nickname"),
                "position": letter,
                "shirt_number": p.get("jersey_number"),
                "minutes": minutes,
                "captain": None,
                "substitute": pid not in starters if starters else None,
                "sb_position": main,
                "sub_position": label,
            }
            played_here = bool(minutes)
            c = counts.get(pid, Counter())
            for column in MATCH_NUMERIC_STATS:
                if column in ("minutes", *NO_EQUIVALENT):
                    continue
                row[column] = c.get(column, 0) if played_here else None
            if played_here and letter != "G":
                row["goals_conceded"] = None  # kept by goalkeepers only
            cards = Counter((card.get("card_type") or "") for card in p.get("cards") or [])
            row["cards_yellow"] = cards["Yellow Card"] if played_here else None
            row["cards_red"] = (cards["Red Card"] + cards["Second Yellow"]) if played_here else None
            rows.append(row)
    return rows


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """The rows with the API Football match table's types."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in MATCH_NUMERIC_STATS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in [
        "fixture_id",
        "league_id",
        "season",
        "team_id",
        "opponent_id",
        "team_goals",
        "opponent_goals",
        "source_player_id",
        "shirt_number",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["captain", "substitute", "is_home"]:
        frame[column] = frame[column].astype("boolean")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True).dt.as_unit("ns")
    for column in [
        "round",
        "player_nickname",
        "sb_position",
        "sub_position",
        "player_name",
        "team_name",
        "opponent_name",
    ]:
        frame[column] = frame[column].astype(object)
    return frame


def read_match_table(store, config) -> pd.DataFrame:
    """Every configured season's matches from the raw cache, parsed into one
    frame (no request is made: ``scout statsbomb download`` fills the cache)."""
    from scout.providers.statsbomb.client import season_ids
    from scout.storage import CacheKey

    def cached(endpoint: str, params: dict) -> Any:
        key = CacheKey.of(SOURCE, endpoint, params)
        if not store.exists(key):
            raise FileNotFoundError(
                f"statsbomb {endpoint} {params} is not cached: run `scout statsbomb download`"
            )
        return json.loads(store.get(key))

    rows: list[dict] = []
    for s in season_ids(cached("competitions", {}), config):
        matches = cached("matches", {"competition_id": s.competition_id, "season_id": s.season_id})
        for match in sorted(matches, key=lambda m: m["match_id"]):
            mid = {"match_id": match["match_id"]}
            rows += parse_match(
                match, cached("lineups", mid), cached("events", mid), s.league_key, s.season
            )
    return rows_to_frame(rows)
