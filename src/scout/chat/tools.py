"""The chat's four tools over the published tables.

Each tool returns plain JSON whose numbers are written the way the page
writes them (``€22m``, ``30% of 142 past players, decile 10``, ``#305``),
so an answer that copies them says what the page says. The filters, the
sorting and the similar player search follow the page's own JavaScript
(``scout/site_template.html``): the same league at forecast, the same age
rules, the same cosine on the 8 bit vectors of ``similar.json`` and the same
pools (his position group today; a year either side for "as he was at an
age"; the middle of the season age for the measured greats).

* ``search_players``: filters and a sort over one table, at most 10 rows
  and the number of players that matched;
* ``player_card``: every fact the page shows for one player, the written
  report included;
* ``similar_players``: the ten nearest of his position group today, or of
  today's players to him (or to a measured great) at an age;
* ``how_to_read``: the dictionary entry of a column and what it cannot say.

Every schema is strict (``additionalProperties: false``), with optional
fields left out of ``required`` rather than nullable, under the API's limit
of 24 optional and 16 union typed parameters across all strict tools.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable

from scout.chat.data import Tables

MAX_ROWS = 10
# scout.scouting.POSITION_GROUPS (a test keeps the two equal; this package
# does not import pandas).
POSITION_GROUPS = {
    "Goalkeeper": "goalkeeper",
    "Centre-Back": "centre_back",
    "Left-Back": "full_back",
    "Right-Back": "full_back",
    "Defensive Midfield": "defensive_mid",
    "Central Midfield": "central_mid",
    "Attacking Midfield": "attacking_mid",
    "Second Striker": "attacking_mid",
    "Left Winger": "winger",
    "Right Winger": "winger",
    "Left Midfield": "winger",
    "Right Midfield": "winger",
    "Centre-Forward": "centre_forward",
}
GROUPS = sorted(set(POSITION_GROUPS.values()))
TABLES = ["candidates", "bargains_prime", "bargains_experienced", "bargains_all"]
FLAG_KEYS = ["low_minutes", "injured", "value_falling"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


# Formatting, as the page does it


def fold(text) -> str:
    """Lower case without accents (the page's ``fold``)."""
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    for a, b in (("ı", "i"), ("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("đ", "d"), ("ł", "l")):
        text = text.replace(a, b)
    return text


def js_round(x: float) -> int:
    """``Math.round``: halves go up, unlike Python's ``round``."""
    return math.floor(x + 0.5)


def money(v) -> str:
    if v is None:
        return "no value"
    if v < 0:
        return "-" + money(-v)
    if v >= 1e6:
        if v >= 1e7:
            return f"€{js_round(v / 1e6)}m"
        return f"€{v / 1e6:.1f}".removesuffix(".0") + "m"
    if v >= 1e3:
        return f"€{js_round(v / 1e3)}k"
    return f"€{js_round(v)}"


def whole(v) -> str:
    return f"{js_round(v):,}"


def day(iso) -> str | None:
    if not iso:
        return None
    y, m, d = str(iso)[:10].split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"


def years(age) -> int | None:
    return None if age is None else math.floor(age)


def chance(pct, players, decile, reason: str) -> str:
    """'30% of 142 past players, decile 10', or why it is blank."""
    if pct is None:
        return reason
    shown = "under 5%" if pct < 5 else f"{js_round(pct)}%"
    tail = f", decile {int(decile)}" if decile is not None else ""
    return f"{shown} of {whole(players)} past players{tail}"


def flag_keys(text, rules: dict[str, str]) -> list[str]:
    return [k for k, rule in rules.items() if isinstance(text, str) and rule in text]


def _so_far(o: dict) -> dict:
    if not o.get("so_far_minutes"):
        return {"as_of": day(o.get("so_far_date")), "minutes": 0, "note": "no minutes yet"}
    out = {
        "as_of": day(o.get("so_far_date")),
        "league": o.get("so_far_league"),
        "matches": int(o["so_far_matches"] or 0),
        "minutes": int(o["so_far_minutes"]),
        "share_of_team_minutes": f"{js_round(o['so_far_minutes_share'] * 100)}%",
        "ability_pct": None
        if o.get("so_far_ability_pct") is None
        else js_round(o["so_far_ability_pct"]),
    }
    if o.get("so_far_goals_per90") is not None:
        out["goals_per90"] = f"{o['so_far_goals_per90']:.2f}"
        out["assists_per90"] = f"{o['so_far_assists_per90']:.2f}"
    out["flags_today"] = o.get("so_far_flags") or "none"
    return out


def _clubs(o: dict) -> dict:
    moved = bool(o.get("moved_since_forecast"))
    return {
        "club_at_forecast": o.get("club_at_forecast"),
        "league": o.get("league"),
        "club_today": o.get("current_club") or "not known",
        "league_today": o.get("current_league") or "not known",
        "moved_since_forecast": moved,
    }


def candidate_view(o: dict, detail: bool = False) -> dict:
    """A candidates row as the page shows it."""
    rank = o.get("value_rank")
    out = {
        "player": o["player"],
        "player_id": o["player_id"],
        "age": years(o.get("age")),
        "position": o.get("position"),
        **_clubs(o),
        "value": money(o.get("value_eur")),
        "value_rank": f"#{whole(rank)}" if rank is not None else "no value rank",
        "fair_value": money(o.get("fair_value_eur")),
        "elite_chance": chance(
            o.get("elite_chance_pct"),
            o.get("elite_chance_players"),
            o.get("elite_decile"),
            "outside the elite pool",
        ),
        "step_up_chance": chance(
            o.get("step_up_chance_pct"),
            o.get("step_up_chance_players"),
            o.get("step_up_decile"),
            "not rated",
        ),
        "big_move_chance": chance(
            o.get("big_move_chance_pct"),
            o.get("big_move_chance_players"),
            o.get("big_move_decile"),
            "at a top 20 club" if rank is not None and rank <= 2000 else "not rated",
        ),
        "fall_back_risk": chance(
            o.get("fall_back_risk_pct"),
            o.get("fall_back_risk_players"),
            None,
            "outside the elite pool",
        ),
        "flags": o.get("flags") or "none",
        "rising": bool(o.get("rising")),
        "ability_pct": None if o.get("ability_pct") is None else js_round(o["ability_pct"]),
    }
    if detail:
        out["explanation"] = o.get("explanation")
        out["season_so_far"] = _so_far(o)
    return out


def bargain_view(o: dict, tags: dict[str, str], detail: bool = False) -> dict:
    """A bargains row as the page shows it."""
    value, fair = o.get("value_eur"), o.get("fair_value_eur")
    cheapest = o.get("value_to_fair_pct")
    out = {
        "list": o.get("list"),
        "list_rank": o.get("list_rank"),
        "player": o["player"],
        "player_id": o["player_id"],
        "age": years(o.get("age")),
        "position": o.get("position"),
        **_clubs(o),
        "value": money(value),
        "value_date": day(o.get("value_date")),
        "fair_value": money(fair),
        "price_against_fair": f"{js_round(o['value_to_fair'] * 100)}% of fair",
        "cheapest_in_position_group": (
            f"cheapest {1 if cheapest < 1 else js_round(cheapest)}% of his position group"
            if cheapest is not None
            else None
        ),
        "fair_minus_value": money(fair - value) if value is not None and fair is not None else None,
        "ability_pct": None if o.get("ability_pct") is None else js_round(o["ability_pct"]),
        "minutes": f"{js_round(o['minutes_share'] * 100)}% of team minutes",
        "contract_until": day(o.get("contract_expiry")) or "not known",
        "flags": o.get("flags") or "none",
        "tags": [tags[k] for k in ("on_loan", "stale_value") if o.get(k) and k in tags],
        "rising": bool(o.get("rising")),
    }
    if detail:
        out["explanation"] = o.get("explanation")
        out["season_so_far"] = _so_far(o)
    return out


# search_players

SORTS: dict[str, tuple[Callable[[dict], float | None], bool, str]] = {
    # name: (value, descending, what it means)
    "table_order": (lambda o: o["_i"], False, "the page's own order"),
    "youngest": (lambda o: o.get("age"), False, "youngest first"),
    "oldest": (lambda o: o.get("age"), True, "oldest first"),
    "value_highest": (lambda o: o.get("value_eur"), True, "highest market value first"),
    "value_lowest": (lambda o: o.get("value_eur"), False, "lowest market value first"),
    "best_value_rank": (lambda o: o.get("value_rank"), False, "best value rank first"),
    "fair_value_highest": (lambda o: o.get("fair_value_eur"), True, "highest fair value first"),
    "cheapest_against_fair": (
        lambda o: _ratio(o),
        False,
        "lowest value against fair value first",
    ),
    "biggest_gap_eur": (lambda o: _gap(o), True, "largest fair value minus value first"),
    "elite_chance": (
        lambda o: _bucket(o, "elite_chance_pct", "elite_decile"),
        True,
        "highest elite chance first",
    ),
    "step_up_chance": (
        lambda o: _bucket(o, "step_up_chance_pct", "step_up_decile"),
        True,
        "highest step up chance first",
    ),
    "big_move_chance": (
        lambda o: _bucket(o, "big_move_chance_pct", "big_move_decile"),
        True,
        "highest big move chance first",
    ),
    "fall_back_risk": (lambda o: o.get("fall_back_risk_pct"), True, "highest fall back risk first"),
    "ability": (lambda o: o.get("ability_pct"), True, "highest ability percentile first"),
    "so_far_minutes": (lambda o: o.get("so_far_minutes"), True, "most minutes this season first"),
}


def _ratio(o: dict) -> float | None:
    if o.get("value_to_fair") is not None:
        return o["value_to_fair"]
    value, fair = o.get("value_eur"), o.get("fair_value_eur")
    return value / fair if value is not None and fair else None


def _gap(o: dict) -> float | None:
    value, fair = o.get("value_eur"), o.get("fair_value_eur")
    return fair - value if value is not None and fair is not None else None


def _bucket(o: dict, pct: str, decile: str) -> float | None:
    """The page's chance sort: the shown rate, then the decile."""
    if o.get(pct) is None:
        return None
    return o[pct] * 100 + (o.get(decile) or 0)


def _table(t: Tables, name: str) -> list[dict]:
    if name == "candidates":
        rows = t.candidates
    elif name == "bargains_all":
        rows = sorted(t.bargains, key=lambda o: (o["list"] != "prime", o["list_rank"]))
    else:
        rows = [o for o in t.bargains if o["list"] == name.removeprefix("bargains_")]
        rows = sorted(rows, key=lambda o: o["list_rank"])
    return [{**o, "_i": i} for i, o in enumerate(rows)]


def _sorted(rows: list[dict], how: str) -> list[dict]:
    value, desc, _ = SORTS[how]
    known = [o for o in rows if value(o) is not None]
    blank = [o for o in rows if value(o) is None]
    known.sort(key=lambda o: ((-value(o) if desc else value(o)), o["_i"]))
    return known + blank


def search_players(t: Tables, args: dict) -> dict:
    table = args["table"]
    if table not in TABLES:
        return {"error": f"table must be one of {TABLES}"}
    how = args.get("sort_by", "table_order")
    if how not in SORTS:
        return {"error": f"sort_by must be one of {sorted(SORTS)}"}
    limit = max(1, min(MAX_ROWS, int(args.get("limit") or MAX_ROWS)))
    rules = t.meta.get("flags", {})
    rows = _table(t, table)
    applied = {}

    def keep(name: str, test: Callable[[dict], bool]) -> None:
        nonlocal rows
        rows = [o for o in rows if test(o)]
        applied[name] = args[name]

    if args.get("position"):
        keep("position", lambda o: o.get("position") == args["position"])
    if args.get("position_group"):
        keep(
            "position_group",
            lambda o: POSITION_GROUPS.get(o.get("position")) == args["position_group"],
        )
    if args.get("age_min") is not None:
        keep("age_min", lambda o: o.get("age") is not None and o["age"] >= args["age_min"])
    if args.get("age_max") is not None:
        keep(
            "age_max",
            lambda o: o.get("age") is not None and math.floor(o["age"]) <= args["age_max"],
        )
    if args.get("league"):
        keep("league", lambda o: fold(o.get("league")) == fold(args["league"]))
    if args.get("club"):
        q = fold(args["club"])
        keep(
            "club",
            lambda o: q in fold(o.get("club_at_forecast")) or q in fold(o.get("current_club")),
        )
    if args.get("flags"):
        keep("flags", lambda o: set(args["flags"]) <= set(flag_keys(o.get("flags"), rules)))
    if args.get("max_flag_count") is not None:
        keep("max_flag_count", lambda o: (o.get("flag_count") or 0) <= args["max_flag_count"])
    if args.get("rising") is not None:
        keep("rising", lambda o: bool(o.get("rising")) == args["rising"])
    if args.get("moved_since_forecast") is not None:
        keep(
            "moved_since_forecast",
            lambda o: bool(o.get("moved_since_forecast")) == args["moved_since_forecast"],
        )
    for name, column in (
        ("min_ability_pct", "ability_pct"),
        ("min_elite_chance_pct", "elite_chance_pct"),
        ("min_step_up_chance_pct", "step_up_chance_pct"),
    ):
        if args.get(name) is not None:
            keep(name, lambda o, c=column, n=name: o.get(c) is not None and o[c] >= args[n])
    ordered = _sorted(rows, how)
    tags = t.meta.get("tags", {})
    view = (
        (lambda o: candidate_view(o))
        if table == "candidates"
        else (lambda o: bargain_view(o, tags))
    )
    shown = ordered[:limit]
    out = {
        "table": table,
        "filters": applied,
        "sort": SORTS[how][2],
        "matched": len(ordered),
        "shown": len(shown),
        "rows": [view(o) for o in shown],
    }
    value = SORTS[how][0]
    if how != "table_order" and len(ordered) > len(shown) and shown:
        last = value(shown[-1])
        if last is not None and value(ordered[len(shown)]) == last:
            out["note"] = "the last row shown ties on this sort with the next one not shown"
    top = [value(o) for o in shown[:2]]
    if how != "table_order" and len(top) == 2 and top[0] is not None and top[0] == top[1]:
        out["tie_at_top"] = True
    if not ordered:
        out["note"] = "no player in this table matches these filters"
    return out


# Finding a player by name


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[\s.\-']+", fold(text)) if w]


def name_score(query: str, name: str) -> int:
    """3 exact, 2 every word of the query in the name, 1 initial and surname
    ('Kenan Yildiz' for 'K. Yıldız', or the other way round), else 0."""
    q, n = fold(query).strip(), fold(name).strip()
    if not q or not n:
        return 0
    if q == n:
        return 3
    qt, nt = _tokens(query), _tokens(name)
    if qt and all(w in nt for w in qt):
        return 2
    if len(qt) > 1 and len(nt) > 1 and qt[-1] == nt[-1] and qt[0][0] == nt[0][0]:
        return 1
    return 0


def people(t: Tables) -> dict[int, dict]:
    """Every player the page shows, by id: in the tables, today's similar
    players and the earlier seasons once in the top 500 (built once per
    load)."""
    if t.people is None:
        t.people = _people(t)
    return t.people


def _people(t: Tables) -> dict[int, dict]:
    people: dict[int, dict] = {}
    for table, rows in (("candidates", t.candidates), ("bargains", t.bargains)):
        for o in rows:
            p = people.setdefault(o["player_id"], {"id": o["player_id"], "name": o["player"]})
            p.setdefault(table, o)
            p.setdefault("club", o.get("current_club") or o.get("club_at_forecast"))
    for o in t.similar.today:
        p = people.setdefault(o["id"], {"id": o["id"], "name": o["name"]})
        p["today"] = o
        p.setdefault("club", o.get("club_now_name") or o.get("club_name"))
    for o in t.similar.history:
        p = people.setdefault(o["id"], {"id": o["id"], "name": o["name"]})
        p.setdefault("seasons", []).append(o)
        p.setdefault("club", o.get("club_name"))
    return people


def find(t: Tables, name: str, club: str | None = None, pool=None) -> list[dict]:
    """The best matching players, the most valuable first."""
    everyone = pool if pool is not None else people(t)
    scored = [(name_score(name, p["name"]), p) for p in everyone.values()]
    best = max((s for s, _ in scored), default=0)
    if best == 0:
        return []
    found = [p for s, p in scored if s == best]
    if club:
        q = fold(club)
        narrowed = [p for p in found if any(q in fold(c) for c in _club_names(p))]
        found = narrowed or found
    return sorted(found, key=lambda p: (_rank(p), p["name"]))


def _club_names(p: dict) -> list[str]:
    names = [p.get("club")]
    for key in ("candidates", "bargains"):
        if key in p:
            names += [p[key].get("club_at_forecast"), p[key].get("current_club")]
    if "today" in p:
        names += [p["today"].get("club_name"), p["today"].get("club_now_name")]
    return [n for n in names if n]


def _rank(p: dict) -> float:
    for key in ("candidates", "bargains"):
        if key in p and p[key].get("value_rank") is not None:
            return p[key]["value_rank"]
    if "today" in p and p["today"].get("value_rank") is not None:
        return p["today"]["value_rank"]
    return 1e9


def _choices(found: list[dict]) -> dict:
    return {
        "matches": [
            {
                "player": p["name"],
                "club": p.get("club"),
                "in_tables": [k for k in ("candidates", "bargains") if k in p],
            }
            for p in found[:MAX_ROWS]
        ],
        "note": "several players match this name; ask which one, naming the club",
    }


# player_card


def report_of(t: Tables, player_id: int) -> dict:
    r = t.reports
    row = (r.get("rows") or {}).get(str(player_id)) if r.get("available") else None
    if not row:
        return {"kind": "none", "note": "no written report for him"}
    if row[0] == "report":
        return {"kind": "written report", "text": row[1], "label": r.get("label")}
    return {
        "kind": "rule",
        "note": "no checked report for these facts this week; the explanation written by "
        "rule is shown instead",
        "text": row[1],
    }


def player_card(t: Tables, args: dict) -> dict:
    found = find(t, args["name"], args.get("club"))
    greats = sorted({g["great"] for g in t.similar.greats if name_score(args["name"], g["great"])})
    if not found:
        if greats:
            return {
                "found": False,
                "measured_greats": greats,
                "note": "not in the tables; he is one of the measured greats, so only "
                "similar_players (mode at_age) can compare today's players with him",
            }
        return {"found": False, "note": "no player of that name is held on this page"}
    if len(found) > 1:
        return _choices(found)
    p = found[0]
    tags = t.meta.get("tags", {})
    out: dict = {"found": True, "player": p["name"], "player_id": p["id"]}
    out["in_tables"] = [k for k in ("candidates", "bargains") if k in p]
    if "candidates" in p:
        out["candidate"] = candidate_view(p["candidates"], detail=True)
    if "bargains" in p:
        out["bargain"] = bargain_view(p["bargains"], tags, detail=True)
    if out["in_tables"]:
        out["report"] = report_of(t, p["id"])
    else:
        o = p.get("today") or p["seasons"][-1]
        out["note"] = (
            "not in the candidates or bargains tables; the page holds him only in the "
            "similar players data"
        )
        out["similar_data"] = _sim_view(o, t.similar)
    if greats:
        out["measured_greats"] = greats
    return out


# similar_players


def similarity(a: dict, b: dict, scale: int) -> float:
    return sum(x * y for x, y in zip(a["_v"], b["_v"], strict=True)) / (scale * scale)


def _nearest(q: dict, pool: list[dict], scale: int, k: int = MAX_ROWS) -> list[tuple]:
    scored = [(similarity(q, o, scale), i, o) for i, o in enumerate(pool)]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(s, o) for s, _, o in scored[:k]]


def season_label(season: int) -> str:
    return f"{season}/{str(season + 1)[-2:]}"


def _club_of(o: dict) -> str:
    now, club = o.get("club_now_name"), o.get("club_name") or ""
    return f"{now} (now; was {club})" if now else club


def _sim_view(o: dict, sim, age_key: str = "age") -> dict:
    return {
        "player": o["name"],
        "season": season_label(o["season"]),
        "age": math.floor(o[age_key]),
        "position": o.get("position"),
        "position_group": o.get("group_name"),
        "club": _club_of(o),
        "league": o.get("league_name"),
        "ability_pct": o.get("ability"),
        "value_rank": f"#{whole(o['value_rank'])}" if o.get("value_rank") else "no value rank",
    }


def _elite_line(t: Tables, player_id: int) -> str:
    c = next((o for o in t.candidates if o["player_id"] == player_id), None)
    if c is None:
        return "not rated (not in the candidates table)"
    if c.get("elite_chance_pct") is None:
        return "outside the elite pool"
    return chance(c["elite_chance_pct"], c["elite_chance_players"], c["elite_decile"], "")


def _listed(found, scale, t: Tables, age_key: str = "age", elite: bool = False) -> list[dict]:
    out = []
    for s, o in found:
        item = {
            "player": o["name"],
            "position": o.get("position"),
            "age": math.floor(o[age_key]),
            "club": _club_of(o),
            "league": o.get("league_name"),
            "ability_pct": o.get("ability"),
            "similarity": f"{s:.2f}",
        }
        if elite:
            item["elite_chance"] = _elite_line(t, o["id"])
        out.append(item)
    return out


def similar_players(t: Tables, args: dict) -> dict:
    sim = t.similar
    if not sim.available:
        return {"error": "the similar players data is not built yet"}
    tested = sim.raw.get("tested") or {}
    caveat = (
        "Similar means plays the same way, not as good: style from box score stats with the "
        "level taken out."
    )
    if tested.get("learned") is not None:
        caveat += (
            f" Tested on {tested['test_seasons'][0]} to {tested['test_seasons'][-1]}, the same "
            f"player's other season is among his five nearest {js_round(tested['learned'] * 100)}%"
            " of the time, so treat the list as a shortlist."
        )
    mode = args["mode"]
    if mode == "at_age":
        if args.get("age") is None:
            return {"error": "mode at_age needs the age to compare at"}
        great = _great_match(sim, args["name"])
        if great:
            return {**_like_great(t, great, int(args["age"])), "caveat": caveat}
    everyone = people(t)
    if mode == "now":
        pool = {pid: p for pid, p in everyone.items() if "today" in p}
    else:
        pool = {pid: p for pid, p in everyone.items() if "today" in p or "seasons" in p}
    found = find(t, args["name"], args.get("club"), pool)
    if not found:
        held = find(t, args["name"], args.get("club"))
        if held:
            return {
                "found": False,
                "note": "he has no style vector this season (it needs 450 league minutes "
                "with match statistics); try mode at_age",
            }
        return {"found": False, "note": "no player of that name is held on this page"}
    if len(found) > 1:
        return _choices(found)
    p = found[0]
    if mode == "now":
        q = p["today"]
        rows = []
        for o in sim.today:
            if o["id"] == q["id"] or o["group"] != q["group"]:
                continue
            a = math.floor(o["age"])
            if args.get("age_min") is not None and a < args["age_min"]:
                continue
            if args.get("age_max") is not None and a > args["age_max"]:
                continue
            if args.get("league") and fold(o.get("league_name")) != fold(args["league"]):
                continue
            if args.get("min_ability_pct") is not None and o["ability"] < args["min_ability_pct"]:
                continue
            rows.append(o)
        found_rows = _nearest(q, rows, sim.scale)
        return {
            "query": _sim_view(q, sim),
            "pool": "today's players of his position group"
            + ("" if rows else "; none match these filters"),
            "nearest": _listed(found_rows, sim.scale, t),
            "caveat": caveat,
        }
    age = int(args["age"])
    seasons = [*p.get("seasons", []), *([p["today"]] if "today" in p else [])]
    q = sorted(seasons, key=lambda o: (abs(math.floor(o["age"]) - age), o["season"]))[0]
    rows = [
        o
        for o in sim.today
        if o["id"] != p["id"]
        and o["group"] == q["group"]
        and age - 1 <= math.floor(o["age"]) <= age + 1
    ]
    out = {
        "query": _sim_view(q, sim),
        "pool": f"today's players of his position group aged {age - 1} to {age + 1}",
        "nearest": _listed(_nearest(q, rows, sim.scale), sim.scale, t),
        "caveat": caveat,
    }
    if math.floor(q["age"]) != age:
        out["note"] = f"he has no season at {age}; the nearest held is at {math.floor(q['age'])}"
    return out


def _great_match(sim, name: str) -> str | None:
    names = sorted({g["great"] for g in sim.greats})
    scored = [(name_score(name, n), n) for n in names]
    best = max((s for s, _ in scored), default=0)
    hits = [n for s, n in scored if s == best and s >= 2]
    return hits[0] if len(hits) == 1 else None


def _like_great(t: Tables, great: str, age: int) -> dict:
    sim = t.similar
    ages = [g for g in sim.greats if g["great"] == great]
    q = next((g for g in ages if g["age"] == age), None)
    held = [
        f"{g['age']} ({season_label(g['season'])} {g['club']}"
        + (", StatsBomb" if g["source"] == "statsbomb" else "")
        + (f", thin: {whole(g['minutes'])} min" if g["thin"] else "")
        + ")"
        for g in ages
    ]
    if q is None:
        return {"found": False, "note": f"no season of {great} at {age}", "ages_held": held}
    groups = sim.raw["groups"]
    pool = [
        o
        for o in sim.today
        if groups[o["group"]] == q["group"] and math.floor(o["age_mid"]) == q["age"]
    ]
    out = {
        "query": {
            "great": great,
            "season": season_label(q["season"]),
            "age": q["age"],
            "club": q["club"],
            "league": q["league"],
            "position_group": q["group"],
            "league_minutes": whole(q["minutes"]),
            "ability_pct_among_peers": q["ability"],
            "source": "StatsBomb data" if q["source"] == "statsbomb" else "our own data",
            "peer_group": q["peers"],
        },
        "pool": f"today's players of his position group aged {q['age']} in the middle of the "
        "season",
        "nearest": _listed(_nearest(q, pool, sim.scale), sim.scale, t, "age_mid", elite=True),
        "greats_label": (sim.raw.get("greats") or {}).get("label"),
    }
    if q["thin"]:
        out["note"] = (
            f"a thin season ({whole(q['minutes'])} minutes): his vector is mostly the position "
            "average, so these neighbours say little"
        )
    return out


# how_to_read


def how_to_read(t: Tables, args: dict, limits: dict[str, str], topics: dict[str, list[str]]):
    want = fold(args["column"]).strip().replace(" ", "_")
    table = args.get("table")
    entries = [
        r
        for r in t.dictionary
        if (table is None or r["table"] == table) and fold(r["column"]) == want
    ]
    if not entries:
        entries = [
            r
            for r in t.dictionary
            if (table is None or r["table"] == table)
            and (want in fold(r["column"]) or fold(r["column"]) in want)
        ][:6]
    keys = set()
    for r in entries:
        keys.update(topics.get(r["column"], []))
    for topic in limits:
        if topic in want or want in topic:
            keys.add(topic)
    out = {
        "entries": [
            {"table": r["table"], "column": r["column"], "meaning": r["meaning"]} for r in entries
        ],
        "limits": [limits[k] for k in sorted(keys)],
    }
    if not entries and not keys:
        out["note"] = "no column or topic of that name"
        out["topics"] = sorted(limits)
        out["columns"] = sorted({r["column"] for r in t.dictionary})
    return out


# The definitions sent to the model


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def definitions(t: Tables) -> list[dict]:
    """The four tool definitions, strict; the league list comes from the data
    and is sorted, so the definitions (and the prompt cache) stay the same
    until the tables change."""
    leagues = sorted(
        {o["league"] for o in t.candidates + t.bargains if o.get("league")}
        | set(t.similar.raw.get("leagues", []))
    )
    rules = t.meta.get("flags", {})
    flag_text = "; ".join(f"{k}: {v}" for k, v in rules.items())
    return [
        {
            "name": "search_players",
            "description": (
                "Filter and sort one of the page's tables and return at most 10 rows, each with "
                "the facts the page shows, plus 'matched', the number of players that pass the "
                "filters (use it for 'how many' questions). Tables: candidates (players ranked "
                "#101 to #2,000 by value with elite, step up, big move chances and fall back "
                "risk), bargains_prime (29 and under), bargains_experienced (30 to 33), "
                "bargains_all (both lists). League is the league of the season the tables "
                "describe; age_min compares the exact age, age_max whole years. Flags: "
                f"{flag_text}."
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "table": {"type": "string", "enum": TABLES},
                    "sort_by": {
                        "type": "string",
                        "enum": list(SORTS),
                        "description": "; ".join(f"{k}: {v[2]}" for k, v in SORTS.items()),
                    },
                    "limit": {"type": "integer", "description": "rows to return, 1 to 10"},
                    "position": {"type": "string", "enum": sorted(POSITION_GROUPS)},
                    "position_group": {
                        "type": "string",
                        "enum": GROUPS,
                        "description": "broad group: winger covers left and right wingers and "
                        "midfielders, full_back both full backs, attacking_mid the second striker",
                    },
                    "age_min": {"type": "integer"},
                    "age_max": {"type": "integer"},
                    "league": {"type": "string", "enum": leagues},
                    "club": {"type": "string", "description": "part of a club name"},
                    "flags": {
                        "type": "array",
                        "items": {"type": "string", "enum": FLAG_KEYS},
                        "description": "players carrying every flag listed",
                    },
                    "max_flag_count": {"type": "integer"},
                    "rising": {"type": "boolean"},
                    "moved_since_forecast": {"type": "boolean"},
                    "min_ability_pct": {"type": "number"},
                    "min_elite_chance_pct": {"type": "number"},
                    "min_step_up_chance_pct": {"type": "number"},
                },
                ["table", "sort_by", "limit"],
            ),
        },
        {
            "name": "player_card",
            "description": (
                "Every fact the page shows for one player found by name (accents and initials "
                "are matched loosely): his rows in candidates and bargains with chances, flags, "
                "value, fair value, ability, club today, the season so far, the explanation and "
                "his written scout report. A player outside both tables but in the similar "
                "players data gets what that data holds. Several matches come back as a list."
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "name": {"type": "string"},
                    "club": {"type": "string", "description": "to tell apart players of one name"},
                },
                ["name"],
            ),
        },
        {
            "name": "similar_players",
            "description": (
                "Players who play most like someone, from the page's style vectors (level "
                "taken out). mode now: the ten nearest of his position group today, with "
                "optional age, league and ability filters. mode at_age: today's players a year "
                "either side of that age nearest to his season at that age; for a measured "
                "great (Lionel Messi, Xavi, Thierry Henry and others) today's players of that "
                "age in the middle of the season, with their elite chance."
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "name": {"type": "string"},
                    "mode": {"type": "string", "enum": ["now", "at_age"]},
                    "age": {"type": "integer", "description": "mode at_age: the age to compare at"},
                    "age_min": {"type": "integer"},
                    "age_max": {"type": "integer"},
                    "league": {"type": "string", "enum": leagues},
                    "min_ability_pct": {"type": "number"},
                    "club": {"type": "string", "description": "to tell apart players of one name"},
                },
                ["name", "mode"],
            ),
        },
        {
            "name": "how_to_read",
            "description": (
                "What a column or topic of the page means (its dictionary entry) and what it "
                "cannot tell: how the chances were measured, their limits, the dates. Topics: "
                "elite_chance, step_up_chance, big_move_chance, fall_back_risk, flags, value, "
                "fair_value, ability, bargains, rising, season_so_far, similar, greats, reports, "
                "explanation, dates."
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "column": {"type": "string", "description": "a column name or a topic"},
                    "table": {"type": "string", "enum": ["candidates", "bargains", "reports"]},
                },
                ["column"],
            ),
        },
    ]
