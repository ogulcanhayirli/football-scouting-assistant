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
  and the number of players that matched; the players table (prompt 20f)
  holds every player of the season, the two lists only theirs. Over the
  bargains it keeps only the players priced below fair value, the
  bargains, unless ``include_above_fair`` asks for the whole ranking
  (prompt 20g);
* ``player_card``: every fact the page shows for one player, the written
  report included; found in the candidates, then the bargains, then the
  players table, whose row says in sentences set by code why he is on no
  list (``scout.players``); a player on the bargains ranking priced above
  his fair value gets ``bargain_verdict``, the code's sentence that he is
  not a bargain (``not_a_bargain``);
* ``similar_players``: the ten nearest of his position group today, or of
  today's players to him (or to a measured great) at an age, or to the
  archetype of a legend drafted from the record (prompt 20d);
* ``how_to_read``: the dictionary entry of a column and what it cannot say.

Both player tools read a name through ``NICKNAMES`` first (prompt 20d), so
R9, CR7, Zizou or La Pulga reach the player, great or archetype the page
holds, and an answer about a great or an archetype carries its label
(``GREATS_LABEL``, ``ARCHETYPE_LABEL``). A name nobody holds is then read
loosely (``fuzzy_score``: one or two letters off in each word), and the
result says so (prompt 20f).

Every schema is strict (``additionalProperties: false``), with optional
fields left out of ``required`` rather than nullable, under the API's limit
of 24 optional and 16 union typed parameters across all strict tools.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher

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
TABLES = ["players", "candidates", "bargains_prime", "bargains_experienced", "bargains_all"]
FUZZY = 0.8  # a misspelt word must keep this much of the held one (difflib ratio)
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
# The labels a comparison with a legend carries into the answer (prompt 20d).
ARCHETYPE_LABEL = "archetype from the record, not measured"
GREATS_LABEL = "measured, cross era comparison is rough"
# Nicknames and the obvious surnames of the greats and the legends, folded
# (``fold``), against the name the page holds them under. A whole question
# name is looked up, never a part of it, so "R. Messi" stays himself. A
# surname shared with a player of today (Maldini, Zidane) goes to the
# legend; the tools then name the players of today it also matches.
NICKNAMES = {
    "r9": "Ronaldo Nazário",
    "il fenomeno": "Ronaldo Nazário",
    "o fenomeno": "Ronaldo Nazário",
    "fenomeno": "Ronaldo Nazário",
    "the phenomenon": "Ronaldo Nazário",
    "ronaldo nazario": "Ronaldo Nazário",
    "ronaldo r9": "Ronaldo Nazário",
    "brazilian ronaldo": "Ronaldo Nazário",
    "cr7": "Cristiano Ronaldo",
    "cristiano": "Cristiano Ronaldo",
    "la pulga": "Lionel Messi",
    "pulga": "Lionel Messi",
    "leo messi": "Lionel Messi",
    "messi": "Lionel Messi",
    "zizou": "Zinedine Zidane",
    "zidane": "Zinedine Zidane",
    "il maestro": "Andrea Pirlo",
    "pirlo": "Andrea Pirlo",
    "rio": "Rio Ferdinand",
    "ferdinand": "Rio Ferdinand",
    "maldini": "Paolo Maldini",
    "baresi": "Franco Baresi",
    "cruyff": "Johan Cruyff",
    "cruijff": "Johan Cruyff",
    "gullit": "Ruud Gullit",
    "xavi hernandez": "Xavi",
    "iniesta": "Andrés Iniesta",
    "henry": "Thierry Henry",
    "titi": "Thierry Henry",
    "puyol": "Carles Puyol",
    "pique": "Gerard Piqué",
    "busquets": "Sergio Busquets",
    "busi": "Sergio Busquets",
    "buffon": "Gianluigi Buffon",
    "gigi": "Gianluigi Buffon",
    "gigi buffon": "Gianluigi Buffon",
    "chiellini": "Giorgio Chiellini",
    "kante": "N'Golo Kanté",
    "modric": "Luka Modrić",
    "kroos": "Toni Kroos",
    "de bruyne": "Kevin De Bruyne",
    "kdb": "Kevin De Bruyne",
    "lewandowski": "Robert Lewandowski",
    "lewy": "Robert Lewandowski",
    "suarez": "Luis Suárez",
    "el pistolero": "Luis Suárez",
    "bergkamp": "Dennis Bergkamp",
    "the iceman": "Dennis Bergkamp",
    "vieira": "Patrick Vieira",
    "pires": "Robert Pirès",
    "van dijk": "Virgil van Dijk",
    "vvd": "Virgil van Dijk",
    "sergio ramos": "Sergio Ramos",
    "dani alves": "Dani Alves",
    "ney": "Neymar",
    # A surname two players of today share, read as the one a fan means; the
    # card names the other under also_on_page (prompt 20g: E. Mbappé).
    "mbappe": "Kylian Mbappé",
    "kylian": "Kylian Mbappé",
}


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
    # The unit follows the rounded number: 999,999.9999 (float noise in 52
    # bargains' values) is €1m, never €1000k, and 999.6 is €1k.
    if v >= 1e6 or js_round(v / 1e3) >= 1000:
        if v >= 1e7:
            return f"€{js_round(v / 1e6)}m"
        return f"€{v / 1e6:.1f}".removesuffix(".0") + "m"
    if js_round(v) >= 1000:
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


def below_fair(o: dict) -> bool:
    """A bargain: a player of the bargains ranking priced below his fair value
    (prompt 20g); a page published before the flag existed falls back to the
    ratio it was set from."""
    flag = o.get("below_fair_value")
    return bool(flag) if flag is not None else o["value_to_fair"] < 1


def not_a_bargain(o: dict) -> str:
    """The code's sentence for a player of the bargains ranking priced above
    his fair value, to be said as written."""
    return (
        f"No: {o['player']} is on the value ranking at #{whole(o['list_rank'])} "
        f"({o.get('list')} list) but priced above fair value, so not a bargain."
    )


def bargain_view(o: dict, tags: dict[str, str], detail: bool = False) -> dict:
    """A bargains row as the page shows it."""
    value, fair = o.get("value_eur"), o.get("fair_value_eur")
    cheapest = o.get("value_to_fair_pct")
    out = {
        "bargain": "yes: priced below his fair value"
        if below_fair(o)
        else "no: priced above his fair value, so on the ranking but not a bargain",
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


def player_view(o: dict, detail: bool = False) -> dict:
    """A players table row (prompt 20f) as the chat gives it: blank chances
    with their reason, and the code's sentence for each list he is not on."""
    rank = o.get("value_rank")
    out = {
        "player": o["player"],
        "player_id": o["player_id"],
        "age": years(o.get("age")),
        "nationality": o.get("nationality"),
        "position": o.get("position"),
        "club": o.get("club"),
        "league": o.get("league"),
        "league_minutes": whole(o["minutes"]) if o.get("minutes") is not None else None,
        "share_of_team_minutes": None
        if o.get("minutes_share") is None
        else f"{js_round(o['minutes_share'] * 100)}%",
        "ability_pct": None if o.get("ability_pct") is None else js_round(o["ability_pct"]),
        "value": money(o.get("value_eur")),
        "value_rank": f"#{whole(rank)}" if rank is not None else "no value rank",
        "fair_value": money(o.get("fair_value_eur"))
        if o.get("fair_value_eur") is not None
        else "no fair value (regular players with a value only)",
        "price_against_fair": f"{js_round(o['value_to_fair'] * 100)}% of fair"
        if o.get("value_to_fair") is not None
        else None,
        "elite_chance": chance(
            o.get("elite_chance_pct"),
            o.get("elite_chance_players"),
            o.get("elite_decile"),
            f"not computed: {o.get('elite_chance_why_blank')}",
        ),
        "big_move_chance": chance(
            o.get("big_move_chance_pct"),
            o.get("big_move_chance_players"),
            o.get("big_move_decile"),
            f"not computed: {o.get('big_move_chance_why_blank')}",
        ),
        "flags": o.get("flags") or "none",
        "on_lists": _on_lists(o),
    }
    if detail:
        out["ability_score"] = o.get("ability_score")
        out["value_date"] = day(o.get("value_date"))
        out["described_on"] = day(o.get("data_date"))
    return out


def _on_lists(o: dict) -> list[str] | str:
    out = ["candidates"] if o.get("in_candidates") else []
    if o.get("in_bargains"):
        ratio = o.get("value_to_fair")
        above = ratio is not None and ratio >= 1
        out.append(
            "bargains ranking, priced above fair value: not a bargain" if above else "bargains"
        )
    return out or "none"


def why_not(o: dict) -> list[str]:
    """The code's sentences for the lists a player is not on."""
    return [o[k] for k in ("why_not_candidate", "why_not_bargain") if o.get(k)]


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
    "minutes": (lambda o: o.get("minutes"), True, "most league minutes in the season first"),
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
    if name == "players":
        rows = t.players
    elif name == "candidates":
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
    table = args.get("table") or "players"
    if table not in TABLES:
        return {"error": f"table must be one of {TABLES}"}
    how = args.get("sort_by", "table_order")
    if how not in SORTS:
        return {"error": f"sort_by must be one of {sorted(SORTS)}"}
    limit = max(1, min(MAX_ROWS, int(args.get("limit") or MAX_ROWS)))
    rules = t.meta.get("flags", {})
    rows = _table(t, table)
    applied = {}
    bargains = table.startswith("bargains")
    above = 0

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
            lambda o: (
                q in fold(o.get("club_at_forecast"))
                or q in fold(o.get("current_club"))
                or q in fold(o.get("club"))
            ),
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
        ("min_minutes", "minutes"),
        ("min_ability_pct", "ability_pct"),
        ("min_elite_chance_pct", "elite_chance_pct"),
        ("min_step_up_chance_pct", "step_up_chance_pct"),
    ):
        if args.get(name) is not None:
            keep(name, lambda o, c=column, n=name: o.get(c) is not None and o[c] >= args[n])
    if bargains and not args.get("include_above_fair"):
        # Bargains are the players priced below fair value (prompt 20g); the rest
        # of the ranking only when asked for.
        above = sum(1 for o in rows if not below_fair(o))
        rows = [o for o in rows if below_fair(o)]
    elif bargains:
        applied["include_above_fair"] = True
    ordered = _sorted(rows, how)
    tags = t.meta.get("tags", {})

    def view(o: dict) -> dict:
        if table == "players":
            return player_view(o)
        return candidate_view(o) if table == "candidates" else bargain_view(o, tags)

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
    if bargains and not args.get("include_above_fair"):
        out["bargains_only"] = (
            f"only bargains, the players priced below fair value; {whole(above)} more of the "
            "ranking match these filters but are priced above fair value (not bargains; "
            "include_above_fair true shows the whole ranking)"
        )
    elif bargains:
        out["whole_ranking"] = (
            "the whole ranking by value against fair value; only rows with bargain yes, priced "
            "below fair value, are bargains"
        )
    if not ordered:
        out["note"] = "no player in this table matches these filters"
        if table != "players":
            out["note"] += "; the players table holds every player of the season"
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


def fuzzy_score(query: str, name: str, nt: list[str] | None = None) -> float:
    """How well a misspelt name fits a held one, 0 when it does not: every
    word of the query must be a held word one or two letters off (difflib
    ratio of ``FUZZY`` or more), the last ones (the surnames) included, or
    the first letter of a held initial ('Erling Halland' fits 'E. Haaland').
    ``nt``: the held name's words, when already split."""
    qt, nt = _tokens(query), nt if nt is not None else _tokens(name)
    if not qt or not nt or len(qt) > len(nt):
        return 0.0
    surname = _spelling(qt[-1], nt[-1])
    if surname < FUZZY:
        return 0.0
    scores = [surname]
    for w in qt[:-1]:
        best = max((1.0 if len(n) == 1 and w[0] == n else _spelling(w, n)) for n in nt[:-1] or nt)
        if best < FUZZY:
            return 0.0
        scores.append(best)
    return sum(scores) / len(scores)


def _spelling(a: str, b: str) -> float:
    """difflib's ratio of two words, 0 below ``FUZZY`` (cheap bounds first)."""
    if a == b:
        return 1.0
    if abs(len(a) - len(b)) > 2 or len(a) < 4:
        return 0.0
    m = SequenceMatcher(None, a, b)
    return m.ratio() if m.real_quick_ratio() >= FUZZY and m.quick_ratio() >= FUZZY else 0.0


def people(t: Tables) -> dict[int, dict]:
    """Every player the page shows, by id: in the tables, the players table,
    today's similar players and the earlier seasons once in the top 500
    (built once per load)."""
    if t.people is None:
        t.people = _people(t)
    return t.people


def _people(t: Tables) -> dict[int, dict]:
    people = _held(t)
    for p in people.values():
        p["words"] = _tokens(p["name"])  # for fuzzy_score, split once
    return people


def _held(t: Tables) -> dict[int, dict]:
    people: dict[int, dict] = {}
    for table, rows in (("candidates", t.candidates), ("bargains", t.bargains)):
        for o in rows:
            p = people.setdefault(o["player_id"], {"id": o["player_id"], "name": o["player"]})
            p.setdefault(table, o)
            p.setdefault("club", o.get("current_club") or o.get("club_at_forecast"))
    for o in t.players:
        p = people.setdefault(o["player_id"], {"id": o["player_id"], "name": o["player"]})
        p["players"] = o
        p.setdefault("club", o.get("club"))
    for o in t.similar.today:
        p = people.setdefault(o["id"], {"id": o["id"], "name": o["name"]})
        p["today"] = o
        p.setdefault("club", o.get("club_now_name") or o.get("club_name"))
    for o in t.similar.history:
        p = people.setdefault(o["id"], {"id": o["id"], "name": o["name"]})
        p.setdefault("seasons", []).append(o)
        p.setdefault("club", o.get("club_name"))
    return people


def find(
    t: Tables, name: str, club: str | None = None, pool=None, loose: bool = False
) -> list[dict]:
    """The best matching players, the most valuable first; with ``loose``,
    a name no one holds is read as the closest misspelling (``fuzzy_score``),
    and the players found carry ``loose: True``."""
    everyone = pool if pool is not None else people(t)
    scored = [(name_score(name, p["name"]), p) for p in everyone.values()]
    best = max((s for s, _ in scored), default=0)
    fuzzy = best == 0 and loose
    if fuzzy:
        scored = [(fuzzy_score(name, p["name"], p.get("words")), p) for p in everyone.values()]
        best = max((s for s, _ in scored), default=0)
    if best == 0:
        return []
    found = [{**p, "loose": True} if fuzzy else p for s, p in scored if s == best]
    if club:
        q = fold(club)
        narrowed = [p for p in found if any(q in fold(c) for c in _club_names(p))]
        found = narrowed or found
    return sorted(found, key=lambda p: (_rank(p), p["name"]))


def resolve(name: str) -> str:
    """The name the page holds for a nickname or a legend's surname, else the
    name as asked."""
    key = re.sub(r"\s+", " ", fold(name).replace("'", "").replace("\u2019", "")).strip()
    return NICKNAMES.get(key, name)


def _also(t: Tables, asked: str, meant: str, pool=None) -> list[str]:
    """Players of the page the name as asked also matches, when a nickname
    took it elsewhere (Maldini: Daniel Maldini of today)."""
    if fold(asked) == fold(meant):
        return []
    found = [p for p in (pool if pool is not None else people(t)).values()]
    return sorted({p["name"] for p in found if name_score(asked, p["name"]) >= 2})[:MAX_ROWS]


def _club_names(p: dict) -> list[str]:
    names = [p.get("club")]
    if "players" in p:
        names.append(p["players"].get("club"))
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
    if "players" in p and p["players"].get("value_rank") is not None:
        return p["players"]["value_rank"]
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
        "note": "several players match this name; ask which one, naming the club"
        + (
            "; no name matched exactly, these are the closest spellings"
            if found[0].get("loose")
            else ""
        ),
    }


# player_card


def report_of(t: Tables, player_id: int) -> dict:
    r = t.reports
    row = (r.get("rows") or {}).get(str(player_id)) if r.get("available") else None
    if not row:
        return {
            "kind": "none",
            "note": "there is no written report for him: reports are written only for the "
            "players of the candidates and bargains tables",
        }
    if row[0] == "report":
        return {"kind": "written report", "text": row[1], "label": r.get("label")}
    return {
        "kind": "rule",
        "note": "no checked report for these facts this week; the explanation written by "
        "rule is shown instead",
        "text": row[1],
    }


def player_card(t: Tables, args: dict) -> dict:
    asked = args["name"]
    name = resolve(asked)
    greats = sorted({g["great"] for g in t.similar.greats if name_score(name, g["great"])})
    legend = _archetype_match(t.similar, name)
    exact = greats or legend is not None  # a great or a legend is never read loosely
    found = find(t, name, args.get("club"), loose=not exact)
    if not found:
        if legend is not None:
            out = {
                "found": False,
                "archetype": archetype_card(t, legend),
                "label": ARCHETYPE_LABEL,
                "note": "not in the tables and never measured: the page holds him only as an "
                "archetype, his style drafted from the historical record; similar_players finds "
                "today's players nearest to it",
            }
            also = _also(t, asked, name)
            if also:
                out["also_on_page"] = also
            return out
        if greats:
            return {
                "found": False,
                "measured_greats": greats,
                "label": GREATS_LABEL,
                "note": "not in the tables; he is one of the measured greats, so only "
                "similar_players (mode at_age) can compare today's players with him",
            }
        return {"found": False, "note": "no player of that name is held on this page"}
    if len(found) > 1:
        return _choices(found)
    p = found[0]
    tags = t.meta.get("tags", {})
    out: dict = {"found": True, "player": p["name"], "player_id": p["id"]}
    if name != asked:
        out["nickname"] = f"{asked} read as {name}"
        also = [n for n in _also(t, asked, name) if n != p["name"]]
        if also:
            out["also_on_page"] = also
    if p.get("loose"):
        out["spelling"] = f"no player is held as {asked}; the closest name held is {p['name']}"
    club = args.get("club")
    if club and not any(fold(club) in fold(c) for c in _club_names(p)):
        out["club"] = (
            f"no {asked} of {club} is held; the player found plays for "
            f"{', '.join(dict.fromkeys(_club_names(p)))}"
        )
    out["in_tables"] = [k for k in ("candidates", "bargains") if k in p]
    row = p.get("players")
    if "candidates" in p:
        out["candidate"] = candidate_view(p["candidates"], detail=True)
    if "bargains" in p and below_fair(p["bargains"]):
        out["bargain"] = bargain_view(p["bargains"], tags, detail=True)
    elif "bargains" in p:
        # set by code, to be said as written (prompt 20g)
        out["bargain_verdict"] = not_a_bargain(p["bargains"])
        out["value_ranking"] = bargain_view(p["bargains"], tags, detail=True)
    if row is not None and out["in_tables"]:
        out["season_facts"] = {
            "nationality": row.get("nationality"),
            "league_minutes": whole(row["minutes"]),
        }
    elif row is not None:
        out["player_row"] = player_view(row, detail=True)
    if row is not None and why_not(row):
        # set by code (scout.players), to be said as written
        out["why_not_on_lists"] = why_not(row)
    if out["in_tables"]:
        out["report"] = report_of(t, p["id"])
    elif row is not None:
        out["note"] = (
            "on neither list: his row of the players table, which holds every player of the "
            "season, is all the page holds on him"
        )
        out["report"] = report_of(t, p["id"])
    else:
        o = p.get("today") or p["seasons"][-1]
        out["note"] = (
            "not in the candidates, bargains or players tables (no league minutes this "
            "season); the page holds him only in the similar players data"
        )
        out["similar_data"] = _sim_view(o, t.similar)
    if greats:
        out["measured_greats"] = greats
    return out


def _archetype_match(sim, name: str) -> dict | None:
    """The legend an asked name means (the calibration drafts are players we
    measure, so they are never a target)."""
    legends = [a for a in sim.archetypes if a.get("kind") == "legends"]
    scored = [(name_score(name, a["name"]), a) for a in legends]
    best = max((s for s, _ in scored), default=0)
    hits = [a for s, a in scored if s == best and s >= 2]
    return hits[0] if len(hits) == 1 else None


def archetype_card(t: Tables, a: dict) -> dict:
    """What the Archetypes tab shows for a legend, in brief."""
    info = a.get("details") or {}
    raw = t.similar.raw.get("archetypes") or {}
    names = raw.get("trait_names") or {}
    traits = sorted(info.get("traits") or [], key=lambda x: -x[1])[:3]
    calibration = raw.get("calibration") or {}
    return {
        "archetype": a["name"],
        "position_group": a["group"],
        "era": info.get("era"),
        "ability_pct_from_the_drafted_traits": a.get("ability"),
        "status": "reviewed by the owner"
        if info.get("reviewed")
        else "drafted from the historical record, not yet reviewed by the owner",
        "strongest_traits": [
            {"trait": names.get(k, k), "percentile": round(v), "why": why} for k, v, why in traits
        ],
        "how_far_to_trust_it": ("" if calibration.get("calibrated") else "illustrative only: ")
        + str(calibration.get("text", "")),
    }


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
    asked = args["name"]
    name = resolve(asked)
    args = {**args, "name": name}
    mode = args["mode"]
    legend = _archetype_match(sim, name)
    if mode == "at_age":
        if args.get("age") is None:
            return {"error": "mode at_age needs the age to compare at"}
        great = _great_match(sim, name)
        if great:
            return {**_like_great(t, great, int(args["age"])), "caveat": caveat}
    everyone = people(t)
    if mode == "now":
        pool = {pid: p for pid, p in everyone.items() if "today" in p}
    else:
        pool = {pid: p for pid, p in everyone.items() if "today" in p or "seasons" in p}
    exact = legend is not None or _great_match(sim, name) is not None
    found = find(t, name, args.get("club"), pool, loose=not exact)
    if not found and legend is not None:
        out = {**_like_archetype(t, legend, args), "caveat": caveat}
        if mode == "at_age":
            out["note"] = (
                "an archetype has no seasons by age: it describes his style in "
                f"{(legend.get('details') or {}).get('era')}"
            )
        also = _also(t, asked, name, pool)
        if also:
            out["also_on_page"] = also
        return out
    if not found:
        held = find(t, name, args.get("club"), loose=not exact)
        if held:
            return {
                "found": False,
                "player": held[0]["name"],
                "note": "he has no style vector this season (it needs 450 league minutes "
                "with match statistics)"
                + ("" if mode == "at_age" else "; try mode at_age")
                + (
                    f"; {whole(with_vector(t))} of the {whole(len(t.players))} players of the "
                    "season have one"
                    if t.players
                    else ""
                ),
            }
        great = _great_match(sim, name)
        if great:
            ages = sorted(g["age"] for g in sim.greats if g["great"] == great)
            return {
                "found": False,
                "measured_great": great,
                "ages_held": ages,
                "note": "a measured great is compared as he was at an age: ask again with mode "
                "at_age and one of the ages held",
            }
        return {"found": False, "note": "no player of that name is held on this page"}
    if len(found) > 1:
        return _choices(found)
    p = found[0]
    spelling = (
        {"spelling": f"no player is held as {asked}; the closest name held is {p['name']}"}
        if p.get("loose")
        else {}
    )
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
        out = {
            **spelling,
            "query": _sim_view(q, sim),
            "pool": "today's players of his position group"
            + ("" if rows else "; none match these filters"),
            "nearest": _listed(found_rows, sim.scale, t),
            "caveat": caveat,
        }
        great = _great_match(sim, name)
        if great:
            out["measured_great"] = (
                f"{great} is also a measured great: mode at_age compares today's players with "
                "him as he was at an age"
            )
        return out
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
        **spelling,
        "query": _sim_view(q, sim),
        "pool": f"today's players of his position group aged {age - 1} to {age + 1}",
        "nearest": _listed(_nearest(q, rows, sim.scale), sim.scale, t),
        "caveat": caveat,
    }
    if math.floor(q["age"]) != age:
        out["note"] = f"he has no season at {age}; the nearest held is at {math.floor(q['age'])}"
    return out


def with_vector(t: Tables) -> int:
    """Players of the players table with a style vector this season."""
    today = {o["id"] for o in t.similar.today}
    return sum(1 for o in t.players if o["player_id"] in today)


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
        "label": GREATS_LABEL,
    }
    if q["thin"]:
        out["note"] = (
            f"a thin season ({whole(q['minutes'])} minutes): his vector is mostly the position "
            "average, so these neighbours say little"
        )
    return out


def _like_archetype(t: Tables, a: dict, args: dict) -> dict:
    """Today's ten nearest of the legend's position group (the Archetypes
    tab's search: ages in the middle of the season)."""
    sim = t.similar
    groups = sim.raw["groups"]
    rows = []
    for o in sim.today:
        if groups[o["group"]] != a["group"]:
            continue
        age = math.floor(o["age_mid"])
        if args.get("age_min") is not None and age < args["age_min"]:
            continue
        if args.get("age_max") is not None and age > args["age_max"]:
            continue
        if args.get("league") and fold(o.get("league_name")) != fold(args["league"]):
            continue
        if args.get("min_ability_pct") is not None and o["ability"] < args["min_ability_pct"]:
            continue
        rows.append(o)
    card = archetype_card(t, a)
    return {
        "query": {k: card[k] for k in ("archetype", "position_group", "era", "status")},
        "pool": f"today's players of his position group ({a['group'].replace('_', ' ')})"
        + ("" if rows else "; none match these filters"),
        "nearest": _listed(_nearest(a, rows, sim.scale), sim.scale, t, "age_mid", elite=True),
        "label": ARCHETYPE_LABEL,
        "how_far_to_trust_it": card["how_far_to_trust_it"],
    }


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
    keys = set(topics.get(want, []))
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


# Strict decoding writes the arguments in the declared order and cannot go
# back, so a filter the model reaches for after a later one is lost (C06 of
# the evaluation: the league written, then no room for the age). Without the
# grammar the model writes table, sort_by and limit, then the league, the
# position, the flags, the ages and the minimums, whatever order the question
# states them in; the properties are declared in that order, and the
# description names it (prompt 20b: 15 of 15 calls kept every filter against
# 9 of 15 without the sentence).
SEARCH_ORDER = (
    "Give the arguments in the order they are listed: table, sort_by, limit, include_above_fair, "
    "then league or club, position, flags, rising or moved, the ages, the minimums."
)
SIMILAR_ORDER = (
    "Give the arguments in the order they are listed: name, mode, age, then league or club, "
    "the ages, the minimum ability."
)


def definitions(t: Tables) -> list[dict]:
    """The four tool definitions, strict; the league list comes from the data
    and is sorted, so the definitions (and the prompt cache) stay the same
    until the tables change."""
    leagues = sorted(
        {o["league"] for o in t.candidates + t.bargains + t.players if o.get("league")}
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
                "filters (use it for 'how many' questions). Tables: players (every player with "
                "league minutes in the season, on a list or not: club, minutes, ability, value, "
                "fair value, flags, the elite and big move chances where computed and why not; "
                "use it for players in general and for any named player), candidates (players "
                "ranked #101 to #2,000 by value with elite, step up, big move chances and fall "
                "back risk), bargains_prime (29 and under), bargains_experienced (30 to 33), "
                "bargains_all (both lists); use a list only when the question is about it or "
                "its chances. The bargains tables return only bargains, players priced below "
                "fair value, unless include_above_fair asks for the whole ranking by value "
                "against fair value. League is the league of the season the tables describe; "
                "age_min "
                f"compares the exact age, age_max whole years. Flags: {flag_text}. "
                f"{SEARCH_ORDER}"
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
                    "include_above_fair": {
                        "type": "boolean",
                        "description": "bargains tables only: true for the whole ranking, also "
                        "the players priced above fair value, who are not bargains; leave it out "
                        "for bargains",
                    },
                    "league": {"type": "string", "enum": leagues},
                    "club": {"type": "string", "description": "part of a club name"},
                    "position_group": {
                        "type": "string",
                        "enum": GROUPS,
                        "description": "broad group: winger covers left and right wingers and "
                        "midfielders, full_back both full backs, attacking_mid the second striker",
                    },
                    "position": {"type": "string", "enum": sorted(POSITION_GROUPS)},
                    "flags": {
                        "type": "array",
                        "items": {"type": "string", "enum": FLAG_KEYS},
                        "description": "players carrying every flag listed",
                    },
                    "max_flag_count": {"type": "integer"},
                    "rising": {"type": "boolean"},
                    "moved_since_forecast": {"type": "boolean"},
                    "age_min": {"type": "integer"},
                    "age_max": {"type": "integer"},
                    "min_minutes": {"type": "integer", "description": "league minutes"},
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
                "Every fact the page shows for one player found by name (accents, initials and "
                "small misspellings are matched loosely): his rows in candidates and bargains "
                "with chances, flags, value, fair value, ability, club today, the season so far, "
                "the explanation and his written scout report; a player on the bargains ranking "
                "priced above fair value gets bargain_verdict, the sentence that he is not a "
                "bargain. Any other player of the season "
                "gets his row of the players table (club, minutes, ability, value, fair value, "
                "flags, the chances where computed) and why_not_on_lists, the sentences that say "
                "why he is on no list. A measured great or a legend held only as an archetype "
                "says so. Nicknames (R9, CR7, Zizou, La Pulga) are understood: pass the name as "
                "asked. Several matches come back as a list."
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
                "age in the middle of the season, with their elite chance. For a legend we hold "
                "no data for (Ronaldo Nazário, Zinedine Zidane, Andrea Pirlo and others) today's "
                "players nearest to his archetype, drafted from the record. Nicknames (R9, CR7, "
                f"Zizou, La Pulga) are understood: pass the name as asked. {SIMILAR_ORDER}"
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "name": {"type": "string"},
                    "mode": {"type": "string", "enum": ["now", "at_age"]},
                    "age": {"type": "integer", "description": "mode at_age: the age to compare at"},
                    "league": {"type": "string", "enum": leagues},
                    "club": {"type": "string", "description": "to tell apart players of one name"},
                    "age_min": {"type": "integer"},
                    "age_max": {"type": "integer"},
                    "min_ability_pct": {"type": "number"},
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
                "fair_value, ability, bargains, rising, season_so_far, similar, greats, "
                "archetypes, reports, explanation, players, dates."
            ),
            "strict": True,
            "input_schema": _schema(
                {
                    "column": {"type": "string", "description": "a column name or a topic"},
                    "table": {
                        "type": "string",
                        "enum": ["players", "candidates", "bargains", "reports"],
                    },
                },
                ["column"],
            ),
        },
    ]
