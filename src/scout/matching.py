"""Match players between a stats source and a valuation source.

Strategy, in order of reliability:

1. Block on exact date of birth. Two different players rarely share a birth
   date *and* a similar name, so this removes almost all false candidates.
2. Score names with rapidfuzz ``token_set_ratio`` on normalised names. Token
   sets handle "M. Salah" against "Mohamed Salah" and long legal names that
   contain the common name.
3. Break ties with club name similarity.
4. When no name scores high enough, accept a candidate whose surname matches
   exactly and whose first name starts with the same letter ("T. Hoban" and
   "Tommie Hoban"), but only if exactly one candidate passes. Nicknames and
   short forms ("Zachary" and "Zak") defeat fuzzy scoring but not this rule.

Names are normalised before scoring. Turkish letters are mapped explicitly
first (``İ`` lowercases to ``i`` plus a combining dot in Python, and ``ı`` has
no ASCII form in some transliterations), then everything is transliterated
to ASCII.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

_TURKISH = str.maketrans(
    {"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ç": "c", "Ç": "c",
     "ö": "o", "Ö": "o", "ü": "u", "Ü": "u"}
)  # fmt: skip
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_CLUB_NOISE = re.compile(r"\b(fc|sk|as|afc|cf|jk|fk|spor kulubu|kulubu|club|a s)\b")


def normalize_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    # The API sometimes sends HTML entities, e.g. "M. O&apos;Riley".
    text = unidecode(html.unescape(name).translate(_TURKISH)).lower()
    text = _NON_ALNUM.sub(" ", text.replace("-", " ").replace(".", " "))
    return " ".join(text.split())


def normalize_club(name: object) -> str:
    return " ".join(_CLUB_NOISE.sub(" ", normalize_name(name)).split())


@dataclass(frozen=True)
class MatchConfig:
    min_score: float = 85.0
    min_margin: float = 5.0  # best candidate must beat the runner up by this much


def _name_score(af_row: pd.Series, tm_name: str) -> float:
    full = normalize_name(f"{af_row.get('firstname') or ''} {af_row.get('lastname') or ''}")
    short = normalize_name(af_row.get("player_name"))
    return max(fuzz.token_set_ratio(full, tm_name), fuzz.token_set_ratio(short, tm_name))


def _surname_initial(af_row: pd.Series, tm_name: str) -> bool:
    """Same surname and same first initial, on normalised names."""
    last = normalize_name(af_row.get("lastname"))
    first = normalize_name(af_row.get("firstname")) or normalize_name(af_row.get("player_name"))
    if not last or not first or not tm_name:
        return False
    tm_tokens = tm_name.split()
    return len(tm_tokens) >= 2 and tm_name.endswith(" " + last) and tm_tokens[0][0] == first[0]


def match_players(
    stats_players: pd.DataFrame,
    valuation_players: pd.DataFrame,
    config: MatchConfig | None = None,
) -> pd.DataFrame:
    """Match each stats player to at most one valuation player.

    ``stats_players`` needs source_player_id, player_name, firstname, lastname,
    birth_date and optionally team_name. ``valuation_players`` needs
    source_player_id, player_name, birth_date and optionally club_names (a
    space separated string of clubs the player appeared for).

    Returns one row per stats player with ``tm_player_id`` (or NA), ``score``
    ``reason`` (matched, no_birth_date, no_candidate_same_dob, low_score or
    ambiguous) and ``rule`` (name_score or surname) for matched rows.
    """
    config = config or MatchConfig()
    tm = valuation_players.copy()
    tm["_name"] = tm["player_name"].map(normalize_name)
    tm["_club"] = tm.get("club_names", pd.Series("", index=tm.index)).map(normalize_club)
    tm["_dob"] = pd.to_datetime(tm["birth_date"]).dt.date
    by_dob = {dob: group for dob, group in tm.groupby("_dob")}

    out = []
    for _, row in stats_players.iterrows():
        record = {"af_player_id": row["source_player_id"], "tm_player_id": pd.NA, "score": 0.0}
        dob = pd.to_datetime(row.get("birth_date"))
        if pd.isna(dob):
            out.append(record | {"reason": "no_birth_date"})
            continue
        candidates = by_dob.get(dob.date())
        if candidates is None:
            out.append(record | {"reason": "no_candidate_same_dob"})
            continue
        club = normalize_club(row.get("team_name"))
        scored = []
        for _, cand in candidates.iterrows():
            name_score = _name_score(row, cand["_name"])
            club_bonus = 0.0
            if club and cand["_club"]:
                club_bonus = 5.0 if fuzz.partial_ratio(club, cand["_club"]) >= 90 else 0.0
            scored.append((name_score + club_bonus, name_score, cand["source_player_id"]))
        scored.sort(reverse=True)
        best_total, best_name, best_id = scored[0]
        record["score"] = round(best_name, 1)
        if best_name < config.min_score:
            rule = [c["source_player_id"] for _, c in candidates.iterrows()
                    if _surname_initial(row, c["_name"])]  # fmt: skip
            if len(rule) == 1:
                out.append(
                    record | {"tm_player_id": rule[0], "reason": "matched", "rule": "surname"}
                )
            else:
                out.append(record | {"reason": "low_score", "tm_candidate_id": best_id})
            continue
        if len(scored) > 1 and best_total - scored[1][0] < config.min_margin:
            out.append(record | {"reason": "ambiguous", "tm_candidate_id": best_id})
            continue
        out.append(record | {"tm_player_id": best_id, "reason": "matched", "rule": "name_score"})

    result = pd.DataFrame(out)
    for column in ("tm_candidate_id", "rule"):
        if column not in result:
            result[column] = pd.NA
    return result
