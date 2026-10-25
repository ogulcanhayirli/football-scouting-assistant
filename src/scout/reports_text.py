"""Generated scout reports with a faithfulness check (prompt 19).

``scout reports`` runs after ``scout lists``. For every player in the two
tables (one report a player, also when he is in both):

* **Fact sheet.** ``fact_sheets`` turns the table columns into one fact per
  line: age, position, club at forecast and today, league, value and fair
  value, ability percentile, each chance with its count of past players and
  what it measured, the flags with their raw numbers, the rising tag, the
  three explanation stats, the season so far, the three most similar players
  of today and, on the bargains, the list, the minutes and the contract. The
  sheet is the only variable part of the request; the system prompt and the
  shared instructions are a fixed prefix with prompt caching.
* **Writer.** Claude Haiku 5.5 writes 120 to 150 words for a fan, through
  the Message Batches API. A report is regenerated only when its fact sheet
  changed (``fact_hash``: the sheet, the prompt version and the model);
  otherwise the stored one is reused.
* **Check, in code** (``check``). Every number of the report (with its percent
  sign or unit) must match a number of the fact sheet within the rounding the
  report wrote it with, and every proper noun must be a name of the fact
  sheet; the length must be 120 to 150 words. A failing report goes back to
  the writer with the violations, at most twice; after that the template
  explanation of ``scout.explain`` is shown and the failure is logged
  (``failures.jsonl``).
* **Judge.** Each run a sample of reports not judged before (``judge_sample``,
  50) goes to Claude Sonnet 5.5 with a structured output: faithful (with the
  offending span), hedging adequate, readability 1 to 5. The verdicts are
  stored, and ``reports/reports_text.md`` lays out 20 of them for the owner,
  whose agreement goes to ``judge/agreement.csv`` (committed) and from there
  to the page.
* **Budget.** ``max_tokens_per_run`` is a hard cap: before a batch is sent,
  each request's worst case (its input estimated from above, plus its
  ``max_tokens``) is added to what the run has used, and requests that could
  pass the cap are held back (their players show the template this week).
  The cost of the run is computed from the usage fields of every reply.

Everything lands in ``data/processed/reports_text/``: ``reports.parquet``
(one row a player), ``runs.jsonl``, ``verdicts.jsonl``, ``failures.jsonl``,
``summary.json`` and ``dictionary.csv``. ``scout site`` rebuilds the fact
sheets and shows a stored report only when its hash still matches, so a
stale report never reaches the page. Tests use a fake client; nothing here
calls the API without ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scout.fsutil import atomic_write_text

log = logging.getLogger(__name__)

OUT = "reports_text"  # under data/processed
PROMPT_VERSION = "19.1"  # part of every fact hash: a new prompt regenerates every report
SIMILAR_SHOWN = 3
OWNER_READS = 20  # judged reports laid out for the owner each run
AGREEMENT = Path("judge") / "agreement.csv"  # committed; the owner's verdicts on the judge
AGREEMENT_COLUMNS = [
    "judged_on",
    "player_id",
    "player",
    "fact_hash",
    "judge_faithful",
    "judge_hedging_adequate",
    "judge_readability",
    "owner_agrees",
    "note",
]
MODEL_NAMES = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5"}


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def label(model: str) -> str:
    """The line under every report on the page."""
    return (
        f"written by {model_name(model)} from the facts above and checked automatically; "
        "odds are historical rates"
    )


# The fact sheet


@dataclass(frozen=True)
class Num:
    """A number of a fact sheet or a report. ``unit``: plain, pct, eur or
    ordinal; ``tag``: what it counts (age, minutes, players, ...) when the
    words around it say so; ``step``: the precision it was written with (0
    for the exact values of the tables)."""

    value: float
    unit: str = "plain"
    tag: str | None = None
    step: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class Fact:
    label: str
    text: str
    numbers: tuple[Num, ...] = ()
    names: tuple[str, ...] = ()  # the proper nouns this fact holds
    people: tuple[str, ...] = ()  # names of people: a surname alone is allowed too


@dataclass
class FactSheet:
    player_id: int
    player: str
    tables: str
    facts: list[Fact]
    template: str
    priority: tuple = ()

    @property
    def text(self) -> str:
        return "\n".join(f"{f.label}: {f.text}" for f in self.facts)

    def hash(self, model: str) -> str:
        blob = f"{PROMPT_VERSION}\n{model}\n{self.text}".encode()
        return hashlib.sha256(blob).hexdigest()

    def numbers(self) -> list[Num]:
        found = []
        for f in self.facts:
            found += f.numbers
            found += extract_numbers(f"{f.label}: {f.text}")
        return found

    def names(self) -> set[str]:
        """Folded name phrases: the names of every fact, a person's single
        names, and the capitalised spans of the free text (explanations,
        flags)."""
        out = set()
        for f in self.facts:
            for name in (*f.names, *f.people):
                folded = fold(name)
                if folded:
                    out.add(folded)
            for name in f.people:
                parts = [p for p in fold(name).split() if len(p) > 1]
                out.update(
                    " ".join(parts[i:j])
                    for i in range(len(parts))
                    for j in range(i + 1, len(parts) + 1)
                )
            for span in capital_spans(f.text):
                out.add(fold(span))
        out.discard("")
        return out

    def vocabulary(self) -> set[str]:
        """Folded ordinary words of the sheet (labels and lower case words):
        a capitalised common word of the report ("Elite") is not a name."""
        words = set()
        for f in self.facts:
            words.update(fold(f.label).split())
            for token in WORD.findall(f.text):
                if token[0].islower():
                    words.update(fold(token).split())
        return words


def money(value: float) -> str:
    """As the page writes it: €30m, €21.6m, €750k."""
    if value >= 1e7:
        return f"€{round(value / 1e6)}m"
    if value >= 1e6:
        return f"€{value / 1e6:.1f}".removesuffix(".0") + "m"
    if value >= 1e3:
        return f"€{round(value / 1e3)}k"
    return f"€{round(value)}"


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def day(value) -> str:
    ts = pd.Timestamp(value)
    return f"{ts.day} {ts.strftime('%B %Y')}"


def season_label(season: int) -> str:
    return f"{season}/{str(season + 1)[-2:]}"


def _present(value) -> bool:
    return value is not None and not pd.isna(value)


def _rate(pct) -> str:
    return "under 5%" if int(pct) == 0 else f"{int(pct)}%"


def _rate_nums(pct, players, decile=None) -> tuple[Num, ...]:
    nums = [Num(float(players), tag="players")]
    if int(pct) > 0:
        nums.append(Num(float(pct), "pct"))
    if decile is not None:
        nums.append(Num(float(decile), tag="decile"))
    return tuple(nums)


CHANCES = {
    # column prefix: (label, what happened to the past players, not rated text)
    "elite_chance": (
        "Elite chance",
        "became one of the 100 most valuable players within 2 seasons",
        "not rated: only the elite pool (ranked 101 to 1,000 by value, 23 or under) gets one",
    ),
    "step_up_chance": (
        "Step up chance",
        "saw their value rank become at least 3 times better, inside the top 500, within 2 seasons",
        "not rated: he is valued below the ranks this chance was measured on",
    ),
    "big_move_chance": (
        "Big move chance",
        "joined a top 20 club by squad value within 24 months",
        "not rated",
    ),
}


def _chance_fact(row: pd.Series, prefix: str) -> Fact:
    title, outcome, blank = CHANCES[prefix]
    pct, players = row.get(f"{prefix}_pct"), row.get(f"{prefix}_players")
    if not _present(pct):
        rank = row.get("value_rank")
        if prefix == "big_move_chance" and _present(rank) and int(rank) <= 2000:
            blank = "not rated: he already plays for a top 20 club"
        return Fact(title, blank)
    decile = row.get(prefix.replace("_chance", "_decile"))
    decile = int(decile) if _present(decile) else None
    band = f"in the same band (decile {decile} of 10)" if decile is not None else "in the same band"
    text = f"of {int(players):,} past players {band}, {_rate(pct)} {outcome}"
    return Fact(title, text, _rate_nums(pct, players, decile))


def _fall_back_fact(row: pd.Series) -> Fact:
    pct, players = row.get("fall_back_risk_pct"), row.get("fall_back_risk_players")
    if not _present(pct):
        return Fact("Fall back risk", "not rated: only the elite pool gets one")
    text = (
        f"of {int(players):,} past elite pool players with as many warning flags, "
        f"{_rate(pct)} saw their value rank become at least 2 times worse 2 seasons later"
    )
    return Fact("Fall back risk", text, _rate_nums(pct, players))


def _club_today(row: pd.Series) -> Fact:
    moved, club, league = row.get("moved_since_forecast"), row.get("current_club"), None
    if _present(row.get("current_league")):
        league = row["current_league"]
    if not _present(moved) or not _present(club):
        return Fact("Club today", "not known")
    where = f"{club}, {league}" if league else str(club)
    if bool(moved):
        return Fact(
            "Club today", f"{where} (he moved after the forecast)", names=(club, league or "")
        )
    return Fact("Club today", f"{where}, the same club", names=(club, league or ""))


def _so_far(row: pd.Series) -> Fact | None:
    when = row.get("so_far_date")
    if not _present(when):
        return None
    minutes = row.get("so_far_minutes")
    if not _present(minutes) or float(minutes) <= 0:
        return Fact("Season so far", f"to {day(when)}: no league minutes in our leagues yet")
    league = row.get("so_far_league")
    nums = [Num(float(minutes), tag="minutes")]
    parts = []
    if _present(row.get("so_far_matches")):
        parts.append(f"{int(row['so_far_matches'])} matches")
        nums.append(Num(float(row["so_far_matches"]), tag="matches"))
    parts.append(f"{int(minutes):,} minutes")
    if _present(row.get("so_far_minutes_share")):
        share = float(row["so_far_minutes_share"]) * 100
        parts.append(f"{share:.0f}% of team minutes")
        nums.append(Num(share, "pct"))
    text = f"to {day(when)}" + (f", {league}" if _present(league) else "") + ": " + ", ".join(parts)
    if _present(row.get("so_far_ability_pct")):
        text += f"; ability so far {ordinal(round(row['so_far_ability_pct']))} percentile"
        nums.append(Num(float(row["so_far_ability_pct"])))
    if _present(row.get("so_far_goals_per90")):
        goals, assists = float(row["so_far_goals_per90"]), float(row["so_far_assists_per90"])
        text += f"; {goals:.2f} goals and {assists:.2f} assists per 90 minutes"
        nums += [Num(goals, tag="goals"), Num(assists, tag="assists")]
    flags = row.get("so_far_flags")
    if _present(flags):
        text += f"; warning flags today: {flags}"
    return Fact("Season so far", text, tuple(nums), names=(league,) if _present(league) else ())


def _similar(rows: pd.DataFrame | None, leagues: dict[str, str]) -> Fact | None:
    if rows is None or rows.empty:
        return None
    parts, names, nums = [], [], []
    for r in rows.sort_values("rank").head(SIMILAR_SHOWN).itertuples(index=False):
        league = leagues.get(r.neighbour_league_key, r.neighbour_league_key)
        age = math.floor(float(r.neighbour_age))
        ability = round(float(r.neighbour_ability_pct))
        parts.append(
            f"{r.neighbour} ({age}, {r.neighbour_club}, {league}, "
            f"ability {ordinal(ability)} percentile)"
        )
        names += [r.neighbour, r.neighbour_club, league]
        nums += [Num(float(age), tag="age"), Num(float(r.neighbour_ability_pct))]
    text = "; ".join(parts) + " (alike in playing style, not in level)"
    people = tuple(names[0::3])
    places = tuple(n for i, n in enumerate(names) if i % 3)
    return Fact("Similar players today", text, tuple(nums), places, people)


def sheet_for(
    player_id: int,
    cand: pd.Series | None,
    barg: pd.Series | None,
    neighbours: pd.DataFrame | None,
    leagues: dict[str, str],
    priority: tuple = (),
) -> FactSheet:
    """One player's fact sheet from his candidates row, his bargains row, or both."""
    row = cand if cand is not None else barg
    facts = [
        Fact("Player", str(row["player"]), people=(row["player"],)),
        Fact("Season described", f"{season_label(int(row['season']))}, at its end"),
    ]
    if _present(row.get("age")):
        age = math.floor(float(row["age"]))
        facts.append(Fact("Age", str(age), (Num(float(age), tag="age"),)))
    if _present(row.get("position")):
        facts.append(Fact("Position", str(row["position"]), names=(row["position"],)))
    facts.append(
        Fact(
            "Club at forecast",
            f"{row['club_at_forecast']}, {row['league']}",
            names=(row["club_at_forecast"], row["league"]),
        )
    )
    facts.append(_club_today(row))
    tables = []
    if cand is not None:
        tables.append("candidates")
        if _present(cand.get("value_rank")):
            rank = int(cand["value_rank"])
            facts.append(
                Fact(
                    "Value rank",
                    f"#{rank:,} among all players we track",
                    (Num(float(rank), tag="rank"),),
                )
            )
    if barg is not None:
        tables.append("bargains")
        listed = "prime list (29 and under)" if barg["list"] == "prime" else "experienced list"
        if barg["list"] != "prime":
            listed = "experienced list (30 to 33)"
        facts.append(
            Fact(
                "Bargain list",
                f"{listed}, #{int(barg['list_rank']):,}",
                (Num(float(barg["list_rank"]), tag="rank"),),
            )
        )
    value, fair = row.get("value_eur"), row.get("fair_value_eur")
    if _present(value):
        facts.append(Fact("Market value", money(float(value)), (Num(float(value), "eur"),)))
    else:
        facts.append(Fact("Market value", "none known"))
    if _present(fair):
        text = money(float(fair))
        nums = [Num(float(fair), "eur")]
        if _present(value) and float(fair) > 0:
            ratio = (
                float(barg["value_to_fair"]) * 100
                if barg is not None and _present(barg.get("value_to_fair"))
                else float(value) / float(fair) * 100
            )
            text += f"; his market value is {ratio:.0f}% of it"
            nums.append(Num(ratio, "pct"))
        if barg is not None and _present(barg.get("value_to_fair_pct")):
            cheap = float(barg["value_to_fair_pct"])
            text += f", among the cheapest {cheap:g}% of listed players of his position group"
            nums.append(Num(cheap, "pct"))
        facts.append(Fact("Fair value (our model)", text, tuple(nums)))
    if _present(row.get("ability_pct")):
        ability = round(float(row["ability_pct"]))
        facts.append(
            Fact(
                "Ability",
                f"{ordinal(ability)} percentile among regular players of his position this season, "
                "from his output per 90 minutes over the last 2 seasons (100 is the best)",
                (Num(float(row["ability_pct"])),),
            )
        )
    if barg is not None and _present(barg.get("minutes_share")):
        share = float(barg["minutes_share"]) * 100
        facts.append(
            Fact("Minutes", f"{share:.0f}% of his club's league minutes", (Num(share, "pct"),))
        )
    if cand is not None:
        facts += [_chance_fact(cand, p) for p in CHANCES]
        facts.append(_fall_back_fact(cand))
    count = int(row["flag_count"]) if _present(row.get("flag_count")) else 0
    flags = str(row["flags"]) if count else "none"
    facts.append(
        Fact(
            "Warning flags",
            f"{count}: {flags}" if count else "none",
            (Num(float(count), tag="flags"),) if count else (),
        )
    )
    if cand is not None:
        rising = bool(cand["rising"]) if _present(cand.get("rising")) else False
        facts.append(
            Fact(
                "Rising",
                "yes: in this season's top 50 of the riser model, which looks for players whose "
                "play may rise clearly over the next 2 seasons"
                if rising
                else "no",
            )
        )
    facts.append(Fact("Standout stats", str(row["explanation"])))
    if barg is not None:
        if _present(barg.get("contract_expiry")):
            facts.append(Fact("Contract until", day(barg["contract_expiry"])))
        from scout.flags import TAGS

        for tag in ("on_loan", "stale_value"):
            if _present(barg.get(tag)) and bool(barg[tag]):
                facts.append(Fact("Tag", TAGS[tag]))
    so_far = _so_far(row)
    if so_far is not None:
        facts.append(so_far)
    similar = _similar(neighbours, leagues)
    if similar is not None:
        facts.append(similar)
    template = str(cand["explanation"] if cand is not None else barg["explanation"])
    return FactSheet(
        player_id=int(player_id),
        player=str(row["player"]),
        tables=" and ".join(tables),
        facts=facts,
        template=template,
        priority=priority,
    )


def fact_sheets(
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    neighbours: pd.DataFrame | None,
    leagues: dict[str, str],
) -> list[FactSheet]:
    """One sheet a player of the two tables, the candidates' order first,
    then the bargains (prime before experienced, by list rank)."""
    cand = cand.reset_index(drop=True)
    barg = barg.reset_index(drop=True)
    by_cand = {int(r["player_id"]): (i, r) for i, r in cand.iterrows()}
    order = barg.assign(_l=barg["list"].ne("prime")).sort_values(["_l", "list_rank"], kind="stable")
    by_barg = {int(r["player_id"]): (i, r) for i, r in order.iterrows()}
    near = {}
    if neighbours is not None and not neighbours.empty:
        near = {int(k): g for k, g in neighbours.groupby("player_id")}
    ids = list(by_cand) + [p for p in by_barg if p not in by_cand]
    out = []
    for n, pid in enumerate(ids):
        c = by_cand.get(pid, (None, None))[1]
        b = by_barg.get(pid, (None, None))[1]
        out.append(sheet_for(pid, c, b, near.get(pid), leagues, priority=(n,)))
    return out


# Text handling

WORD = re.compile(r"[^\W\d_](?:[\w'\u2019-]*[^\W_])?")


def fold(text: str) -> str:
    """Lower case without accents, hyphens and punctuation as spaces."""
    text = str(text).replace("\u2019", "'")
    for a, b in (("ı", "i"), ("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("đ", "d"), ("ł", "l")):
        text = text.replace(a, b).replace(a.upper(), b.upper())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"'s\b", "", text)
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def words(text: str) -> int:
    return len([w for w in re.findall(r"\S+", text) if re.search(r"\w", w)])


# Numbers

SCALES = {"m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9, "k": 1e3, "thousand": 1e3}
NUMBER = re.compile(
    r"(?<![\w.,])(?P<cur>€\s?)?(?P<sign>[-+\u2212])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?P<dec>\.\d+)?(?P<ord>st|nd|rd|th)?"
    r"(?:\s?(?P<unit>%|per ?cent\b|percent\b|million\b|m\b|bn\b|billion\b|k\b|thousand\b))?"
)
NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "fifty": 50,
    "hundred": 100,
    "twice": 2,
}
WORD_NUMBER = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
NEXT_TAGS = {
    **dict.fromkeys(["year", "years", "year-old", "years-old"], "age"),
    **dict.fromkeys(["season", "seasons"], "seasons"),
    **dict.fromkeys(["minute", "minutes", "mins"], "minutes"),
    **dict.fromkeys(["match", "matches", "game", "games", "appearances"], "matches"),
    **dict.fromkeys(["player", "players"], "players"),
    **dict.fromkeys(["flag", "flags"], "flags"),
    **dict.fromkeys(["goal", "goals"], "goals"),
    **dict.fromkeys(["assist", "assists"], "assists"),
    **dict.fromkeys(["month", "months"], "months"),
}
PREV_TAGS = {"aged": "age", "age": "age", "decile": "decile", "rank": "rank", "ranked": "rank"}
EURO_AFTER = re.compile(r"\s*(euros?|eur)\b", re.IGNORECASE)
LETTERS = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tag(text: str, start: int, end: int) -> str | None:
    if text[max(0, start - 1) : start] == "#":
        return "rank"
    for w in LETTERS.findall(text[end : end + 40])[:2]:
        if w.lower() in NEXT_TAGS:
            return NEXT_TAGS[w.lower()]
    for w in reversed(LETTERS.findall(text[max(0, start - 30) : start])[-2:]):
        if w.lower() in PREV_TAGS:
            return PREV_TAGS[w.lower()]
    return None


def extract_numbers(text: str) -> list[Num]:
    """Every number with its unit, its precision and what it counts."""
    found = []
    for m in NUMBER.finditer(text):
        digits = m["int"].replace(",", "")
        decimals = len(m["dec"]) - 1 if m["dec"] else 0
        value = float(digits + (m["dec"] or ""))
        raw_unit = (m["unit"] or "").strip().lower().replace(" ", "")
        scale = SCALES.get(raw_unit, 1.0)
        euro = bool(m["cur"]) or EURO_AFTER.match(text, m.end()) is not None
        if raw_unit in ("%", "percent"):
            unit = "pct"
        elif euro or raw_unit in SCALES:
            unit = "eur"
        elif m["ord"]:
            unit = "ordinal"
        else:
            unit = "plain"
        if m["sign"] in ("-", "\u2212"):
            value = -value
        found.append(
            Num(
                value * scale,
                unit,
                _tag(text, m.start(), m.end()),
                10.0**-decimals * scale,
                m.group(0).strip(),
            )
        )
    for m in WORD_NUMBER.finditer(text):
        value = float(NUMBER_WORDS[m[1].lower()])
        found.append(Num(value, "plain", _tag(text, m.start(), m.end()), 1.0, m[1]))
    return found


def _compatible(said: Num, fact: Num) -> bool:
    if "eur" in (said.unit, fact.unit) and said.unit != fact.unit:
        return False
    if said.unit == "pct" and fact.unit != "pct":
        return False
    if said.unit == "ordinal" and fact.unit not in ("plain", "ordinal"):
        return False
    return said.tag is None or fact.tag is None or said.tag == fact.tag


def number_in(said: Num, facts: Iterable[Num]) -> bool:
    """True when some fact number, written with the report's precision, is
    the report's number: 21.6 for 21.632, never 21.7; signs are words in a
    report ("fell 25%"), so they are compared without."""
    for fact in facts:
        if not _compatible(said, fact):
            continue
        tolerance = said.step / 2 + 1e-9 * max(1.0, abs(fact.value))
        if abs(abs(said.value) - abs(fact.value)) <= tolerance:
            return True
    return False


# Proper nouns

CONNECTORS = {
    "de", "da", "do", "dos", "das", "del", "della", "di", "van", "der", "den", "von", "le",
    "la", "du", "el", "al", "bin", "ten", "ter", "y",
}  # fmt: skip
# Ordinary words that open a sentence; never a name on their own there.
STARTERS = {
    "a", "after", "against", "all", "also", "although", "among", "an", "and", "as", "at",
    "before", "both", "but", "by", "despite", "each", "even", "for", "from", "he", "her",
    "his", "historically", "however", "if", "in", "it", "its", "just", "last", "meanwhile",
    "most", "no", "not", "now", "of", "on", "only", "or", "other", "overall", "past",
    "since", "so", "still", "that", "the", "their", "there", "these", "this", "those",
    "though", "to", "today", "under", "what", "when", "where", "while", "with", "yet",
}  # fmt: skip
ALWAYS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
}  # fmt: skip


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    possessive: bool
    numeric: bool = False


# A number of one or two digits right after a capitalised word belongs to the
# name ("Ligue 1", "Schalke 04"), unless a unit or a count follows it.
NAME_NUMBER = re.compile(r"\d{1,2}(?![\d%/.,]\d|[\d%])")


def _tokens(text: str) -> list[_Token]:
    out = []
    for m in WORD.finditer(text):
        word, possessive = m.group(0), False
        for s in ("'s", "\u2019s"):
            if word.endswith(s) and len(word) > 2:
                word, possessive = word[: -len(s)], True
        if word[0].isupper():
            cut = re.search(r"-(?=[^\W\d_])(?=[a-zß-ÿ])", word)  # "Como-based" is Como
            if cut:
                word, possessive = word[: cut.start()], True
        out.append(_Token(word, m.start(), m.start() + len(word), possessive))
    for m in NAME_NUMBER.finditer(text):
        if (
            m.start()
            and text[m.start() - 1] == " "
            and (m.start() < 2 or text[m.start() - 2] != " ")
        ):
            after = LETTERS.findall(text[m.end() : m.end() + 20])[:1]
            if not after or after[0].lower() not in NEXT_TAGS:
                out.append(_Token(m.group(0), m.start(), m.end(), False, numeric=True))
    return sorted(out, key=lambda t: t.start)


def _capital(token: _Token) -> bool:
    return not token.numeric and token.text[0].isupper()


def _initial(tokens: list[_Token], i: int, text: str) -> bool:
    """An initial ("R. Bürki"): one capital and a dot, not ending a run of
    capitalised words ("Serie A. He")."""
    tok = tokens[i]
    if not (len(tok.text) == 1 and tok.text.isupper() and text[tok.end : tok.end + 1] == "."):
        return False
    return i == 0 or not _capital(tokens[i - 1]) or text[tokens[i - 1].end : tok.start] != " "


def capital_spans(text: str) -> list[str]:
    """The runs of capitalised words of a text (connectors such as "van" or
    "de" inside), as written."""
    return [text[s[0].start : s[-1].end] for s, _ in _spans(text)]


def _spans(text: str) -> list[tuple[list[_Token], bool]]:
    """Capitalised runs and whether each opens a sentence."""
    tokens = _tokens(text)
    spans, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if not _capital(tok):
            i += 1
            continue
        before = text[tokens[i - 1].end : tok.start] if i else text[: tok.start]
        if i == 0:
            opens = True
        else:
            ends = re.search(r"[!?:;\n]|\.(?!\d)", before)
            opens = bool(ends) and not (before == ". " and _initial(tokens, i - 1, text))
        span = [tok]
        j = i + 1
        while j < len(tokens) and not span[-1].possessive:
            gap = text[span[-1].end : tokens[j].start]
            initial = len(span) == 1 and _initial(tokens, j - 1, text)
            joined = gap in (" ", "-", "") or (gap == ". " and initial)
            if not joined:
                break
            nxt = tokens[j]
            if _capital(nxt) or (nxt.numeric and gap == " "):
                span.append(nxt)
                j += 1
                continue
            linked = j + 1 < len(tokens) and _capital(tokens[j + 1])
            if (
                linked
                and nxt.text.lower() in CONNECTORS
                and text[nxt.end : tokens[j + 1].start] == " "
            ):
                span += [nxt, tokens[j + 1]]
                j += 2
                continue
            break
        spans.append((span, opens))
        i = j
    return spans


def _phrase(tokens: list[_Token]) -> str:
    return fold(" ".join(t.text for t in tokens))


def _covered(tokens: list[_Token], names: set[str], vocabulary: set[str]) -> bool:
    """The tokens split into names of the sheet, its ordinary words, months
    and days, from the left, longest first."""
    i = 0
    while i < len(tokens):
        for j in range(len(tokens), i, -1):
            phrase = _phrase(tokens[i:j])
            single = j == i + 1 and (
                phrase in vocabulary or phrase in ALWAYS or tokens[i].numeric
            )  # a number alone is the number check's business
            if phrase in names or single:
                i = j
                break
        else:
            return False
    return True


def unknown_names(
    text: str, sheet: FactSheet, gazetteer: Iterable[str] = (), names=None, vocabulary=None
) -> list[str]:
    """The proper nouns of a report that are not names of its fact sheet.

    A capitalised run inside a sentence is a proper noun. One that opens a
    sentence may be an ordinary word ("Despite", "Strong"), so its first word
    is let go unless it is a known name (``gazetteer``: every player, club and
    league of the tables), and a single word opening a sentence counts only
    when it is a known name."""
    names = sheet.names() if names is None else names
    vocabulary = sheet.vocabulary() if vocabulary is None else vocabulary
    known = set(gazetteer)
    bad = []
    for tokens, opens in _spans(text):
        if opens:
            first = _phrase(tokens[:1])
            if first in STARTERS:
                tokens = tokens[1:]
                if not tokens:
                    continue
            elif first not in known and (
                len(tokens) == 1 or _covered(tokens[1:], names, vocabulary)
            ):
                continue
        if not _covered(tokens, names, vocabulary):
            bad.append(text[tokens[0].start : tokens[-1].end])
    return bad


# The check


@dataclass
class Check:
    passed: bool
    violations: list[str]
    words: int
    numbers: int
    names: int


def check(
    report: str,
    sheet: FactSheet,
    words_min: int = 120,
    words_max: int = 150,
    gazetteer: Iterable[str] = (),
) -> Check:
    """Every number of the report in the fact sheet within its rounding,
    every proper noun a name of the sheet, and the length in range."""
    violations = []
    allowed = sheet.numbers()
    said = extract_numbers(report)
    for num in said:
        if not number_in(num, allowed):
            violations.append(
                f'the number "{num.text}" is not in the fact sheet, or not rounded correctly'
            )
    spans = _spans(report)
    for name in unknown_names(report, sheet, gazetteer):
        violations.append(f'the name "{name}" is not in the fact sheet')
    n = words(report)
    if not words_min <= n <= words_max:
        violations.append(f"the report has {n} words; it must have {words_min} to {words_max}")
    return Check(not violations, violations, n, len(said), len(spans))


# The prompts (the fixed prefix: never put anything that changes in here)

WRITER_SYSTEM = (
    "You write short scouting reports for a football fan who follows the game closely but is "
    "not a professional scout or a statistician. Each report describes one player from a fact "
    "sheet that our own models and tables produced. You write plain, friendly English, like a "
    "knowledgeable friend, and you never claim more than the facts say."
)


def _unwrap(text: str) -> str:
    """Paragraphs written over several source lines, joined back: a line
    break inside a paragraph becomes a space, a list item ("- ") keeps its
    own line."""
    paragraphs = []
    for block in text.strip().split("\n\n"):
        lines = []
        for line in block.split("\n"):
            if lines and not line.startswith("- "):
                lines[-1] += " " + line.strip()
            else:
                lines.append(line.strip())
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


WRITER_INSTRUCTIONS = _unwrap(
    """
How to write the report

1. Length and form. Write 120 to 150 words of plain prose in one or two paragraphs. No title,
no headings, no bullet points, no bold, no emoji. Return only the report.

2. Use the fact sheet and nothing else. Every number you write must appear in the fact sheet.
Copy numbers as the sheet writes them, as digits. You may round a number only correctly and
only to fewer decimals (0.47 may become 0.5, never 0.4). Do not calculate anything new: no
differences, sums, ratios or averages of your own, and no number words such as "three" for a
number the sheet writes as 3.

3. Names. Every name you write (player, club, league, competition, country or person) must
appear in the fact sheet, spelled as it is there. Do not mention a nationality, a coach, a
former club, a rumour, a data source or any player who is not in the sheet. When unsure
whether something is in the sheet, leave it out.

4. Odds are history, not predictions. Each chance and risk in the sheet is a historical rate:
of the past players our tables put in the same band, the share who later did something. Say
it that way, for example "of <count> past players in his band, <rate> became one of the 100
most valuable players within 2 seasons", with the count and the rate from the sheet. Never
write that he will do something, never call a rate his chance, probability or likelihood,
never say he is likely, set, bound, destined, guaranteed or certain to do anything, and never
promise a transfer or a rise. Keep the count of past players next to each rate you mention.

5. What to cover. Open with who he is: age, position, club and league. Then the most telling
of: his ability percentile, his market value against the fair value, the rates that are
rated, the warning flags, the rising tag and the standout stats. Mention every warning flag
with its own number, or say he raises none. If he moved after the forecast, say so. Use the
season so far only as context: it is not part of the forecast. You may name one or two of the
similar players; if you do, say the likeness is in playing style, not level.

6. What the labels mean.
- Ability: where his output per 90 minutes ranks among the regular players of his position
this season; 100 is the best.
- Market value is the market's valuation. Fair value is what our model, trained on earlier
seasons, says a player with his record is worth. A value well below fair value means the
market prices him cheaply, well above means dearly; neither is a verdict on him.
- Elite chance: past players in the same band who became one of the most valuable players in
the world. Only young players already among the more valuable get one.
- Step up chance: past players whose value rank became several times better.
- Big move chance: past players who joined one of the clubs with the most valuable squads.
- Fall back risk: past players with as many warning flags whose value rank became much worse.
- Warning flags: reasons for doubt, each with his own number: low minutes, injuries, a falling
value.
- Rising: he is in the top group of a model that looks for players whose play may rise
clearly. It is a tag, not a rate.
- Standout stats: where he stands furthest from players of his position and age band.
- Similar players today: players whose style of play is closest to his.
- Bargain list: players whose market value is low against their fair value; the prime list is
the younger one, the experienced list the older one.
- Not rated: the table gives no rate there. Say so if it matters; do not invent a reason.

7. Tone. Measured and concrete. Prefer football words to statistical ones. No hype, no
judgement of character, no advice to buy or sell.

The fact sheet comes in the next message, one fact per line as "label: value".
"""
)

JUDGE_SYSTEM = (
    "You check short scouting reports that another model wrote for a football fan from a fact "
    "sheet of our own tables. You are strict, concrete and fair."
)

JUDGE_INSTRUCTIONS = _unwrap(
    """
Read the fact sheet, then the report, and give three verdicts.

faithful: true only when every claim of the report is supported by the fact sheet: each
number, each name, the direction of each comparison (market value above or below fair
value), which warning flags are raised, whether he moved, which chances are rated, what each
chance measured. A plain football paraphrase of a fact is fine. Anything the sheet does not
support is not: an added reason, a nationality, a trait, a transfer, a club or player not in
the sheet, an invented or wrongly rounded number, a reversed comparison, a rate given to the
wrong outcome.

offending_span: when faithful is false, the shortest exact passage of the report, copied
character for character, that the sheet does not support; when faithful is true, an empty
string.

hedging_adequate: true when every chance and risk is presented as what happened to past
players in the same band (a historical rate), never as a prediction or a probability for this
player, and no sentence promises that he will do something.

readability: from 1 to 5 for a football fan. 5: clear, natural and well ordered. 4: clear
with small stiffness. 3: understandable but list like or awkward. 2: hard to follow. 1:
confusing.

reason: one short sentence on the most important point of your verdicts.
"""
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "offending_span": {"type": "string"},
        "hedging_adequate": {"type": "boolean"},
        "readability": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["faithful", "offending_span", "hedging_adequate", "readability", "reason"],
    "additionalProperties": False,
}

FEEDBACK = (
    "An automatic check compared your report with the fact sheet and rejected it:\n{problems}\n"
    "Write the whole report again from the fact sheet, fixing every point. Return only the report."
)


def _system(system: str, instructions: str) -> list[dict]:
    """The fixed prefix, cached for an hour (a batch can take longer than five minutes)."""
    return [
        {"type": "text", "text": system},
        {"type": "text", "text": instructions, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]


def writer_params(cfg, sheet: FactSheet, previous: str = "", problems=()) -> dict:
    messages = [{"role": "user", "content": sheet.text}]
    if previous and problems:
        listed = "\n".join(f"- {p}" for p in problems)
        messages += [
            {"role": "assistant", "content": previous},
            {"role": "user", "content": FEEDBACK.format(problems=listed)},
        ]
    return {
        "model": cfg.writer_model,
        "max_tokens": cfg.writer_max_tokens,
        "system": _system(WRITER_SYSTEM, WRITER_INSTRUCTIONS),
        "output_config": {"effort": cfg.writer_effort},
        "messages": messages,
    }


def judge_params(cfg, sheet_text: str, report: str) -> dict:
    content = f"Fact sheet:\n{sheet_text}\n\nReport:\n{report}"
    return {
        "model": cfg.judge_model,
        "max_tokens": cfg.judge_max_tokens,
        "system": _system(JUDGE_SYSTEM, JUDGE_INSTRUCTIONS),
        "output_config": {
            "effort": cfg.judge_effort,
            "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
        },
        "messages": [{"role": "user", "content": content}],
    }


# Tokens and money

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_write_5m",
    "cache_write_1h",
)


def usage_of(message) -> dict[str, int]:
    """The usage fields of a reply as plain integers."""
    u = getattr(message, "usage", None)
    out = {k: int(getattr(u, k, 0) or 0) for k in USAGE_FIELDS[:4]}
    created = getattr(u, "cache_creation", None)
    w5 = int(getattr(created, "ephemeral_5m_input_tokens", 0) or 0)
    w1 = int(getattr(created, "ephemeral_1h_input_tokens", 0) or 0)
    if created is None or w5 + w1 != out["cache_creation_input_tokens"]:
        w5, w1 = 0, out["cache_creation_input_tokens"]  # we only ask for the hour
    out["cache_write_5m"], out["cache_write_1h"] = w5, w1
    return out


def billed_tokens(usage: dict) -> int:
    return sum(usage.get(k, 0) for k in USAGE_FIELDS[:4])


def cost_usd(usage: dict, model: str, cfg, batch: bool = True) -> float:
    """USD of one reply (or summed usage) from its usage fields: input at the
    input rate, cache writes at 1.25 times (five minutes) or 2 times (one
    hour), cache reads at 0.1 times, output at the output rate; half of it
    through the Message Batches API."""
    price = cfg.prices[model]
    inp = (
        usage.get("input_tokens", 0)
        + cfg.cache_write_5m * usage.get("cache_write_5m", 0)
        + cfg.cache_write_1h * usage.get("cache_write_1h", 0)
        + cfg.cache_read * usage.get("cache_read_input_tokens", 0)
    )
    total = (inp * price.input + usage.get("output_tokens", 0) * price.output) / 1e6
    return total * (cfg.batch_discount if batch else 1.0)


def estimate_tokens(params: dict) -> int:
    """An upper bound on a request's input tokens: one per two bytes of its
    text (English runs nearer four bytes a token, digits about two)."""
    blob = json.dumps([params.get("system"), params.get("messages")], ensure_ascii=False)
    return len(blob.encode()) // 2 + 1


def worst_case(params: dict) -> int:
    return estimate_tokens(params) + int(params["max_tokens"])


def _add(total: dict, usage: dict) -> None:
    for k, v in usage.items():
        total[k] = total.get(k, 0) + v


@dataclass
class Budget:
    """The hard token cap of one run."""

    cap: int
    used: int = 0
    held_back: int = 0

    def take(self, requests: list[dict], reserve: int = 0) -> tuple[list[dict], list[dict]]:
        """The requests whose worst case still fits under the cap (keeping
        ``reserve`` free), in order; the rest are held back."""
        sent, held, planned = [], [], self.used + reserve
        for r in requests:
            cost = worst_case(r["params"])
            if planned + cost <= self.cap:
                sent.append(r)
                planned += cost
            else:
                held.append(r)
        self.held_back += len(held)
        return sent, held

    def spend(self, usage: dict) -> None:
        self.used += billed_tokens(usage)


# The Message Batches API


@dataclass
class Reply:
    custom_id: str
    status: str  # succeeded, refused, truncated, errored, canceled, expired, missing
    text: str = ""
    usage: dict = field(default_factory=dict)
    error: str = ""


def _reply(item) -> Reply:
    result = item.result
    kind = getattr(result, "type", "missing")
    if kind != "succeeded":
        error = getattr(getattr(result, "error", None), "type", "") or ""
        return Reply(item.custom_id, kind, error=str(error))
    message = result.message
    text = "".join(
        getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text"
    ).strip()
    usage = usage_of(message)
    stop = getattr(message, "stop_reason", None)
    status = {"refusal": "refused", "max_tokens": "truncated"}.get(stop, "succeeded")
    return Reply(item.custom_id, status, text, usage)


CANCEL_GRACE = 600  # seconds for a cancelled batch to end and give what it finished


@dataclass
class Batches:
    """Sends a list of requests as one batch and waits for it, until the
    run's deadline; a batch still running then is cancelled, and the
    requests it did finish are kept."""

    client: object
    poll_seconds: float
    deadline: float  # on ``clock``
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    created: list[str] = field(default_factory=list)

    def run(self, requests: list[dict]) -> dict[str, Reply]:
        if not requests:
            return {}
        if self.clock() > self.deadline:
            log.warning("reports: past the run's deadline, %d requests not sent", len(requests))
            return {r["custom_id"]: Reply(r["custom_id"], "missing") for r in requests}
        batches = self.client.messages.batches
        batch = batches.create(requests=requests)
        self.created.append(batch.id)
        log.info("reports: batch %s, %d requests", batch.id, len(requests))
        cancelled_at = None
        batch = batches.retrieve(batch.id)
        while batch.processing_status != "ended":
            now = self.clock()
            if cancelled_at is None and now > self.deadline:
                log.warning("reports: batch %s past the run's deadline, cancelled", batch.id)
                batches.cancel(batch.id)
                cancelled_at = now
            elif cancelled_at is not None and now > cancelled_at + CANCEL_GRACE:
                log.warning("reports: batch %s did not end after its cancel", batch.id)
                break
            self.sleep(self.poll_seconds)
            batch = batches.retrieve(batch.id)
        replies = {}
        if batch.processing_status == "ended":
            replies = {item.custom_id: _reply(item) for item in batches.results(batch.id)}
        for r in requests:
            replies.setdefault(r["custom_id"], Reply(r["custom_id"], "missing"))
        return replies


# Writing and checking


@dataclass
class Outcome:
    player_id: int
    report: str
    source: str  # model or template
    status: str  # passed, failed, held (cap or deadline: tried again next run)
    attempts: int
    first_try: bool | None  # passed the check on the first try; None when never written
    violations: list[str]
    generated_at: str


def write_reports(
    sheets: list[FactSheet],
    cfg,
    runner: Batches,
    budget: Budget,
    gazetteer: Iterable[str] = (),
    reserve: int = 0,
    now: str = "",
    usage: dict | None = None,
) -> dict[int, Outcome]:
    """The writer's rounds: every sheet once, then the failures again with
    their violations, ``cfg.retries`` times at most."""
    gazetteer = set(gazetteer)
    by_id = {s.player_id: s for s in sheets}
    state = {s.player_id: {"previous": "", "problems": [], "attempts": 0} for s in sheets}
    done: dict[int, Outcome] = {}
    first: dict[int, bool] = {}
    todo = [s.player_id for s in sheets]
    for round_ in range(cfg.retries + 1):
        if not todo:
            break
        requests = [
            {
                "custom_id": f"w{round_}-{pid}",
                "params": writer_params(
                    cfg, by_id[pid], state[pid]["previous"], state[pid]["problems"]
                ),
            }
            for pid in todo
        ]
        sent, held = budget.take(requests, reserve)
        for r in held:
            pid = int(r["custom_id"].split("-", 1)[1])
            done[pid] = _held(by_id[pid], state[pid], first.get(pid), now, "the token cap")
        replies = runner.run(sent)
        retry = []
        for r in sent:
            pid = int(r["custom_id"].split("-", 1)[1])
            reply = replies[r["custom_id"]]
            budget.spend(reply.usage)
            if usage is not None:
                _add(usage, reply.usage)
            st = state[pid]
            if reply.status == "missing" or reply.status in ("canceled", "expired"):
                done[pid] = _held(by_id[pid], st, first.get(pid), now, "the run's deadline")
                continue
            st["attempts"] += 1
            if reply.status == "succeeded" and reply.text:
                result = check(reply.text, by_id[pid], cfg.words_min, cfg.words_max, gazetteer)
                problems = result.violations
            else:
                problems = [f"the request ended as {reply.status} {reply.error}".strip()]
            first.setdefault(pid, not problems)
            if not problems:
                done[pid] = Outcome(pid, reply.text, "model", "passed", st["attempts"],
                                    first[pid], [], now)  # fmt: skip
                continue
            st["previous"], st["problems"] = reply.text, problems
            retry.append(pid)
        todo = retry
    for pid in todo:
        st = state[pid]
        sheet = by_id[pid]
        log.warning(
            "reports: %s (%d) failed the check %d times: %s; the template is shown",
            sheet.player,
            pid,
            st["attempts"],
            "; ".join(st["problems"]),
        )
        done[pid] = Outcome(pid, sheet.template, "template", "failed", st["attempts"],
                            first.get(pid), st["problems"], now)  # fmt: skip
    return done


def _held(sheet: FactSheet, st: dict, first, now: str, why: str) -> Outcome:
    return Outcome(sheet.player_id, sheet.template, "template", "held", st["attempts"], first,
                   [f"not written: {why}"], now)  # fmt: skip


@dataclass
class Verdict:
    faithful: bool
    offending_span: str
    hedging_adequate: bool
    readability: int
    reason: str


def parse_verdict(text: str) -> Verdict | None:
    try:
        data = json.loads(text)
        verdict = Verdict(
            bool(data["faithful"]),
            str(data.get("offending_span", "")),
            bool(data["hedging_adequate"]),
            int(data["readability"]),
            str(data.get("reason", "")),
        )
    except (ValueError, KeyError, TypeError):
        return None
    return verdict if 1 <= verdict.readability <= 5 else None


def judge_reports(
    items: list[tuple[FactSheet, str]],
    cfg,
    runner: Batches,
    budget: Budget,
    usage: dict | None = None,
) -> dict[int, Verdict]:
    requests = [
        {"custom_id": f"j-{sheet.player_id}", "params": judge_params(cfg, sheet.text, report)}
        for sheet, report in items
    ]
    sent, held = budget.take(requests)
    if held:
        log.warning("reports: %d judge requests held back by the token cap", len(held))
    replies = runner.run(sent)
    verdicts = {}
    for r in sent:
        reply = replies[r["custom_id"]]
        budget.spend(reply.usage)
        if usage is not None:
            _add(usage, reply.usage)
        verdict = parse_verdict(reply.text) if reply.status == "succeeded" else None
        if verdict is None:
            log.warning("reports: no verdict for %s (%s)", r["custom_id"], reply.status)
            continue
        verdicts[int(r["custom_id"].split("-", 1)[1])] = verdict
    return verdicts


# Storage

REPORT_COLUMNS = {
    "player_id": "The player's API Football id, as in the two tables.",
    "player": "The player's name as API Football writes it.",
    "tables": "The tables he is in: candidates, bargains, or both (one report a player).",
    "fact_hash": "A fingerprint of his fact sheet, the prompt version and the writer model; the "
    "report is written again only when it changes, and the page shows a report only when it "
    "matches today's sheet.",
    "fact_sheet": "The facts the report was written from, one per line, all taken from the two "
    "tables, the similar players and the season so far.",
    "report": "What the page shows: the written report when it passed the check, otherwise the "
    "template explanation of the tables.",
    "source": "model when the report was written by the writer model and passed the check; "
    "template when the template explanation is shown instead.",
    "status": "passed (written and checked), failed (failed the check after every retry, the "
    "template is shown) or held (not written this run because of the token cap or the deadline; "
    "tried again next run).",
    "model": "The writer model's id.",
    "generated_at": "When the report was written (UTC).",
    "attempts": "How many times the writer was asked for this sheet (one, plus at most two "
    "retries with the check's findings).",
    "first_try": "True when the first report passed the check.",
    "violations": "What the check found on the last failed attempt: numbers not in the fact "
    "sheet or rounded wrongly, names not in it, the length; blank when it passed.",
    "judge_model": "The judge model's id, for the reports in a weekly sample.",
    "judged_at": "When the judge read the report (UTC).",
    "judge_faithful": "The judge's verdict: every claim is supported by the fact sheet.",
    "judge_offending_span": "The passage the judge found unsupported; blank when faithful.",
    "judge_hedging_adequate": "The judge's verdict: every chance reads as a historical rate, "
    "never as a prediction.",
    "judge_readability": "The judge's readability score for a football fan, 1 (confusing) to 5 "
    "(clear and natural).",
    "judge_reason": "The judge's one sentence reason.",
}
JUDGE_COLUMNS = [c for c in REPORT_COLUMNS if c.startswith("judge")] + ["judged_at"]


def dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [{"table": "reports", "column": c, "meaning": m} for c, m in REPORT_COLUMNS.items()]
    )


def out_dir(processed: Path) -> Path:
    return Path(processed) / OUT


def load_stored(processed: Path) -> pd.DataFrame:
    path = out_dir(processed) / "reports.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=list(REPORT_COLUMNS))
    return pd.read_parquet(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    new = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
    atomic_write_text(path, old + new)


# The run


def inputs(settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, str]]:
    """The two tables, today's neighbours and the league names."""
    from scout.embeddings import OUT as EMBEDDINGS
    from scout.lists import league_names

    processed = settings.paths.processed_dir
    lists = processed / "lists"
    cand = pd.read_parquet(lists / "candidates.parquet")
    barg = pd.read_parquet(lists / "bargains.parquet")
    near_path = processed / EMBEDDINGS / "neighbours.parquet"
    near = pd.read_parquet(near_path) if near_path.is_file() else None
    return cand, barg, near, league_names(settings)


def gazetteer(cand: pd.DataFrame, barg: pd.DataFrame, near, settings) -> set[str]:
    """Every name the tables know, folded: players, clubs, leagues and cups.
    Used only to tell a name that opens a sentence from an ordinary word."""
    values = set()
    for frame, columns in (
        (cand, ["player", "club_at_forecast", "current_club", "league", "current_league"]),
        (barg, ["player", "club_at_forecast", "current_club", "league", "current_league"]),
        (near, ["neighbour", "neighbour_club"]),
    ):
        if frame is None:
            continue
        for c in columns:
            if c in frame:
                values.update(str(v) for v in frame[c].dropna().unique())
    values.update(lg.name for lg in settings.leagues)
    values.update(c.name for c in settings.competitions)
    return {f for f in map(fold, values) if len(f) > 2 and f not in STARTERS}


def plan(sheets: list[FactSheet], stored: pd.DataFrame, model: str) -> tuple[list, dict]:
    """The sheets to write (new or changed since the last run, or held back
    then) and the stored rows to keep, by player."""
    keep, todo = {}, []
    by_id = {}
    if not stored.empty:
        by_id = {int(r["player_id"]): r for _, r in stored.iterrows()}
    for sheet in sheets:
        old = by_id.get(sheet.player_id)
        if (
            old is not None
            and str(old["fact_hash"]) == sheet.hash(model)
            and str(old["status"]) in ("passed", "failed")
        ):
            keep[sheet.player_id] = old
        else:
            todo.append(sheet)
    return todo, keep


# For estimates only (``--dry-run``, docs/DEPLOY.md); a run's cost comes from its usage fields.
BYTES_PER_TOKEN = 3.3  # English and figures on the Claude 5 tokenizers, roughly
WRITER_OUTPUT_GUESS = 450  # a 150 word report is about 220 tokens; the rest is thinking
JUDGE_OUTPUT_GUESS = 800  # the verdict is short; thinking at medium effort is most of it


def estimate_run(settings) -> dict:
    """What a run would send, without sending it: the sheets to write, the
    worst case the cap counts, and a rough cost for a full regeneration and
    for this run (every prefix read from the cache, no retries; with no
    cache hit at all, add the prefix at the input rate each time)."""
    cfg = settings.reports_text
    cand, barg, near, leagues = inputs(settings)
    sheets = fact_sheets(cand, barg, near, leagues)
    todo, keep = plan(sheets, load_stored(settings.paths.processed_dir), cfg.writer_model)
    prefix = len((WRITER_SYSTEM + WRITER_INSTRUCTIONS).encode()) / BYTES_PER_TOKEN
    judge_prefix = len((JUDGE_SYSTEM + JUDGE_INSTRUCTIONS).encode()) / BYTES_PER_TOKEN

    def writer_cost(items: list[FactSheet]) -> float:
        sheet = sum(len(s.text.encode()) for s in items) / BYTES_PER_TOKEN
        usage = {
            "input_tokens": sheet,
            "cache_read_input_tokens": prefix * max(len(items) - 1, 0),
            "cache_write_1h": prefix if items else 0,
            "output_tokens": WRITER_OUTPUT_GUESS * len(items),
        }
        return cost_usd(usage, cfg.writer_model, cfg)

    n = min(cfg.judge_sample, len(sheets))
    judge_in = (prefix / 4 + judge_prefix + 2 * 1612 / BYTES_PER_TOKEN) * n
    judge = cost_usd(
        {"input_tokens": judge_in, "output_tokens": JUDGE_OUTPUT_GUESS * n}, cfg.judge_model, cfg
    )
    worst = sum(worst_case(writer_params(cfg, s)) for s in todo)
    return {
        "fact sheets": len(sheets),
        "to write": len(todo),
        "reused": len(keep),
        "prefix tokens (estimate)": round(prefix),
        "sheet tokens, mean (estimate)": round(
            sum(len(s.text.encode()) for s in sheets) / BYTES_PER_TOKEN / max(len(sheets), 1)
        ),
        "worst case tokens of the first round (the cap counts these)": f"{worst:,}",
        "token cap": f"{cfg.max_tokens_per_run:,}",
        "full regeneration, writer (estimate, USD)": round(writer_cost(sheets), 2),
        "this run, writer (estimate, USD)": round(writer_cost(todo), 2),
        f"judge, {n} reports (estimate, USD)": round(judge, 2),
    }


def judge_sample(rows: pd.DataFrame, size: int, seed: str) -> list[int]:
    """Up to ``size`` written reports that no judge has read, at random
    (seeded by the run's day, so a rerun picks the same)."""
    if rows.empty or size <= 0:
        return []
    pool = rows[(rows["source"] == "model") & rows["judged_at"].isna()]
    ids = sorted(int(p) for p in pool["player_id"])
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(size, len(ids))))


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s")


def run_reports(settings, client=None, now: pd.Timestamp | None = None, sleep=time.sleep) -> dict:
    """``scout reports``. Without a client and without ``ANTHROPIC_API_KEY``
    it logs one line and returns an empty summary: the page then shows the
    reports still matching their sheets and the template elsewhere."""
    cfg = settings.reports_text
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            log.warning(
                "reports: ANTHROPIC_API_KEY is not set; skipped (the page shows the stored "
                "reports that still match their facts, the template elsewhere)"
            )
            return {}
        import anthropic

        client = anthropic.Anthropic()
    now = now or _now()
    stamp = now.isoformat()
    processed = settings.paths.processed_dir
    cand, barg, near, leagues = inputs(settings)
    sheets = fact_sheets(cand, barg, near, leagues)
    names = gazetteer(cand, barg, near, settings)
    stored = load_stored(processed)
    todo, keep = plan(sheets, stored, cfg.writer_model)
    log.info("reports: %d fact sheets, %d to write, %d reused", len(sheets), len(todo), len(keep))
    budget = Budget(cfg.max_tokens_per_run)
    runner = Batches(client, cfg.poll_seconds, time.monotonic() + cfg.max_run_minutes * 60, sleep)
    reserve = cfg.judge_sample * (cfg.judge_max_tokens + 6000)  # the judge's worst case, kept free
    writer_usage, judge_usage = {}, {}
    outcomes = write_reports(todo, cfg, runner, budget, names, reserve, stamp, writer_usage)

    rows = []
    by_sheet = {s.player_id: s for s in sheets}
    for sheet in sheets:
        pid = sheet.player_id
        if pid in outcomes:
            o = outcomes[pid]
            rows.append(
                {
                    "player_id": pid,
                    "player": sheet.player,
                    "tables": sheet.tables,
                    "fact_hash": sheet.hash(cfg.writer_model),
                    "fact_sheet": sheet.text,
                    "report": o.report,
                    "source": o.source,
                    "status": o.status,
                    "model": cfg.writer_model,
                    "generated_at": o.generated_at,
                    "attempts": o.attempts,
                    "first_try": o.first_try,
                    "violations": "; ".join(o.violations),
                    **dict.fromkeys(JUDGE_COLUMNS),
                }
            )
        else:
            rows.append({c: keep[pid].get(c) for c in REPORT_COLUMNS})
    table = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))

    sample = judge_sample(table, cfg.judge_sample, now.strftime("%Y-%m-%d"))
    items = [
        (by_sheet[pid], table.loc[table["player_id"] == pid, "report"].iloc[0]) for pid in sample
    ]
    verdicts = judge_reports(items, cfg, runner, budget, judge_usage) if items else {}
    for pid, v in verdicts.items():
        at = table["player_id"] == pid
        table.loc[at, "judge_model"] = cfg.judge_model
        table.loc[at, "judged_at"] = stamp
        table.loc[at, "judge_faithful"] = v.faithful
        table.loc[at, "judge_offending_span"] = v.offending_span
        table.loc[at, "judge_hedging_adequate"] = v.hedging_adequate
        table.loc[at, "judge_readability"] = v.readability
        table.loc[at, "judge_reason"] = v.reason

    summary = summarize(outcomes, table, verdicts, writer_usage, judge_usage, budget, cfg, stamp)
    summary["batches"] = runner.created
    save(processed, table, outcomes, verdicts, summary, stamp)
    path = write_owner_report(settings.paths.reports_dir, table, sample, verdicts, summary)
    log.info("reports: %s", cost_line(summary))
    log.info("Report: %s", path)
    return summary


def summarize(outcomes, table, verdicts, writer_usage, judge_usage, budget, cfg, stamp) -> dict:
    written = [o for o in outcomes.values() if o.first_try is not None]
    writer_cost = cost_usd(writer_usage, cfg.writer_model, cfg) if writer_usage else 0.0
    judge_cost = cost_usd(judge_usage, cfg.judge_model, cfg) if judge_usage else 0.0
    shown = table["source"].value_counts()
    return {
        "run_at": stamp,
        "sheets": len(table),
        "reused": len(table) - len(outcomes),
        "written": len(written),
        "passed_first_try": sum(bool(o.first_try) for o in written),
        "failed_first_try": sum(not o.first_try for o in written),
        "passed_after_retries": sum(o.status == "passed" and not o.first_try for o in written),
        "failed_after_retries": sum(o.status == "failed" for o in outcomes.values()),
        "held": sum(o.status == "held" for o in outcomes.values()),
        "shown_model": int(shown.get("model", 0)),
        "shown_template": int(shown.get("template", 0)),
        "judged": len(verdicts),
        "judge_faithful": sum(v.faithful for v in verdicts.values()),
        "judge_hedging_adequate": sum(v.hedging_adequate for v in verdicts.values()),
        "judge_readability": {
            str(k): sum(v.readability == k for v in verdicts.values()) for k in range(1, 6)
        },
        "tokens_used": budget.used,
        "token_cap": budget.cap,
        "held_by_cap": budget.held_back,
        "writer_model": cfg.writer_model,
        "judge_model": cfg.judge_model,
        "writer_usage": writer_usage,
        "judge_usage": judge_usage,
        "writer_cost_usd": round(writer_cost, 4),
        "judge_cost_usd": round(judge_cost, 4),
        "cost_usd": round(writer_cost + judge_cost, 4),
    }


def cost_line(summary: dict) -> str:
    """The one line of the weekly run's summary."""
    if not summary:
        return "Reports: skipped (no ANTHROPIC_API_KEY)."
    return (
        f"Reports: {summary['written']} written ({summary['failed_first_try']} failed the check "
        f"first, {summary['failed_after_retries']} still after retries, {summary['held']} held), "
        f"{summary['reused']} reused, {summary['judged']} judged; cost ${summary['cost_usd']:.2f} "
        f"(writer ${summary['writer_cost_usd']:.2f}, judge ${summary['judge_cost_usd']:.2f}); "
        f"{summary['tokens_used']:,} of {summary['token_cap']:,} tokens."
    )


TYPES = {
    "player_id": "Int64",
    "attempts": "Int64",
    "judge_readability": "Int64",
    "first_try": "boolean",
    "judge_faithful": "boolean",
    "judge_hedging_adequate": "boolean",
}


def typed(table: pd.DataFrame) -> pd.DataFrame:
    """One dtype a column, whether its rows were kept or written this run."""
    out = table.copy()
    for c in out.columns:
        kind = TYPES.get(c, "string")
        values = out[c].astype(object).where(out[c].notna(), None)
        out[c] = pd.array(values.tolist(), dtype=kind)
    return out


def save(processed: Path, table, outcomes, verdicts, summary, stamp) -> Path:
    folder = out_dir(processed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "reports.parquet"
    tmp = path.with_name(path.name + ".tmp")
    typed(table).to_parquet(tmp, index=False)
    tmp.replace(path)
    dictionary().to_csv(folder / "dictionary.csv", index=False)
    atomic_write_text(folder / "summary.json", json.dumps(summary, indent=2) + "\n")
    _append_jsonl(folder / "runs.jsonl", [summary])
    failed = [
        {
            "run_at": stamp,
            "player_id": o.player_id,
            "attempts": o.attempts,
            "violations": o.violations,
        }
        for o in outcomes.values()
        if o.status == "failed"
    ]
    _append_jsonl(folder / "failures.jsonl", failed)
    judged = table[table["player_id"].isin(list(verdicts))]
    _append_jsonl(
        folder / "verdicts.jsonl",
        [
            {c: r[c] for c in ["player_id", "player", "fact_hash", "report", *JUDGE_COLUMNS]}
            for _, r in judged.iterrows()
        ],
    )
    return path


# For the owner


def load_agreement(path: Path = AGREEMENT) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=AGREEMENT_COLUMNS)
    return pd.read_csv(path, dtype={"owner_agrees": "string", "note": "string"})


def agreement(path: Path = AGREEMENT) -> dict | None:
    """The owner's agreement with the judge, once he has recorded any."""
    table = load_agreement(path)
    said = table["owner_agrees"].dropna().str.strip().str.lower()
    said = said[said.isin(["yes", "no"])]
    if said.empty:
        return None
    return {"read": len(said), "agreed": int((said == "yes").sum())}


def owner_order(table: pd.DataFrame, sample: list[int], verdicts: dict) -> list[int]:
    """The judged reports the owner reads: the ones the judge faulted first,
    then the rest in the sample's order, ``OWNER_READS`` at most."""
    judged = [pid for pid in sample if pid in verdicts]
    faulted = [p for p in judged if not (verdicts[p].faithful and verdicts[p].hedging_adequate)]
    return (faulted + [p for p in judged if p not in faulted])[:OWNER_READS]


def write_owner_report(
    reports_dir: Path, table: pd.DataFrame, sample: list[int], verdicts: dict, summary: dict
) -> Path:
    """``reports/reports_text.md`` and ``reports/judge_sample.csv``: the
    run in numbers, the judge's verdicts, and 20 judged reports with their
    facts for the owner to read and agree or disagree with."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    when = summary["run_at"][:10]
    read = owner_order(table, sample, verdicts)
    rows = table.set_index("player_id")
    sheet_rows = []
    parts = [
        "# Scout reports",
        "",
        f"Run of {when} (`scout reports`, `src/scout/reports_text.py`). Each player of the two",
        "tables has a report of 120 to 150 words written by",
        f"{model_name(summary['writer_model'])} from his fact sheet, checked in code: every",
        "number must be a number of the sheet within its rounding and every name a name of",
        "the sheet. A failing report goes back with the findings at most twice; then the",
        "template explanation is shown.",
        "",
        "## The run",
        "",
        f"* Fact sheets: {summary['sheets']:,}; reused unchanged: {summary['reused']:,}; "
        f"written: {summary['written']:,}.",
        f"* Failed the check on the first try: {summary['failed_first_try']:,}; passed after a "
        f"retry: {summary['passed_after_retries']:,}; still failing after the retries (template "
        f"shown): {summary['failed_after_retries']:,}; held back (token cap or deadline): "
        f"{summary['held']:,}.",
        f"* On the page: {summary['shown_model']:,} written reports, "
        f"{summary['shown_template']:,} template explanations.",
        f"* {cost_line(summary)}",
        "",
        f"## The judge ({model_name(summary['judge_model'])}, {summary['judged']} reports)",
        "",
    ]
    if summary["judged"]:
        r = summary["judge_readability"]
        parts += [
            f"* Faithful: {summary['judge_faithful']} of {summary['judged']}.",
            f"* Hedging adequate: {summary['judge_hedging_adequate']} of {summary['judged']}.",
            "* Readability: " + ", ".join(f"{k}: {r[str(k)]}" for k in range(5, 0, -1)) + ".",
            "",
        ]
        spans = [(p, v) for p, v in verdicts.items() if not v.faithful]
        if spans:
            parts += ["Unfaithful passages:", ""]
            parts += [
                f'* {rows.at[p, "player"]}: "{v.offending_span}" ({v.reason})' for p, v in spans
            ]
            parts.append("")
    else:
        parts += ["No report was judged in this run.", ""]
    parts += [
        f"## For the owner: {len(read)} judged reports",
        "",
        "Read each report against its facts, then the judge's verdict. In",
        "`reports/judge_sample.csv` write yes in `owner_agrees` where you agree with the",
        "verdict (all three parts) and no where you do not, with a note if you like, and",
        "append those rows to `judge/agreement.csv` (committed). The page's \"how to read",
        'this" box shows the agreement rate once that file holds an answer.',
        "",
    ]
    for n, pid in enumerate(read, start=1):
        row, v = rows.loc[pid], verdicts[pid]
        parts += [
            f"### {n}. {row['player']} ({row['tables']})",
            "",
            "```text",
            row["fact_sheet"],
            "```",
            "",
            "> " + row["report"].replace("\n", "\n> "),
            "",
            f"Judge: faithful {'yes' if v.faithful else 'no'}"
            + (f' ("{v.offending_span}")' if v.offending_span else "")
            + f", hedging adequate {'yes' if v.hedging_adequate else 'no'}, readability "
            f"{v.readability}. {v.reason}",
            "",
        ]
        sheet_rows.append(
            {
                "judged_on": when,
                "player_id": pid,
                "player": row["player"],
                "fact_hash": row["fact_hash"][:12],
                "judge_faithful": "yes" if v.faithful else "no",
                "judge_hedging_adequate": "yes" if v.hedging_adequate else "no",
                "judge_readability": v.readability,
                "owner_agrees": "",
                "note": "",
            }
        )
    pd.DataFrame(sheet_rows, columns=AGREEMENT_COLUMNS).to_csv(
        reports_dir / "judge_sample.csv", index=False
    )
    path = reports_dir / "reports_text.md"
    atomic_write_text(path, "\n".join(parts))
    return path


# The page


def page_reports(
    settings, cand: pd.DataFrame, barg: pd.DataFrame, agreement_path: Path = AGREEMENT
) -> dict:
    """``data/reports.json``: for every player of the tables his written
    report when the stored one matches today's fact sheet, else his template
    explanation; the label, and the judge and owner numbers for the "how to
    read this" box."""
    from scout.embeddings import OUT as EMBEDDINGS
    from scout.lists import league_names

    cfg = settings.reports_text
    processed = settings.paths.processed_dir
    stored = load_stored(processed)
    agreed = agreement(agreement_path)
    if stored.empty:
        return {"available": False, "agreement": agreed}
    near_path = processed / EMBEDDINGS / "neighbours.parquet"
    near = pd.read_parquet(near_path) if near_path.is_file() else None
    sheets = fact_sheets(cand, barg, near, league_names(settings))
    by_id = {int(r["player_id"]): r for _, r in stored.iterrows()}
    rows, written = {}, 0
    for sheet in sheets:
        old = by_id.get(sheet.player_id)
        fresh = old is not None and str(old["fact_hash"]) == sheet.hash(cfg.writer_model)
        if fresh and str(old["source"]) == "model":
            rows[str(sheet.player_id)] = ["report", old["report"]]
            written += 1
        else:
            rows[str(sheet.player_id)] = ["template", sheet.template]
    judged = stored[stored["judged_at"].notna()] if "judged_at" in stored else stored.iloc[:0]
    return {
        "available": True,
        "label": label(cfg.writer_model),
        "writer": model_name(cfg.writer_model),
        "judge": model_name(cfg.judge_model),
        "written": written,
        "template": len(rows) - written,
        "judged": len(judged),
        "judge_faithful": int(judged["judge_faithful"].fillna(False).astype(bool).sum()),
        "agreement": agreed,
        "rows": rows,
    }
