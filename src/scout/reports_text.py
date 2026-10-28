"""Generated scout reports with a faithfulness check (prompt 19).

``scout reports`` runs after ``scout lists``. For every player in the two
tables (one report a player, also when he is in both):

* **Fact sheet.** ``fact_sheets`` turns the table columns into one fact per
  line: age, position, club at forecast and today, league, value and fair
  value, ability percentile, each chance with its count of past players and
  what it measured, the flags with their raw numbers, the rising tag, the
  three explanation stats, the three most similar players of today (nearest
  first) and, on the bargains, the list, the minutes and the contract. Since
  prompt 19c it holds only what the report may cite, the forecast and the
  club today: the season so far changes with every match and is shown next
  to the report on the page instead, so a match never rewrites a report. The
  sheet is the only variable part of the request; the system prompt and the
  shared instructions are a fixed prefix with prompt caching.
* **Writer.** Claude Haiku 5.5 writes about 130 words (never more than 150,
  with an example report of that length in the prompt) for a fan, through
  the Message Batches API. A report is regenerated only when its fact sheet
  changed (``fact_hash``: the sheet, the prompt version and the model);
  otherwise the stored one is reused.
* **Check, in code** (``check``). Every number of the report (with its percent
  sign or unit) must match a number of the fact sheet within the rounding the
  report wrote it with and, since prompt 19c, of the kind its words say (a
  chance's rate, a share of fair value, of minutes, a percentile...); a
  similar player's number only where he is named, a rate with the count of
  its own chance. Every proper noun must be a name of the fact sheet (no
  first name added to an initial), only the first similar player may be
  called the closest, cheap and dear must follow the fair value ratio, and
  the season so far stays out; the length must be 90 to 170 words (prompt 19b: the first real run
  held 120 to 150 and sent 94% of the players to the template, nearly all
  for a few words over). A failing report goes back to the writer with the
  violations, its word count and the cut needed, at most twice; after that
  the template explanation of ``scout.explain`` is shown and the failure is
  logged (``failures.jsonl``). A player whose report ended on the template
  is asked for again on the next run, once (``template_runs``), then not
  until the prompt version or the writer model changes.
* **Judge.** Once in ``judge_min_days`` (6) a sample of reports not judged
  before (``judge_sample``, 50) goes to Claude Sonnet 5.5 with the sheet's
  field names and a structured output: faithful, every unsupported span with
  the field it contradicts, hedging adequate, readability 1 to 5. Code
  checks each span against the report and each field against the sheet
  (``verify``); a verdict that fails is set aside. The verdicts are stored,
  and ``reports/reports_text.md`` lays out 20 of them for the owner, whose
  agreement goes to ``judge/agreement.csv`` (committed) and from there to the
  page; ``judge_test`` scores the judge against the hand sorting of
  ``judge/sorting.csv``.
* **Slow batches** (prompt 19c). The step waits at most ``max_run_minutes``
  (20); a batch still running is kept in ``pending.json`` and collected by
  the next run before anything new is sent, its replies checked, retried,
  judged and stored as a fast batch's are.
* **Budget.** ``max_tokens_per_run`` is a hard cap: before a batch is sent,
  each request's worst case (its input estimated from above, plus its
  ``max_tokens``) is added to what the run has used, and requests that could
  pass the cap are held back (their players show the template this week).
  The cost of the run is computed from the usage fields of every reply.

Everything lands in ``data/processed/reports_text/``: ``reports.parquet``
(one row a player), ``runs.jsonl``, ``verdicts.jsonl``, ``failures.jsonl``,
``summary.json``, ``dictionary.csv`` and ``pending.json``. ``scout site`` rebuilds the fact
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
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from scout.fsutil import atomic_write_text

log = logging.getLogger(__name__)

OUT = "reports_text"  # under data/processed
PROMPT_VERSION = "19.3"  # part of every fact hash: a new prompt regenerates every report
SIMILAR_SHOWN = 3
WORDS_TARGET = 130  # what the prompt asks for
WORDS_ASKED = 150  # the most the prompt allows; the check allows ``words_max`` (170)
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
    for the exact values of the tables); ``kind``: what a percentage or a
    percentile measures (``KINDS``) when the words around it say so;
    ``field``: the key of the sheet's fact it belongs to (sheet numbers only);
    ``at``: where it starts in the report (report numbers only)."""

    value: float
    unit: str = "plain"
    tag: str | None = None
    step: float = 0.0
    text: str = ""
    kind: str | None = None
    field: str | None = None
    at: int = -1


@dataclass(frozen=True)
class Fact:
    label: str
    text: str
    numbers: tuple[Num, ...] = ()
    names: tuple[str, ...] = ()  # the proper nouns this fact holds
    people: tuple[str, ...] = ()  # names of people: a surname alone is allowed too

    @property
    def key(self) -> str:
        return field_key(self.label)


# The field name of each line of a sheet: what the judge names when a report
# contradicts a line (prompt 19c). ``season_so_far`` is only on sheets of
# prompt version 19.2 and before, kept so their verdicts can be read again.
LABEL_KEYS = {
    "Player": "player",
    "Season described": "season",
    "Age": "age",
    "Position": "position",
    "Club at forecast": "club_at_forecast",
    "Club today": "club_today",
    "Value rank": "value_rank",
    "Bargain list": "bargain_list",
    "Market value": "market_value",
    "Fair value (our model)": "fair_value",
    "Ability": "ability",
    "Minutes": "minutes",
    "Elite chance": "elite_chance",
    "Step up chance": "step_up_chance",
    "Big move chance": "big_move_chance",
    "Fall back risk": "fall_back_risk",
    "Warning flags": "warning_flags",
    "Rising": "rising",
    "Standout stats": "standout_stats",
    "Contract until": "contract",
    "Tag": "tag",
    "Season so far": "season_so_far",
    "Similar players today": "similar_players",
}
FIELD_KEYS = list(LABEL_KEYS.values())
CHANCE_KEYS = ("elite_chance", "step_up_chance", "big_move_chance", "fall_back_risk")


def field_key(label: str) -> str:
    return LABEL_KEYS.get(label, fold(label).replace(" ", "_"))


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

    @property
    def keyed_text(self) -> str:
        """The sheet as the judge reads it: each line opened by its field name."""
        return keyed(self.text)

    def numbers(self) -> list[Num]:
        """Every number of the sheet with the field it belongs to: the ones
        given with each fact, and the ones its text holds (flags and stats
        written as sentences), whose kind is read from the words around them
        just as in a report. Read once a sheet: the check asks for them for
        every number of every attempt."""
        cached = self.__dict__.get("_numbers")
        if cached is None or cached[0] != len(self.facts):
            found = []
            for f in self.facts:
                for n in (*f.numbers, *extract_numbers(f"{f.label}: {f.text}")):
                    found.append(replace(n, field=f.key))
            cached = self.__dict__["_numbers"] = (len(self.facts), found)
        return cached[1]

    def similar(self) -> list[str]:
        """The similar players, nearest first."""
        for f in self.facts:
            if f.key == "similar_players":
                return list(f.people)
        return []

    def fact(self, key: str) -> Fact | None:
        return next((f for f in self.facts if f.key == key), None)

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
        nums.append(Num(float(pct), "pct", kind="chance"))
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
        nums += [Num(float(age), tag="age"), Num(float(r.neighbour_ability_pct), kind="percentile")]
    # The order is said (prompt 19c): run 9's judge faulted "X is the closest"
    # because the sheet did not say who was, and four of nine such reports had
    # picked the second or third.
    text = "nearest first: " + "; ".join(parts) + " (alike in playing style, not in level)"
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
            shown = round(ratio)
            side = "above" if shown > 100 else "below" if shown < 100 else "at"
            text += f"; his market value is {ratio:.0f}% of it ({side} our fair value)"
            nums.append(Num(ratio, "pct", kind="fair_ratio"))
        if barg is not None and _present(barg.get("value_to_fair_pct")):
            # Run 9's judge read "among the cheapest 97.2%" as cheap; past the
            # middle the line now counts from the dear end.
            cheap = float(barg["value_to_fair_pct"])
            if cheap <= 50:
                text += f", among the cheapest {cheap:g}% of listed players of his position group"
                nums.append(Num(cheap, "pct", kind="cheapest"))
            else:
                dear = round(100 - cheap, 1)
                text += f", among the dearest {dear:g}% of listed players of his position group"
                nums.append(Num(dear, "pct", kind="cheapest"))
        facts.append(Fact("Fair value (our model)", text, tuple(nums)))
    if _present(row.get("ability_pct")):
        ability = round(float(row["ability_pct"]))
        facts.append(
            Fact(
                "Ability",
                f"{ordinal(ability)} percentile among regular players of his position in "
                f"{season_label(int(row['season']))}, from his output per 90 minutes over the last "
                "2 seasons (100 is the best)",
                (Num(float(row["ability_pct"]), kind="percentile"),),
            )
        )
    if barg is not None and _present(barg.get("minutes_share")):
        share = float(barg["minutes_share"]) * 100
        facts.append(
            Fact(
                "Minutes",
                f"{share:.0f}% of his club's league minutes",
                (Num(share, "pct", kind="minutes_share"),),
            )
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
    # No season so far (prompt 19c): it changes with every match, so a sheet
    # holding it would be written again every day; the page shows it as its
    # own block next to the report.
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

# Straight and curly apostrophes, the modifier letter and the left quote some
# sources use in N'Guessan or O'Brien: one mark, never a word break (prompt 19c).
APOSTROPHES = "'\u2019\u2018\u02bc"
WORD = re.compile(r"[^\W\d_](?:[\w" + APOSTROPHES + r"-]*[^\W_])?")


def fold(text: str) -> str:
    """Lower case without accents, hyphens and punctuation as spaces; an
    apostrophe inside a name is dropped, not a space ("N'Guessan" is
    "nguessan"), and a possessive "'s" goes."""
    text = re.sub("[" + APOSTROPHES + "]", "'", str(text))
    for a, b in (("ı", "i"), ("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("đ", "d"), ("ł", "l")):
        text = text.replace(a, b).replace(a.upper(), b.upper())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
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
PREV_TAGS = {"aged": "age", "age": "age", "decile": "decile", "rank": "rank", "ranked": "rank",
             "rating": "rating"}  # fmt: skip
EURO_AFTER = re.compile(r"\s*(euros?|eur)\b", re.IGNORECASE)
LETTERS = re.compile(r"[A-Za-z][A-Za-z'-]*")


CLAUSE = re.compile(r"[.;:!?,](?:\s|$)|[()]")


def _around(text: str, start: int, end: int) -> tuple[list[str], list[str]]:
    """The lower case words before (up to three) and after (up to five) a
    number, within its clause (a comma ends one too)."""
    after = text[end : end + 60]
    cut = CLAUSE.search(after)
    after = after[: cut.start()] if cut else after
    before = text[max(0, start - 40) : start]
    cuts = list(CLAUSE.finditer(before))
    before = before[cuts[-1].end() :] if cuts else before
    return (
        [w.lower() for w in LETTERS.findall(before)][-3:],
        [w.lower() for w in LETTERS.findall(after)][:5],
    )


def _tag(text: str, start: int, end: int) -> str | None:
    """What a count counts, from the words next to it within its clause: a
    word before it that names a count ("aged 24 to 26", "decile 9 of 10",
    "a match rating of 7.28") first, then one after it ("142 players")."""
    if text[max(0, start - 1) : start] == "#":
        return "rank"
    before, after = _around(text, start, end)
    if "rating" in after[:3]:  # "a 7.28 average match rating" counts no matches
        return "rating"
    if before[-1:] and before[-1] in PREV_TAGS:
        return PREV_TAGS[before[-1]]
    linked = before[-1:] and before[-1] in ("to", "of", "or")
    if linked and before[-2:-1] and before[-2] in PREV_TAGS:
        return PREV_TAGS[before[-2]]
    for i, w in enumerate(after[:2]):
        if w == "per":  # "0.4 per 90 minutes" is a rate, not minutes
            break
        # "aged 21 to 23 this season" counts no seasons: those two only right after
        if w in NEXT_TAGS and (i == 0 or NEXT_TAGS[w] not in ("seasons", "age")):
            return NEXT_TAGS[w]
    return None


# What a percentage or a percentile measures (prompt 19c), read from the words
# around it in a report and in the sheet's own sentences alike:
KINDS = {
    "chance": "a historical rate of a chance or of the fall back risk",
    "fair_ratio": "market value as a share of fair value",
    "cheapest": "his place among the listed players by value against fair value",
    "minutes_share": "a share of team minutes, or the flag's threshold",
    "value_change": "the change of his value on a year before",
    "peer_rank": "top or bottom so many percent of his peers",
    "stat": "a share of his own actions (passes, duels, his team's shots)",
    "percentile": "an ability percentile",
}
OUTCOME_VERBS = {"became", "become", "saw", "joined", "went", "made", "reached", "entered", "got"}
FALLING = {"fell", "dropped", "down", "lost", "falling", "fallen", "drop", "fall", "declined"}


def _kind(text: str, start: int, end: int, unit: str) -> str | None:
    b, a = _around(text, start, end)
    if unit != "pct":
        if a[:1] == ["percentile"] or b[-1:] == ["percentile"] or b[-2:] == ["percentile", "of"]:
            return "percentile"
        return None
    if b[-1:] and b[-1] in ("top", "bottom"):
        return "peer_rank"
    if set(b[-2:]) & {"cheapest", "dearest"}:
        return "cheapest"
    if (a[:1] and a[0] in OUTCOME_VERBS | {"rate", "chance", "risk"}) or (
        a[:2] == ["of", "them"] and a[2:3] and a[2] in OUTCOME_VERBS
    ):  # not "completes 65% of them"
        return "chance"
    if a[:1] == ["of"] and "minutes" in a and "per" not in a[: a.index("minutes")]:
        return "minutes_share"
    if b[-1:] and b[-1] in ("played", "plays", "playing"):
        return "minutes_share"
    fair_words = {"fair", "model", "model's"} & set(a)
    if a[:1] == ["of"] and (
        a[1:2] in (["it"], ["that"], ["this"], ["our"], ["fair"], ["what"]) or fair_words
    ):
        return "fair_ratio"
    if (a[:1] and a[0] in ("lower", "down", "less")) or a[:3] == ["on", "a", "year"]:
        return "value_change"
    if set(b) & FALLING:
        return "value_change"
    if a[:1] == ["of"] and a[1:2] and a[1] in ("his", "the", "all", "team's"):
        return "stat"
    return None


def extract_numbers(text: str) -> list[Num]:
    """Every number with its unit, its precision, what it counts, what kind
    of percentage it is, and where it stands."""
    found = []
    labels = [m.span() for m in SEASON_LABEL.finditer(text)]  # checked as labels, not numbers
    for m in NUMBER.finditer(text):
        if any(a <= m.start() < b for a, b in labels):
            continue
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
                _kind(text, m.start(), m.end(), unit),
                at=m.start(),
            )
        )
    for m in WORD_NUMBER.finditer(text):
        value = float(NUMBER_WORDS[m[1].lower()])
        found.append(Num(value, "plain", _tag(text, m.start(), m.end()), 1.0, m[1], at=m.start()))
    return found


def _compatible(said: Num, fact: Num) -> bool:
    """Typed (prompt 19c): a percentage whose words say what it measures
    matches only a sheet percentage of that kind, a percentile only a
    percentile, money only money; a count that names what it counts never
    matches a sheet number that names something else (prompt 19). A number
    whose words say nothing matches any sheet number of its unit."""
    if "eur" in (said.unit, fact.unit) and said.unit != fact.unit:
        return False
    if said.unit == "pct" and fact.unit != "pct":
        return False
    if said.unit == "ordinal" and fact.unit not in ("plain", "ordinal"):
        return False
    if said.kind is not None and fact.kind != said.kind:
        return False
    if said.unit in ("pct", "eur"):
        return True
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
        for s in [a + "s" for a in APOSTROPHES]:
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
    for given in given_names(text, sheet):
        if not any(given in b or b in given for b in bad):
            bad.append(given)
    return bad


def given_names(text: str, sheet: FactSheet) -> list[str]:
    """A first name the report put before a surname the sheet gives with an
    initial ("Harry Armstrong" for H. Armstrong). Opening a sentence it passed
    the name check as a possible ordinary word; run 9's judge found four."""
    found = []
    for person in (sheet.player, *sheet.similar()):
        m = re.match(r"^([^\W\d_])\.\s+(.+)$", str(person))
        if not m:
            continue
        initial, rest = fold(m[1]), m[2]
        pattern = r"(?<![\w.])([^\W\d_][\w" + APOSTROPHES + r"-]+)\s+" + re.escape(rest) + r"(?!\w)"
        for hit in re.finditer(pattern, text):
            word = fold(hit[1])
            if word[:1] == initial and word not in STARTERS and hit[1][0].isupper():
                found.append(hit.group(0))
    return found


# Claims (prompt 19c): where run 9's judge found real errors that neither the
# number nor the name check could see.

SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\b[^\W\d_]\.)(?<!\bSt\.)(?<!\s\d\.)\s+(?=[A-Z0-9\"'(])")
CLOSEST = re.compile(r"\b(closest|nearest|most similar|most alike)\b", re.IGNORECASE)
SO_FAR = re.compile(r"\bso far\b|\bseason in progress\b", re.IGNORECASE)
SEASON_LABEL = re.compile(r"\b20\d\d/\d\d\b")
CHEAP = re.compile(r"\b(cheaply|below)\b", re.IGNORECASE)
DEAR = re.compile(r"\b(dearly|above)\b", re.IGNORECASE)
CHANCE_WORDS = {
    "elite_chance": ("elite", "100 most valuable"),
    "step_up_chance": ("step up", "times better"),
    "big_move_chance": ("big move", "top 20 club"),
    "fall_back_risk": ("fall back", "times worse", "as many warning flags", "as many flags"),
}


def sentences(text: str) -> list[tuple[int, int]]:
    """Where each sentence starts and ends; an initial ("M. Jaber"), "St."
    and "1. FC Köln" do not end one."""
    out, start = [], 0
    for m in SENTENCE_END.finditer(text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return out


def _mentions(person: str, sentence: str, own: set[str]) -> bool:
    """The sentence names this similar player: his full name or one of his
    names (not an initial, nor a name he shares with the player himself)."""
    folded = f" {fold(sentence)} "
    name = fold(person)
    if f" {name} " in folded:
        return True
    parts = [p for p in name.split() if len(p) > 2 and p not in own]
    return any(f" {p} " in folded for p in parts)


def claim_violations(report: str, sheet: FactSheet, said: list[Num]) -> list[str]:
    """The claims the sheet's own order and fields decide."""
    out = []
    similar = sheet.similar()
    own = set(fold(sheet.player).split())
    spans = sentences(report)
    for start, end in spans:
        text = report[start:end]
        if similar and CLOSEST.search(text):
            named = [i for i, p in enumerate(similar) if _mentions(p, text, own)]
            if named and named != list(range(len(named))):
                who = " and ".join(similar[i] for i in named)
                out.append(
                    f"the report calls {who} the closest in playing style, but the sheet lists "
                    f"the similar players nearest first and {similar[0]} is the first"
                )
        out += _chance_pairs(text, sheet, [n for n in said if start <= n.at < end], start)
        out += _direction(text, sheet)
    if sheet.fact("season_so_far") is None and SO_FAR.search(report):
        out.append(
            "the report mentions the season so far, which is not in the fact sheet (the page "
            "shows it next to the report); describe only the season of the sheet"
        )
    described = sheet.fact("season")
    own_season = set(SEASON_LABEL.findall(described.text)) if described else set()
    for label_ in sorted(set(SEASON_LABEL.findall(report)) - own_season):
        out.append(f'the season "{label_}" is not the season of the fact sheet')
    return out


def _chance_pairs(text: str, sheet: FactSheet, said: list[Num], offset: int) -> list[str]:
    """Each count of past players and the rate after it must be the count and
    the rate of one chance; when the words name the chance, of that one."""
    counts = [n for n in said if n.tag == "players"]
    rates = [n for n in said if n.kind == "chance"]
    nums = {k: [n for n in sheet.numbers() if n.field == k] for k in CHANCE_KEYS}
    out, begin = [], offset
    for i, count in enumerate(counts):
        stop = counts[i + 1].at if i + 1 < len(counts) else offset + len(text)
        rate = next((r for r in rates if count.at < r.at < stop), None)
        if rate is None:
            continue
        holds = [k for k in CHANCE_KEYS if number_in(count, nums[k]) and number_in(rate, nums[k])]
        segment = text[begin - offset : stop - offset].lower()
        named = [k for k, ws in CHANCE_WORDS.items() if any(w in segment for w in ws)]
        begin = rate.at + len(rate.text)
        if not holds:
            out.append(
                f'the rate "{rate.text}" and the count "{count.text}" are not the rate and the '
                "count of one chance on the fact sheet"
            )
        elif len(named) == 1 and named[0] not in holds:
            out.append(
                f'the rate "{rate.text}" of {count.text} past players is not the '
                f"{sheet.fact(named[0]).label.lower() if sheet.fact(named[0]) else named[0]}"
            )
    return out


def _unnegated(pattern: re.Pattern, text: str) -> re.Match | None:
    """The first match not after a "not" or a "than" ("not priced cheaply",
    "dearly rather than cheaply")."""
    for m in pattern.finditer(text):
        before = [w.lower() for w in LETTERS.findall(text[max(0, m.start() - 30) : m.start()])]
        if not set(before[-3:]) & {"not", "never", "neither", "nor", "than", "isn't", "wasn't"}:
            return m
    return None


def _direction(text: str, sheet: FactSheet) -> list[str]:
    """Cheap is below fair value and dear above it."""
    if "fair" not in text.lower():
        return []
    ratio = next(
        (n.value for n in sheet.numbers() if n.kind == "fair_ratio" and n.field == "fair_value"),
        None,
    )
    if ratio is None:
        return []
    cheap, dear = (_unnegated(p, text) for p in (CHEAP, DEAR))
    shown = round(ratio)
    if cheap and not dear and shown >= 100:
        word = cheap.group(0)
    elif dear and not cheap and shown <= 100:
        word = dear.group(0)
    else:
        return []
    return [
        f'the report says "{word}", but his market value is {shown}% of his fair value '
        "(below 100% is cheap, above it dear)"
    ]


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
    words_min: int = 90,
    words_max: int = 170,
    gazetteer: Iterable[str] = (),
) -> Check:
    """Every number of the report in the fact sheet within its rounding and
    of the kind its words say (prompt 19c), a similar player's number only in
    a sentence that names a similar player, every proper noun a name of the
    sheet, the claims of ``claim_violations``, and the length in range."""
    violations = []
    allowed = sheet.numbers()
    own = [n for n in allowed if n.field != "similar_players"]
    similar = sheet.similar()
    mine = set(fold(sheet.player).split())
    said = extract_numbers(report)
    bounds = sentences(report)
    for num in said:
        start, end = next(((s, e) for s, e in bounds if s <= num.at < e), (0, len(report)))
        names_similar = any(_mentions(p, report[start:end], mine) for p in similar)
        pool = allowed if names_similar else own
        if number_in(num, pool):
            continue
        loose = replace(num, kind=None, tag=None)
        if num.kind and number_in(loose, pool):
            what = f", but not as {KINDS[num.kind]}"
        elif number_in(num, allowed):
            what = " only for a similar player, and this sentence names none"
        elif num.tag and number_in(loose, pool):
            what = f", but not as a count of {num.tag}"
        else:
            what = None
        violations.append(
            f'the number "{num.text}" is on the fact sheet{what}'
            if what
            else f'the number "{num.text}" is not in the fact sheet, or not rounded correctly'
        )
    spans = _spans(report)
    for name in unknown_names(report, sheet, gazetteer):
        violations.append(f'the name "{name}" is not in the fact sheet')
    violations += claim_violations(report, sheet, said)
    n = words(report)
    if not words_min <= n <= words_max:
        violations.append(f"the report has {n} words; it must have {words_min} to {words_max}")
    return Check(not violations, violations, n, len(said), len(spans))


def length_note(n: int, words_min: int = 90) -> str | None:
    """The retry's word count and the cut (or addition) it needs, for any
    report outside what the prompt asks, whatever else failed: the first
    real run's retries were told the limit but not the count to cut, and
    grew instead of shrinking."""
    if n > WORDS_ASKED:
        return (
            f"your report has {n} words; the limit is {WORDS_ASKED}: cut at least "
            f"{n - WORDS_ASKED} words, and aim for about {WORDS_TARGET} (cut about "
            f"{n - WORDS_TARGET}) by dropping the least telling facts, not by squeezing sentences"
        )
    if n < words_min:
        return (
            f"your report has {n} words; write about {WORDS_TARGET} (add about {WORDS_TARGET - n})"
        )
    return None


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


# What the sheet's labels mean: the writer is told it, and the judge reads the
# sheet the same way (run 9's judge faulted 8 reports that followed it).
LABEL_MEANINGS = _unwrap(
    """
- Ability: where his output per 90 minutes ranks among the regular players of his position
in the season described; 100 is the best.
- Market value is the market's valuation. Fair value is what our model, trained on earlier
seasons, says a player with his record is worth. A market value above fair value (over 100%
of it) means the market prices him dearly, below it (under 100%) cheaply; neither is a
verdict on him. His place among the listed players ("among the cheapest" or "the dearest" so
many percent) never turns a value above fair value into a cheap one.
- Elite chance: past players in the same band who became one of the most valuable players in
the world. Only young players already among the more valuable get one. "His elite band",
"his step up band" and the like name the band of past players a chance was measured on.
- Step up chance: past players whose value rank became several times better.
- Big move chance: past players who joined one of the clubs with the most valuable squads.
- Fall back risk: past players with as many warning flags whose value rank became much worse.
- Warning flags: reasons for doubt, each with his own number: low minutes, injuries, a falling
value. They are the flags at the forecast.
- Rising: he is in the top group of a model that looks for players whose play may rise
clearly. It is a tag, not a rate.
- Standout stats: where he stands furthest from players of his position and age band; "top"
and "bottom" so many percent are written as the sheet writes them.
- Similar players today: players whose style of play is closest to his, nearest first: the
first one is the closest.
- Bargain list: players whose market value is low against their fair value; the prime list is
the younger one, the experienced list the older one.
- Not rated: the table gives no rate there. Say so if it matters, with the sheet's own
condition he does not meet (his rank or his age) if you give a reason; invent no other.
"""
)

WRITER_INSTRUCTIONS = _unwrap(
    """
How to write the report

1. Length and form. Write about 130 words, never more than 150, of plain prose in one or two
paragraphs. A report over 150 words is rejected, so leave out the least telling facts rather
than squeezing them all in: you do not have to use every line of the sheet. No title, no
headings, no bullet points, no bold, no emoji. Return only the report.

2. Use the fact sheet and nothing else. Every number you write must appear in the fact sheet,
as the fact it is there: a rate as that chance's rate, a percentile as his ability, a share of
minutes as his minutes. Copy numbers as the sheet writes them, as digits. You may round a
number only correctly and only to fewer decimals (0.47 may become 0.5, never 0.4). Do not
calculate anything new: no differences, sums, ratios or averages of your own, and no number
words such as "three" for a number the sheet writes as 3.

3. Names. Every name you write (player, club, league, competition, country or person) must
appear in the fact sheet, spelled exactly as it is there. Never add a first name to an
initial: write H. Armstrong, never Harry Armstrong, even when you know it. Do not mention a
nationality, a coach, a former club, a rumour, a data source or any player who is not in the
sheet. When unsure whether something is in the sheet, leave it out.

4. Odds are history, not predictions. Each chance and risk in the sheet is a historical rate:
of the past players our tables put in the same band, the share who later did something. Say
it that way, for example "of <count> past players in his band, <rate> became one of the 100
most valuable players within 2 seasons", with the count and the rate of the same line of the
sheet. Never write that he will do something, never call a rate his chance, probability or
likelihood, never say he is likely, set, bound, destined, guaranteed or certain to do
anything, and never promise a transfer or a rise. Keep the count of past players next to
each rate you mention.

5. What to cover. Open with who he is: age, position, club and league. Then the most telling
of: his ability percentile, his market value against the fair value, the rates that are
rated, the warning flags, the rising tag and the standout stats. Mention every warning flag
with its own number, or say he raises none. If he moved after the forecast, say so. The
sheet describes the season at its end and his club today, nothing of the season now in
progress: never write about this season's matches, minutes or form, nor "so far" (the page
shows them next to your report). You may name one or two of the similar players; if you do,
say the likeness is in playing style, not level, and call only the first one the closest.

6. What the labels mean.
- {labels}

7. Tone. Measured and concrete. Prefer football words to statistical ones. No hype, no
judgement of character, no advice to buy or sell.

8. An example of the length and the shape: 132 words, about a made up player from a made up
fact sheet. Never copy its names or its numbers; use only your own fact sheet.

Marco Ferri is a 21-year-old central midfielder at Riverton in the Example League, where he
still plays. His ability sits in the 84th percentile of the regular midfielders in 2025/26.
The market values him at €8m, below the €12.4m our model calls fair, so he is priced cheaply.
Of 156 past players in his elite band, 15% became one of the 100 most valuable players within
2 seasons, and of 880 in his step up band, 20% saw their value rank become at least 3 times
better. He raises one warning flag: he played 41% of his team's minutes. He wins 68% of his
duels, top 5% of midfielders aged 21 to 23. J. Ortiz, the first of his similar players, is the
closest to him in playing style, not in level.

The fact sheet comes in the next message, one fact per line as "label: value".
"""
).replace("- {labels}", LABEL_MEANINGS)

JUDGE_SYSTEM = (
    "You check short scouting reports that another model wrote for a football fan from a fact "
    "sheet of our own tables. You are strict, concrete and fair."
)

JUDGE_INSTRUCTIONS = _unwrap(
    """
Read the fact sheet, then the report, and give your verdicts. Each line of the sheet opens with
its field name in square brackets, for example "[elite_chance] Elite chance: ...".

faithful: true only when every claim of the report is supported by the fact sheet: each
number, each name, the direction of each comparison (market value above or below fair
value), which warning flags are raised, whether he moved, which chances are rated, what each
chance measured. A plain football paraphrase of a fact is fine. Anything the sheet does not
support is not: an added reason, a first name added to an initial, a nationality, a trait, a
transfer, a club or player not in the sheet, an invented or wrongly rounded number, a number
of one line told as the fact of another, a reversed comparison, a rate given to the wrong
outcome, a similar player called the closest who is not the first.

issues: one entry for each claim the sheet does not support, none when faithful is true and
at least one when it is false. Each entry has:
- span: the shortest exact passage of the report that makes the claim, copied character for
character;
- field: the field name of the sheet line the claim contradicts or misuses, or "none" when no
line covers it (an added name, reason, trait, transfer or season);
- problem: "contradicts" when a line says otherwise, "unsupported" when no line says it;
- explanation: one short sentence.
Code checks every span against the report and every field against the sheet; a verdict whose
span is not in the report or whose field is not on the sheet is set aside.

hedging_adequate: true when every chance and risk is presented as what happened to past
players in the same band (a historical rate), never as a prediction or a probability for this
player, and no sentence promises that he will do something.

readability: from 1 to 5 for a football fan. 5: clear, natural and well ordered. 4: clear
with small stiffness. 3: understandable but list like or awkward. 2: hard to follow. 1:
confusing.

reason: one short sentence on the most important point of your verdicts.

The writer was told what the labels mean; read the sheet the same way:
- {labels}
"""
).replace("- {labels}", LABEL_MEANINGS)

ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "span": {"type": "string"},
        "field": {"type": "string", "enum": [*FIELD_KEYS, "none"]},
        "problem": {"type": "string", "enum": ["contradicts", "unsupported"]},
        "explanation": {"type": "string"},
    },
    "required": ["span", "field", "problem", "explanation"],
    "additionalProperties": False,
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "issues": {"type": "array", "items": ISSUE_SCHEMA},
        "hedging_adequate": {"type": "boolean"},
        "readability": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["faithful", "issues", "hedging_adequate", "readability", "reason"],
    "additionalProperties": False,
}


def keyed(text: str) -> str:
    """A sheet's text with each line opened by its field name."""
    lines = []
    for line in text.split("\n"):
        label_, _, _ = line.partition(": ")
        lines.append(f"[{field_key(label_)}] {line}")
    return "\n".join(lines)


def sheet_keys(text: str) -> set[str]:
    return {field_key(line.partition(": ")[0]) for line in text.split("\n") if line}


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
    """The judge's request: the sheet with its field names, then the report."""
    content = f"Fact sheet:\n{keyed(sheet_text)}\n\nReport:\n{report}"
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


@dataclass
class Pending:
    """A batch still running when the step stopped waiting (prompt 19c):
    its id and what each of its requests was for, kept in ``pending.json``
    so the next run collects it before it sends anything new."""

    id: str
    kind: str  # writer or judge
    created_at: str
    requests: dict[str, dict]  # custom_id -> the player, his sheet's hash, the attempt


@dataclass
class Batches:
    """Sends a list of requests as one batch and waits for it until the
    step's deadline (``max_run_minutes``, 20). A batch still running then is
    not cancelled: it is kept in ``pending`` and its requests come back as
    pending, for the next run to collect (run 12 waited 100 minutes on one)."""

    client: object
    poll_seconds: float
    deadline: float  # on ``clock``
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    created: list[str] = field(default_factory=list)
    pending: list[Pending] = field(default_factory=list)
    now: str = ""

    def _wait(self, batch_id: str) -> dict[str, Reply] | None:
        batches = self.client.messages.batches
        batch = batches.retrieve(batch_id)
        while batch.processing_status != "ended":
            if self.clock() > self.deadline:
                return None
            self.sleep(self.poll_seconds)
            batch = batches.retrieve(batch_id)
        return {item.custom_id: _reply(item) for item in batches.results(batch_id)}

    def run(self, requests: list[dict], kind: str = "writer", meta=None) -> dict[str, Reply]:
        if not requests:
            return {}
        if self.clock() > self.deadline:
            log.warning("reports: past the step's deadline, %d requests not sent", len(requests))
            return {r["custom_id"]: Reply(r["custom_id"], "missing") for r in requests}
        batch = self.client.messages.batches.create(requests=requests)
        self.created.append(batch.id)
        log.info("reports: batch %s, %d requests", batch.id, len(requests))
        replies = self._wait(batch.id)
        if replies is None:
            log.warning(
                "reports: batch %s still running after the step's wait; the next run collects it",
                batch.id,
            )
            meta = meta or {}
            asked = {r["custom_id"]: meta.get(r["custom_id"], {}) for r in requests}
            self.pending.append(Pending(batch.id, kind, self.now, asked))
            return {cid: Reply(cid, "pending") for cid in asked}
        for r in requests:
            replies.setdefault(r["custom_id"], Reply(r["custom_id"], "missing"))
        return replies

    def collect(self, pending: Pending) -> dict[str, Reply] | None:
        """An earlier run's batch: its replies once it has ended (waiting for
        it within this step's deadline), None while it still runs. A batch
        that can no longer be read (results are kept 29 days) gives every
        request back as missing, to be written again."""
        try:
            replies = self._wait(pending.id)
        except Exception as exc:  # the SDK's not found and the like; never stop the run
            log.warning("reports: batch %s cannot be read (%s); asked for again", pending.id, exc)
            replies = {}
        if replies is None:
            return None
        for cid in pending.requests:
            replies.setdefault(cid, Reply(cid, "missing"))
        return replies


# Writing and checking


@dataclass
class Outcome:
    player_id: int
    report: str
    source: str  # model or template
    status: str  # passed, failed, held (cap or deadline), pending (in a batch still running)
    attempts: int
    first_try: bool | None  # passed the check on the first try; None when never written
    violations: list[str]
    generated_at: str
    words: list[int] = field(default_factory=list)  # each written attempt's word count


def write_reports(
    sheets: list[FactSheet],
    cfg,
    runner: Batches,
    budget: Budget,
    gazetteer: Iterable[str] = (),
    reserve: int = 0,
    now: str = "",
    usage: dict | None = None,
    carried: dict[int, tuple[dict, Reply]] | None = None,
) -> dict[int, Outcome]:
    """The writer's rounds: every sheet once, then the failures again with
    their violations, ``cfg.retries`` times at most. ``carried``: the replies
    of an earlier run's batch collected now, with what their request was
    (the attempt, the earlier word counts, the first try's result); each is
    checked as that attempt and its retries join this run's rounds, so a
    slow batch ends exactly as a fast one would have."""
    gazetteer = set(gazetteer)
    by_id = {s.player_id: s for s in sheets}
    carried = {pid: c for pid, c in (carried or {}).items() if pid in by_id}
    state = {
        s.player_id: {"previous": "", "feedback": [], "problems": [], "attempts": 0, "words": [],
                      "first": None}
        for s in sheets
    }  # fmt: skip
    done: dict[int, Outcome] = {}

    def take(pid: int, reply: Reply) -> bool:
        """Checks one reply; True when it goes back to the writer."""
        st, sheet = state[pid], by_id[pid]
        budget.spend(reply.usage)
        if usage is not None:
            _add(usage, reply.usage)
        if reply.status == "pending":
            done[pid] = _held(sheet, st, now, "a batch still running", "pending")
            return False
        if reply.status in ("missing", "canceled", "expired"):
            done[pid] = _held(sheet, st, now, "the run's deadline")
            return False
        st["attempts"] += 1
        note = None
        if reply.status == "succeeded" and reply.text:
            result = check(reply.text, sheet, cfg.words_min, cfg.words_max, gazetteer)
            problems = result.violations
            st["words"].append(result.words)
            note = length_note(result.words, cfg.words_min)
        else:
            problems = [f"the request ended as {reply.status} {reply.error}".strip()]
        if st["first"] is None:
            st["first"] = not problems
        if not problems:
            done[pid] = Outcome(pid, reply.text, "model", "passed", st["attempts"], st["first"],
                                [], now, st["words"])  # fmt: skip
            return False
        st["previous"], st["problems"] = reply.text, problems
        # What the writer is told: the violations, with the length one replaced
        # by the count and the cut, which also comes when only a number failed.
        feedback = [p for p in problems if not p.startswith("the report has ")]
        st["feedback"] = feedback + ([note] if note else [])
        if st["attempts"] <= cfg.retries:
            return True
        log.warning(
            "reports: %s (%d) failed the check %d times: %s; the template is shown",
            sheet.player,
            pid,
            st["attempts"],
            "; ".join(problems),
        )
        done[pid] = Outcome(pid, sheet.template, "template", "failed", st["attempts"],
                            st["first"], problems, now, st["words"])  # fmt: skip
        return False

    todo = []
    for pid, (meta, reply) in carried.items():
        st = state[pid]
        st["attempts"] = int(meta.get("attempts", 0))
        st["words"] = list(meta.get("words", []))
        st["first"] = meta.get("first")
        if take(pid, reply):
            todo.append(pid)
    todo += [s.player_id for s in sheets if s.player_id not in carried]
    while todo:
        requests, meta = [], {}
        for pid in todo:
            st, cid = state[pid], f"w{state[pid]['attempts']}-{pid}"
            params = writer_params(cfg, by_id[pid], st["previous"], st["feedback"])
            requests.append({"custom_id": cid, "params": params})
            meta[cid] = {
                "player_id": pid,
                "fact_hash": by_id[pid].hash(cfg.writer_model),
                "attempts": st["attempts"],
                "words": st["words"],
                "first": st["first"],
            }
        sent, held = budget.take(requests, reserve)
        for r in held:
            pid = int(r["custom_id"].split("-", 1)[1])
            done[pid] = _held(by_id[pid], state[pid], now, "the token cap")
        replies = runner.run(sent, "writer", meta)
        todo = []
        for r in sent:
            pid = int(r["custom_id"].split("-", 1)[1])
            if take(pid, replies[r["custom_id"]]):
                todo.append(pid)
    return done


def _held(sheet: FactSheet, st: dict, now: str, why: str, status: str = "held") -> Outcome:
    return Outcome(sheet.player_id, sheet.template, "template", status, st["attempts"],
                   st["first"], [f"not written: {why}"], now, st["words"])  # fmt: skip


@dataclass
class Verdict:
    faithful: bool
    offending_span: str  # the first issue's span; blank when faithful
    hedging_adequate: bool
    readability: int
    reason: str
    issues: list[dict] = field(default_factory=list)  # span, field, problem, explanation
    problems: list[str] = field(default_factory=list)  # what ``verify`` found; empty: checkable

    @property
    def checkable(self) -> bool:
        return not self.problems


def parse_verdict(text: str) -> Verdict | None:
    try:
        data = json.loads(text)
        issues = [
            {k: str(i[k]) for k in ("span", "field", "problem", "explanation")}
            for i in data["issues"]
        ]
        verdict = Verdict(
            bool(data["faithful"]),
            issues[0]["span"] if issues else "",
            bool(data["hedging_adequate"]),
            int(data["readability"]),
            str(data.get("reason", "")),
            issues,
        )
    except (ValueError, KeyError, TypeError):
        return None
    return verdict if 1 <= verdict.readability <= 5 else None


def _plain(text: str) -> str:
    """Whitespace, quotes, apostrophes and dashes made alike, for finding a
    quoted span in its report."""
    text = re.sub("[" + APOSTROPHES + "]", "'", str(text))
    text = re.sub("[\u201c\u201d\u201e]", '"', text)
    text = re.sub("[\u2010-\u2015]", "-", text)
    return " ".join(text.split()).strip(" .,;:")


def verify(verdict: Verdict, report: str, sheet_text: str) -> list[str]:
    """What makes a verdict uncheckable: faithful with issues or unfaithful
    without one, a span that is not in the report, a field that is not a line
    of the sheet (prompt 19c)."""
    found = []
    if verdict.faithful and verdict.issues:
        found.append("faithful, yet it lists issues")
    if not verdict.faithful and not verdict.issues:
        found.append("not faithful, yet it lists no issue")
    keys, body = sheet_keys(sheet_text), _plain(report)
    for issue in verdict.issues:
        span = _plain(issue["span"])
        if not span or span not in body:
            found.append(f'the span "{issue["span"]}" is not in the report')
        if issue["field"] != "none" and issue["field"] not in keys:
            found.append(f'the field "{issue["field"]}" is not a line of the sheet')
        if issue["field"] == "none" and issue["problem"] == "contradicts":
            found.append("it contradicts no field")
    return found


def verdict_from(reply: Reply, sheet_text: str, report: str) -> Verdict | None:
    """A reply's verdict with what ``verify`` found; None without one."""
    verdict = parse_verdict(reply.text) if reply.status == "succeeded" else None
    if verdict is not None:
        verdict.problems = verify(verdict, report, sheet_text)
        if verdict.problems:
            log.warning("reports: a verdict set aside: %s", "; ".join(verdict.problems))
    return verdict


def report_sha(report: str) -> str:
    return hashlib.sha256(str(report).encode()).hexdigest()[:16]


def judge_reports(
    items: list[tuple[FactSheet, str]],
    cfg,
    runner: Batches,
    budget: Budget,
    usage: dict | None = None,
) -> dict[int, Verdict]:
    requests, meta, by_cid = [], {}, {}
    for sheet, report in items:
        cid = f"j-{sheet.player_id}"
        requests.append({"custom_id": cid, "params": judge_params(cfg, sheet.text, report)})
        meta[cid] = {"player_id": sheet.player_id, "fact_hash": sheet.hash(cfg.writer_model),
                     "report_sha": report_sha(report)}  # fmt: skip
        by_cid[cid] = (sheet, report)
    sent, held = budget.take(requests)
    if held:
        log.warning("reports: %d judge requests held back by the token cap", len(held))
    replies = runner.run(sent, "judge", meta)
    verdicts = {}
    for r in sent:
        reply = replies[r["custom_id"]]
        budget.spend(reply.usage)
        if usage is not None:
            _add(usage, reply.usage)
        if reply.status == "pending":
            continue
        sheet, report = by_cid[r["custom_id"]]
        verdict = verdict_from(reply, sheet.text, report)
        if verdict is None:
            log.warning("reports: no verdict for %s (%s)", r["custom_id"], reply.status)
            continue
        verdicts[sheet.player_id] = verdict
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
    "tables and the similar players: the forecast and the club today, never the season so far "
    "(since prompt 19c).",
    "report": "What the page shows: the written report when it passed the check, otherwise the "
    "template explanation of the tables.",
    "source": "model when the report was written by the writer model and passed the check; "
    "template when the template explanation is shown instead.",
    "status": "passed (written and checked), failed (failed the check after every retry, the "
    "template is shown), held (not written this run because of the token cap or the deadline; "
    "tried again next run) or pending (in a batch still running when the run stopped waiting; "
    "the next run collects it).",
    "model": "The writer model's id.",
    "generated_at": "When the report was written (UTC).",
    "attempts": "How many times the writer was asked for this sheet in its last run (one, plus "
    "at most two retries with the check's findings).",
    "prompt_version": "The version of the writer's instructions the report was written with.",
    "template_runs": "Runs in a row this player's report ended on the template under this prompt "
    "version (0 once one passes); after 2 the runs stop asking for it until the prompt or the "
    "writer model changes.",
    "first_try": "True when the first report passed the check.",
    "violations": "What the check found on the last failed attempt: numbers not in the fact "
    "sheet or rounded wrongly, names not in it, the length; blank when it passed.",
    "judge_model": "The judge model's id, for the reports in a weekly sample.",
    "judged_at": "When the judge read the report (UTC).",
    "judge_faithful": "The judge's verdict: every claim is supported by the fact sheet.",
    "judge_offending_span": "The passage the judge found unsupported (its first issue); blank "
    "when faithful.",
    "judge_issues": "Every passage the judge found unsupported, as JSON: the span, the field of "
    "the sheet it contradicts (or none), contradicts or unsupported, and why.",
    "judge_checkable": "True when code found every span of the verdict in the report and every "
    "field on the sheet, and the verdict agrees with its issues; a verdict that is not is set "
    "aside and not counted.",
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


PENDING = "pending.json"  # the batches a run left running, for the next run to collect


def load_pending(processed: Path) -> list[Pending]:
    path = out_dir(processed) / PENDING
    if not path.is_file():
        return []
    return [Pending(**b) for b in json.loads(path.read_text(encoding="utf-8"))["batches"]]


def save_pending(processed: Path, pending: list[Pending]) -> None:
    folder = out_dir(processed)
    folder.mkdir(parents=True, exist_ok=True)
    body = {"batches": [p.__dict__ for p in pending]}
    atomic_write_text(folder / PENDING, json.dumps(body, indent=2, default=str) + "\n")


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


def _template_runs(old, model: str) -> int:
    """Runs in a row the stored row ended on the template under this prompt
    version and writer model; a row of the first run (prompt 19.1) had no
    count and another prompt, so it starts over."""
    if old is None or str(old.get("status")) != "failed":
        return 0
    if str(old.get("prompt_version")) != PROMPT_VERSION or str(old.get("model")) != model:
        return 0
    runs = old.get("template_runs")
    return 1 if runs is None or pd.isna(runs) else int(runs)


def plan(
    sheets: list[FactSheet], stored: pd.DataFrame, model: str, template_runs: int = 2
) -> tuple[list, dict]:
    """The sheets to write and the stored rows to keep, by player. A passed
    report is kept while its sheet is unchanged. A report that ended on the
    template is asked for again on the next run, sheet changed or not, until
    ``template_runs`` runs in a row ended so under this prompt version and
    writer model; then it is kept (the page shows the explanation written by
    rule) until the prompt or the model changes, not retried every week."""
    keep, todo = {}, []
    by_id = {}
    if not stored.empty:
        by_id = {int(r["player_id"]): r for _, r in stored.iterrows()}
    for sheet in sheets:
        old = by_id.get(sheet.player_id)
        if old is None:
            todo.append(sheet)
            continue
        unchanged = str(old["status"]) == "passed" and str(old["fact_hash"]) == sheet.hash(model)
        given_up = _template_runs(old, model) >= template_runs
        if unchanged or given_up:
            keep[sheet.player_id] = old
        else:
            todo.append(sheet)
    return todo, keep


# For estimates only (``--dry-run``, docs/DEPLOY.md); a run's cost comes from its usage fields.
BYTES_PER_TOKEN = 3.3  # English and figures on the Claude 5 tokenizers, roughly
WRITER_OUTPUT_GUESS = 450  # a 150 word report is about 220 tokens; the rest is thinking
JUDGE_OUTPUT_GUESS = 150  # runs 9 to 12 measured 110 to 125 a verdict; the issues add some


def estimate_run(settings) -> dict:
    """What a run would send, without sending it: the sheets to write, the
    worst case the cap counts, and a rough cost for a full regeneration and
    for this run (every prefix read from the cache, no retries; with no
    cache hit at all, add the prefix at the input rate each time)."""
    cfg = settings.reports_text
    cand, barg, near, leagues = inputs(settings)
    sheets = fact_sheets(cand, barg, near, leagues)
    todo, keep = plan(
        sheets, load_stored(settings.paths.processed_dir), cfg.writer_model, cfg.template_runs
    )
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


def judge_due(stored: pd.DataFrame, waiting: list[Pending], now: pd.Timestamp, days: float) -> bool:
    """The weekly sample is due when no sample was judged or sent to the
    judge in the last ``days`` days (prompt 19c: daily runs from 6 November
    must not judge 50 reports a day)."""
    stamps = []
    if not stored.empty and "judged_at" in stored:
        stamps += [pd.Timestamp(t) for t in stored["judged_at"].dropna()]
    stamps += [pd.Timestamp(p.created_at) for p in waiting if p.kind == "judge" and p.created_at]
    if not stamps:
        return True
    return now - max(stamps) >= pd.Timedelta(days=days) - pd.Timedelta(hours=1)


def _row(sheet: FactSheet, o: Outcome, model: str, old) -> dict:
    runs = _template_runs(old, model)
    return {
        "player_id": sheet.player_id,
        "player": sheet.player,
        "tables": sheet.tables,
        "fact_hash": sheet.hash(model),
        "fact_sheet": sheet.text,
        "report": o.report,
        "source": o.source,
        "status": o.status,
        "model": model,
        "generated_at": o.generated_at,
        "attempts": o.attempts,
        "prompt_version": PROMPT_VERSION,
        "template_runs": {"passed": 0, "failed": runs + 1}.get(o.status, runs),
        "first_try": o.first_try,
        "violations": "; ".join(o.violations),
        **dict.fromkeys(JUDGE_COLUMNS),
    }


def _apply(table: pd.DataFrame, pid: int, v: Verdict, model: str, stamp: str) -> None:
    at = table["player_id"] == pid
    table.loc[at, "judge_model"] = model
    table.loc[at, "judged_at"] = stamp
    table.loc[at, "judge_faithful"] = v.faithful
    table.loc[at, "judge_offending_span"] = v.offending_span
    table.loc[at, "judge_issues"] = json.dumps(v.issues, ensure_ascii=False)
    table.loc[at, "judge_checkable"] = v.checkable
    table.loc[at, "judge_hedging_adequate"] = v.hedging_adequate
    table.loc[at, "judge_readability"] = v.readability
    table.loc[at, "judge_reason"] = v.reason


def run_reports(
    settings,
    client=None,
    now: pd.Timestamp | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """``scout reports``. Without a client and without ``ANTHROPIC_API_KEY``
    it logs one line and returns an empty summary: the page then shows the
    reports still matching their sheets and the template elsewhere.

    The step waits at most ``max_run_minutes`` (20) on the Message Batches
    API. It first collects the batches an earlier run left running
    (``pending.json``): their writer replies are checked as the attempt they
    were and their verdicts applied, as if they had come back in time; a
    reply whose sheet changed since is dropped. A batch still running is
    left for the run after, and its players are not asked for again."""
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
    model = cfg.writer_model
    processed = settings.paths.processed_dir
    cand, barg, near, leagues = inputs(settings)
    sheets = fact_sheets(cand, barg, near, leagues)
    by_sheet = {s.player_id: s for s in sheets}
    names = gazetteer(cand, barg, near, settings)
    stored = load_stored(processed)
    before = {int(r["player_id"]): r for _, r in stored.iterrows()} if not stored.empty else {}
    runner = Batches(client, cfg.poll_seconds, clock() + cfg.max_run_minutes * 60, sleep, clock,
                     now=stamp)  # fmt: skip
    budget = Budget(cfg.max_tokens_per_run)
    writer_usage, judge_usage = {}, {}

    # What earlier runs left running, collected before anything new is sent.
    waiting = load_pending(processed)
    carried_w, carried_j, still, collected, stale = {}, {}, [], 0, 0
    for p in waiting:
        replies = runner.collect(p)
        if replies is None:
            still.append(p)
            continue
        log.info("reports: collected batch %s (%s, %d requests)", p.id, p.kind, len(replies))
        for cid, reply in replies.items():
            meta = p.requests.get(cid, {})
            pid = meta.get("player_id")
            sheet = by_sheet.get(pid)
            if sheet is None or meta.get("fact_hash") != sheet.hash(model):
                stale += 1  # his facts changed since: written again from today's sheet
                _add(writer_usage if p.kind == "writer" else judge_usage, reply.usage)
                budget.spend(reply.usage)
                continue
            collected += 1
            if p.kind == "writer":
                carried_w[pid] = (meta, reply)
            else:
                carried_j[pid] = (meta, reply, p.created_at)
    in_flight = {
        m["player_id"]
        for p in still
        if p.kind == "writer"
        for m in p.requests.values()
        if m.get("player_id") in by_sheet
        and m.get("fact_hash") == by_sheet[m["player_id"]].hash(model)
        and m.get("player_id") in before
    }
    judging = {m.get("player_id") for p in still if p.kind == "judge" for m in p.requests.values()}

    todo, keep = plan(sheets, stored, model, cfg.template_runs)
    for sheet in [t for t in todo if t.player_id in in_flight]:
        keep[sheet.player_id] = before[sheet.player_id]
    todo = [t for t in todo if t.player_id not in in_flight]
    log.info(
        "reports: %d fact sheets, %d to write (%d collected), %d reused, %d still in a batch",
        len(sheets), len(todo), len(carried_w), len(keep) - len(in_flight), len(in_flight),
    )  # fmt: skip
    reserve = cfg.judge_sample * (cfg.judge_max_tokens + 6000)  # the judge's worst case, kept free
    outcomes = write_reports(
        todo, cfg, runner, budget, names, reserve, stamp, writer_usage, carried=carried_w
    )

    rows = []
    for sheet in sheets:
        pid = sheet.player_id
        if pid in outcomes:
            rows.append(_row(sheet, outcomes[pid], model, before.get(pid)))
        else:
            rows.append({c: keep[pid].get(c) for c in REPORT_COLUMNS})
    table = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))

    # The verdicts of a judge batch an earlier run left running, on the
    # reports they read, if those are still the ones shown.
    verdicts: dict[int, Verdict] = {}
    for pid, (meta, reply, sent_at) in carried_j.items():
        budget.spend(reply.usage)
        _add(judge_usage, reply.usage)
        row = table.loc[table["player_id"] == pid].iloc[0]
        if row["source"] != "model" or report_sha(row["report"]) != meta.get("report_sha"):
            continue
        verdict = verdict_from(reply, row["fact_sheet"], row["report"])
        if verdict is not None:
            verdicts[pid] = verdict
            _apply(table, pid, verdict, cfg.judge_model, sent_at or stamp)

    due = not carried_j and judge_due(stored, waiting, now, cfg.judge_min_days)
    sample = []
    if due:
        pool = table[~table["player_id"].isin(judging)]
        sample = judge_sample(pool, cfg.judge_sample, now.strftime("%Y-%m-%d"))
        items = [
            (by_sheet[pid], table.loc[table["player_id"] == pid, "report"].iloc[0])
            for pid in sample
        ]
        fresh = judge_reports(items, cfg, runner, budget, judge_usage) if items else {}
        for pid, v in fresh.items():
            _apply(table, pid, v, cfg.judge_model, stamp)
        verdicts.update(fresh)
    else:
        log.info("reports: the judge's sample is not due (one a %g days)", cfg.judge_min_days)

    pending = still + runner.pending
    summary = summarize(outcomes, table, verdicts, writer_usage, judge_usage, budget, cfg, stamp)
    summary["batches"] = runner.created
    summary["collected"] = collected
    summary["collected_stale"] = stale
    summary["pending_players"] = len(in_flight) + sum(
        o.status == "pending" for o in outcomes.values()
    )
    summary["batches_pending"] = [p.id for p in pending]
    summary["judge_due"] = due
    save(processed, table, outcomes, verdicts, summary, stamp)
    save_pending(processed, pending)
    shown = sample or sorted(verdicts)
    path = write_owner_report(settings.paths.reports_dir, table, shown, verdicts, summary)
    log.info("reports: %s", cost_line(summary))
    log.info("Report: %s", path)
    return summary


def word_spread(counts: list[int]) -> dict:
    """Least, 10th percentile, median, 90th percentile and most."""
    if not counts:
        return {}
    q = pd.Series(counts, dtype=float).quantile([0.1, 0.5, 0.9])
    return {"n": len(counts), "min": min(counts), "p10": round(q[0.1]),
            "median": round(q[0.5]), "p90": round(q[0.9]), "max": max(counts)}  # fmt: skip


def summarize(outcomes, table, verdicts, writer_usage, judge_usage, budget, cfg, stamp) -> dict:
    written = [o for o in outcomes.values() if o.first_try is not None]
    first_words = [o.words[0] for o in outcomes.values() if o.words]
    passed_words = [o.words[-1] for o in outcomes.values() if o.status == "passed" and o.words]
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
        "words_first_try": word_spread(first_words),
        "over_asked_first_try": sum(n > WORDS_ASKED for n in first_words),
        "outside_check_first_try": sum(
            not cfg.words_min <= n <= cfg.words_max for n in first_words
        ),
        "words_passed": word_spread(passed_words),
        "shown_model": int(shown.get("model", 0)),
        "shown_template": int(shown.get("template", 0)),
        "judged": len(verdicts),
        "judge_checkable": sum(v.checkable for v in verdicts.values()),
        "judge_faithful": sum(v.faithful for v in verdicts.values() if v.checkable),
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
        f"{summary['reused']} reused, {summary.get('collected', 0)} collected from an earlier "
        f"batch, {summary.get('pending_players', 0)} in a batch still running, "
        f"{summary['judged']} judged; cost ${summary['cost_usd']:.2f} "
        f"(writer ${summary['writer_cost_usd']:.2f}, judge ${summary['judge_cost_usd']:.2f}); "
        f"{summary['tokens_used']:,} of {summary['token_cap']:,} tokens."
    )


TYPES = {
    "player_id": "Int64",
    "attempts": "Int64",
    "template_runs": "Int64",
    "judge_readability": "Int64",
    "first_try": "boolean",
    "judge_faithful": "boolean",
    "judge_hedging_adequate": "boolean",
    "judge_checkable": "boolean",
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
            {
                c: r[c]
                for c in [
                    "player_id",
                    "player",
                    "fact_hash",
                    "fact_sheet",
                    "report",
                    *JUDGE_COLUMNS,
                ]
            }
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


def _spread_text(spread: dict | None) -> str:
    if not spread:
        return "none"
    return (f"median {spread['median']} (10th to 90th percentile {spread['p10']} to "
            f"{spread['p90']}, {spread['min']} to {spread['max']})")  # fmt: skip


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
        "tables has a report of about 130 words (90 to 170 pass the check) written by",
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
        f"* Words on the first try: {_spread_text(summary.get('words_first_try'))}; over "
        f"{WORDS_ASKED}: {summary.get('over_asked_first_try', 0):,}; outside the check's "
        f"range: {summary.get('outside_check_first_try', 0):,}. Passed reports: "
        f"{_spread_text(summary.get('words_passed'))}.",
        f"* On the page: {summary['shown_model']:,} written reports, "
        f"{summary['shown_template']:,} template explanations.",
        f"* {cost_line(summary)}",
        "",
        f"## The judge ({model_name(summary['judge_model'])}, {summary['judged']} reports)",
        "",
    ]
    if summary["judged"]:
        r = summary["judge_readability"]
        checkable = summary.get("judge_checkable", summary["judged"])
        parts += [
            f"* Checkable (every span found in its report, every field on its sheet): "
            f"{checkable} of {summary['judged']}; the rest are set aside.",
            f"* Faithful: {summary['judge_faithful']} of the {checkable} checkable.",
            f"* Hedging adequate: {summary['judge_hedging_adequate']} of {summary['judged']}.",
            "* Readability: " + ", ".join(f"{k}: {r[str(k)]}" for k in range(5, 0, -1)) + ".",
            "",
        ]
        faulted = [(p, v) for p, v in verdicts.items() if not v.faithful or not v.checkable]
        if faulted:
            parts += ["Unfaithful passages, with the field of the sheet each one contradicts:", ""]
            for p, v in faulted:
                issues = "; ".join(
                    f'"{i["span"]}" ({i["field"]}, {i["problem"]}: {i["explanation"]})'
                    for i in v.issues
                )
                aside = f" Set aside: {'; '.join(v.problems)}." if v.problems else ""
                parts.append(f"* {rows.at[p, 'player']}: {issues or v.reason}.{aside}")
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
            + "".join(f' ("{i["span"]}", field {i["field"]})' for i in v.issues)
            + ("" if v.checkable else f" (set aside: {'; '.join(v.problems)})")
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


# The judge against a hand sorting (prompt 19c)

SORTING = Path("judge") / "sorting.csv"  # committed: run 9's 24 unfaithful verdicts, sorted by hand
SCORED = {"real_error": False, "judge_mistake": True}  # the verdict the sorting expects


def _overlaps(a: str, b: str) -> bool:
    a, b = _plain(a).lower(), _plain(b).lower()
    return bool(a) and bool(b) and (a in b or b in a)


def judge_test(
    settings, client=None, limit: int = 20, sorting: Path = SORTING, sleep=time.sleep
) -> dict:
    """``scout reports --judge-test``: the judge, as it is now, on the first
    ``limit`` reports of the hand sorting, through the Message Batches API
    (about 0.05 USD for 20). A real error agrees when the verdict is
    checkable, unfaithful, and one of its spans overlaps the sorted span; a
    judge mistake agrees when the verdict is faithful; a sheet gap is shown,
    not scored. The reports and their sheets come from ``verdicts.jsonl`` and
    ``reports.parquet``; once a full rewrite has replaced a sheet, read the
    older ``reports.parquet`` (the bucket keeps its versions) into place
    first. Writes ``reports/judge_test.md``."""
    cfg = settings.reports_text
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            log.warning("reports: ANTHROPIC_API_KEY is not set; the judge test is skipped")
            return {}
        import anthropic

        client = anthropic.Anthropic()
    processed = settings.paths.processed_dir
    wanted = pd.read_csv(sorting, dtype={"fact_hash": str}).fillna("").head(limit)
    folder = out_dir(processed)
    kept = folder / "verdicts.jsonl"
    seen = pd.read_json(kept, lines=True) if kept.is_file() else pd.DataFrame()
    stored = load_stored(processed)
    items, missing = [], []
    for _, w in wanted.iterrows():
        prefix = str(w["fact_hash"])
        match = (
            seen[seen["fact_hash"].astype(str).str.startswith(prefix)] if not seen.empty else seen
        )
        sheets = stored[stored["fact_hash"].astype(str).str.startswith(prefix)]
        if match.empty or sheets.empty:
            missing.append(str(w["player"]))
            continue
        items.append((w, sheets.iloc[0]["fact_sheet"], match.iloc[-1]["report"]))
    runner = Batches(client, cfg.poll_seconds, time.monotonic() + cfg.max_run_minutes * 60, sleep)
    requests = [
        {"custom_id": f"t-{int(w['player_id'])}", "params": judge_params(cfg, sheet, report)}
        for w, sheet, report in items
    ]
    replies = runner.run(requests, "judge")
    usage, rows = {}, []
    for (w, sheet, report), r in zip(items, requests, strict=True):
        reply = replies[r["custom_id"]]
        _add(usage, reply.usage)
        v = verdict_from(reply, sheet, report)
        expected = SCORED.get(str(w["category"]))
        if v is None or expected is None:
            agreed = None
        elif expected:
            agreed = v.faithful
        else:
            agreed = (
                not v.faithful
                and v.checkable
                and any(_overlaps(i["span"], w["span"]) for i in v.issues)
            )
        rows.append((w, v, agreed))
    scored = [a for _, _, a in rows if a is not None]
    summary = {
        "asked": len(items),
        "not_found": missing,
        "verdicts": sum(v is not None for _, v, _ in rows),
        "checkable": sum(v is not None and v.checkable for _, v, _ in rows),
        "scored": len(scored),
        "agreed": sum(scored),
        "cost_usd": round(cost_usd(usage, cfg.judge_model, cfg), 4) if usage else 0.0,
    }
    lines = [
        "# The judge against the hand sorting",
        "",
        f"`scout reports --judge-test` on {len(items)} reports of `{sorting}` "
        f"({model_name(cfg.judge_model)}): agreed on {summary['agreed']} of {summary['scored']} "
        f"scored (real errors found at the sorted span, judge mistakes now faithful); "
        f"{summary['checkable']} of {summary['verdicts']} verdicts checkable; "
        f"cost {summary['cost_usd']:.2f} USD."
        + (f" Not found: {', '.join(missing)}." if missing else ""),
        "",
        "| Player | Sorted as | Field | Judge | Issues | Agrees |",
        "|---|---|---|---|---|---|",
    ]
    for w, v, agreed in rows:
        said = "no verdict" if v is None else ("faithful" if v.faithful else "not faithful")
        if v is not None and not v.checkable:
            said += " (set aside)"
        issues = "" if v is None else "; ".join(f'"{i["span"]}" ({i["field"]})' for i in v.issues)
        mark = "" if agreed is None else ("yes" if agreed else "no")
        lines.append(
            f"| {w['player']} | {w['category']} | {w['field']} | {said} | "
            f"{issues.replace('|', '/')} | {mark} |"
        )
    path = Path(settings.paths.reports_dir) / "judge_test.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines) + "\n")
    log.info("reports: judge test %s, %s", summary, path)
    return summary


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
    if "judge_checkable" in judged:  # a verdict code could not check is not counted (prompt 19c)
        judged = judged[judged["judge_checkable"].astype("boolean").fillna(True).astype(bool)]
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
