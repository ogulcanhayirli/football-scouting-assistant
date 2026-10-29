"""The third published table (prompt 20f): every player of the newest season.

The candidates and the bargains hold about 3,400 players; the chat and the
page also answer about everyone else in the data. ``players`` has one row
per player with at least one league minute in the newest season (2025/26,
or the 2025 calendar year) in the leagues we track, which covers every
player of the two lists (a handful of rows the API gives no name are left
out: nobody can ask for them). Each row is the league row where he played most
minutes; for a regular player it is the very row the lists read
(``scout.lists.newest_players``), so his numbers agree with theirs.

* **What it holds:** name, age, nationality, position, club, league, league
  minutes and his share of the team's, the ability score (version 2, in its
  position group's units) and its percentile, market value with its date
  and rank, fair value and the value to fair value ratio, the elite and big
  move chances copied from the candidates where they are computed, the
  three warning flags with his numbers, whether he is on each list and
  whether the page shows a written report for him, and the day the row
  describes (``data_date``, the feature date).
* **Why a cell is blank.** Ability, fair value and every chance exist only
  for regular players (450 league minutes or more), as everywhere in the
  models. A blank chance carries the reason in ``elite_chance_why_blank``
  or ``big_move_chance_why_blank``; a player on no list carries one
  sentence each, set here and nowhere else, in ``why_not_candidate`` and
  ``why_not_bargain``. Each sentence states the rules of the lists as the
  code applies them (``scout.lists``, ``scout.bargains``), every rule he
  fails in one list ("a, b and c"): a candidate is a regular ranked #101 to #2,000 by
  value, or one of the riser model's top 50 valued lower; a bargain is a
  regular of 33 or under with ability at the 70th percentile of his
  position or better, half of his team's minutes or more and both a value
  and a fair value. The bargains list ranks every such player, also those
  priced above their fair value, so "value above fair value" is never a
  reason here; a player on that ranking priced above his fair value is on
  the list but not a bargain (``below_fair_value`` of the bargains table,
  prompt 20g), which the chat says from his bargains row.
* ``report_available``: the page shows a written report for him (only
  players of the two lists get one). ``scout lists`` sets it from the stored
  reports; ``scout site`` sets it again from the page's own ``reports.json``
  before publishing, since ``scout reports`` runs between the two.

``scout lists`` writes it next to the two lists, as Parquet and CSV, checked
by ``PLAYERS_TABLE`` (``scout.contracts.schemas``). ``scout site`` also
writes ``data/players.json`` (``compact``): the same rows with the repeated
texts as codes, which the page's Players section and the chat read.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scout import flags
from scout.bargains import EXPERIENCED_MAX_AGE, MIN_ABILITY_PCT, MIN_MINUTES_SHARE
from scout.features.build import REGULAR_MINUTES
from scout.scouting import ELITE_MAX_AGE, ELITE_RANK, ELITE_REACH, KEYS

log = logging.getLogger(__name__)

MIN_MINUTES = 1  # at least one league minute in the season
TOP_RANK, LAST_RANK = 101, 2000  # scout.lists.RANKS

# The reasons, each a short clause; a sentence joins every clause that applies.
FEW_MINUTES = f"fewer than {REGULAR_MINUTES} league minutes this season"
NO_VALUE = "no value on record"
TOP_100 = f"already among the {ELITE_RANK} most valuable"
OLDER = f"older than {ELITE_MAX_AGE}"
NOT_RISER = "not one of the riser model's top 50"
NOT_CANDIDATE = "not a candidate: "
NOT_BARGAIN = "not a bargain: "

COMPACT_CODED = [
    "nationality",
    "position",
    "club",
    "league",
    "flags",
    "elite_chance_why_blank",
    "big_move_chance_why_blank",
    "why_not_candidate",
    "why_not_bargain",
]

DICTIONARY = {
    "season": "The season the row describes (2025 is 2025/26, or the 2025 calendar year).",
    "player_id": "The player's API Football id, for joining with the other tables.",
    "player": "The player's name as API Football writes it.",
    "age": "His age in years at the feature date (data_date).",
    "nationality": "His nationality as API Football gives it.",
    "position": "His Transfermarkt position, such as Centre-Forward or Left-Back.",
    "club": "The club he played most league minutes for that season (the club at forecast "
    "of the two lists).",
    "league": "The league of that club; a player of two of our leagues is shown in the one "
    "where he played most.",
    "minutes": "His league minutes that season in that league.",
    "minutes_share": "The share of his club's league minutes he played.",
    "ability_score": "The version 2 ability score: a per 90 composite of the last two "
    "seasons for his position group, adjusted for league and team strength and shrunk by "
    "minutes. Its units differ by position group, so compare players through ability_pct; "
    f"blank under {REGULAR_MINUTES} league minutes.",
    "ability_pct": "That score as a percentile among regulars of his position group that "
    f"season (100 is the best); blank under {REGULAR_MINUTES} league minutes.",
    "value_eur": "His Transfermarkt market value at the feature date, in euros; blank when "
    "Transfermarkt holds none for him.",
    "value_date": "The date of that market value.",
    "value_rank": "Where his market value ranks among all players we track (1 is the dearest).",
    "fair_value_eur": "What the fair value model, trained on earlier seasons only, says a "
    "player with his record is worth, in euros; only for regular players "
    f"({REGULAR_MINUTES} league minutes or more) with a market value.",
    "value_to_fair": "Market value divided by fair value: 0.40 means the market asks 40% of "
    "the model's price.",
    "elite_decile": "As in the candidates table; blank when the elite chance is blank.",
    "elite_chance_pct": "The elite chance of the candidates table, copied; it exists only "
    f"for the elite pool (regulars ranked #{ELITE_RANK + 1} to #{ELITE_REACH:,} by value and "
    f"{ELITE_MAX_AGE} or under).",
    "elite_chance_players": "How many past players that elite chance is based on.",
    "elite_chance_why_blank": "Why the elite chance is blank, every reason that applies "
    f"(for example older than {ELITE_MAX_AGE}, or already among the {ELITE_RANK} most "
    "valuable); blank when it is given.",
    "big_move_decile": "As in the candidates table; blank when the big move chance is blank.",
    "big_move_chance_pct": "The big move chance of the candidates table, copied; it exists "
    f"only for candidates ranked #{TOP_RANK} to #{LAST_RANK:,} who are not at a top 20 club.",
    "big_move_chance_players": "How many past players that big move chance is based on.",
    "big_move_chance_why_blank": "Why the big move chance is blank, every reason that "
    "applies; blank when it is given.",
    "flag_count": "How many warning flags he raises: low minutes, injured, value falling "
    "(the flags behind the fall back risk).",
    "flags": "The raised flags, each with his own number, or none.",
    "in_candidates": "True when he is a row of the candidates table.",
    "in_bargains": "True when he is a row of the bargains table (prime or experienced), "
    "the ranking by value against fair value; he is a bargain only when his value_to_fair is "
    "also under 1 (below_fair_value of that table).",
    "why_not_candidate": "For a player outside the candidates table, the one sentence that "
    "says why, from the table's rules (a regular ranked #101 to #2,000 by value, or one of "
    "the riser model's top 50 valued lower); blank for a candidate.",
    "why_not_bargain": "For a player outside the bargains table, the one sentence that says "
    "why, from the list's rules (a regular of 33 or under, ability at the 70th percentile of "
    "his position or better, half of his team's minutes or more, a value and a fair value); "
    "blank for a listed player. The list ranks every such player, also those priced above "
    "their fair value.",
    "report_available": "True when the page shows a written report for him (only players of "
    "the two lists get one).",
    "data_date": "The day the row describes him (the feature date: 30 June 2026, or 1 "
    "January 2026 in the calendar year leagues).",
}
COLUMNS = list(DICTIONARY)


def main_rows(scouting: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per player with a league minute in ``season``: for a regular
    the row the lists read (``newest_players``), else the row with most
    minutes."""
    from scout.lists import newest_players

    rows = scouting[(scouting["season"] == season) & (scouting["minutes"] >= MIN_MINUTES)]
    nameless = rows["player_name"].isna()
    if nameless.any():  # a few the API names nowhere (5 on 28 October 2026, none regular)
        log.info("players: %d rows without a name left out", int(nameless.sum()))
        rows = rows[~nameless]
    regular = newest_players(scouting, season)
    rest = rows[~rows["source_player_id"].isin(regular["source_player_id"])]
    # Equal minutes in two leagues: the league key decides, never the row order (prompt 20i).
    rest = rest.sort_values(["minutes", "league_key"], kind="stable").drop_duplicates(
        "source_player_id", keep="last"
    )
    return pd.concat([regular, rest], ignore_index=True)


def _join(prefix: str, parts: list[str]) -> str | None:
    """'a', 'a and b', 'a, b and c'."""
    if not parts:
        return None
    return prefix + (", ".join(parts[:-1]) + " and " if len(parts) > 1 else "") + parts[-1]


def elite_reason(regular: bool, age: float, rank: float, given: bool) -> str | None:
    if given:
        return None
    parts = [FEW_MINUTES] if not regular else []
    if pd.notna(age) and age >= ELITE_MAX_AGE + 1:
        parts.append(OLDER)
    if pd.isna(rank):
        parts.append(NO_VALUE)
    elif rank <= ELITE_RANK:
        parts.append(TOP_100)
    elif rank > ELITE_REACH:
        parts.append(f"valued below #{ELITE_REACH:,}")
    return _join("", parts) or "not rated"


def big_move_reason(regular: bool, rank: float, candidate: bool, given: bool) -> str | None:
    if given:
        return None
    parts = [FEW_MINUTES] if not regular else []
    if pd.isna(rank):
        parts.append(NO_VALUE)
    elif rank <= ELITE_RANK:
        parts.append(TOP_100)
    elif rank > LAST_RANK:
        parts.append(f"valued below #{LAST_RANK:,}")
    elif candidate:
        parts.append("at a top 20 club")
    return _join("", parts) or "not rated"


def candidate_reason(regular: bool, rank: float) -> str:
    parts = [FEW_MINUTES] if not regular else []
    if pd.isna(rank):
        parts += [NO_VALUE, NOT_RISER]
    elif rank <= ELITE_RANK:
        parts.append(TOP_100)
    elif rank > LAST_RANK:
        parts += [f"valued below #{LAST_RANK:,}", NOT_RISER]
    return _join(NOT_CANDIDATE, parts) or NOT_CANDIDATE.removesuffix(": ")


def bargain_reason(o) -> str:
    parts = []
    if not o.regular:
        parts.append(FEW_MINUTES)
    else:
        if pd.isna(o.group):
            parts.append("no Transfermarkt position")  # so no ability score either
        elif pd.isna(o.ability_pct):
            parts.append("no ability score")
        elif o.ability_pct < MIN_ABILITY_PCT:
            parts.append(f"ability below the {MIN_ABILITY_PCT:.0f}th percentile of his position")
        if pd.notna(o.minutes_share) and o.minutes_share < MIN_MINUTES_SHARE:
            parts.append("under half of his team's league minutes")
    if pd.notna(o.age) and o.age >= EXPERIENCED_MAX_AGE:
        parts.append(f"{EXPERIENCED_MAX_AGE} or older")
    if pd.isna(o.value_eur) or o.value_eur <= 0:
        parts.append(NO_VALUE)
    elif o.regular and pd.isna(o.fair_value_eur):
        parts.append("no fair value")
    return _join(NOT_BARGAIN, parts) or NOT_BARGAIN.removesuffix(": ")


def players_table(
    scouting: pd.DataFrame,
    features: pd.DataFrame,
    fair_value: pd.DataFrame,
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    leagues: dict[str, str],
    reports: dict | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """The players table of ``season`` (the newest when None). ``fair_value``
    is ``backtest_fair_value``; ``reports`` the page's reports
    (``scout.reports_text.page_reports``), None when there are none."""
    from scout.lists import tenths

    season = int(scouting["season"].max()) if season is None else season
    rows = main_rows(scouting, season)
    extra = features.assign(season=features["season"].astype(int))[[*KEYS, "nationality"]]
    rows = rows.merge(extra, on=KEYS, how="left")
    fair = fair_value.assign(
        season=fair_value["season"].astype(int),
        fair=np.exp(fair_value["pred_lightgbm"].astype(float)),
    )
    rows = rows.merge(fair[[*KEYS, "fair"]], on=KEYS, how="left")
    regular = rows["regular"].astype(bool)
    value = rows["market_value"].astype(float)
    out = pd.DataFrame(
        {
            "season": rows["season"].astype(int),
            "player_id": rows["source_player_id"].astype("Int64"),
            "player": rows["player_name"],
            "age": tenths(rows["age"]),
            "nationality": rows["nationality"],
            "position": rows["sub_position"]
            .astype("string")
            .fillna(rows["position"].astype("string")),
            "club": rows["team_name"],
            "league": rows["league_key"].map(leagues).fillna(rows["league_key"]),
            "minutes": rows["minutes"].astype(float).round(0).astype("Int64"),
            "minutes_share": rows["minutes_share"].astype(float).round(2),
            "ability_score": rows["adj_rating"].astype(float).where(regular).round(2),
            "ability_pct": rows["ability_pct"].astype(float).where(regular).round(0),
            "value_eur": value,
            "value_date": pd.to_datetime(rows["value_date"]).where(value.notna()),
            "value_rank": rows["value_rank"].astype("Int64"),
            "fair_value_eur": rows["fair"].round(-3),
            "value_to_fair": (value / rows["fair"]).round(3),
        }
    )
    on_list = cand.set_index("player_id")
    for name in ("elite", "big_move"):
        for part in ("decile", "chance_pct", "chance_players"):
            column = f"{name}_{part}"
            out[column] = out["player_id"].map(on_list[column]).astype("Int64")
    raised = flags.describe(rows)
    out["flag_count"] = raised["flag_count"]
    out["flags"] = raised["flags"]
    out["in_candidates"] = out["player_id"].isin(cand["player_id"]).astype(bool)
    out["in_bargains"] = out["player_id"].isin(barg["player_id"]).astype(bool)
    out["regular"] = regular.to_numpy()
    out["group"] = rows["group"].to_numpy()
    out["elite_chance_why_blank"] = [
        elite_reason(r, a, k, pd.notna(p))
        for r, a, k, p in zip(
            out["regular"], out["age"], out["value_rank"], out["elite_chance_pct"], strict=True
        )
    ]
    out["big_move_chance_why_blank"] = [
        big_move_reason(r, k, c, pd.notna(p))
        for r, k, c, p in zip(
            out["regular"],
            out["value_rank"],
            out["in_candidates"],
            out["big_move_chance_pct"],
            strict=True,
        )
    ]
    out["why_not_candidate"] = [
        None if c else candidate_reason(r, k)
        for c, r, k in zip(out["in_candidates"], out["regular"], out["value_rank"], strict=True)
    ]
    out["why_not_bargain"] = [
        None if o.in_bargains else bargain_reason(o) for o in out.itertuples(index=False)
    ]
    out["report_available"] = report_flags(out["player_id"], reports)
    out["data_date"] = pd.to_datetime(rows["feature_date"])
    missing = set(cand["player_id"]) | set(barg["player_id"])
    missing -= set(out["player_id"])
    if missing:
        raise ValueError(f"{len(missing)} list players have no row: {sorted(missing)[:5]}")
    out = out.sort_values(
        ["value_rank", "minutes", "player_id"], ascending=[True, False, True], na_position="last"
    )
    return out[COLUMNS].reset_index(drop=True)


def report_flags(player_id: pd.Series, reports: dict | None) -> pd.Series:
    """True where the page shows his written report (``reports.json``)."""
    rows = (reports or {}).get("rows") or {}
    written = {int(k) for k, v in rows.items() if v and v[0] == "report"}
    return player_id.astype("Int64").isin(written).to_numpy()


def compact(table: pd.DataFrame) -> dict:
    """``data/players.json``: the rows as lists, the repeated texts (club,
    league, position, nationality, flags and the reasons) as positions in
    ``texts``, dates as ISO days, missing values as null."""
    frame = table.drop(columns="season")
    texts = sorted({v for c in COMPACT_CODED for v in frame[c].dropna().unique()})
    where = {v: i for i, v in enumerate(texts)}
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")
    frame = frame.astype(object).where(frame.notna(), None)
    columns = []
    for column in frame.columns:
        values = [v.item() if isinstance(v, np.generic) else v for v in frame[column]]
        if column in COMPACT_CODED:
            values = [None if v is None else where[v] for v in values]
        columns.append(values)
    rows = [list(row) for row in zip(*columns, strict=True)]
    return {
        "season": int(table["season"].max()),
        "columns": list(frame.columns),
        "coded": COMPACT_CODED,
        "texts": texts,
        "rows": rows,
    }
