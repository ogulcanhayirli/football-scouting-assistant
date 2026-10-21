"""One line explanations for both tables (prompt 14).

For each player, the three stats where he stands furthest from his peers in
the current season, as plain sentences with his own number and where it
ranks: "Takes 28% of his team's shots, top 10% of centre forwards aged 21 to
23." Rule based templates, no language model.

* **Peers** are the regular players (450+ minutes) of the same season, the
  same position group (``scout.scouting.POSITION_GROUPS``; the broad position
  when Transfermarkt has no detailed one) and the same age band
  (``scout.scouting.AGE_BANDS``), over all 37 leagues. When fewer than
  ``MIN_PEERS`` of them have a stat, the age band is dropped for it; under
  ``MIN_PEERS`` even then, the stat is not used.
* **Percentiles** are oriented so that high is good: for a stat where less
  is better (injuries, yellow cards), the player who missed most matches is
  at the bottom. "Top 4%" means 4% of his peers are at least as good, so a
  value shared by many (no yellow card) is never at the top. A stat is
  notable in the ``NOTABLE`` outer tenths (top or bottom 10%); the three
  furthest from the middle are shown, ties broken by the order of ``STATS``.
* **One table for flags and explanation.** ``peer_table`` holds the raw
  number of every stat, the three flag columns among them, and
  ``scout.flags.describe`` reads its flags from the same rows. A flag column
  is only told on the side its flag says: low minutes, injuries and a falling
  value appear on the bad side only when the flag is raised, and on the good
  side only when it is not, so the two can never contradict each other.
* A player with no notable stat gets a neutral sentence, never an empty
  cell: "No stat stands out against other full backs aged 24 to 26 this
  season (12 compared)."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scout.flags import FLAGS, Flag
from scout.scouting import AGE_BANDS, AGE_LABELS, KEYS, POSITION_GROUPS

MIN_PEERS = 30
NOTABLE = 10.0  # percentiles from the top or the bottom
SHOWN = 3

PEERS = {
    "goalkeeper": "goalkeepers",
    "centre_back": "centre backs",
    "full_back": "full backs",
    "defensive_mid": "defensive midfielders",
    "central_mid": "central midfielders",
    "attacking_mid": "attacking midfielders",
    "winger": "wingers",
    "centre_forward": "centre forwards",
    "Goalkeeper": "goalkeepers",
    "Defender": "defenders",
    "Midfielder": "midfielders",
    "Attacker": "forwards",
}
KEEPERS = {"goalkeeper", "Goalkeeper"}


@dataclass(frozen=True)
class Stat:
    column: str
    text: str  # a format string for the raw number ``v``
    higher_is_better: bool = True
    keepers: bool | None = False  # True: goalkeepers only; False: outfield only; None: all

    def applies(self, peer_group: pd.Series) -> pd.Series:
        keeper = peer_group.isin(KEEPERS)
        if self.keepers is None:
            return pd.Series(True, index=peer_group.index)
        return keeper if self.keepers else ~keeper


# The three flag columns use the same rounding as ``scout.flags.raw_text``,
# so the number in the explanation is the number in the flag.
STATS = [
    Stat("goals_per90", "Scores {v:.2f} goals per 90 minutes"),
    Stat("assists_per90", "Gives {v:.2f} assists per 90 minutes"),
    Stat("shot_share", "Takes {v:.0%} of his team's shots"),
    Stat("passes_key_per90", "Makes {v:.1f} key passes per 90 minutes"),
    Stat("dribbles_success_per90", "Completes {v:.1f} dribbles per 90 minutes"),
    Stat("ball_wins_per90", "Wins the ball back {v:.1f} times per 90 minutes"),
    Stat("duel_win_rate", "Wins {v:.0%} of his duels"),
    Stat("dribbled_past_per90", "Is dribbled past {v:.1f} times per 90 minutes", False),
    Stat("saves_per90", "Makes {v:.1f} saves per 90 minutes", True, True),
    Stat("goals_conceded_per90", "Concedes {v:.2f} goals per 90 minutes", False, True),
    Stat("pass_accuracy", "Completes {v:.0%} of his passes", True, None),
    Stat("passes_total_per90", "Plays {v:.0f} passes per 90 minutes", True, None),
    Stat("rating", "Averages a match rating of {v:.2f}", True, None),
    Stat("cards_yellow_per90", "Picks up {v:.2f} yellow cards per 90 minutes", False, None),
    Stat("minutes_share", "Plays {v:.0%} of team minutes", True, None),
    Stat(
        "injury_matches_missed_12m",
        "Missed {v:.0f} matches injured in the last 12 months",
        False,
        None,
    ),
    Stat("value_change_1y", "Market value {c:+.0%} on a year before", True, None),
    Stat(
        "intl_minutes_12m",
        "Played {v:,.0f} minutes for his national team in the last 12 months",
        True,
        None,
    ),
    Stat(
        "ucl_minutes_12m",
        "Played {v:,.0f} Champions League minutes in the last 12 months",
        True,
        None,
    ),
]
FLAG_OF = {f.column: (name, f) for name, f in FLAGS.items()}


def peer_group(rows: pd.DataFrame) -> pd.Series:
    detailed = rows["sub_position"].map(POSITION_GROUPS) if "sub_position" in rows else None
    broad = rows["position"] if "position" in rows else pd.Series(np.nan, index=rows.index)
    group = broad if detailed is None else detailed.fillna(broad)
    return group.fillna("unknown").astype(str)


def peer_label(group: str, band: str | None) -> str:
    plural = PEERS.get(group, "players in his position")
    return plural if band is None else f"{plural} aged {band}"


def derived(rows: pd.DataFrame) -> pd.DataFrame:
    """Columns built from the feature row: ``shot_share`` (his league shots
    over his team's) and ``ball_wins_per90`` (tackles plus interceptions)."""
    out = pd.DataFrame(index=rows.index)
    if {"shots_total", "team_shots_pm", "team_matches"} <= set(rows):
        team = rows["team_shots_pm"].astype(float) * rows["team_matches"].astype(float)
        share = rows["shots_total"].astype(float) / team.where(team > 0)
        out["shot_share"] = share.clip(upper=1.0)
    if {"tackles_total_per90", "interceptions_per90"} <= set(rows):
        out["ball_wins_per90"] = rows["tackles_total_per90"].astype(float) + rows[
            "interceptions_per90"
        ].astype(float)
    return out


def population(features: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """One row per regular player of ``season`` (the newest when None): the
    league row with most minutes, as in the lists."""
    season = int(features["season"].max()) if season is None else season
    rows = features[(features["season"].astype(int) == season) & features["regular"].astype(bool)]
    return rows.sort_values("minutes").drop_duplicates("source_player_id", keep="last")


def peer_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per player row: KEYS, ``peer_group``, ``age_band``, and for every stat
    in ``STATS`` present in ``rows`` its raw number (``<stat>``) and good side
    percentile (``<stat>__pct``, 0 to 100, high is good) with the peer label
    it was ranked in (``<stat>__peers``). ``rows`` is one season's
    population, one row per player."""
    rows = rows.reset_index(drop=True)
    extra = derived(rows)
    rows = pd.concat([rows.drop(columns=extra.columns, errors="ignore"), extra], axis=1)
    out = rows[[k for k in KEYS if k in rows]].copy()
    out["peer_group"] = peer_group(rows)
    out["age_band"] = (
        pd.cut(rows["age"].astype(float), AGE_BANDS, right=False, labels=AGE_LABELS)
        .astype(str)
        .replace("nan", "unknown")
    )
    for stat in STATS:
        if stat.column not in rows:
            continue
        v = rows[stat.column].astype(float).where(stat.applies(out["peer_group"]))
        out[stat.column] = rows[stat.column].astype(float)
        good = v if stat.higher_is_better else -v
        fine = _ranked(good, [out["peer_group"], out["age_band"]])
        coarse = _ranked(good, [out["peer_group"]])
        use_fine = fine["n"] >= MIN_PEERS
        pct = fine["pct"].where(use_fine, coarse["pct"].where(coarse["n"] >= MIN_PEERS))
        out[f"{stat.column}__pct"] = pct
        out[f"{stat.column}__peers"] = [
            peer_label(g, b if fine_ok else None) if pd.notna(p) else None
            for g, b, fine_ok, p in zip(
                out["peer_group"], out["age_band"], use_fine, pct, strict=True
            )
        ]
    return out


def _ranked(good: pd.Series, by: list[pd.Series]) -> pd.DataFrame:
    """Good side percentile (0 to 100) and peer count within groups. Ties
    count against the claim: "top 4%" means 4% of peers are at least as
    good, so a player sharing a common value (no yellow card, every minute
    in goal) with many peers is in the middle, never at the top."""
    grouped = good.groupby(by)
    n = grouped.transform("count")
    top = (n - grouped.rank(method="min") + 1) / n  # share at least as good
    bottom = grouped.rank(method="max") / n  # share at most as good
    pct = pd.Series(50.0, index=good.index)
    pct = pct.where(~(bottom < 0.5), bottom * 100)
    pct = pct.where(~(top < 0.5), 100 - top * 100)
    return pd.DataFrame({"pct": pct.where(good.notna()), "n": n.where(good.notna(), 0)})


def rank_text(pct: float) -> str:
    """'top 3%' or 'bottom 8%' from a good side percentile."""
    if pct >= 50:
        return f"top {max(1, math.ceil(100 - pct - 1e-9))}%"
    return f"bottom {max(1, math.ceil(pct - 1e-9))}%"


def sentence(stat: Stat, value: float, pct: float, peers: str) -> str:
    if stat.column == "value_change_1y" and value >= 2:  # never flagged: the flag is for a fall
        times = f"{value:.1f}" if value < 10 else f"{value:.0f}"
        phrase = f"Market value {times} times what it was a year before"
    else:
        phrase = stat.text.format(v=value, c=value - 1)
    return f"{phrase}, {rank_text(pct)} of {peers}."


def _allowed(stat: Stat, pct: float, flagged: dict[str, bool]) -> bool:
    """A flag column only on the side its flag tells."""
    if stat.column not in FLAG_OF:
        return True
    name, _ = FLAG_OF[stat.column]
    good = pct >= 50
    return good != flagged[name]


def explain_row(table: pd.DataFrame, idx, flagged: dict[str, bool]) -> str:
    found = []
    for order, stat in enumerate(STATS):
        key = f"{stat.column}__pct"
        if key not in table:
            continue
        pct = table.at[idx, key]
        if pd.isna(pct) or not (pct >= 100 - NOTABLE or pct <= NOTABLE):
            continue
        if not _allowed(stat, pct, flagged):
            continue
        found.append((-abs(pct - 50), order, stat, pct))
    if not found:
        return neutral(table, idx)
    found.sort(key=lambda t: (t[0], t[1]))
    return " ".join(
        sentence(stat, table.at[idx, stat.column], pct, table.at[idx, f"{stat.column}__peers"])
        for _, _, stat, pct in found[:SHOWN]
    )


def neutral(table: pd.DataFrame, idx) -> str:
    known = [s for s in STATS if pd.notna(table.at[idx, f"{s.column}__pct"])]
    group = table.at[idx, "peer_group"]
    band = table.at[idx, "age_band"]
    peers = peer_label(group, None if band == "unknown" else band)
    if not known:
        return f"Too few stats recorded this season to compare him with other {peers}."
    return f"No stat stands out against other {peers} this season ({len(known)} compared)."


def explain(table: pd.DataFrame, flags: dict[str, Flag] = FLAGS) -> pd.Series:
    """The explanation per row of a ``peer_table``."""
    for s in STATS:  # a stat absent from the table is never notable
        if f"{s.column}__pct" not in table:
            table = table.assign(**{f"{s.column}__pct": np.nan, f"{s.column}__peers": None})
    raised = pd.DataFrame(
        {
            name: f.raised(table[f.column]) if f.column in table else False
            for name, f in flags.items()
        },
        index=table.index,
    )
    return pd.Series(
        [explain_row(table, idx, raised.loc[idx].to_dict()) for idx in table.index],
        index=table.index,
        name="explanation",
    )


def flags_and_explanation(table: pd.DataFrame) -> pd.DataFrame:
    """``flag_count``, ``flags`` and ``explanation`` from one peer table."""
    from scout.flags import describe

    return pd.concat([describe(table), explain(table)], axis=1)


def for_rows(profile: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The peer table rows of ``rows`` (matched on KEYS), in ``rows``' order.
    A row missing from the profile (not a regular of that season) is ranked
    on its own numbers as unknown, which gives the neutral sentence and
    flags from its own columns."""
    keyed = rows[KEYS].reset_index(names="_row").merge(profile, on=KEYS, how="left")
    keyed = keyed.set_index("_row").reindex(rows.index)
    for f in FLAGS.values():  # a missing profile row still carries its flag numbers
        if f.column in rows:
            keyed[f.column] = keyed[f.column].fillna(rows[f.column].astype(float))
    keyed["peer_group"] = keyed["peer_group"].fillna("unknown")
    keyed["age_band"] = keyed["age_band"].fillna("unknown")
    return keyed
