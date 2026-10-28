"""The two user facing tables (decision 5) and the one readable report.

* **candidates**: one row per regular player ranked #101 to #2,000 by value
  in the newest season, with his elite chance (elite pool only), step up
  chance, fall back risk (elite pool only), flags, explanation, rising tag,
  big move chance, current and fair value, and ability percentile. Sorted by
  elite decile, then step up decile (the chance order while the rates rise
  with the decile). The riser model's top 50 are added as
  rows when they are valued below #2,000 (or have no value), with the
  chances blank: those were never measured there (prompt 14).
* **bargains**: the prime and experienced lists of ``scout.bargains`` with
  the shared flags, the explanation and the two context tags. The lists
  rank every eligible player by value against fair value; only a player
  priced below his fair value (``below_fair_value``, prompt 20g) is called
  a bargain, the rest are on the ranking but not bargains.
* **players** (prompt 20f, ``scout.players``): every player with a league
  minute in the newest season, the two lists' players among them, with why
  he is on no list.

Flags and explanation come from one peer table (``scout.explain``), so the
sentence and the flags of a player always quote the same numbers.

Every chance is a historical rate, not a model probability: of the players
shown in the same bucket (score decile, or flag count) by the past
forecasts that ``scout forecast history`` rebuilt with the models and
buckets of their time, the share who made it two seasons on (prompt 14b;
``scout.candidates.realized_tables``), pooled over the last five scored
seasons since prompt 14c (2019 to 2023 for the 2025 table), rounded to the
nearest 5%, never shown for a bucket of fewer than 30 players with a known
outcome (such a bucket merges with its neighbour), with that number of
players. Step up and big move deciles are taken among players ranked #101
to #2,000, the players this table shows. Until prompt 14b the rate pooled
the backtest rows of every test season, which ran high at the top of the
elite chance; ``reports/candidates.md`` shows the in sample tables, and
the prompt 14b rate (every scored season) against the current one.

Both tables go to ``data/processed/lists/`` as Parquet and CSV with
``dictionary.csv``; ``reports/candidates.md`` shows their top 50 and every
backtest number next to its baseline. ``scout lists`` reads local files
only; run it after ``backtest``, ``moves``, ``scouting`` and ``forecast
history``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scout import bargains, candidates, explain
from scout.candidates import (
    COUNT_BUCKETS,
    DECILES,
    ChanceTable,
    chance_table,
    score_deciles,
    show_rate,
)
from scout.flags import ABILITY, ABILITY_MIN_RATIO, TAGS
from scout.fsutil import atomic_write_text
from scout.models.backtest import summarize
from scout.models.moves import summarize_moves
from scout.scouting import (
    ELITE_MAX_AGE,
    HORIZON_SEASONS,
    KEYS,
    RISER_LEVEL,
    RISER_MIN_SHARE,
    elite_pool,
    evaluate,
    season_precision,
)

log = logging.getLogger(__name__)

RANKS = (101, 2000)  # the candidates table: value rank #101 to #2,000
RISING_TOP = 50  # the riser model's top 50 of the season carry the rising tag
REPORT_SIZE = 50

EXPLANATION = (
    f"The {explain.SHOWN} stats where he stands furthest from regulars of his position and "
    "age band this season, each with his number and where it ranks (top means better, "
    "even for stats where less is better); a neutral sentence when nothing is in the top "
    f"or bottom {explain.NOTABLE:.0f}%. It reads the same numbers as the flags."
)

DICTIONARY = {
    "candidates": {
        "season": "The season the row describes (2025 is 2025/26, or the 2025 calendar year).",
        "player_id": "The player's API Football id, for joining with other tables.",
        "player": "The player's name as API Football writes it.",
        "age": "His age in years at the feature date, the day the row describes him.",
        "position": "His Transfermarkt position, such as Centre-Forward or Left-Back.",
        "club_at_forecast": "The club he played most league minutes for that season, the "
        "club the forecast describes.",
        "league": "The league of that club.",
        "value_rank": "Where his market value ranks among all players we track (1 is the dearest); "
        "blank when he has no market value.",
        "elite_decile": "Where the elite model puts him among this season's elite pool, from 1 "
        "(least likely) to 10 (most likely); blank outside the pool.",
        "elite_chance_pct": "Of the players that past forecasts of the last five scored "
        "seasons, rebuilt with the models of their time, showed in the same elite decile, the "
        "share who became one of the 100 most valuable players within two seasons, rounded to "
        f"5; blank outside the elite pool (ranked 101 to 1,000 and {ELITE_MAX_AGE} or under).",
        "elite_chance_players": "How many players of those past forecasts, with a known "
        "outcome, that elite chance is based on.",
        "step_up_decile": "Where the step up model puts him among this season's candidates, "
        "from 1 to 10.",
        "step_up_chance_pct": "Of the players ranked 101 to 2,000 that past forecasts of the "
        "last five scored seasons showed in the same step up decile, the share whose value "
        "rank became at least three times better, inside the top 500, within two seasons, "
        "rounded to 5.",
        "step_up_chance_players": "How many players of those past forecasts, with a known "
        "outcome, that step up chance is based on.",
        "fall_back_risk_pct": "Of the elite pool players that past forecasts of the last five "
        "scored seasons showed with the same number of flags, the share whose value rank was "
        "at least twice as bad two seasons later, rounded to 5; blank outside the elite pool, "
        "where it was never measured.",
        "fall_back_risk_players": "How many players of those past forecasts, with a known "
        "outcome, that fall back risk is based on.",
        "flag_count": "How many warning flags he raises: low minutes, injured, value falling.",
        "flags": "The raised flags, each with his own number, or none.",
        "explanation": EXPLANATION,
        "rising": f"True when the riser model has him in this season's top {RISING_TOP}: "
        "likely to play clearly better in each of the next two seasons and end at the "
        f"{RISER_LEVEL:.0f}th percentile of his position or better (players with at least "
        f"{RISER_MIN_SHARE:.0%} of team minutes). Risers valued below #2,000 are added as "
        "rows of their own, with the chances blank.",
        "big_move_decile": "Where the big move model puts him among this season's candidates, "
        "from 1 to 10; blank when he already plays for a top 20 club.",
        "big_move_chance_pct": "Of the players ranked 101 to 2,000 that past forecasts of the "
        "last five scored seasons showed in the same big move decile, the share who joined a "
        "top 20 club (by squad value) within 24 months, rounded to 5.",
        "big_move_chance_players": "How many players of those past forecasts, with a known "
        "outcome, that big move chance is based on.",
        "value_eur": "His Transfermarkt market value at the feature date, in euros.",
        "fair_value_eur": "What the fair value model, trained on earlier seasons only, says a "
        "player with his record is worth, in euros.",
        "ability_pct": "His ability score as a percentile among regulars of his position that "
        "season (100 is the best), from per 90 stats of the last two seasons.",
    },
    "bargains": {
        "season": "The season the row describes.",
        "list": "prime (29 and under) or experienced (30 to 33).",
        "list_rank": "His place on that list, 1 being the cheapest against the fair value.",
        "player_id": "The player's API Football id.",
        "player": "The player's name as API Football writes it.",
        "age": "His age in years at the feature date.",
        "position": "His Transfermarkt position.",
        "club_at_forecast": "The club he played most league minutes for that season, the "
        "club the forecast describes.",
        "league": "The league of that club.",
        "ability_pct": "His ability percentile within his position (70 or better to be listed).",
        "minutes_share": "The share of his club's league minutes he played (half or more to be "
        "listed).",
        "value_eur": "His Transfermarkt market value at the feature date, in euros.",
        "fair_value_eur": "What the fair value model, trained on earlier seasons only, says he "
        "is worth, in euros.",
        "value_to_fair": "Market value divided by fair value: 0.40 means the market asks 40% of "
        "the model's price.",
        "value_to_fair_pct": "That ratio as a percentile among listed players of his position "
        "group (1 is the cheapest); the lists are sorted by it.",
        "below_fair_value": "True when his market value is below his fair value (value_to_fair "
        "under 1): only such a player is called a bargain. The lists rank every eligible player "
        "by value against fair value, also those priced above it, who are on the ranking but "
        "not bargains.",
        "contract_expiry": "When his contract ends, from the Transfermarkt snapshot, shown only "
        "when it ends after the snapshot date.",
        "flag_count": "How many warning flags he raises: low minutes, injured, value falling.",
        "flags": "The raised flags, each with his own number, or none.",
        "explanation": EXPLANATION,
        "on_loan": "True when he is on loan: " + TAGS["on_loan"].split(": ", 1)[1] + ".",
        "stale_value": "True when his market value is "
        + TAGS["stale_value"].split(": ", 1)[1]
        + ", so the cheap price may be out of date.",
        "value_date": "The date of his market value.",
    },
}


# The present (prompt 16): today's club and the season in progress, for the
# newest season's tables only (``scout.present``); never in the past
# forecasts that ``scout forecast history`` rebuilds.
PRESENT = {
    "current_club": "The club that lists him in its squad today (API Football squads, "
    "fetched on the day in so_far_date's pull); blank when no club we follow lists him.",
    "current_league": "The league of that club when it is one of ours; blank otherwise.",
    "moved_since_forecast": "True when his club today is not the club at forecast: a "
    "transfer, a loan or a loan's end since the feature date (30 June 2026, or 1 January "
    "2026 in the calendar year leagues); blank when his club today is not known.",
}
SO_FAR = {
    "so_far_date": "The season so far block describes the season in progress up to this "
    "day (2026/27, or the 2026 calendar year), from the league matches cached then.",
    "so_far_league": "The league of the season so far numbers (where he played most "
    "minutes this season).",
    "so_far_matches": "League matches he played in so far; blank with no minutes.",
    "so_far_minutes": "League minutes in the season in progress; 0 when he has none in our "
    "leagues (he may play elsewhere).",
    "so_far_minutes_share": "His share of his club's league minutes so far.",
    "so_far_ability_pct": "The one season version 2 ability score so far, with the stored "
    "fit (nothing refitted), as a percentile among this season's regulars of his position "
    "(450 minutes or more); blank under 450 minutes.",
    "so_far_goals_per90": "Goals per 90 minutes so far; blank for goalkeepers and centre "
    "backs, and under 270 minutes.",
    "so_far_assists_per90": "Assists per 90 minutes so far; blank for goalkeepers and centre "
    "backs, and under 270 minutes.",
    "so_far_flag_count": "The warning flags as they stand on so_far_date: low minutes on "
    "this season's share, injured on matches missed in the 12 months before it, value "
    "falling as at the forecast (no newer valuation exists).",
    "so_far_flags": "Those flags, each with his own number, or none.",
}


def _with_present(columns: dict[str, str]) -> dict[str, str]:
    out = {}
    for column, meaning in columns.items():
        out[column] = meaning
        if column == "league":
            out.update(PRESENT)
    return out | SO_FAR


DICTIONARY = {table: _with_present(columns) for table, columns in DICTIONARY.items()}
PRESENT_COLUMNS = [*PRESENT, *SO_FAR]


def base_columns(table: str) -> list[str]:
    """The columns a season's table carries before the present is added."""
    return [c for c in DICTIONARY[table] if c not in PRESENT_COLUMNS]


def tenths(age: pd.Series) -> pd.Series:
    """Age cut, not rounded, to tenths: 33.96 shows as 33.9, still 33."""
    return np.floor(age.astype(float) * 10) / 10


def league_names(settings) -> dict[str, str]:
    return {lg.key: lg.name for lg in settings.leagues}


# Inputs


def load(settings) -> dict:
    models = settings.paths.processed_dir / "models"
    files = {
        "scouting": "scouting",
        "elite": "backtest_elite",
        "step_up": "backtest_step_up",
        "riser": "backtest_riser",
        "big_move": "backtest_big_move_24m",
        "big_move_now": "big_move_current",
        "fair_value": "backtest_fair_value",
    }
    out = {name: pd.read_parquet(models / f"{file}.parquet") for name, file in files.items()}
    for frame in out.values():
        frame["season"] = frame["season"].astype(int)
    return out


def in_ranks(value_rank: pd.Series) -> pd.Series:
    return value_rank.between(*RANKS)


def chance_tables(inputs: dict, through: int | None = None) -> dict[str, ChanceTable]:
    """The in sample tables (the shown rates until prompt 14b): elite on the
    elite pool; step up and big move on past players ranked #101 to #2,000,
    all from the pooled backtest rows; fall back by flag count from the
    scouting rows. ``through``: only seasons up to it (a forecast for season
    S uses S minus 2, the last season whose two year outcome is known by
    then)."""

    def upto(frame: pd.DataFrame) -> pd.DataFrame:
        return frame if through is None else frame[frame["season"] <= through]

    ranks = inputs["scouting"][[*KEYS, "value_rank"]]
    elite = upto(inputs["elite"])
    step = upto(inputs["step_up"])
    move = (
        upto(inputs["big_move"]).drop(columns="value_rank", errors="ignore").merge(ranks, on=KEYS)
    )
    fall = candidates.pool_summary(upto(inputs["scouting"]))
    return {
        "elite": chance_table(elite[elite_pool(elite["value_rank"], elite["age"])]),
        "step_up": chance_table(step[in_ranks(step["value_rank"])]),
        "big_move": chance_table(move[in_ranks(move["value_rank"])], "score_lightgbm"),
        "fall_back": ChanceTable({b: b for b in COUNT_BUCKETS}, fall["count"]),
    }


def load_outcomes(settings) -> pd.DataFrame:
    """The scored past forecasts of ``scout forecast history``."""
    from scout.forecast import outcomes_path

    path = outcomes_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `scout forecast history` first")
    return pd.read_parquet(path)


def newest_players(scouting: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """The season's regular players (the newest when None), one row each (the
    league where he played most minutes)."""
    season = scouting["season"].max() if season is None else season
    newest = scouting[(scouting["season"] == season) & scouting["regular"].astype(bool)]
    return newest.sort_values("minutes").drop_duplicates("source_player_id", keep="last")


def rising_ids(scouting: pd.DataFrame, top: int = RISING_TOP, season: int | None = None) -> set:
    """The riser model's top ``top`` of the season (the newest when None),
    among players with a one season score and half of team minutes (the riser
    pool)."""
    newest = newest_players(scouting, season)
    pool = newest[
        (newest["minutes_share"].astype(float) >= RISER_MIN_SHARE)
        & newest["ability_pct_1s"].notna()
        & newest["score_riser"].notna()
    ]
    return set(pool.sort_values("score_riser", ascending=False)["source_player_id"].head(top))


def _chance(
    rows: pd.DataFrame, score: pd.Series, table: ChanceTable, name: str, mask: pd.Series
) -> pd.DataFrame:
    """Decile within the current rows in ``mask``, and its historical rate."""
    use = mask & score.notna()
    out = pd.DataFrame(index=rows.index)
    deciles = score_deciles(score[use], rows.loc[use, "season"])
    found = table.lookup(deciles)
    out[f"{name}_decile"] = deciles.reindex(rows.index).astype("Int64")
    out[f"{name}_chance_rate"] = found["rate"].reindex(rows.index)
    out[f"{name}_chance_pct"] = found["pct"].reindex(rows.index).astype("Int64")
    out[f"{name}_chance_players"] = found["players"].reindex(rows.index).astype("Int64")
    return out


COMPUTED = [
    c
    for c in DICTIONARY["candidates"]
    if c.endswith(("_decile", "_pct", "_players")) and c != "ability_pct"
] + ["flag_count", "flags", "explanation", "rising"]


def candidates_table(
    inputs: dict,
    tables: dict,
    leagues: dict[str, str],
    profile: pd.DataFrame | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """The candidates of ``season`` (the newest when None). ``profile`` is the
    season's peer table (``scout.explain.peer_table``); without one it is
    built from the scouting rows, which carry the flag columns but few stats."""
    s = inputs["scouting"]
    season = int(s["season"].max()) if season is None else season
    players = newest_players(s, season)
    if profile is None:
        profile = explain.peer_table(players)
    risers = rising_ids(s, season=season)
    ranked = in_ranks(players["value_rank"])
    below = players["source_player_id"].isin(risers) & ~(players["value_rank"] <= RANKS[1])
    rows = players[ranked | below].copy()
    fair = inputs["fair_value"].assign(fair_value_eur=lambda d: np.exp(d["pred_lightgbm"]))
    rows = rows.merge(fair[[*KEYS, "fair_value_eur"]], on=KEYS, how="left")
    move = inputs["big_move_now"][[*KEYS, "score"]].rename(columns={"score": "score_big_move"})
    rows = rows.merge(move, on=KEYS, how="left")

    pool = elite_pool(rows["value_rank"], rows["age"])
    ranked = in_ranks(rows["value_rank"])  # added risers have no chances
    parts = [
        _chance(rows, rows["score_elite"], tables["elite"], "elite", pool),
        _chance(rows, rows["score_step_up"], tables["step_up"], "step_up", ranked),
        _chance(rows, rows["score_big_move"], tables["big_move"], "big_move", ranked),
    ]
    rows = pd.concat([rows, *parts], axis=1)

    flags = explain.flags_and_explanation(explain.for_rows(profile, rows))
    bucket = candidates.count_bucket(flags["flag_count"]).where(pool)
    fall = tables["fall_back"].lookup(bucket[pool])
    rows["flag_count"] = flags["flag_count"]
    rows["flags"] = flags["flags"]
    rows["explanation"] = flags["explanation"]
    rows["fall_back_risk_rate"] = fall["rate"].reindex(rows.index)
    rows["fall_back_risk_pct"] = fall["pct"].reindex(rows.index).astype("Int64")
    rows["fall_back_risk_players"] = fall["players"].reindex(rows.index).astype("Int64")
    rows["rising"] = rows["source_player_id"].isin(risers)

    # By decile, not by the shown rate: the rates are calibration and can dip
    # out of order (prompt 14c, elite decile 1 above decile 2); while they
    # rise with the decile, as in prompt 14b, both give the same order.
    rows = rows.sort_values(
        ["elite_decile", "step_up_decile", "score_elite", "score_step_up"],
        ascending=False,
        na_position="last",
    )
    out = pd.DataFrame(
        {
            "season": rows["season"],
            "player_id": rows["source_player_id"].astype("Int64"),
            "player": rows["player_name"],
            "age": tenths(rows["age"]),
            "position": rows["sub_position"]
            .astype("string")
            .fillna(rows["position"].astype("string")),
            "club_at_forecast": rows["team_name"],
            "league": rows["league_key"].map(leagues).fillna(rows["league_key"]),
            "value_rank": rows["value_rank"].astype("Int64"),
            **{c: rows[c] for c in COMPUTED if c in rows},
            "value_eur": rows["market_value"],
            "fair_value_eur": rows["fair_value_eur"].round(-3),
            "ability_pct": rows["ability_pct"].round(0),
        }
    )
    return out[base_columns("candidates")].reset_index(drop=True)


def bargains_table(lists: dict[str, pd.DataFrame], leagues: dict[str, str]) -> pd.DataFrame:
    frames = []
    for name, frame in lists.items():
        frames.append(
            pd.DataFrame(
                {
                    "season": frame["season"],
                    "list": name,
                    "list_rank": frame["list_rank"],
                    "player_id": frame["source_player_id"].astype("Int64"),
                    "player": frame["player_name"],
                    "age": tenths(frame["age"]),
                    "position": frame["sub_position"]
                    .astype("string")
                    .fillna(frame["position"].astype("string")),
                    "club_at_forecast": frame["team_name"],
                    "league": frame["league_key"].map(leagues).fillna(frame["league_key"]),
                    "ability_pct": frame["ability_pct"].round(0),
                    "minutes_share": frame["minutes_share"].astype(float).round(2),
                    "value_eur": frame["value"],
                    "fair_value_eur": frame["predicted"].round(-3),
                    "value_to_fair": frame["ratio"].round(3),
                    "value_to_fair_pct": frame["ratio_pct"].round(1),
                    "below_fair_value": frame["ratio"].round(3) < 1,
                    "contract_expiry": frame["contract_expiry"],
                    "flag_count": frame["flag_count"],
                    "flags": frame["flags"],
                    "explanation": frame["explanation"],
                    "on_loan": frame["on_loan"],
                    "stale_value": frame["stale_value"],
                    "value_date": frame["value_date"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[base_columns("bargains")]


def dictionary(present: bool = True) -> pd.DataFrame:
    """One row per column; without ``present``, the columns of a past season's
    two tables (the players table, prompt 20f, exists for the newest season only)."""
    from scout.players import DICTIONARY as PLAYERS

    tables = {**DICTIONARY, "players": PLAYERS} if present else DICTIONARY
    return pd.DataFrame(
        [
            {"table": table, "column": column, "meaning": meaning}
            for table, columns in tables.items()
            for column, meaning in columns.items()
            if present or column not in PRESENT_COLUMNS
        ]
    )


# Backtest numbers, each next to its baseline


def model_table(inputs: dict) -> pd.DataFrame:
    """One row per model and measure: the model, its baseline, the base rate."""
    rows = []
    fair = summarize(inputs["fair_value"]).set_index("model")
    for measure, label in (
        ("mae_log", "error (log units, lower is better)"),
        ("median_ape", "median error"),
        ("spearman", "rank correlation"),
    ):
        rows.append(
            {
                "model": "fair value",
                "rows": int(fair.loc["lightgbm", "rows"]),
                "measure": label,
                "model_value": fair.loc["lightgbm", measure],
                "baseline": fair.loc["group_median", measure],
                "base_rate": np.nan,
            }
        )
    moves = summarize_moves(inputs["big_move"]).set_index("model")
    for measure, label in (
        ("auc", "AUC"),
        ("average_precision", "average precision"),
        ("precision_top50_per_season", "top 50 precision"),
    ):
        rows.append(
            {
                "model": "big move",
                "rows": int(moves.loc["lightgbm", "rows"]),
                "measure": label,
                "model_value": moves.loc["lightgbm", measure],
                "baseline": moves.loc["group_rate", measure],
                "base_rate": moves.loc["lightgbm", "base_rate"],
            }
        )
    elite = inputs["elite"]
    sets = {
        "elite, all candidates (training pool)": elite,
        f"elite, elite pool ({ELITE_MAX_AGE} or under)": elite[
            elite_pool(elite["value_rank"], elite["age"])
        ],
        "step up": inputs["step_up"],
        "riser": inputs["riser"],
    }
    for name, p in sets.items():
        model, base = evaluate(p), evaluate(p, score="score_baseline")
        for measure, label in (
            ("auc", "AUC"),
            ("average_precision", "average precision"),
            ("precision_top50", "top 50 precision"),
        ):
            rows.append(
                {
                    "model": name,
                    "rows": model["rows"],
                    "measure": label,
                    "model_value": model[measure],
                    "baseline": base[measure],
                    "base_rate": model["base_rate"],
                }
            )
    return pd.DataFrame(rows)


def _chance_md(table: ChanceTable, y_rate: float) -> str:
    pooled = table.pooled.reindex(table.buckets)
    cells = {
        b: show_rate(r if pd.notna(r) else None, int(n))
        for b, r, n in zip(pooled.index, pooled["rate"], pooled["players"], strict=True)
    }
    frame = pd.DataFrame([cells], index=["rate"])
    frame.insert(0, "all (base rate)", f"{y_rate:.1%}")
    return frame.to_markdown()


def _effects_md(effects: pd.DataFrame) -> str:
    out = effects.copy()
    for c in ("share_with", "rate_with", "rate_without"):
        out[c] = out[c].map(lambda v: f"{v:.1%}")
    out["ratio"] = out["ratio"].round(2)
    return out.to_markdown(index=False)


def _money(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        value_m=(frame["value_eur"] / 1e6).round(2),
        fair_value_m=(frame["fair_value_eur"] / 1e6).round(2),
    )


CANDIDATE_SHOW = [
    "player",
    "age",
    "position",
    "club_at_forecast",
    "current_club",
    "league",
    "value_rank",
    "elite_chance",
    "step_up_chance",
    "fall_back_risk",
    "flags",
    "rising",
    "big_move_chance",
    "value_m",
    "fair_value_m",
    "ability_pct",
    "explanation",
]
BARGAIN_SHOW = [
    "list_rank",
    "player",
    "age",
    "position",
    "club_at_forecast",
    "current_club",
    "league",
    "ability_pct",
    "value_m",
    "fair_value_m",
    "value_to_fair",
    "minutes_share",
    "contract_expiry",
    "flags",
    "on_loan",
    "stale_value",
    "explanation",
]


def _cell(pct, players) -> str:
    if pd.isna(pct):
        return ""
    return f"{'under 5' if pct == 0 else int(pct)}% (n={int(players):,})"


def readable_candidates(table: pd.DataFrame) -> pd.DataFrame:
    out = _money(table)
    for name in ("elite_chance", "step_up_chance", "big_move_chance"):
        out[name] = [
            _cell(p, n) for p, n in zip(out[f"{name}_pct"], out[f"{name}_players"], strict=True)
        ]
    out["fall_back_risk"] = [
        _cell(p, n)
        for p, n in zip(out["fall_back_risk_pct"], out["fall_back_risk_players"], strict=True)
    ]
    out["rising"] = out["rising"].map({True: "rising", False: ""})
    return out[[c for c in CANDIDATE_SHOW if c in out]]


def readable_bargains(table: pd.DataFrame) -> pd.DataFrame:
    out = _money(table)
    out["contract_expiry"] = pd.to_datetime(out["contract_expiry"]).dt.strftime("%Y-%m-%d")
    out["contract_expiry"] = out["contract_expiry"].fillna("")
    for tag in ("on_loan", "stale_value"):
        out[tag] = out[tag].map({True: "yes", False: ""})
    return out[[c for c in BARGAIN_SHOW if c in out]]


CHANCES = {
    "elite": "Elite chance (elite pool, by decile of the elite score)",
    "step_up": "Step up chance (ranked #101 to #2,000, by decile)",
    "big_move": "Big move chance (ranked #101 to #2,000, not at a top club, by decile)",
    "fall_back": "Fall back risk (elite pool, by flag count)",
}


def revision_md(old: dict, new: dict) -> str:
    """Old (every scored season) against new (the last five) for every
    bucket, with counts, the buckets merged under the floor and those whose
    rate is out of order."""
    parts = []
    for name, title in CHANCES.items():
        keys = COUNT_BUCKETS if name == "fall_back" else list(range(1, DECILES + 1))
        parts += [
            f"{title}:",
            "",
            candidates.compare(old[name], new[name], keys).to_markdown(index=False),
            "",
        ]
        merged = candidates.merges(new[name])
        parts += [
            f"Merged under the {candidates.MIN_BUCKET} player floor: " + ", ".join(merged) + "."
            if merged
            else f"No bucket is under the {candidates.MIN_BUCKET} player floor; none merged.",
            "",
        ]
        out_of_order = candidates.not_monotonic(new[name])
        parts += [
            "Not monotonic, shown as it is: " + "; ".join(out_of_order) + "."
            if out_of_order
            else "The realized rates rise from bucket to bucket.",
            "",
        ]
    return "\n".join(parts)


def write_report(
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    inputs: dict,
    tables: dict,
    bargain_results: pd.DataFrame,
    reports: Path,
    size: int = REPORT_SIZE,
    old_tables: dict | None = None,
    in_sample: dict | None = None,
) -> Path:
    season = int(cand["season"].max())
    added = cand[~in_ranks(cand["value_rank"].astype(float))]
    s = candidates.pool_summary(inputs["scouting"])
    pool_rows, y = s["rows"], s["y"]
    models = model_table(inputs)
    models["model_value"] = models["model_value"].round(3)
    models["baseline"] = models["baseline"].round(3)
    models["base_rate"] = models["base_rate"].round(3)
    elite_pool_rows = inputs["elite"][
        elite_pool(inputs["elite"]["value_rank"], inputs["elite"]["age"])
    ]
    step = inputs["step_up"][in_ranks(inputs["step_up"]["value_rank"])]
    move = inputs["big_move"].merge(inputs["scouting"][[*KEYS, "value_rank"]], on=KEYS)
    move = move[in_ranks(move["value_rank"])]
    verdict = "passes" if s["ability_passes"] else "fails"
    through = season - HORIZON_SEASONS
    window = f"{through - candidates.WINDOW + 1} to {through}"
    in_sample = in_sample or tables
    pooled_bargains = bargain_results[bargain_results["season"] == "pooled"].drop(columns="season")
    pooled_bargains = pooled_bargains.assign(
        outcome=pooled_bargains["outcome"].map(bargains.OUTCOMES)
    )
    dict_md = dictionary()
    parts = [
        "# Candidates and bargains",
        "",
        f"Season {season} (feature date 30 June 2026; 1 January 2026 for Brazil and",
        "Argentina). Built by `scout lists` from local files (`src/scout/lists.py`); the",
        "full tables are `data/processed/lists/candidates` and `bargains` (Parquet, CSV),",
        "explained column by column in the dictionary at the end.",
        "",
        "**How to read it.** Every chance is what happened to past forecasts: of the",
        "players that our tables, rebuilt for each past season with the models of that",
        "time, put in the same tenth (decile) of a model's score (or with the same number",
        "of flags), the share who made it two seasons on, over the last five seasons whose",
        "outcome is known, rounded to 5%, with the number of players behind it (n). Elite",
        "chance and fall back risk",
        f"are only given inside the elite pool (ranked #101 to #1,000, {ELITE_MAX_AGE} or",
        "under), where they were measured. Flags are reasons for doubt, each with the",
        "player's own number. Rising: in the riser model's top 50 of the season. The",
        "explanation names the three stats where he stands furthest from regulars of his",
        "position and age band; it reads the same numbers as the flags.",
        "",
        f"## Candidates: top {size} of {len(cand):,} (value rank #101 to #2,000)",
        "",
        readable_candidates(cand.head(size)).to_markdown(index=False),
        "",
        f"### Risers valued below #{RANKS[1]:,}: {len(added):,} rows added",
        "",
        f"Of the riser model's top {RISING_TOP}, these are valued below #{RANKS[1]:,} or have "
        "no market value. They are rows of the table with the chances blank (never measured",
        "there); risers inside the top 100 are not listed.",
        "",
        readable_candidates(added).to_markdown(index=False) if len(added) else "None.",
        "",
    ]
    for name, title in bargains.LISTS.items():
        frame = barg[barg["list"] == name]
        parts += [
            f"## Bargains, {title.lower()}: top {min(size, len(frame))} of {len(frame):,}",
            "",
            readable_bargains(frame.head(size)).to_markdown(index=False),
            "",
        ]
    parts += [
        "## Backtest: every number next to its baseline",
        "",
        "Each test season is predicted by a model trained only on earlier seasons. Fair",
        "value baseline: the median of the same league, position and age band. Big move",
        "baseline: the move rate of the same league, position and age band. Elite, step",
        "up and riser baseline: the success rate of players with a similar value rank and",
        "age. Top 50 precision is the share of successes in each season's top 50.",
        "",
        models.to_markdown(index=False),
        "",
        "### Shown chances: realized rates of the last five scored seasons (prompt 14c)",
        "",
        "Every chance in the table is the realized rate of the forecasts that",
        "`scout forecast history` rebuilt (`data/processed/forecasts/history/`), each with",
        "the models and buckets available at the time and scored two seasons on, pooled",
        f"over the last {candidates.WINDOW} scored seasons: {window}. Prompt 14b pooled every",
        f"scored season, 2015 to {through} (old); out of sample that leaned on 2015 to 2017",
        "and showed the top elite decile and the top fall back bucket too high for recent",
        "seasons (`reports/forecast_history.md`). A bucket under the 30 player floor merges",
        "with its neighbour and is named below. n is the number of forecast players with",
        "a known outcome behind the rate. Old against new, every bucket:",
        "",
        revision_md(old_tables, tables) if old_tables else "",
        "",
        "### In sample chance tables (backtest rows pooled, shown until prompt 14b)",
        "",
        f"Elite chance, elite pool, by decile of the elite score ({len(elite_pool_rows):,} "
        "players):",
        "",
        _chance_md(in_sample["elite"], elite_pool_rows["y"].mean()),
        "",
        f"Step up chance, ranked #101 to #2,000 ({len(step):,} players):",
        "",
        _chance_md(in_sample["step_up"], step["y"].mean()),
        "",
        f"Big move chance, ranked #101 to #2,000 and not at a top club ({len(move):,} players):",
        "",
        _chance_md(in_sample["big_move"], move["y"].mean()),
        "",
        "### Fall back risk (elite pool)",
        "",
        f"{len(pool_rows):,} players of the elite pool, seasons {int(pool_rows['season'].min())}"
        f" to {int(pool_rows['season'].max())}; {y.mean():.1%} fell back (value rank at least",
        "twice as bad two seasons later). Each flag's rate with it against without it:",
        "",
        _effects_md(s["effects"]),
        "",
        f"The ability flag ({ABILITY.rule}) raises the risk {s['ability_ratio']:.2f} times; "
        f"the owner's rule needs {ABILITY_MIN_RATIO}, so it {verdict} and "
        + (
            "counts."
            if s["ability_passes"]
            else "is left out of the count; the ability percentile is a column."
        ),
        "",
        "Fall back rate by flag count, in sample (shown until prompt 14b; two or more flags",
        "are one bucket since prompt 14, as three or more held few players):",
        "",
        candidates.wide(s["count"], candidates.COUNT_BUCKETS).to_markdown(),
        "",
        "### Risers per season",
        "",
        season_precision(inputs["riser"]).round(3).to_markdown(),
        "",
        "### Bargains: did past lists pay off?",
        "",
        f"Each season's top {bargains.TOP_K} of each list against players of the same season,",
        "ability band and age band who passed the same filters (`matched_rate`). Value up:",
        "market value 24 months later is higher. Club with higher Elo: within two seasons he",
        "plays for another club rated above his club now. `z` above 2 is unlikely to be luck.",
        "Part of the value rise is a low price correcting itself; the move to a stronger",
        "club is the cleaner test.",
        "",
        pooled_bargains.round(3).to_markdown(index=False),
        "",
        "## Data dictionary",
        "",
    ]
    for table in DICTIONARY:
        part = dict_md[dict_md["table"] == table].drop(columns="table")
        parts += [f"### {table}", "", part.to_markdown(index=False), ""]
    path = reports / "candidates.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        for suffix in ("parquet", "csv"):
            path = out_dir / f"{name}.{suffix}"
            tmp = path.with_name(path.name + ".tmp")
            if suffix == "parquet":
                frame.to_parquet(tmp, index=False)
            else:
                frame.to_csv(tmp, index=False)
            tmp.replace(path)
            written.append(path)
    return written


def with_explanation(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """A bargains list with flags and explanation from the peer table."""
    found = explain.flags_and_explanation(explain.for_rows(profile, frame))
    return frame.assign(
        flag_count=found["flag_count"], flags=found["flags"], explanation=found["explanation"]
    )


def season_tables(
    inputs: dict,
    features: pd.DataFrame,
    bargain_rows: pd.DataFrame,
    profiles: pd.DataFrame,
    snapshot: pd.Timestamp | None,
    leagues: dict[str, str],
    season: int | None = None,
    outcomes: pd.DataFrame | None = None,
    window: int | None = candidates.WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(candidates, bargains, chance tables) of one season, the newest when
    None. Chances are the realized rates of the scored past forecasts in
    ``outcomes`` (``scout forecast history``), or without them the in sample
    rates of the backtest; either from seasons up to ``season`` minus 2
    only, so a past season's tables are what we could have published then.
    ``window``: the realized rates pool that many scored seasons (None:
    every one, the prompt 14b rate)."""
    season = int(inputs["scouting"]["season"].max()) if season is None else season
    through = season - HORIZON_SEASONS
    if outcomes is None:
        tables = chance_tables(inputs, through=through)
    else:
        tables = candidates.realized_tables(outcomes, through=through, window=window)
    profile = explain.peer_table(explain.population(features, season))
    cand = candidates_table(inputs, tables, leagues, profile, season)
    lists = bargains.current_lists(bargain_rows, profiles, snapshot, season)
    lists = {name: with_explanation(frame, profile) for name, frame in lists.items()}
    return cand, bargains_table(lists, leagues), tables


def all_players(
    settings,
    inputs: dict,
    features: pd.DataFrame,
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    leagues: dict[str, str],
) -> pd.DataFrame:
    """The players table (prompt 20f, ``scout.players``), checked by its
    contract; ``report_available`` from the reports stored so far."""
    from scout import reports_text
    from scout.contracts.schemas import PLAYERS_TABLE, validate
    from scout.players import players_table

    reports = reports_text.page_reports(settings, cand, barg)
    table = players_table(
        inputs["scouting"], features, inputs["fair_value"], cand, barg, leagues, reports
    )
    return validate(PLAYERS_TABLE, table)


def run_lists(settings) -> Path:
    """``scout lists``: both tables, the dictionary and reports/candidates.md."""
    inputs = load(settings)
    leagues = league_names(settings)
    data, profiles, _, snapshot, features = bargains.load(settings)
    rows = bargains.eligible(data)
    outcomes = load_outcomes(settings)
    cand, barg, tables = season_tables(
        inputs, features, rows, profiles, snapshot, leagues, outcomes=outcomes
    )
    from scout.contracts.schemas import BARGAINS_TABLE, validate

    barg = validate(BARGAINS_TABLE, barg)
    season = int(cand["season"].max())
    through = season - HORIZON_SEASONS
    old_tables = candidates.realized_tables(outcomes, through=through, window=None)
    in_sample = chance_tables(inputs, through=through)
    added = int((~in_ranks(cand["value_rank"].astype(float))).sum())
    log.info("lists: %d risers valued below #%d added to the candidates", added, RANKS[1])
    results = bargains.backtest(rows)
    results.to_parquet(settings.paths.processed_dir / "models" / "bargains_backtest.parquet")

    from scout.present import add_present

    cand, barg = add_present(settings, cand, barg, inputs["scouting"])
    everyone = all_players(settings, inputs, features, cand, barg, leagues)
    out_dir = settings.paths.processed_dir / "lists"
    write_tables(
        {"candidates": cand, "bargains": barg, "players": everyone, "dictionary": dictionary()},
        out_dir,
    )
    log.info(
        "lists: %d candidates, %d bargains, %d players in %s",
        len(cand),
        len(barg),
        len(everyone),
        out_dir,
    )
    path = write_report(
        cand,
        barg,
        inputs,
        tables,
        results,
        settings.paths.reports_dir,
        old_tables=old_tables,
        in_sample=in_sample,
    )
    log.info("Report: %s", path)
    return path
