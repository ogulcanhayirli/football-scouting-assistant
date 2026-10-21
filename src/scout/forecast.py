"""Registered forecasts (prompt 14): freeze the two tables, score them later.

A forecast folder holds what the lists promised on one day, exactly as
published, so it can be scored against what happened without anyone being
able to adjust it afterwards:

    forecasts/<date>/candidates.csv, bargains.csv, dictionary.csv
    forecasts/<date>/forecast.json   season, feature dates, valuation date
    forecasts/<date>/README.md       the promises in words

``scout forecast freeze`` copies today's ``data/processed/lists`` there and
refuses to overwrite a folder that exists. ``scout forecast score FOLDER
--as-of DATE`` reads a folder and the local tables as they stand at a later
feature date and reports, for each promise, the bucket against what
happened, next to the base rate of that season's table:

* **elite chance** (elite pool rows): share whose value rank reached the top
  100 in a later season;
* **fall back risk** (elite pool rows, by flag count bucket): share whose
  value rank at the end was at least twice as bad;
* **step up chance** (rows ranked #101 to #2,000): share whose value rank
  became at least three times better and inside the top 500;
* **big move chance** (rows not at a top 20 club): share who joined a top 20
  club (by squad value in the forecast season) since the feature date,
  returns to a former club excluded;
* **bargains**: share of each list's top 50 whose market value rose, against
  the whole list.

Outcomes use only rows dated on or before ``as_of``: later seasons' value
ranks (up to two seasons on) from ``scouting.parquet`` and moves from the
transfers table. Before two seasons have passed the scores are interim and
the report says how many seasons it saw. A player with no later row (he left
the 37 leagues or lost his market value) is unknown, never a miss; the known
count stands next to every share.

``scout forecast history`` rebuilds the forecast of every season 2015 to
2023 in the same folder format (``data/processed/forecasts/history``), each
from backtest scores and chance tables of seasons whose outcomes were known
by then, and scores it two seasons on, so the command is tested on nine
seasons before the first registered forecast comes due in 2027.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scout import bargains
from scout.fsutil import atomic_write_text
from scout.scouting import (
    ELITE_RANK,
    FALL_BACK_FACTOR,
    HORIZON_SEASONS,
    STEP_UP_FACTOR,
    STEP_UP_RANK,
)

log = logging.getLogger(__name__)

TABLES = ("candidates", "bargains", "dictionary")
META = "forecast.json"
VALUATION_END = "2026-06-12"  # the Transfermarkt snapshot's last valuation (CLAUDE.md)
HISTORY = range(2015, 2024)
TOP_K = bargains.TOP_K

PROMISES = {
    "elite": ("elite_chance_pct", "entered the top 100 by value"),
    "fall_back": ("fall_back_risk_pct", "fell back (value rank at least twice as bad)"),
    "step_up": ("step_up_chance_pct", "stepped up (rank 3 times better, inside the top 500)"),
    "big_move": ("big_move_chance_pct", "joined a top 20 club"),
}


# Writing and reading a folder


def feature_dates(scouting: pd.DataFrame, season: int, leagues: dict[str, str]) -> dict:
    rows = scouting[scouting["season"].astype(int) == season]
    dates = rows.groupby("league_key")["feature_date"].max()
    out: dict[str, list[str]] = {}
    for key, date in dates.items():
        out.setdefault(pd.Timestamp(date).strftime("%Y-%m-%d"), []).append(leagues.get(key, key))
    return {d: sorted(v) for d, v in sorted(out.items())}


def meta_for(
    scouting: pd.DataFrame,
    season: int,
    leagues: dict[str, str],
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    created: str,
    kind: str,
) -> dict:
    rows = scouting[scouting["season"].astype(int) == season]
    return {
        "kind": kind,
        "created": created,
        "season": season,
        "feature_dates": feature_dates(scouting, season, leagues),
        "valuation_end": VALUATION_END if kind == "registered" else None,
        "latest_value_date": pd.Timestamp(rows["value_date"].max()).strftime("%Y-%m-%d"),
        "horizon_seasons": HORIZON_SEASONS,
        "rows": {"candidates": len(cand), "bargains": len(barg)},
    }


def write_folder(
    folder: Path, cand: pd.DataFrame, barg: pd.DataFrame, words: pd.DataFrame, meta: dict
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(TABLES, (cand, barg, words), strict=True):
        tmp = folder / f"{name}.csv.tmp"
        frame.to_csv(tmp, index=False)
        tmp.replace(folder / f"{name}.csv")
    atomic_write_text(folder / META, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return folder


def read_folder(folder: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((folder / META).read_text())
    cand = pd.read_csv(folder / "candidates.csv")
    barg = pd.read_csv(folder / "bargains.csv")
    return meta, cand, barg


def readme(meta: dict, words: pd.DataFrame) -> str:
    dates = "; ".join(f"{d} for {', '.join(v)}" for d, v in meta["feature_dates"].items())
    rows = meta["rows"]
    lines = [
        f"# Registered forecast of {meta['created']}",
        "",
        "The two user facing tables exactly as `scout lists` wrote them on "
        f"{meta['created']}, frozen so they can be scored when the outcomes are known. "
        "Never edit these files; a new forecast gets a new dated folder.",
        "",
        f"* Season: {meta['season']} ({meta['season']}/{(meta['season'] + 1) % 100:02d}, or "
        "the calendar year for calendar year leagues).",
        f"* Feature date (the day each row describes the player): {dates}. No row uses "
        "anything dated after it.",
        f"* Market values: the Transfermarkt snapshot, whose valuations end on "
        f"{meta['valuation_end']} (the latest value in these tables is dated "
        f"{meta['latest_value_date']}).",
        f"* Rows: {rows['candidates']:,} candidates, {rows['bargains']:,} bargains.",
        f"* Due: {meta['horizon_seasons']} seasons on, at the feature dates of season "
        f"{meta['season'] + meta['horizon_seasons']} (two years after the dates above); an "
        "interim score is possible after one season.",
        "* Score it with `uv run scout --config configs/backfill.ultra.yaml forecast score "
        f"forecasts/{meta['created']} --as-of <later feature date>`.",
        "",
        "## What each column promises",
        "",
        "Every chance is a historical rate: of past players in the same bucket, the share "
        "who made it. A well calibrated forecast sees about that share of this season's "
        "players in the bucket make it too.",
        "",
        "* `elite_chance_pct`: that share of the bucket's players reach the 100 most "
        "valuable players within two seasons (elite pool only: ranked #101 to #1,000, 23 or "
        "under). Rebuilt forecasts of 2015 to 2023 saw 30.5% of the players shown 40% make "
        "it, so the top bucket runs high (MODELS.md).",
        "* `fall_back_risk_pct`: that share of the elite pool players with the same flag "
        "count (0, 1, 2 or more) have a value rank at least twice as bad two seasons on.",
        "* `step_up_chance_pct`: that share of the bucket's players (ranked #101 to #2,000) "
        "see their value rank become at least three times better, inside the top 500, within "
        "two seasons.",
        "* `big_move_chance_pct`: that share of the bucket's players (not at a top 20 club) "
        "join a top 20 club by squad value within 24 months, a return to a former club not "
        "counted.",
        "* `rising`: the riser model's top 50; the riser backtest saw about 9% of such a "
        "top 50 rise, against a base rate under 2%.",
        "* `bargains`: the top 50 of each list (`list_rank` 1 to 50) gain market value over "
        "24 months more often than the rest of the list.",
        "* `flags`, `explanation`, values and ability are descriptions, not promises.",
        "",
        "## Columns",
        "",
        words.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def freeze(lists_dir: Path, scouting: pd.DataFrame, leagues: dict, folder: Path) -> Path:
    """Copy today's tables into ``folder``; refuse to overwrite one."""
    if folder.exists():
        raise FileExistsError(f"{folder} exists: a registered forecast is never overwritten")
    cand = pd.read_csv(lists_dir / "candidates.csv")
    barg = pd.read_csv(lists_dir / "bargains.csv")
    words = pd.read_csv(lists_dir / "dictionary.csv")
    season = int(cand["season"].max())
    meta = meta_for(scouting, season, leagues, cand, barg, folder.name, "registered")
    folder.mkdir(parents=True)
    for name in TABLES:
        shutil.copyfile(lists_dir / f"{name}.csv", folder / f"{name}.csv")
    atomic_write_text(folder / META, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(folder / "README.md", readme(meta, words))
    return folder


# Outcomes


def later_rows(scouting: pd.DataFrame, season: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Each player's row in seasons ``season + 1`` to ``season + 2`` dated on
    or before ``as_of`` (the league with most minutes), with ``ahead``."""
    s = scouting.assign(season=scouting["season"].astype(int))
    rows = s[s["season"].between(season + 1, season + HORIZON_SEASONS)]
    rows = rows[rows["feature_date"] <= as_of]
    rows = rows.sort_values("minutes").drop_duplicates(["source_player_id", "season"], keep="last")
    return rows.assign(ahead=rows["season"] - season)


def rank_outcomes(
    rank_now: pd.Series, player: pd.Series, later: pd.DataFrame, seen: int
) -> pd.DataFrame:
    """elite, step_up and fall_back per forecast row (missing when the
    player has no ranked later row; fall back needs the last season seen)."""
    ranked = later.dropna(subset=["value_rank"])
    best = ranked.groupby("source_player_id")["value_rank"].min()
    end = ranked[ranked["ahead"] == seen].set_index("source_player_id")["value_rank"]
    pid = player.astype("int64")
    b = pid.map(best).astype(float)
    e = pid.map(end).astype(float)
    r = rank_now.astype(float)
    known = b.notna() & r.notna()
    return pd.DataFrame(
        {
            "elite": (b <= ELITE_RANK).astype(float).where(known),
            "step_up": ((b <= STEP_UP_RANK) & (r / b >= STEP_UP_FACTOR)).astype(float).where(known),
            "fall_back": (e >= FALL_BACK_FACTOR * r).astype(float).where(e.notna() & r.notna()),
        },
        index=rank_now.index,
    )


def move_outcomes(
    cand: pd.DataFrame,
    season: int,
    scouting: pd.DataFrame,
    transfers: pd.DataFrame,
    as_of: pd.Timestamp,
    fallback_date: pd.Timestamp,
) -> pd.Series:
    """1 when the player joined a top 20 club of the forecast season after
    his feature date and by ``as_of`` (within 24 months), else 0."""
    from scout.models.moves import move_labels, top_clubs

    s = scouting.assign(season=scouting["season"].astype(int))
    now = s[s["season"] == season].sort_values("minutes")
    now = now.drop_duplicates("source_player_id", keep="last").set_index("source_player_id")
    pid = cand["player_id"].astype("int64")
    rows = pd.DataFrame(
        {
            "league_key": pid.map(now["league_key"]).fillna("unknown").to_numpy(),
            "season": season,
            "source_player_id": pid.to_numpy(),
            "feature_date": pid.map(now["feature_date"]).fillna(fallback_date).to_numpy(),
            "team_id": pid.map(now["team_id"]).to_numpy(),
        },
        index=cand.index,
    )
    seen = transfers[transfers["transfer_date"] <= as_of]
    labels = move_labels(rows, seen, top_clubs(s), pd.Timestamp.max)
    moved = labels["move_date"].notna() & (labels["move_date"] <= as_of)
    return moved.astype(float).where(~labels["at_top_club"].astype(bool))


def value_outcome(barg: pd.DataFrame, later: pd.DataFrame, seen: int) -> pd.Series:
    """1 when the market value in the last season seen is above the listed
    value."""
    end = later[later["ahead"] == seen].set_index("source_player_id")["market_value"]
    after = barg["player_id"].astype("int64").map(end).astype(float)
    return (after > barg["value_eur"].astype(float)).astype(float).where(after.notna())


def outcomes(
    meta: dict,
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    scouting: pd.DataFrame,
    transfers: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(candidates with elite, step_up, fall_back, big_move outcomes;
    bargains with gained_value; later seasons seen)."""
    season = int(meta["season"])
    later = later_rows(scouting, season, as_of)
    seen = int(later["ahead"].max()) if len(later) else 0
    first = pd.Timestamp(min(meta["feature_dates"]))
    cand = cand.copy()
    out = rank_outcomes(cand["value_rank"], cand["player_id"], later, seen)
    cand[out.columns] = out
    cand["big_move"] = move_outcomes(cand, season, scouting, transfers, as_of, first)
    barg = barg.assign(gained_value=value_outcome(barg, later, seen))
    return cand, barg, seen


# Scoring


def _label(pct) -> str:
    return "under 5%" if pct == 0 else f"{int(pct)}%"


def promise_table(frame: pd.DataFrame, promise: str, outcome: str) -> pd.DataFrame:
    """Per promised bucket: players, known, promised and actual share, then
    the table's base rate (``all``)."""
    rows = frame[frame[promise].notna()]
    out = []
    for pct, group in sorted(rows.groupby(promise), key=lambda t: t[0]):
        y = group[outcome].dropna()
        out.append(
            {
                "bucket": _label(pct),
                "promised": float(pct) / 100,
                "players": len(group),
                "known": len(y),
                "actual": y.mean() if len(y) else np.nan,
            }
        )
    y = rows[outcome].dropna()
    out.append(
        {
            "bucket": "all (base rate)",
            "promised": np.nan,
            "players": len(rows),
            "known": len(y),
            "actual": y.mean() if len(y) else np.nan,
        }
    )
    return pd.DataFrame(out)


def bargain_table(barg: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in bargains.LISTS:
        rows = barg[barg["list"] == name]
        for label, part in ((f"top {TOP_K}", rows[rows["list_rank"] <= TOP_K]), ("all", rows)):
            y = part["gained_value"].dropna()
            out.append(
                {
                    "list": name,
                    "rows": label if label != "all" else "all (base rate)",
                    "players": len(part),
                    "known": len(y),
                    "value_rose": y.mean() if len(y) else np.nan,
                }
            )
    return pd.DataFrame(out)


def score(
    meta: dict,
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    scouting: pd.DataFrame,
    transfers: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict:
    cand, barg, seen = outcomes(meta, cand, barg, scouting, transfers, as_of)
    tables = {name: promise_table(cand, column, name) for name, (column, _) in PROMISES.items()}
    tables["bargains"] = bargain_table(barg)
    return {"tables": tables, "seen": seen, "candidates": cand, "bargains": barg}


def _md(frame: pd.DataFrame) -> str:
    out = frame.copy()
    if "bucket" in out:  # the bucket is the promise
        out = out.drop(columns="promised")
    for column in ("promised", "actual", "value_rose", "base_rate"):
        if column in out:
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    return out.to_markdown(index=False)


def report(meta: dict, result: dict, as_of: pd.Timestamp, title: str) -> str:
    seen = result["seen"]
    state = "final" if seen >= HORIZON_SEASONS else f"interim: {seen} of {HORIZON_SEASONS} seasons"
    parts = [
        f"# {title}",
        "",
        f"Forecast of season {meta['season']} (created {meta['created']}), scored with data "
        f"dated up to {as_of.date()} ({state}). Each bucket's promised share against the "
        "share that happened among players whose outcome is known, then the base rate of "
        "the season's table. Big moves count up to the as of date or 24 months, whichever "
        "is earlier.",
        "",
    ]
    for name, (_, words) in PROMISES.items():
        parts += [f"## {name.replace('_', ' ')}: {words}", "", _md(result["tables"][name]), ""]
    parts += [
        f"## bargains: market value rose (top {TOP_K} of each list against the whole list)",
        "",
        _md(result["tables"]["bargains"]),
        "",
    ]
    return "\n".join(parts)


def load_outcome_data(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scout.backfill.transfers import transfers_path

    scouting = pd.read_parquet(settings.paths.processed_dir / "models" / "scouting.parquet")
    transfers = pd.read_parquet(transfers_path(settings))
    return scouting, transfers


def run_score(settings, folder: Path, as_of: str | None = None) -> Path:
    """``scout forecast score``: one folder against the local tables."""
    scouting, transfers = load_outcome_data(settings)
    when = pd.Timestamp(as_of) if as_of else pd.Timestamp(scouting["feature_date"].max())
    meta, cand, barg = read_folder(folder)
    result = score(meta, cand, barg, scouting, transfers, when)
    text = report(meta, result, when, f"Forecast {folder.name} scored at {when.date()}")
    path = settings.paths.reports_dir / f"forecast_{folder.name}_at_{when.date()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)
    return path


# History: the same folders for 2015 to 2023, scored two seasons on


def history_inputs(inputs: dict, season: int) -> dict:
    """The inputs as ``scout lists`` would have seen them for ``season``:
    the backtest's out of sample scores in place of the current ones."""
    s = inputs["scouting"].copy()
    names = {"elite": "score_elite", "step_up": "score_step_up", "riser": "score_riser"}
    at = s["season"] == season
    for source, column in names.items():
        p = inputs[source]
        p = p[p["season"] == season].set_index(["league_key", "season", "source_player_id"])[
            "score"
        ]
        keys = pd.MultiIndex.from_frame(s.loc[at, ["league_key", "season", "source_player_id"]])
        s.loc[at, column] = p.reindex(keys).to_numpy()
    move = inputs["big_move"]
    move = move[move["season"] == season]
    return {
        **inputs,
        "scouting": s,
        "big_move_now": move[["league_key", "season", "source_player_id"]].assign(
            score=move["score_lightgbm"].to_numpy()
        ),
    }


def label_agreement(cand: pd.DataFrame, barg: pd.DataFrame, inputs: dict, labels) -> dict:
    """Share of rows where the scored outcome equals the stored label (both
    known): the scorer's own check against the targets the models learned."""
    s = inputs["scouting"]
    season = int(cand["season"].iloc[0])
    now = s[s["season"] == season].sort_values("minutes")
    now = now.drop_duplicates("source_player_id", keep="last").set_index("source_player_id")
    out = {}
    pid = cand["player_id"].astype("int64")
    for name in ("elite", "step_up", "fall_back"):
        stored = pid.map(now[name]).astype(float)
        both = stored.notna() & cand[name].notna()
        out[name] = (float((stored[both] == cand.loc[both, name]).mean()), int(both.sum()))
    move = labels["big_move"]
    move = move[move["season"] == season].drop_duplicates("source_player_id")
    stored = pid.map(move.set_index("source_player_id")["big_move_24m"]).astype(float)
    both = stored.notna() & cand["big_move"].notna()
    out["big_move"] = (float((stored[both] == cand.loc[both, "big_move"]).mean()), int(both.sum()))
    grow = labels["growth"]
    grow = grow[grow["season"] == season].drop_duplicates("source_player_id")
    stored = barg["player_id"].astype("int64").map(grow.set_index("source_player_id")["g"])
    stored = (stored.astype(float) > 0).astype(float).where(stored.notna())
    both = stored.notna() & barg["gained_value"].notna()
    out["gained_value"] = (
        float((stored[both] == barg.loc[both, "gained_value"]).mean()),
        int(both.sum()),
    )
    return out


def pooled(results: dict[int, dict]) -> dict:
    """Every season's rows together: the same tables over 2015 to 2023."""
    cand = pd.concat([r["candidates"] for r in results.values()], ignore_index=True)
    barg = pd.concat([r["bargains"] for r in results.values()], ignore_index=True)
    tables = {name: promise_table(cand, column, name) for name, (column, _) in PROMISES.items()}
    tables["bargains"] = bargain_table(barg)
    return tables


def per_season(results: dict[int, dict]) -> pd.DataFrame:
    """Per season and promise: the base rate and the top bucket's promise and
    outcome."""
    rows = []
    for season, r in results.items():
        for name, table in r["tables"].items():
            if name == "bargains":
                for lst, part in table.groupby("list"):
                    top, base = part.iloc[0], part.iloc[1]
                    rows.append(
                        {
                            "season": season,
                            "promise": f"bargains, {lst}",
                            "base_rate": base["value_rose"],
                            "top_promised": np.nan,
                            "top_actual": top["value_rose"],
                            "top_known": int(top["known"]),
                        }
                    )
                continue
            buckets, base = table.iloc[:-1], table.iloc[-1]
            top = buckets.iloc[-1] if len(buckets) else None
            rows.append(
                {
                    "season": season,
                    "promise": name.replace("_", " "),
                    "base_rate": base["actual"],
                    "top_promised": np.nan if top is None else top["promised"],
                    "top_actual": np.nan if top is None else top["actual"],
                    "top_known": 0 if top is None else int(top["known"]),
                }
            )
    return pd.DataFrame(rows)


def run_history(settings, seasons=HISTORY) -> Path:
    """``scout forecast history``: rebuild, write and score every season."""
    from scout import lists
    from scout.tables import read_feature_table

    inputs = lists.load(settings)
    leagues = lists.league_names(settings)
    data, profiles, _, _, features = bargains.load(settings)
    rows = bargains.eligible(data)
    scouting, transfers = load_outcome_data(settings)
    label_root = settings.paths.processed_dir / "features" / "player_season_labels"
    growth = read_feature_table(label_root)[["season", "source_player_id", "log_growth_24m"]]
    growth = growth.assign(season=growth["season"].astype(int)).rename(
        columns={"log_growth_24m": "g"}
    )
    moves = pd.read_parquet(settings.paths.processed_dir / "models" / "big_move_labels.parquet")
    moves = moves.assign(season=moves["season"].astype(int))
    root = settings.paths.processed_dir / "forecasts" / "history"
    results, checks = {}, {}
    for season in seasons:
        past = history_inputs(inputs, season)
        cand, barg, _ = lists.season_tables(past, features, rows, profiles, None, leagues, season)
        meta = meta_for(scouting, season, leagues, cand, barg, f"history-{season}", "history")
        folder = write_folder(root / str(season), cand, barg, lists.dictionary(), meta)
        meta, cand, barg = read_folder(folder)  # score what was written, as for a real one
        rows_end = scouting[scouting["season"].astype(int) == season + HORIZON_SEASONS]
        as_of = pd.Timestamp(rows_end["feature_date"].max())
        result = score(meta, cand, barg, scouting, transfers, as_of)
        results[season] = result
        checks[season] = label_agreement(
            result["candidates"], result["bargains"], past, {"big_move": moves, "growth": growth}
        )
        log.info("forecast history %d: %d candidates, %d bargains", season, len(cand), len(barg))
    return write_history_report(results, checks, settings.paths.reports_dir)


def write_history_report(results: dict, checks: dict, reports: Path) -> Path:
    tables = pooled(results)
    first, last = min(results), max(results)
    agree = pd.DataFrame(
        [
            {"season": s, "outcome": k, "agreement": v[0], "rows": v[1]}
            for s, c in checks.items()
            for k, v in c.items()
        ]
    )
    agree_all = agree.groupby("outcome").apply(
        lambda g: pd.Series(
            {
                "agreement": (g["agreement"] * g["rows"]).sum() / max(g["rows"].sum(), 1),
                "rows": int(g["rows"].sum()),
            }
        ),
        include_groups=False,
    )
    parts = [
        "# Forecast scoring on the history",
        "",
        f"`scout forecast history`: the two tables of every season {first} to {last} rebuilt "
        "in the folder format of a registered forecast (`data/processed/forecasts/history/"
        "<season>/`), each from the backtest's out of sample scores and from chance tables "
        "of seasons whose outcomes were known by then (two seasons earlier and before), then "
        "scored by `scout forecast score`'s code two seasons on. Pooled over the seasons:",
        "",
    ]
    for name, (_, words) in PROMISES.items():
        parts += [f"## {name.replace('_', ' ')}: {words}", "", _md(tables[name]), ""]
    parts += [
        f"## bargains: market value rose (top {TOP_K} of each list against the whole list)",
        "",
        _md(tables["bargains"]),
        "",
        "## Per season: base rate and the top bucket",
        "",
        _md(
            per_season(results).rename(columns={"top_promised": "promised", "top_actual": "actual"})
        ),
        "",
        "## Check: scored outcomes against the stored labels",
        "",
        "The scorer reads later seasons' value ranks from the feature rows; the model "
        "labels read the Transfermarkt valuation two years after the feature date. Share of "
        "rows where both are known and agree:",
        "",
        agree_all.assign(agreement=agree_all["agreement"].map(lambda v: f"{v:.1%}")).to_markdown(),
        "",
    ]
    path = reports / "forecast_history.md"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(parts))
    return path
