"""The January window forecast (prompt 16): two lists frozen before the
window opens and scored on the transfers once it has closed.

``scout forecast window-freeze`` writes ``forecasts/window-2027-01/``:

* ``big_move.csv``: the big move model's top 50 of the newest season
  (``big_move_current.parquet``), and ``step_up.csv``: the step up model's
  top 50 among players ranked #101 to #2,000 by value (``scouting.parquet``);
* ``pool_big_move.csv`` and ``pool_step_up.csv``: every player each list was
  chosen from (player id and club today), for the base rate;
* ``top_clubs.csv``: the top 20 clubs by Elo and the top 20 by squad value,
  the clubs a player has to join;
* ``window.json`` and ``README.md``: the rule, the window, the freeze day and
  the counts.

The rule: **joins a top 20 club by Elo or value rank between 1 January and
3 February 2027.** A move counts when its date lies in the window, it goes
to one of the clubs in ``top_clubs.csv`` other than the player's club at
the freeze, and it is a transfer, a loan or a free move (a loan return is
not a new move, as in the big move model). Players whose club at the freeze
is already one of those clubs are left out of both pools. Top 20 by Elo:
the clubs of our leagues with the highest rating after their latest match
(``elo_matches.parquet``), among clubs that played in the last 12 months;
top 20 by value: the big move model's own clubs (``top_clubs``, squad value
in the newest season).

``scout forecast window-score FOLDER`` reads the transfers (the full table
and every dated snapshot of ``scout current``), and writes
``reports/window_<folder>.md``: each list's hits against the rate of its
pool. Before 3 February 2027 has passed in the newest transfer snapshot, the
report says it is interim.

A folder is frozen once; ``--refreeze`` (planned for 15 December 2026)
rewrites the lists and keeps the earlier freeze's counts and day in
``window.json``, with the earlier files in git.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from scout.fsutil import atomic_write_text

log = logging.getLogger(__name__)

WINDOW_START = pd.Timestamp("2027-01-01")
WINDOW_END = pd.Timestamp("2027-02-03")
FOLDER = "window-2027-01"
TOP = 50
TOP_N = 20
RULE = "joins a top 20 club by Elo or value rank between 1 January and 3 February 2027"
COUNTED_KINDS = {"transfer", "loan", "free", "unknown"}
RANKS = (101, 2000)
ELO_RECENT = pd.Timedelta(days=365)


def top_elo_clubs(elo_matches: pd.DataFrame, leagues: set[str], day: pd.Timestamp) -> pd.DataFrame:
    """team_id, team_name, elo, by_elo_rank: the ``TOP_N`` highest ratings
    after each club's latest league match before ``day``."""
    m = elo_matches[elo_matches["kind"].eq("league") & elo_matches["league_key"].isin(leagues)]
    m = m[pd.to_datetime(m["kickoff"], utc=True).dt.tz_convert(None) < day]
    sides = []
    for side in ("home", "away"):
        sides.append(
            pd.DataFrame(
                {
                    "team_id": m[f"{side}_id"].astype("Int64"),
                    "team_name": m[f"{side}_name"],
                    "kickoff": pd.to_datetime(m["kickoff"], utc=True).dt.tz_convert(None),
                    "elo": m[f"{side}_post"].astype(float),
                }
            )
        )
    clubs = pd.concat(sides).sort_values("kickoff").drop_duplicates("team_id", keep="last")
    clubs = clubs[clubs["kickoff"] >= day - ELO_RECENT]
    clubs = clubs.sort_values("elo", ascending=False).head(TOP_N).reset_index(drop=True)
    clubs["by_elo_rank"] = range(1, len(clubs) + 1)
    return clubs[["team_id", "team_name", "elo", "by_elo_rank"]]


def top_club_table(by_value: pd.DataFrame, by_elo: pd.DataFrame) -> pd.DataFrame:
    value = by_value.rename(columns={"rank": "by_value_rank"})[
        ["team_id", "team_name", "by_value_rank"]
    ].astype({"team_id": "Int64"})
    out = value.merge(by_elo[["team_id", "team_name", "by_elo_rank"]], on="team_id", how="outer")
    out["team_name"] = out["team_name_x"].fillna(out["team_name_y"])
    out = out[["team_id", "team_name", "by_value_rank", "by_elo_rank"]]
    out["by_value_rank"] = out["by_value_rank"].astype("Int64")
    out["by_elo_rank"] = out["by_elo_rank"].astype("Int64")
    return out.sort_values(["by_elo_rank", "by_value_rank"]).reset_index(drop=True)


def _list(pool: pd.DataFrame, score: str) -> pd.DataFrame:
    top = pool.sort_values(score, ascending=False).head(TOP).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def window_lists(
    big_move: pd.DataFrame,
    scouting: pd.DataFrame,
    current: pd.DataFrame | None,
    top: pd.DataFrame,
    leagues: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """The two pools with each player's club today and the two top 50 lists."""
    from scout.lists import newest_players

    clubs_now = (
        current.set_index("source_player_id")[["current_club_id", "current_club"]]
        if current is not None
        else pd.DataFrame(columns=["current_club_id", "current_club"])
    )
    newest = newest_players(scouting)
    team_of = newest.set_index("source_player_id")["team_id"]
    top_ids = set(top["team_id"].dropna().astype(int))

    def frame(rows: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
        pid = rows["source_player_id"].astype("Int64")
        out = pd.DataFrame(
            {
                "player_id": pid.to_numpy(),
                "player": rows["player_name"].to_numpy(),
                "age": rows["age"].astype(float).round(1).to_numpy(),
                "position": rows["position"].astype(str).to_numpy(),
                "club_at_forecast": rows["team_name"].to_numpy(),
                "league": rows["league_key"].map(leagues).fillna(rows["league_key"]).to_numpy(),
                "value_eur": rows["market_value"].to_numpy(),
                "score": score.round(4).to_numpy(),
            }
        )
        now_id = pid.map(clubs_now["current_club_id"]).astype("Int64").to_numpy()
        fallback = pid.map(team_of).astype("Int64").to_numpy()
        out["club_id_at_freeze"] = pd.array(now_id, dtype="Int64").fillna(
            pd.array(fallback, dtype="Int64")
        )
        out["club_at_freeze"] = pid.map(clubs_now["current_club"]).to_numpy()
        out["club_at_freeze"] = out["club_at_freeze"].fillna(out["club_at_forecast"])
        at_top = out["club_id_at_freeze"].isin(top_ids).fillna(False)
        return out[~at_top].reset_index(drop=True)

    move_pool = frame(big_move, big_move["score"])
    ranked = newest["value_rank"].between(*RANKS) & newest["score_step_up"].notna()
    step = newest[ranked]
    step_pool = frame(step, step["score_step_up"])
    step_pool["value_rank"] = _value_ranks(step, step_pool)
    return {
        "pool_big_move": move_pool,
        "pool_step_up": step_pool,
        "big_move": _list(move_pool, "score"),
        "step_up": _list(step_pool, "score"),
    }


def _value_ranks(step: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    ranks = step.set_index(step["source_player_id"].astype("Int64"))["value_rank"]
    ranks = ranks[~ranks.index.duplicated()]
    return pool["player_id"].map(ranks).astype("Int64")


LIST_COLUMNS = [
    "rank",
    "player_id",
    "player",
    "age",
    "position",
    "club_at_forecast",
    "club_at_freeze",
    "club_id_at_freeze",
    "league",
    "value_eur",
    "score",
]
POOL_COLUMNS = ["player_id", "club_id_at_freeze"]


def readme(meta: dict) -> str:
    lines = [
        f"# Window forecast {meta['folder']}",
        "",
        f"Frozen on {meta['frozen']}. **Rule: {meta['rule']}.**",
        "",
        "Two lists, each scored against the rate of the pool it was chosen from:",
        "",
        f"* `big_move.csv`: the big move model's top {TOP} of season {meta['season']} "
        f"({meta['lists']['big_move']} players; pool {meta['pools']['big_move']:,}, "
        "`pool_big_move.csv`): regular players not at a top 20 club by squad value at the "
        "feature date.",
        f"* `step_up.csv`: the step up model's top {TOP} among players ranked "
        f"#{RANKS[0]} to #{RANKS[1]:,} by value ({meta['lists']['step_up']} players; pool "
        f"{meta['pools']['step_up']:,}, `pool_step_up.csv`).",
        "",
        "Both pools leave out players whose club at the freeze (the squads of the freeze "
        "day, else the club at forecast) is already one of the clubs in `top_clubs.csv`: "
        f"the top {TOP_N} by Elo (rating after the latest league match, clubs that played in "
        f"the 12 months before the freeze) and the top {TOP_N} by squad value in season "
        f"{meta['season']} (the big move model's clubs), {meta['top_clubs']} clubs in all.",
        "",
        "A move counts when it is dated 1 January to 3 February 2027, goes to one of those "
        "clubs other than the club at the freeze, and is a transfer, a loan or a free move "
        "(a loan return is not a new move). A player with no move in that window is a miss; "
        "the transfers come from API Football's `/transfers` per club.",
        "",
        "Score with `scout forecast window-score forecasts/window-2027-01` once a transfer "
        "snapshot after 3 February 2027 exists (`scout current`). The lists are refreshed "
        "and frozen again on 15 December 2026 (`scout forecast window-freeze --refreeze`); "
        "this freeze stays in git and in `window.json`.",
        "",
        "Columns of the lists: `rank`, `player_id` (API Football), `player`, `age` (at the "
        "feature date), `position`, `club_at_forecast` (most league minutes in season "
        f"{meta['season']}), `club_at_freeze` and `club_id_at_freeze`, `league`, "
        "`value_eur` (Transfermarkt, up to 12 June 2026), `score` (the model's score, not a "
        "probability).",
        "",
    ]
    if meta.get("earlier"):
        lines += ["## Earlier freezes", ""]
        lines += [
            f"* {e['frozen']}: lists {e['lists']}, pools {e['pools']}" for e in meta["earlier"]
        ]
        lines.append("")
    return "\n".join(lines)


def freeze_window(
    settings,
    folder: Path,
    frozen: str,
    refreeze: bool = False,
) -> Path:
    from scout.lists import league_names
    from scout.models.moves import top_clubs
    from scout.tables import read_feature_table

    folder = Path(folder)
    meta_path = folder / "window.json"
    earlier = []
    if meta_path.exists():
        if not refreeze:
            raise FileExistsError(f"{folder} is already frozen; pass --refreeze on 15 December")
        old = json.loads(meta_path.read_text())
        earlier = [*old.get("earlier", []), {k: old[k] for k in ("frozen", "lists", "pools")}]
    processed = settings.paths.processed_dir
    features = read_feature_table(processed / "features" / "player_season_features")
    season = int(features["season"].max())
    by_value = top_clubs(features)
    by_value = by_value[by_value["season"] == season]
    elo = pd.read_parquet(processed / "elo" / "elo_matches.parquet")
    by_elo = top_elo_clubs(elo, {lg.key for lg in settings.leagues}, pd.Timestamp(frozen))
    top = top_club_table(by_value, by_elo)
    current_path = processed / "current" / "current_club.parquet"
    current = pd.read_parquet(current_path) if current_path.exists() else None
    big_move = pd.read_parquet(processed / "models" / "big_move_current.parquet")
    scouting = pd.read_parquet(processed / "models" / "scouting.parquet")
    lists = window_lists(big_move, scouting, current, top, league_names(settings))

    folder.mkdir(parents=True, exist_ok=True)
    for name in ("big_move", "step_up"):
        columns = [*LIST_COLUMNS, *(["value_rank"] if name == "step_up" else [])]
        lists[name][columns].to_csv(folder / f"{name}.csv", index=False)
        lists[f"pool_{name}"][POOL_COLUMNS].to_csv(folder / f"pool_{name}.csv", index=False)
    top.to_csv(folder / "top_clubs.csv", index=False)
    meta = {
        "folder": folder.name,
        "rule": RULE,
        "window": [WINDOW_START.strftime("%Y-%m-%d"), WINDOW_END.strftime("%Y-%m-%d")],
        "frozen": frozen,
        "season": season,
        "counted_kinds": sorted(COUNTED_KINDS),
        "top_clubs": len(top),
        "lists": {n: len(lists[n]) for n in ("big_move", "step_up")},
        "pools": {n: len(lists[f"pool_{n}"]) for n in ("big_move", "step_up")},
        "current_club_known": current is not None,
        "refreeze_planned": "2026-12-15",
        "earlier": earlier,
    }
    atomic_write_text(meta_path, json.dumps(meta, indent=2) + "\n")
    atomic_write_text(folder / "README.md", readme(meta))
    log.info("window: froze %s (%s), pools %s", folder, meta["lists"], meta["pools"])
    return folder


def all_transfers(settings) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """The transfers table and every snapshot of ``scout current``, deduplicated,
    and the newest snapshot day."""
    from scout.backfill.transfers import transfers_path
    from scout.providers.api_football.transfers import TRANSFER_KEY

    frames = []
    path = transfers_path(settings)
    if path.exists():
        frames.append(pd.read_parquet(path))
    snaps = sorted((settings.paths.processed_dir / "current" / "transfers").glob("as_of=*.parquet"))
    frames += [pd.read_parquet(p) for p in snaps]
    if not frames:
        return pd.DataFrame(), None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(TRANSFER_KEY)
    newest = pd.Timestamp(snaps[-1].stem.split("=", 1)[1]) if snaps else None
    return out, newest


def window_hits(pool: pd.DataFrame, transfers: pd.DataFrame, top_ids: set[int]) -> pd.Series:
    """1 when the pool player joined a top club (not his own) inside the window."""
    t = transfers[
        transfers["transfer_date"].between(WINDOW_START, WINDOW_END)
        & transfers["kind"].isin(COUNTED_KINDS)
        & transfers["team_in_id"].isin(top_ids)
    ]
    moves = pool[["player_id", "club_id_at_freeze"]].merge(
        t[["source_player_id", "team_in_id"]].rename(columns={"source_player_id": "player_id"}),
        on="player_id",
    )
    moves = moves[moves["team_in_id"] != moves["club_id_at_freeze"]]
    return pool["player_id"].isin(set(moves["player_id"])).astype(int)


def score_window(folder: Path, transfers: pd.DataFrame, newest: pd.Timestamp | None) -> str:
    folder = Path(folder)
    meta = json.loads((folder / "window.json").read_text())
    top = pd.read_csv(folder / "top_clubs.csv")
    top_ids = set(top["team_id"].dropna().astype(int))
    t = transfers.copy()
    for column in ("source_player_id", "team_in_id"):
        t[column] = t[column].astype("Int64")
    final = newest is not None and newest > WINDOW_END
    lines = [
        f"# Window forecast {meta['folder']}, scored",
        "",
        f"Rule: {meta['rule']}. Frozen on {meta['frozen']}. Newest transfer snapshot: "
        f"{newest.date() if newest is not None else 'none'}.",
        "",
        (
            "Final: the newest snapshot is after the window closed."
            if final
            else "**Interim**: the window has not closed in the newest transfer snapshot."
        ),
        "",
        "| List | Players | Joined a top club | Rate | Pool | Pool joined | Pool rate | Lift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("big_move", "step_up"):
        listed = pd.read_csv(folder / f"{name}.csv")
        pool = pd.read_csv(folder / f"pool_{name}.csv")
        for frame in (listed, pool):
            frame["player_id"] = frame["player_id"].astype("Int64")
            frame["club_id_at_freeze"] = frame["club_id_at_freeze"].astype("Int64")
        hit = window_hits(listed, t, top_ids)
        base = window_hits(pool, t, top_ids)
        rate, base_rate = hit.mean(), base.mean()
        lift = f"{rate / base_rate:.1f}x" if base_rate > 0 else "n/a"
        lines.append(
            f"| {name} | {len(listed)} | {int(hit.sum())} | {rate:.1%} | {len(pool):,} | "
            f"{int(base.sum())} | {base_rate:.2%} | {lift} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_window_score(settings, folder: Path) -> Path:
    transfers, newest = all_transfers(settings)
    text = score_window(folder, transfers, newest)
    reports = settings.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"window_{Path(folder).name}.md"
    atomic_write_text(path, text)
    return path
