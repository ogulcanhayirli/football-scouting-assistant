"""The static page (decision 9): one HTML file with the two tables embedded.

``scout site`` reads the user facing tables of ``scout lists``
(``data/processed/lists/``) and writes ``data/site/``:

* ``index.html``: both tables as JSON inside the page, sorted, filtered and
  searched by a little plain JavaScript (``site_template.html``), with a
  "how to read this" box: the chances are calibrated odds from past seasons,
  not predictions beyond the market, the valuation date, the forecast date
  and the five season window; each player's club today and his season so
  far (prompt 16), as of the day of the last pull;
* ``data/``: the same two tables as CSV and Parquet, and the dictionary;
* ``data/similar.json`` (prompt 18): the similar players panel and search,
  loaded by the page only when they are opened: today's players and every
  earlier season of players once ranked in the top ``HISTORY_TOP`` by value,
  each with name, age, club, league, ability and his embedding as 8 bit
  integers (``scout embed``). Without embeddings it says so and the page
  shows a note. Since prompt 18b it also holds the measured greats
  (``scout greats``): each great's season vector by age, for the page to
  search today's players of that age; today's rows carry their age in the
  middle of the season for it;
* ``statsbomb-logo.png``: the logo the StatsBomb user agreement asks every
  published analysis of its data to carry, written only with the greats
  (no greats are shown without it).

Only the two tables, the dictionary, the similar players file and the logo
are published, never raw match stats, API responses or StatsBomb data (the
greats are results computed from it). The rising tag of a
bargain is the riser model's top 50 of ``scouting.parquet``
(``scout.lists.rising_ids``), the same set the candidates table marks;
nothing else is read from the model outputs.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scout.candidates import WINDOW
from scout.flags import FLAGS, TAGS
from scout.fsutil import atomic_write_text
from scout.scouting import ELITE_MAX_AGE, ELITE_RANK, ELITE_REACH, HORIZON_SEASONS

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).with_name("site_template.html")
PUBLISHED = ("candidates", "bargains")  # each as .csv and .parquet, plus dictionary.csv
SIMILAR = "similar.json"  # in data/, next to the downloads
LOGO = "statsbomb-logo.png"  # next to the page, only with the greats
HISTORY_TOP = 500  # "looks like X at Y": every season of players once ranked this high
SCALE = 127  # unit vectors as 8 bit integers

# The columns each tab carries; the rest of the tables stay in the downloads.
CANDIDATE_COLUMNS = [
    "player",
    "player_id",
    "age",
    "position",
    "club_at_forecast",
    "league",
    "current_club",
    "current_league",
    "moved_since_forecast",
    "value_rank",
    "value_eur",
    "fair_value_eur",
    "elite_decile",
    "elite_chance_pct",
    "elite_chance_players",
    "step_up_decile",
    "step_up_chance_pct",
    "step_up_chance_players",
    "big_move_decile",
    "big_move_chance_pct",
    "big_move_chance_players",
    "fall_back_risk_pct",
    "fall_back_risk_players",
    "flag_count",
    "flags",
    "explanation",
    "rising",
    "ability_pct",
    "so_far_matches",
    "so_far_minutes",
    "so_far_minutes_share",
    "so_far_ability_pct",
    "so_far_goals_per90",
    "so_far_assists_per90",
    "so_far_flag_count",
    "so_far_flags",
]
BARGAIN_COLUMNS = [
    "list",
    "list_rank",
    "player",
    "player_id",
    "age",
    "position",
    "club_at_forecast",
    "league",
    "current_club",
    "current_league",
    "moved_since_forecast",
    "value_eur",
    "fair_value_eur",
    "value_to_fair",
    "value_to_fair_pct",
    "ability_pct",
    "minutes_share",
    "contract_expiry",
    "flag_count",
    "flags",
    "explanation",
    "on_loan",
    "stale_value",
    "value_date",
    "rising",
    "so_far_matches",
    "so_far_minutes",
    "so_far_minutes_share",
    "so_far_ability_pct",
    "so_far_goals_per90",
    "so_far_assists_per90",
    "so_far_flag_count",
    "so_far_flags",
]


def flag_keys(flags: pd.Series) -> list[list[str]]:
    """The raised flags of each row by name, read from the flag text."""
    return [
        [name for name, flag in FLAGS.items() if isinstance(text, str) and flag.rule in text]
        for text in flags
    ]


def _records(frame: pd.DataFrame, columns: list[str]) -> dict:
    """Column names and one list per row, dates as ISO days, missing values as null."""
    frame = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")
    frame = frame.astype(object).where(frame.notna(), None)
    rows = [
        [v.item() if isinstance(v, np.generic) else v for v in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows}


def registered_forecasts(root: Path) -> list[dict]:
    """The committed forecast folders (``forecasts/<date>/forecast.json``), newest first."""
    found = []
    for meta in sorted(Path(root).glob("*/forecast.json"), reverse=True):
        info = json.loads(meta.read_text())
        found.append(
            {
                "date": meta.parent.name,
                "season": info.get("season"),
                "revisions": len(info.get("revisions", [])),
            }
        )
    return found


def fall_back_buckets(cand: pd.DataFrame) -> list[dict]:
    """Fall back risk by flag count as the table shows it (elite pool rows)."""
    pool = cand[cand["fall_back_risk_pct"].notna()]
    count = pool["flag_count"].clip(upper=2)
    out = []
    for n, group in pool.groupby(count):
        out.append(
            {
                "flags": "2 or more" if n >= 2 else str(int(n)),
                "pct": int(group["fall_back_risk_pct"].iloc[0]),
                "players": int(group["fall_back_risk_players"].iloc[0]),
            }
        )
    return out


def _day(value) -> str | None:
    return pd.Timestamp(value).strftime("%Y-%m-%d") if pd.notna(value) else None


def page_data(
    cand: pd.DataFrame,
    barg: pd.DataFrame,
    words: pd.DataFrame,
    *,
    rising: set,
    feature_dates: dict[str, list[str]],
    valuation_end: pd.Timestamp,
    latest_value_date: pd.Timestamp | None,
    built: pd.Timestamp,
    forecasts: list[dict],
    tracked: int,
    greats: bool = False,
) -> dict:
    """Everything the page shows, as plain JSON values."""
    season = int(cand["season"].max())
    through = season - HORIZON_SEASONS
    cand = cand.assign(flag_keys=flag_keys(cand["flags"]))
    barg = barg.assign(
        flag_keys=flag_keys(barg["flags"]), rising=barg["player_id"].isin(rising).to_numpy()
    )
    return {
        "season": season,
        "season_label": f"{season}/{str(season + 1)[-2:]}",
        "built": built.strftime("%Y-%m-%d"),
        "feature_dates": feature_dates,
        "valuation_end": valuation_end.strftime("%Y-%m-%d"),
        "latest_value_date": (
            latest_value_date.strftime("%Y-%m-%d") if latest_value_date is not None else None
        ),
        "window": {"seasons": WINDOW, "first": through - WINDOW + 1, "last": through},
        "horizon": HORIZON_SEASONS,
        "elite_pool": {"from": ELITE_RANK + 1, "to": ELITE_REACH, "max_age": ELITE_MAX_AGE},
        "fall_back": fall_back_buckets(cand),
        "forecasts": forecasts,
        "flags": {name: flag.rule for name, flag in FLAGS.items()},
        "tags": TAGS,
        "so_far_date": _day(cand["so_far_date"].max()) if "so_far_date" in cand else None,
        "leagues": int(cand["league"].nunique()),
        "tracked": tracked,
        "greats": greats,
        "candidates": _records(cand, [*CANDIDATE_COLUMNS, "flag_keys"]),
        "bargains": _records(barg, [*BARGAIN_COLUMNS, "flag_keys"]),
        "dictionary": _records(words, ["table", "column", "meaning"]),
    }


def _codes(values: pd.Series, table: list[str]) -> list:
    """Each value as its position in ``table`` (null when missing)."""
    where = {v: i for i, v in enumerate(table)}
    return [where.get(v) if isinstance(v, str) else None for v in values]


def _int8(frame: pd.DataFrame, columns: list[str]) -> str:
    from scout.embeddings import unit

    u = unit(frame[columns].to_numpy(dtype=float))
    q = np.clip(np.round(u * SCALE), -SCALE, SCALE).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def greats_data(greats: pd.DataFrame | None, info: dict | None, leagues: dict[str, str]) -> dict:
    """The measured greats block of the similar players file: one row per
    great and age (in the middle of the season), the vectors as 8 bit
    integers, the label and the caveats the tab shows."""
    if greats is None or greats.empty or info is None:
        return {"available": False}
    emb = [c for c in greats.columns if c.startswith("emb_")]
    rows = greats.sort_values(["great", "age_years"], kind="stable")
    parts = [
        rows["great"].tolist(),
        rows["season"].astype(int).tolist(),
        rows["age_years"].astype(int).tolist(),
        rows["source"].tolist(),
        rows["team"].tolist(),
        rows["league_key"].map(leagues).fillna(rows["league_key"]).tolist(),
        rows["group"].tolist(),
        rows["minutes"].astype(float).round(0).astype(int).tolist(),
        rows["thin"].astype(bool).tolist(),
        rows["ability_pct"].astype(float).round(0).astype(int).tolist(),
        rows["peers"].tolist(),
    ]
    return {
        "available": True,
        "label": info["label"],
        "attribution": info["attribution"],
        "excluded": info.get("excluded", []),
        "identity_hit": info.get("identity_hit"),
        "logo": LOGO,
        "columns": [
            "great",
            "season",
            "age",
            "source",
            "club",
            "league",
            "group",
            "minutes",
            "thin",
            "ability",
            "peers",
        ],
        "rows": [list(r) for r in zip(*parts, strict=True)],
        "vectors": _int8(rows, emb),
    }


def similar_data(
    vectors: pd.DataFrame | None,
    current: pd.DataFrame | None,
    leagues: dict[str, str],
    tested: dict | None = None,
    greats: dict | None = None,
    calendar_leagues=(),
) -> dict:
    """The similar players file: today's players (main rows of the newest
    season) and the earlier seasons of players once ranked in the top
    ``HISTORY_TOP`` by value, rows as lists, vectors as one base64 string
    of 8 bit integers per table (row after row); and the measured greats
    (``greats_data``), with today's age in the middle of the season."""
    from scout.greats import mid_season_age

    if vectors is None or vectors.empty:
        return {"available": False}
    emb = [c for c in vectors.columns if c.startswith("emb_")]
    columns = emb or [c for c in vectors.columns if c.startswith("style_")]
    main = vectors[vectors["main"].astype(bool)].copy()
    newest = int(main["season"].max())
    main["league"] = main["league_key"].map(leagues).fillna(main["league_key"])
    if current is not None:
        cur = current.set_index("source_player_id")
        pid = main["source_player_id"].astype(int)
        main["club_now"] = pid.map(cur["current_club"])
        now_id = pid.map(cur["current_club_id"]).astype("Int64")
        moved = now_id.notna() & (now_id != main["team_id"].astype("Int64"))
        main["club_now"] = main["club_now"].where(moved.fillna(False).astype(bool))
    else:
        main["club_now"] = None
    today = main[main["season"] == newest].sort_values("player_name", kind="stable")
    today = today.assign(age_mid=mid_season_age(today, calendar_leagues))
    once_top = main.loc[main["value_rank"] <= HISTORY_TOP, "source_player_id"].unique()
    history = main[(main["season"] < newest) & main["source_player_id"].isin(once_top)]
    history = history.sort_values(["source_player_id", "season"], kind="stable")
    groups = sorted(main["group"].dropna().unique())
    league_list = sorted(main["league"].dropna().unique())
    clubs = sorted(
        set(today["team_name"].dropna())
        | set(today["club_now"].dropna())
        | set(history["team_name"].dropna())
    )

    def table(rows: pd.DataFrame, with_now: bool) -> dict:
        out = {
            "columns": ["id", "name", "season", "age", "group", "position", "club", "league"]
            + (["club_now"] if with_now else [])
            + ["ability", "value_rank"]
            + (["age_mid"] if with_now else []),
            "rows": [],
            "vectors": _int8(rows, columns),
        }
        parts = [
            rows["source_player_id"].astype(int).tolist(),
            rows["player_name"].tolist(),
            rows["season"].astype(int).tolist(),
            (np.floor(rows["age"].astype(float) * 10) / 10).tolist(),  # cut, so whole years stay
            _codes(rows["group"], groups),
            rows["sub_position"].where(rows["sub_position"].notna(), None).tolist(),
            _codes(rows["team_name"], clubs),
            _codes(rows["league"], league_list),
        ]
        if with_now:
            parts.append(_codes(rows["club_now"], clubs))
        parts.append(rows["ability_pct"].astype(float).round(0).astype(int).tolist())
        parts.append([int(v) if pd.notna(v) else None for v in rows["value_rank"]])
        if with_now:
            parts.append((np.floor(rows["age_mid"].astype(float) * 10) / 10).tolist())
        out["rows"] = [list(r) for r in zip(*parts, strict=True)]
        return out

    return {
        "available": True,
        "season": newest,
        "dim": len(columns),
        "scale": SCALE,
        "method": "learned" if emb else "baseline",
        "tested": tested or {},
        "history_top": HISTORY_TOP,
        "groups": groups,
        "leagues": league_list,
        "clubs": clubs,
        "today": table(today, with_now=True),
        "history": table(history, with_now=False),
        "greats": greats or {"available": False},
    }


def render(data: dict, template: Path = TEMPLATE) -> str:
    """The page: the template with the data as a JSON script block. ``</`` is
    escaped so no text in the tables can end the block."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    html = template.read_text(encoding="utf-8")
    return html.replace("__TITLE__", f"Scouting lists {data['season_label']}").replace(
        "__DATA__", payload
    )


def build_site(settings, out_dir: Path | None = None, forecasts_root: Path = Path("forecasts")):
    """``scout site``: ``index.html`` and the downloads in ``data/site/``."""
    from scout.forecast import feature_dates
    from scout.lists import league_names, rising_ids
    from scout.providers.transfermarkt import TransfermarktValuationProvider

    processed = settings.paths.processed_dir
    lists_dir = processed / "lists"
    out_dir = Path(out_dir) if out_dir else settings.paths.processed_dir.parent / "site"
    cand = pd.read_parquet(lists_dir / "candidates.parquet")
    barg = pd.read_parquet(lists_dir / "bargains.parquet")
    words = pd.read_csv(lists_dir / "dictionary.csv")
    scouting = pd.read_parquet(processed / "models" / "scouting.parquet")
    season = int(cand["season"].max())
    newest = scouting[scouting["season"].astype(int) == season]
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    latest = newest["value_date"].max()
    greats, info, logo = _greats_inputs(processed)
    greats_ready = all(x is not None for x in (greats, info, logo))
    data = page_data(
        cand,
        barg,
        words,
        rising=rising_ids(scouting, season=season),
        feature_dates=feature_dates(scouting, season, league_names(settings)),
        valuation_end=tm.data_end(),
        latest_value_date=pd.Timestamp(latest) if pd.notna(latest) else None,
        built=pd.Timestamp.now(tz="UTC"),
        forecasts=registered_forecasts(forecasts_root),
        tracked=len(settings.leagues),
        greats=greats_ready,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "index.html"
    atomic_write_text(page, render(data))
    downloads = out_dir / "data"
    downloads.mkdir(exist_ok=True)
    written = [page]
    for name in PUBLISHED:
        for suffix in ("csv", "parquet"):
            target = downloads / f"{name}.{suffix}"
            shutil.copyfile(lists_dir / f"{name}.{suffix}", target)
            written.append(target)
    shutil.copyfile(lists_dir / "dictionary.csv", downloads / "dictionary.csv")
    written.append(downloads / "dictionary.csv")
    vectors, current, tested = _similar_inputs(processed)
    names = league_names(settings)
    calendar = [lg.key for lg in settings.leagues if lg.calendar_year]
    great_block = greats_data(greats, info, names) if greats_ready else {"available": False}
    similar = similar_data(vectors, current, names, tested, great_block, calendar)
    target = downloads / SIMILAR
    atomic_write_text(target, json.dumps(similar, ensure_ascii=False, separators=(",", ":")))
    written.append(target)
    stale_logo = out_dir / LOGO
    if great_block["available"]:
        shutil.copyfile(logo, stale_logo)
        written.append(stale_logo)
    else:
        stale_logo.unlink(missing_ok=True)  # no greats, no logo
        if greats is not None:
            log.warning("site: greats left out: no %s next to them", LOGO)
    if not similar["available"]:
        log.warning("site: no embeddings; run `scout embed` (the similar panel shows a note)")
    log.info(
        "site: %s (%.0f KB) and %d downloads", page, page.stat().st_size / 1e3, len(written) - 1
    )
    return written


def _similar_inputs(processed: Path) -> tuple:
    """Vectors, today's clubs and the stored model's test numbers, each None
    when missing."""
    from scout.embeddings import OUT, STORED
    from scout.models.store import store_dir

    path = Path(processed) / OUT / "vectors.parquet"
    vectors = pd.read_parquet(path) if path.is_file() else None
    club_path = Path(processed) / "current" / "current_club.parquet"
    current = pd.read_parquet(club_path) if club_path.is_file() else None
    stored = store_dir(processed) / STORED
    tested = json.loads(stored.read_text()).get("tested") if stored.is_file() else None
    return vectors, current, tested


def _greats_inputs(processed: Path) -> tuple:
    """The greats' table, its notes and the logo (``scout greats``), each None
    when missing."""
    from scout.greats import GREATS_OUT

    folder = Path(processed) / GREATS_OUT
    table = folder / "greats.parquet"
    info = folder / "greats.json"
    logo = folder / LOGO
    return (
        pd.read_parquet(table) if table.is_file() else None,
        json.loads(info.read_text()) if info.is_file() else None,
        logo if logo.is_file() else None,
    )
