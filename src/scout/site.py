"""The static page (decision 9): one HTML file with the two tables embedded.

``scout site`` reads the user facing tables of ``scout lists``
(``data/processed/lists/``) and writes ``data/site/``:

* ``index.html``: both tables as JSON inside the page, sorted, filtered and
  searched by a little plain JavaScript (``site_template.html``), with a
  "how to read this" box: the chances are calibrated odds from past seasons,
  not predictions beyond the market, the valuation date, the forecast date
  and the five season window;
* ``data/``: the same two tables as CSV and Parquet, and the dictionary.

Only the two tables and the dictionary are published, never raw match stats
or API responses. The rising tag of a bargain is the riser model's top 50 of
``scouting.parquet`` (``scout.lists.rising_ids``), the same set the
candidates table marks; nothing else is read from the model outputs.
"""

from __future__ import annotations

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

# The columns each tab carries; the rest of the tables stay in the downloads.
CANDIDATE_COLUMNS = [
    "player",
    "age",
    "position",
    "club",
    "league",
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
]
BARGAIN_COLUMNS = [
    "list",
    "list_rank",
    "player",
    "age",
    "position",
    "club",
    "league",
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
        "leagues": int(cand["league"].nunique()),
        "tracked": tracked,
        "candidates": _records(cand, [*CANDIDATE_COLUMNS, "flag_keys"]),
        "bargains": _records(barg, [*BARGAIN_COLUMNS, "flag_keys"]),
        "dictionary": _records(words, ["table", "column", "meaning"]),
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
    log.info(
        "site: %s (%.0f KB) and %d downloads", page, page.stat().st_size / 1e3, len(written) - 1
    )
    return written
