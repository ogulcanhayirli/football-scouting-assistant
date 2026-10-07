"""Coverage report built straight from the raw cache.

Reading the cache instead of the Parquet output means partial seasons show
up too, which is what you want when deciding whether to resume a run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scout.backfill.runner import BackfillResult
from scout.config import Settings
from scout.fsutil import atomic_write_text
from scout.providers.api_football.adapter import (
    NUMERIC_STATS,
    parse_players_page,
    rows_to_frame,
)
from scout.providers.api_football.client import SOURCE
from scout.storage import RawStore

MIN_MINUTES = 450  # five full matches; below this per 90 rates are mostly noise
REPORTED_FIELDS = ["rating", *NUMERIC_STATS]


def _completeness(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    return {c: round(float(frame[c].notna().mean()), 3) for c in REPORTED_FIELDS}


def build_coverage(
    settings: Settings, store: RawStore, result: BackfillResult | None = None
) -> dict[str, Any]:
    by_id = {lg.api_football_id: lg for lg in settings.leagues}
    pages: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for key in store.keys(SOURCE, "players"):
        params = dict(key.params)
        league = by_id.get(int(params.get("league", -1)))
        if league is None:
            continue
        pages[(league.key, int(params["season"]))].append(json.loads(store.get(key)))

    statuses = {(o.league_key, o.season): o.status for o in (result.outcomes if result else [])}
    seasons = []
    for (league_key, season), payloads in sorted(pages.items()):
        league = settings.league(league_key)
        rows = [r for p in payloads for r in parse_players_page(p, league, season)]
        frame = rows_to_frame(rows)
        played = frame[frame["minutes"].fillna(0) > 0]
        total = max(int((p.get("paging") or {}).get("total") or 1) for p in payloads)
        seasons.append(
            {
                "league": league_key,
                "season": season,
                "status": statuses.get((league_key, season))
                or ("complete" if len(payloads) >= total else "partial"),
                "pages_cached": len(payloads),
                "pages_total": total,
                "rows": len(frame),
                "players": int(frame["source_player_id"].nunique()),
                "players_with_minutes": int(played["source_player_id"].nunique()),
                f"players_min_{MIN_MINUTES}": int(
                    played.loc[played["minutes"] >= MIN_MINUTES, "source_player_id"].nunique()
                ),
                "field_completeness": _completeness(played),
                "field_completeness_by_position": {
                    str(pos): _completeness(group)
                    for pos, group in played.groupby("position", dropna=False)
                },
            }
        )
    for (league_key, season), status in statuses.items():
        if (league_key, season) not in pages:
            seasons.append({"league": league_key, "season": season, "status": status})

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": settings.api_football.plan,
        "requests_made_this_run": result.requests_made if result else None,
        "stopped_reason": result.stopped_reason if result else None,
        "min_minutes": MIN_MINUTES,
        "seasons": sorted(seasons, key=lambda s: (s["league"], s["season"])),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Backfill coverage report",
        "",
        f"Generated {report['generated_at']} on the `{report['plan']}` plan.",
        f"Requests made this run: {report['requests_made_this_run']}. "
        f"Stopped: {report['stopped_reason'] or 'no'}.",
        "",
        "| League | Season | Status | Pages | Rows | Players | Played | "
        f"{report['min_minutes']}+ min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["seasons"]:
        pages = f"{s.get('pages_cached', 0)}/{s.get('pages_total', '?')}"
        lines.append(
            f"| {s['league']} | {s['season']} | {s['status']} | {pages} | {s.get('rows', 0)} | "
            f"{s.get('players', 0)} | {s.get('players_with_minutes', 0)} | "
            f"{s.get(f'players_min_{report["min_minutes"]}', 0)} |"
        )
    for s in report["seasons"]:
        comp = s.get("field_completeness_by_position")
        if not comp:
            continue
        positions = sorted(comp)
        lines += [
            "",
            f"## {s['league']} {s['season']}: share of non null values (players with minutes)",
            "",
            "| Field | All | " + " | ".join(positions) + " |",
            "|---|---|" + "---|" * len(positions),
        ]
        for field in REPORTED_FIELDS:
            cells = [f"{comp[p].get(field, 0):.0%}" for p in positions]
            overall = s["field_completeness"].get(field, 0)
            lines.append(f"| {field} | {overall:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_coverage_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir = Path(reports_dir)
    json_path = reports_dir / "coverage_report.json"
    md_path = reports_dir / "coverage_report.md"
    atomic_write_text(json_path, json.dumps(report, indent=2, ensure_ascii=False))
    atomic_write_text(md_path, _markdown(report))
    return json_path, md_path
