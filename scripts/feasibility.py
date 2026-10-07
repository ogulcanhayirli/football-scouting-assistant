"""Feasibility check on the free tier, spending as few requests as possible.

Usage:
    uv run python scripts/feasibility.py --config configs/backfill.free.yaml --max-requests 40

Every response goes through the cached client, so rerunning this script costs
nothing for steps that already succeeded. Output:

* reports/feasibility.json            all numbers, machine readable
* reports/feasibility_tables.md       the tables FEASIBILITY.md is written from
* reports/match_review_<league>.csv   matched pairs to check by hand (precision)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from scout.cli import build_client
from scout.config import Settings, load_settings
from scout.feasibility import (
    estimate_requests,
    field_completeness,
    valuation_horizon_coverage,
)
from scout.fsutil import atomic_write_text
from scout.matching import match_players
from scout.providers.api_football import (
    ApiFootballClient,
    ApiFootballError,
    parse_players_page,
)
from scout.providers.api_football.adapter import parse_seasons, rows_to_frame
from scout.providers.transfermarkt import TransfermarktValuationProvider, download_snapshot

log = logging.getLogger("feasibility")
TM_DATA_END = pd.Timestamp("2026-06-12")  # last valuation date stated in the dataset README
REVIEW_SAMPLE = 50


class Budget:
    """Stops the script from spending more than ``limit`` new requests."""

    def __init__(self, client: ApiFootballClient, limit: int) -> None:
        self.client, self.limit, self.start = client, limit, client.requests_made

    @property
    def spent(self) -> int:
        return self.client.requests_made - self.start

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.client.is_cached(endpoint, params) and self.spent >= self.limit:
            raise RuntimeError(f"feasibility budget of {self.limit} requests reached")
        return self.client.get(endpoint, params).payload


def probe(budget: Budget, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch and report ok or the API's own error message, instead of raising."""
    try:
        payload = budget.get(endpoint, params)
        return {"ok": True, "payload": payload}
    except ApiFootballError as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def step_seasons(budget: Budget, settings: Settings, findings: dict) -> dict[str, list[int]]:
    """(a) seasons listed per league, and which ones the free plan actually serves."""
    accessible: dict[str, list[int]] = {}
    status = probe(budget, "status", {})
    findings["status"] = status.get("payload", {}).get("response") or status
    for league in settings.leagues:
        res = probe(budget, "leagues", {"id": league.api_football_id})
        if not res["ok"]:
            findings.setdefault("seasons", {})[league.key] = res
            continue
        seasons = parse_seasons(res["payload"], league)
        covered = [s.season for s in seasons if s.has_player_stats]
        finished = [s.season for s in seasons if s.has_player_stats and not s.current]
        probes = {}
        # Newest finished season first, then step back until refused, then the oldest.
        to_try = sorted(set(finished[-4:] + finished[:1]), reverse=True)
        for season in to_try:
            p = probe(
                budget, "players", {"league": league.api_football_id, "season": season, "page": 1}
            )
            probes[season] = (
                {"ok": True} if p["ok"] else {k: p[k] for k in ("ok", "error", "message")}
            )
        accessible[league.key] = [s for s, p in probes.items() if p["ok"]]
        findings.setdefault("seasons", {})[league.key] = {
            "listed": [s.season for s in seasons],
            "with_player_coverage": covered,
            "current": [s.season for s in seasons if s.current],
            "free_tier_probes": probes,
        }
    return accessible


def step_fields(budget: Budget, settings: Settings, accessible: dict, findings: dict):
    """(b) stat fields returned and their completeness, from up to three pages per league."""
    frames, pages_known = [], {}
    for league in settings.leagues:
        if not accessible.get(league.key):
            continue
        season = max(accessible[league.key])
        params = {"league": league.api_football_id, "season": season}
        first = budget.get("players", params | {"page": 1})
        total = int(first["paging"]["total"])
        pages_known[(league.key, season)] = total
        rows = parse_players_page(first, league, season)
        for page in sorted({(total + 1) // 2, total} - {1}):
            try:
                rows += parse_players_page(
                    budget.get("players", params | {"page": page}), league, season
                )
            except (ApiFootballError, RuntimeError) as exc:
                findings.setdefault("field_errors", []).append(f"{league.key} p{page}: {exc}")
        frames.append(rows_to_frame(rows))
        raw_item = (first.get("response") or [{}])[0]
        findings.setdefault("raw_field_names", {})[league.key] = {
            group: sorted(values) if isinstance(values, dict) else values
            for group, values in ((raw_item.get("statistics") or [{}])[0]).items()
        }
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not stats.empty:
        findings["field_completeness"] = field_completeness(stats).to_dict(orient="records")
    # Every season we probed successfully also tells us its page count for free.
    for league in settings.leagues:
        for season in accessible.get(league.key, []):
            params = {"league": league.api_football_id, "season": season, "page": 1}
            if budget.client.is_cached("players", params):
                pages_known[(league.key, season)] = int(
                    budget.get("players", params)["paging"]["total"]
                )
    return stats, pages_known


def step_fixtures(budget: Budget, settings: Settings, accessible: dict, findings: dict):
    """Fixture count per season, needed for the match level estimate."""
    counts = {}
    for league in settings.leagues:
        if not accessible.get(league.key):
            continue
        season = max(accessible[league.key])
        res = probe(budget, "fixtures", {"league": league.api_football_id, "season": season})
        if res["ok"]:
            counts[league.key] = len(res["payload"].get("response") or [])
    findings["fixtures_per_season"] = counts
    return counts


def step_transfermarkt(settings: Settings, stats: pd.DataFrame, findings: dict, download: bool):
    """(c) valuation depth and (d) player matching."""
    tm_dir = settings.paths.raw_dir / "transfermarkt"
    if download:
        download_snapshot(settings.transfermarkt, settings.paths.raw_dir)
    tm = TransfermarktValuationProvider(tm_dir)
    apps = tm.league_appearances(settings.leagues)
    vals = tm.valuations(settings.leagues)
    players = tm.players(settings.leagues)

    depth = valuation_horizon_coverage(apps, vals, data_end=TM_DATA_END)
    findings["valuation_depth"] = depth.to_dict(orient="records")
    findings["valuation_date_range"] = [
        str(vals["valuation_date"].min()),
        str(vals["valuation_date"].max()),
    ]

    if stats.empty:
        return
    clubs = pd.read_csv(tm_dir / "clubs.csv.gz", usecols=["club_id", "name"])
    club_names = (
        apps.merge(clubs, left_on="club_id", right_on="club_id", how="left")
        .groupby("source_player_id")["name"]
        .agg(lambda s: " ".join(sorted(set(map(str, s.dropna())))))
        .rename("club_names")
    )
    findings["matching"] = {}
    for league in settings.leagues:
        league_stats = stats[stats["league_key"] == league.key].drop_duplicates("source_player_id")
        if league_stats.empty:
            continue
        ids = apps.loc[apps["league_key"] == league.key, "source_player_id"].unique()
        candidates = players[players["source_player_id"].isin(ids)].join(
            club_names, on="source_player_id"
        )
        result = match_players(league_stats, candidates)
        reasons = result["reason"].value_counts().to_dict()
        findings["matching"][league.key] = {
            "sample_players": len(result),
            "matched": int(reasons.get("matched", 0)),
            "match_rate": round(reasons.get("matched", 0) / len(result), 3),
            "reasons": {k: int(v) for k, v in reasons.items()},
        }
        review = (
            result[result["reason"] == "matched"]
            .sample(n=min(REVIEW_SAMPLE, int(reasons.get("matched", 0))), random_state=7)
            .merge(
                league_stats[
                    [
                        "source_player_id",
                        "player_name",
                        "firstname",
                        "lastname",
                        "birth_date",
                        "team_name",
                    ]
                ],
                left_on="af_player_id",
                right_on="source_player_id",
            )
            .merge(
                candidates[["source_player_id", "player_name", "birth_date", "club_names"]],
                left_on="tm_player_id",
                right_on="source_player_id",
                suffixes=("_af", "_tm"),
            )
            .drop(columns=["source_player_id_af", "source_player_id_tm"])
        )
        review["correct"] = ""  # filled in by hand
        review.to_csv(settings.paths.reports_dir / f"match_review_{league.key}.csv", index=False)
        unmatched = result[result["reason"] != "matched"].merge(
            league_stats[["source_player_id", "player_name", "birth_date", "team_name"]],
            left_on="af_player_id",
            right_on="source_player_id",
        )
        unmatched.to_csv(
            settings.paths.reports_dir / f"match_unmatched_{league.key}.csv", index=False
        )


def step_estimate(settings: Settings, findings: dict, pages_known: dict, fixtures: dict):
    """(e) requests for the full backfill, season level and match level."""
    seasons = {
        lg: info.get("with_player_coverage", [])
        for lg, info in findings.get("seasons", {}).items()
        if isinstance(info, dict) and "with_player_coverage" in info
    }
    finished = {
        lg: [s for s in ss if s not in findings["seasons"][lg].get("current", [])]
        for lg, ss in seasons.items()
    }
    table = estimate_requests(pages_known, finished, fixtures)
    findings["request_estimate"] = table.to_dict(orient="records")
    findings["pages_measured"] = {f"{k[0]} {k[1]}": v for k, v in pages_known.items()}


def write_tables(findings: dict, path: Path) -> None:
    parts = ["# Feasibility tables (generated)", ""]
    for key in ["field_completeness", "valuation_depth", "request_estimate"]:
        if findings.get(key):
            parts += [f"## {key}", "", pd.DataFrame(findings[key]).to_markdown(index=False), ""]
    parts += ["## other findings", "", "```json"]
    other = {
        k: v
        for k, v in findings.items()
        if k not in {"field_completeness", "valuation_depth", "request_estimate"}
    }
    parts += [json.dumps(other, indent=2, default=str, ensure_ascii=False), "```", ""]
    atomic_write_text(path, "\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/backfill.free.yaml"))
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument(
        "--no-download", action="store_true", help="Use transfermarkt files already on disk"
    )
    parser.add_argument("--skip-transfermarkt", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings(args.config)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    findings: dict[str, Any] = {}

    with build_client(settings) as client:
        budget = Budget(client, args.max_requests)
        accessible = step_seasons(budget, settings, findings)
        stats, pages_known = step_fields(budget, settings, accessible, findings)
        fixtures = step_fixtures(budget, settings, accessible, findings)
        findings["requests_spent"] = budget.spent
        findings["quota_left_today"] = client.guard.remaining_today

    if not args.skip_transfermarkt:
        step_transfermarkt(settings, stats, findings, download=not args.no_download)
    step_estimate(settings, findings, pages_known, fixtures)

    out = settings.paths.reports_dir
    atomic_write_text(
        out / "feasibility.json", json.dumps(findings, indent=2, default=str, ensure_ascii=False)
    )
    write_tables(findings, out / "feasibility_tables.md")
    log.info("Spent %d requests. Wrote %s", findings["requests_spent"], out / "feasibility.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
