"""Probe the competitions of owner decision 6 before the full download.

Usage:
    uv run python scripts/probe_expansion.py --config configs/backfill.ultra.yaml

For each competition: ``/leagues?id=`` confirms the id and lists the seasons
with their coverage flags; per season from ``--first-season`` on, one
``/fixtures`` request counts the matches, one ``/players`` page 1 gives the
page count, and one finished fixture is probed on ``/fixtures/players`` and
``/fixtures/lineups``. Every request goes through the cached client, so the
probe responses are reused by the later backfill and a rerun costs nothing.

Output: ``configs/expansion_probe.csv`` (one row per competition and season)
and ``configs/expansion_probe.md`` (the per competition request estimate).
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scout.cli import build_client
from scout.config import load_settings
from scout.fsutil import atomic_write_text
from scout.providers.api_football import ApiFootballClient, ApiFootballError
from scout.providers.api_football.matches import finished_fixtures

log = logging.getLogger("probe_expansion")

# (key, expected name, expected country, API Football id). Ids were looked up
# beforehand; the probe checks name and country against /leagues and stops on a mismatch.
COMPETITIONS: list[tuple[str, str, str, int]] = [
    ("austrian_bundesliga", "Bundesliga", "Austria", 218),
    ("swiss_super_league", "Super League", "Switzerland", 207),
    ("croatian_hnl", "HNL", "Croatia", 210),
    ("serbian_superliga", "Super Liga", "Serbia", 286),
    ("czech_first_league", "Czech Liga", "Czech-Republic", 345),
    ("norwegian_eliteserien", "Eliteserien", "Norway", 103),
    ("swedish_allsvenskan", "Allsvenskan", "Sweden", 113),
    ("polish_ekstraklasa", "Ekstraklasa", "Poland", 106),
    ("liga_mx", "Liga MX", "Mexico", 262),
    ("mls", "Major League Soccer", "USA", 253),
    ("colombian_primera_a", "Primera A", "Colombia", 239),
    ("uruguayan_primera", "Primera División", "Uruguay", 268),
    ("j1_league", "J1 League", "Japan", 98),
    ("k_league_1", "K League 1", "South-Korea", 292),
    ("saudi_pro_league", "Pro League", "Saudi-Arabia", 307),
    ("bundesliga_2", "2. Bundesliga", "Germany", 79),
    ("segunda_division", "Segunda División", "Spain", 141),
    ("serie_b", "Serie B", "Italy", 136),
    ("ligue_2", "Ligue 2", "France", 62),
    ("eerste_divisie", "Eerste Divisie", "Netherlands", 89),
    ("uefa_youth_league", "UEFA Youth League", "World", 14),
    ("uefa_u21_championship", "UEFA U21 Championship", "World", 38),
    ("fifa_u20_world_cup", "World Cup - U20", "World", 490),
    ("fifa_u17_world_cup", "World Cup - U17", "World", 587),
    ("olympics_men", "Olympics Men", "World", 480),
]


@dataclass
class SeasonProbe:
    key: str
    league_id: int
    season: int
    current: bool
    flag_player_stats: bool
    flag_lineups: bool
    fixtures: int
    finished: int
    probe_fixture: int | None
    fixture_players_rows: int | None
    lineups_teams: int | None
    players_pages: int | None
    error: str = ""


def _get(client: ApiFootballClient, endpoint: str, params: dict[str, Any]) -> dict | None:
    try:
        return client.get(endpoint, params).payload
    except ApiFootballError as exc:
        log.warning("%s %s: %s", endpoint, params, exc)
        return None


def probe_season(client, key: str, league_id: int, s: dict) -> SeasonProbe:
    season = int(s["year"])
    cov = (s.get("coverage") or {}).get("fixtures") or {}
    row = SeasonProbe(
        key, league_id, season, bool(s.get("current")),
        bool(cov.get("statistics_players")), bool(cov.get("lineups")),
        0, 0, None, None, None, None,
    )  # fmt: skip
    fixtures = _get(client, "fixtures", {"league": league_id, "season": season})
    if fixtures is None:
        row.error = "fixtures refused"
        return row
    row.fixtures = len(fixtures.get("response") or [])
    done = sorted(
        finished_fixtures(fixtures), key=lambda f: (f.get("fixture") or {}).get("timestamp") or 0
    )
    row.finished = len(done)
    players = _get(client, "players", {"league": league_id, "season": season, "page": 1})
    if players is not None:
        row.players_pages = int((players.get("paging") or {}).get("total") or 0)
        if not players.get("response"):
            row.players_pages = 0
    if done:
        # The middle finished match: not an opening round fixture added late, not one
        # whose stats might still be filled in.
        fid = int(done[len(done) // 2]["fixture"]["id"])
        row.probe_fixture = fid
        fp = _get(client, "fixtures/players", {"fixture": fid})
        if fp is not None:
            row.fixture_players_rows = sum(
                len(t.get("players") or []) for t in fp.get("response") or []
            )
        lu = _get(client, "fixtures/lineups", {"fixture": fid})
        if lu is not None:
            row.lineups_teams = sum(1 for t in lu.get("response") or [] if t.get("startXI"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/backfill.ultra.yaml")
    ap.add_argument("--first-season", type=int, default=2010)
    ap.add_argument("--out", default="configs/expansion_probe")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings(args.config)

    rows: list[SeasonProbe] = []
    names: dict[str, str] = {}
    with build_client(settings) as client:
        start = client.requests_made
        for key, name, country, league_id in COMPETITIONS:
            payload = _get(client, "leagues", {"id": league_id})
            items = (payload or {}).get("response") or []
            if not items:
                raise SystemExit(f"{key}: /leagues?id={league_id} returned nothing")
            got_name, got_country = items[0]["league"]["name"], items[0]["country"]["name"]
            if (got_name, got_country) != (name, country):
                raise SystemExit(f"{key}: id {league_id} is {got_name} ({got_country})")
            names[key] = f"{got_name} ({got_country})"
            for s in items[0].get("seasons") or []:
                if int(s["year"]) >= args.first_season:
                    rows.append(probe_season(client, key, league_id, s))
            log.info("%s done, %d new requests so far", key, client.requests_made - start)
        spent = client.requests_made - start

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    atomic_write_text(out.with_suffix(".md"), render(rows, names, spent, args.first_season))
    log.info("probe spent %d new requests", spent)
    return 0


def _span(seasons: list[int]) -> str:
    return f"{min(seasons)} to {max(seasons)} ({len(seasons)})" if seasons else "none"


def render(rows: list[SeasonProbe], names: dict[str, str], spent: int, first: int) -> str:
    """Per competition estimate. Fixture players and lineups are counted only for the
    seasons whose probe fixture returned data; the current season counts all its
    fixtures, since the final pull of 6 November will fetch the ones played by then."""
    head = (
        "| Competition | Id | Seasons | Fixture player stats | Lineups "
        "| Fixtures | Fixture players | Lineups req | Players pages | Total |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|\n"
    )
    lines, totals = [], [0, 0, 0, 0, 0]
    for key, name in names.items():
        rs = [r for r in rows if r.key == key]
        fp = [r for r in rs if r.fixture_players_rows]
        lu = [r for r in rs if r.lineups_teams]
        n_fix = len(rs)
        n_fp = sum(r.fixtures for r in fp)
        n_lu = sum(r.fixtures for r in lu)
        n_pp = sum(r.players_pages or 0 for r in rs)
        tot = n_fix + n_fp + n_lu + n_pp
        for i, v in enumerate((n_fix, n_fp, n_lu, n_pp, tot)):
            totals[i] += v
        lines.append(
            f"| {name} | {rs[0].league_id if rs else ''} | {_span([r.season for r in rs])} "
            f"| {_span([r.season for r in fp])} | {_span([r.season for r in lu])} "
            f"| {n_fix:,} | {n_fp:,} | {n_lu:,} | {n_pp:,} | {tot:,} |"
        )
    lines.append(
        f"| **Total** | | | | | {totals[0]:,} | {totals[1]:,} | {totals[2]:,} "
        f"| {totals[3]:,} | **{totals[4]:,}** |"
    )
    return (
        "# Expansion probe (owner decision 6)\n\n"
        f"Seasons from {first} on, as listed by `/leagues`. Generated by "
        "`scripts/probe_expansion.py`; per season detail in `expansion_probe.csv`.\n\n"
        "Requests: *Fixtures* is one `/fixtures` call per season; *Fixture players* and "
        "*Lineups req* are one call per fixture, counted only for seasons whose probe "
        "fixture (the middle finished match) returned data; *Players pages* is the "
        "`/players` page count per season. Probe requests already made are cached and "
        "are not subtracted, so the totals slightly overstate what remains.\n\n"
        f"{head}" + "\n".join(lines) + f"\n\nThe probe itself spent {spent:,} new requests.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
