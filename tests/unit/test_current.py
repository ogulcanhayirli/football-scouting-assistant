"""``run_current`` on fakes: the season in progress of leagues and competitions,
lineups, injuries and the season pages (6 November 2026 additions)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scout.backfill.current import run_current
from scout.config import Settings
from scout.providers.api_football.adapter import SeasonInfo

ROOT = Path(__file__).resolve().parents[2]


class FakeGuard:
    remaining_today = 1000


class FakeStore:
    def keys(self, source, endpoint):
        return []


class FakeClient:
    def __init__(self):
        self.requests_made = 0
        self.guard = FakeGuard()
        self.store = FakeStore()


class FakeSeasons:
    def list_seasons(self, league):
        return [SeasonInfo(league.key, 2026, None, None, True, True, True)]


def match(fid, league_id):
    return {
        "fixture": {"id": fid, "date": "2026-10-03T14:00:00+00:00", "status": {"short": "FT"}},
        "league": {"id": league_id, "season": 2026, "round": "Regular Season - 7"},
        "teams": {"home": {"id": 1, "name": "Home"}, "away": {"id": 2, "name": "Away"}},
        "goals": {"home": 1, "away": 0},
        "players": [],
        "lineups": [
            {
                "team": {"id": 1, "name": "Home"},
                "coach": {"id": 901, "name": "Coach"},
                "formation": "4-4-2",
                "startXI": [
                    {"player": {"id": 10, "name": "A", "number": 9, "pos": "F", "grid": "4:1"}}
                ],
                "substitutes": [],
            }
        ],
    }


class FakeProvider:
    as_of = "2026-11-06"

    def __init__(self, page_payload):
        self.client = FakeClient()
        self.calls: list[tuple] = []
        self.page_payload = page_payload

    def _hit(self, *call):
        self.calls.append(call)
        self.client.requests_made += 1

    def teams(self, league, season):
        self._hit("teams", league.key)
        return {"response": []}

    def fixtures(self, league, season):
        self._hit("fixtures", league.key)
        return {"response": [match(1000 + league.api_football_id, league.api_football_id)]}

    def fixtures_by_id(self, ids):
        self._hit("fixtures_by_id", tuple(ids))
        return {"response": [match(i, i - 1000) for i in ids]}

    def injuries(self, league, season):
        self._hit("injuries", league.key)
        return {"response": []}

    def player_page(self, league, season, page):
        self._hit("page", league.key, page)
        if page == 1:
            return self.page_payload
        return {"paging": self.page_payload["paging"], "response": []}


@pytest.fixture
def settings(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "backfill.ultra.yaml").read_text(encoding="utf-8"))
    raw["leagues"] = [lg for lg in raw["leagues"] if lg["key"] == "epl"]
    raw["competitions"] = [c for c in raw["competitions"] if c["key"] == "champions_league"]
    raw["paths"] = {
        k: str(tmp_path / k) for k in ("raw_dir", "processed_dir", "state_dir", "reports_dir")
    }
    return Settings.model_validate(raw)


def test_season_in_progress_covers_leagues_and_competitions(settings, load_fixture):
    page = load_fixture("api_football/players_epl_2023_p1.json")  # paging total 2
    provider = FakeProvider(page)
    result = run_current(
        settings, FakeSeasons(), provider, steps=("clubs", "matches", "injuries", "pages")
    )
    assert result.complete, result.failed
    kinds = {(c[0], c[1]) for c in provider.calls if c[0] in ("fixtures", "injuries")}
    assert kinds == {
        ("fixtures", "epl"),
        ("fixtures", "champions_league"),
        ("injuries", "epl"),
        ("injuries", "champions_league"),
    }
    pages = [c for c in provider.calls if c[0] == "page"]
    assert pages == [
        ("page", "epl", 1),
        ("page", "epl", 2),
        ("page", "champions_league", 1),
        ("page", "champions_league", 2),
    ]
    out = Path(settings.paths.processed_dir) / "current"
    lineups = pd.read_parquet(
        out / "match_lineups" / "league=champions_league" / "season=2026.parquet"
    )
    assert lineups["source_player_id"].tolist() == [10]
    assert (out / "player_season_stats" / "league=epl" / "season=2026.parquet").exists()
    # Competitions' season pages are context: cached raw, no table.
    assert not (out / "player_season_stats" / "league=champions_league").exists()
