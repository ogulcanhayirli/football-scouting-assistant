"""``scout update`` (the free tier update) and the S3 backed raw store, on fakes."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from scout.config import load_settings
from scout.providers.api_football.client import FetchResult
from scout.storage import CacheKey, LocalRawStore, RawMeta, S3BackedRawStore
from scout.update import ordered_competitions, run_update, window

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 11, 9)


def match(fid, league_id, day="2026-11-08", status="FT"):
    return {
        "fixture": {
            "id": fid,
            "date": f"{day}T15:00:00+00:00",
            "timestamp": fid,
            "status": {"short": status},
        },
        "league": {"id": league_id, "season": 2026, "round": "Regular Season - 11"},
        "teams": {"home": {"id": 1, "name": "Home"}, "away": {"id": 2, "name": "Away"}},
        "goals": {"home": 1, "away": 0},
        "players": [
            {
                "team": {"id": 1, "name": "Home"},
                "players": [
                    {
                        "player": {"id": fid * 10, "name": f"P {fid}"},
                        "statistics": [
                            {"games": {"minutes": 90, "position": "M"}, "goals": {"total": 1}}
                        ],
                    }
                ],
            }
        ],
        "lineups": [],
    }


class FakeGuard:
    remaining_today = 100


class FakeClient:
    """Serves /fixtures?date= and /fixtures?ids= from a dict of matches by day."""

    def __init__(self, days: dict[str, list[dict]], store):
        self.days = days
        self.by_id = {m["fixture"]["id"]: m for ms in days.values() for m in ms}
        self.store = store
        self.guard = FakeGuard()
        self.requests_made = 0
        self.asked: list[dict] = []

    def is_cached(self, endpoint, params, snapshot=None):
        return self.store.exists(CacheKey.of("api_football", endpoint, params))

    def get(self, endpoint, params, snapshot=None):
        key = CacheKey.of("api_football", endpoint, params)
        if self.store.exists(key):
            return FetchResult(json.loads(self.store.get(key)), True, key)
        self.requests_made += 1
        self.asked.append(params)
        if "date" in params:
            payload = {"response": self.days.get(params["date"], [])}
        else:
            payload = {"response": [self.by_id[int(i)] for i in params["ids"].split("-")]}
        self.store.put(key, json.dumps(payload).encode(), RawMeta("u", {}, "t", 200, {}))
        return FetchResult(payload, False, key)


@pytest.fixture
def settings(tmp_path):
    s = load_settings(ROOT / "configs" / "update.free.yaml")
    paths = {k: tmp_path / k for k in ("raw_dir", "processed_dir", "state_dir", "reports_dir")}
    return s.model_copy(update={"paths": s.paths.model_copy(update=paths)})


def test_free_config_inherits_the_leagues_and_sets_the_free_quota(settings):
    ultra = load_settings(ROOT / "configs" / "backfill.ultra.yaml")
    assert [lg.key for lg in settings.leagues] == [lg.key for lg in ultra.leagues]
    assert settings.api_football.quota.daily_limit == 100
    assert settings.update.max_requests < settings.api_football.quota.daily_limit
    order = [c.key for c in ordered_competitions(settings)]
    assert order[:2] == ["champions_league", "epl"]
    assert set(order) == {c.key for c in [*ultra.leagues, *ultra.competitions]}


def test_window_is_yesterday_back():
    assert window(TODAY, 3) == [date(2026, 11, 8), date(2026, 11, 7), date(2026, 11, 6)]


def test_update_fetches_by_priority_within_budget_and_queues_the_rest(settings):
    settings = settings.model_copy(
        update={"update": settings.update.model_copy(update={"days": 2, "max_requests": 3})}
    )
    # 25 Eerste Divisie matches (last in line) and 25 Premier League ones (first).
    days = {
        "2026-11-08": [match(5000 + i, 89) for i in range(25)]
        + [match(1000 + i, 39) for i in range(25)]
        + [match(9999, 39, status="PST"), match(8888, 999)],  # postponed, not our league
        "2026-11-07": [],
    }
    client = FakeClient(days, LocalRawStore(settings.paths.raw_dir))
    result = run_update(settings, client, TODAY)
    assert result.requests_made == 3  # two days and one batch of 20
    assert result.in_window == 50
    assert result.fetched == 20
    assert result.queued == 30
    assert result.stopped_reason == "budget"
    first_batch = next(p for p in client.asked if "ids" in p)["ids"].split("-")
    assert {int(i) for i in first_batch} == set(range(1000, 1020))  # Premier League first
    part = (
        settings.paths.processed_dir
        / "current"
        / "player_match_stats"
        / "league=epl"
        / "season=2026.parquet"
    )
    assert len(pd.read_parquet(part)) == 20

    # The next run: the days are cached, the queue goes on, old rows stay.
    settings = settings.model_copy(
        update={"update": settings.update.model_copy(update={"max_requests": 10})}
    )
    client2 = FakeClient(days, client.store)
    again = run_update(settings, client2, TODAY)
    assert again.requests_made == 2  # 30 left: the 5 Premier League ones first, in two batches
    assert again.queued == 0
    assert len(pd.read_parquet(part)) == 25
    assert again.per_league["epl"] == {"finished": 25, "held": 25}
    assert again.per_league["eerste_divisie"] == {"finished": 25, "held": 25}


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()


def test_s3_store_copies_down_and_uploads(tmp_path):
    s3 = FakeS3()
    key = CacheKey.of("api_football", "fixtures", {"date": "2026-11-08"})
    s3.objects["raw/api_football/fixtures/date=2026-11-08.json"] = b'{"response": []}'
    s3.objects["raw/api_football/fixtures/date=2026-11-08.meta.json"] = b"{}"
    store = S3BackedRawStore(LocalRawStore(tmp_path), s3, "bucket")
    assert store.exists(key)  # copied down, not asked again
    assert store.get(key) == b'{"response": []}'

    new = CacheKey.of("api_football", "fixtures", {"date": "2026-11-07"})
    assert not store.exists(new)
    store.put(new, b'{"response": [1]}', RawMeta("u", {}, "t", 200, {}))
    assert s3.objects["raw/api_football/fixtures/date=2026-11-07.json"] == b'{"response": [1]}'
    assert store.uploaded[-1].endswith("date=2026-11-07.json")  # body last
