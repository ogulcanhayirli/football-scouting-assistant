import copy
import json

import httpx
import pandas as pd
import pytest
import respx

from scout.backfill import build_coverage, run_backfill, write_coverage_report
from scout.backfill.runner import parquet_path
from scout.config import Settings
from scout.providers.api_football import ApiFootballClient, ApiFootballStatsProvider
from scout.ratelimit import QuotaGuard
from scout.storage import LocalRawStore

BASE = "https://v3.football.api-sports.io"
PAGES = 3


@pytest.fixture
def settings(tmp_path, epl, super_lig):
    return Settings.model_validate(
        {
            "api_football": {"quota": {"daily_limit": 100, "per_minute_limit": 1000}},
            "leagues": [epl.model_dump(), super_lig.model_dump()],
            "paths": {
                "raw_dir": tmp_path / "raw",
                "processed_dir": tmp_path / "processed",
                "state_dir": tmp_path / "state",
                "reports_dir": tmp_path / "reports",
            },
            "backfill": {"seasons": [2023]},
        }
    )


@pytest.fixture
def fake_api(load_fixture):
    """Serve PAGES pages for 2023, a plan error for 2015, and the leagues list."""
    page = load_fixture("api_football/players_epl_2023_p1.json")
    plan_error = load_fixture("api_football/error_plan.json")
    leagues = load_fixture("api_football/leagues_39.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/leagues":
            return httpx.Response(200, json=leagues)
        params = request.url.params
        if params["season"] != "2023":
            return httpx.Response(200, json=plan_error)
        number = int(params["page"])
        body = copy.deepcopy(page)
        body["paging"] = {"current": number, "total": PAGES}
        for item in body["response"]:  # unique player ids per page
            item["player"]["id"] += number * 10
        return httpx.Response(200, json=body)

    with respx.mock(assert_all_called=False) as router:
        route = router.get(url__startswith=BASE).mock(side_effect=handler)
        yield route


def make_provider(settings, daily_limit=100, clock=None):
    kwargs = {"clock": clock} if clock else {}
    guard = QuotaGuard(
        daily_limit=daily_limit,
        per_minute_limit=1000,
        state_path=settings.paths.state_dir / "quota.json",
        **kwargs,
    )
    client = ApiFootballClient(
        api_key="k",
        store=LocalRawStore(settings.paths.raw_dir),
        guard=guard,
        base_url=BASE,
        sleep=lambda _: None,
    )
    return ApiFootballStatsProvider(client)


def test_backfill_end_to_end_and_idempotent(settings, fake_api):
    result = run_backfill(settings, make_provider(settings), league_keys=["epl"])
    assert result.complete
    assert result.requests_made == PAGES
    frame = pd.read_parquet(parquet_path(settings, "epl", 2023))
    assert frame["source_player_id"].nunique() == 3 * PAGES

    again = run_backfill(settings, make_provider(settings), league_keys=["epl"])
    assert again.requests_made == 0
    assert fake_api.call_count == PAGES


def test_backfill_stops_on_daily_budget_and_resumes(settings, fake_api):
    day = [1_800_000_000.0]
    first = run_backfill(
        settings, make_provider(settings, daily_limit=2, clock=lambda: day[0]), league_keys=["epl"]
    )
    assert first.stopped_reason.startswith("quota")
    assert first.requests_made == 2
    assert not parquet_path(settings, "epl", 2023).exists()

    day[0] += 86_400  # next UTC day
    second = run_backfill(
        settings, make_provider(settings, daily_limit=2, clock=lambda: day[0]), league_keys=["epl"]
    )
    assert second.complete
    assert second.requests_made == 1  # only the missing page
    assert fake_api.call_count == PAGES


def test_plan_refusal_is_recorded_not_cached(settings, fake_api):
    result = run_backfill(settings, make_provider(settings), league_keys=["epl"], seasons=[2015])
    assert [o.status for o in result.outcomes] == ["refused_by_plan"]
    assert result.complete  # a refused season is an expected outcome on the free tier
    assert not (settings.paths.raw_dir / "api_football" / "players").exists()


def test_plan_refusal_stops_run_when_configured(settings, fake_api):
    settings.backfill.stop_on_plan_error = True
    result = run_backfill(
        settings, make_provider(settings), league_keys=["epl"], seasons=[2015, 2023]
    )
    assert result.stopped_reason.startswith("plan")
    assert len(result.outcomes) == 1


def test_all_seasons_skips_current_and_uncovered(settings, fake_api):
    settings.backfill.seasons = "all"
    from scout.backfill.runner import target_seasons

    seasons = target_seasons(make_provider(settings), settings.leagues[0], settings)
    assert seasons == list(range(2012, 2026))


def test_coverage_report(settings, fake_api):
    provider = make_provider(settings)
    result = run_backfill(settings, provider, league_keys=["epl"])
    report = build_coverage(settings, provider.client.store, result)
    (season,) = report["seasons"]
    assert season["status"] == "complete"
    assert season["pages_cached"] == season["pages_total"] == PAGES
    assert season["players"] == 3 * PAGES
    assert set(season["field_completeness_by_position"]) == {"Defender", "Midfielder", "Attacker"}
    json_path, md_path = write_coverage_report(report, settings.paths.reports_dir)
    assert json.loads(json_path.read_text())["seasons"][0]["league"] == "epl"
    assert "| epl | 2023 | complete | 3/3 |" in md_path.read_text()


def test_pages_for_another_competition_are_no_player_data_not_a_failure(settings, fake_api):
    # The fake API serves EPL stats (league 39) to every request, so a Süper Lig
    # request gets pages with no stats for its own league, as Liga Profesional
    # 2023 does with the Copa de la Liga.
    result = run_backfill(settings, make_provider(settings), league_keys=["super_lig"])
    assert [o.status for o in result.outcomes] == ["no_player_data"]
    assert result.complete
    assert not parquet_path(settings, "super_lig", 2023).exists()
