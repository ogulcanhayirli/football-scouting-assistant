import httpx
import pytest
import respx

from scout.providers.api_football import (
    ApiFootballClient,
    ApiFootballStatsProvider,
    AuthError,
    PlanRestrictionError,
    QuotaExceededError,
    parse_players_page,
)
from scout.providers.api_football.adapter import COLUMNS, parse_seasons, rows_to_frame
from scout.providers.api_football.client import classify_errors
from scout.ratelimit import QuotaGuard
from scout.storage import LocalRawStore

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-ratelimit-requests-remaining": "90", "x-ratelimit-remaining": "9"}


@pytest.fixture
def client(tmp_path):
    guard = QuotaGuard(daily_limit=100, per_minute_limit=10, state_path=tmp_path / "q.json")
    c = ApiFootballClient(
        api_key="test-key",
        store=LocalRawStore(tmp_path / "raw"),
        guard=guard,
        base_url=BASE,
        sleep=lambda _: None,
    )
    yield c
    c.close()


# Parsing


def test_parse_page_keeps_only_requested_league(load_fixture, epl):
    rows = parse_players_page(load_fixture("api_football/players_epl_2023_p1.json"), epl, 2023)
    # Player 2 has two league entries (mid season move) and one cup entry.
    assert [r["source_player_id"] for r in rows] == [900001, 900002, 900002, 900003]
    assert {r["league_id"] for r in rows} == {39}


def test_parse_page_maps_fields_and_fixes_source_typos(load_fixture, epl):
    rows = parse_players_page(load_fixture("api_football/players_epl_2023_p1.json"), epl, 2023)
    first = rows[0]
    assert first["appearances"] == 20  # source key "appearences"
    assert first["tackles_total"] == 45
    assert first["interceptions"] == 31
    assert first["height_cm"] == 181.0
    assert str(first["birth_date"]) == "2001-03-14"
    assert "penalty_committed" in first  # source key "commited"


def test_parse_keeps_turkish_characters(load_fixture, super_lig):
    payload = load_fixture("api_football/players_super_lig_2023_p1.json")
    names = [r["player_name"] for r in parse_players_page(payload, super_lig, 2023)]
    assert names == ["Ç. Şahinoğlu", "İ. Yılmazer"]


def test_parse_maps_forward_to_attacker(load_fixture, epl):
    payload = load_fixture("api_football/players_epl_2023_p1.json")
    for item in payload["response"]:
        for stat in item["statistics"]:
            if stat["games"]["position"] == "Attacker":
                stat["games"]["position"] = "Forward"
    positions = {r["position"] for r in parse_players_page(payload, epl, 2023)}
    assert "Forward" not in positions
    assert "Attacker" in positions


def test_rows_to_frame_types(load_fixture, epl):
    rows = parse_players_page(load_fixture("api_football/players_epl_2023_p1.json"), epl, 2023)
    frame = rows_to_frame(rows)
    assert list(frame.columns) == COLUMNS
    assert str(frame["minutes"].dtype) == "Float64"
    assert str(frame["rating"].dtype) == "Float64"  # "7.012" strings become numbers
    assert frame["rating"].isna().sum() == 1
    assert frame["height_cm"].isna().sum() == 1


def test_parse_seasons(load_fixture, epl):
    seasons = parse_seasons(load_fixture("api_football/leagues_39.json"), epl)
    assert seasons[0].season == 2010
    assert seasons[-1].current
    assert [s.season for s in seasons if not s.has_player_stats] == [2010, 2011]


# Error classification


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("error_plan", PlanRestrictionError),
        ("error_quota", QuotaExceededError),
        ("error_rate_limit", QuotaExceededError),
        ("error_token", AuthError),
    ],
)
def test_classify_errors(load_fixture, fixture, expected):
    error = classify_errors(load_fixture(f"api_football/{fixture}.json")["errors"])
    assert type(error) is expected


def test_empty_errors_are_ok():
    assert classify_errors([]) is None
    assert classify_errors({}) is None


# HTTP client with mocked transport


@respx.mock
def test_client_caches_and_never_refetches(client, load_fixture):
    payload = load_fixture("api_football/players_epl_2023_p1.json")
    route = respx.get(f"{BASE}/players").mock(
        return_value=httpx.Response(200, json=payload, headers=HEADERS)
    )
    params = {"league": 39, "season": 2023, "page": 1}
    first = client.get("players", params)
    second = client.get("players", params)
    assert route.call_count == 1
    assert not first.from_cache
    assert second.from_cache
    assert second.payload == payload
    assert route.calls[0].request.headers["x-apisports-key"] == "test-key"


@respx.mock
def test_client_does_not_cache_plan_errors(client, load_fixture):
    respx.get(f"{BASE}/players").mock(
        return_value=httpx.Response(200, json=load_fixture("api_football/error_plan.json"))
    )
    params = {"league": 39, "season": 2015, "page": 1}
    with pytest.raises(PlanRestrictionError):
        client.get("players", params)
    assert not client.is_cached("players", params)


@respx.mock
def test_client_retries_server_errors_then_succeeds(client, load_fixture):
    payload = load_fixture("api_football/players_super_lig_2023_p1.json")
    route = respx.get(f"{BASE}/players").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=payload)]
    )
    result = client.get("players", {"league": 203, "season": 2023, "page": 1})
    assert route.call_count == 2
    assert result.payload["results"] == 2
    assert client.guard.used_today == 2  # both attempts count against the quota


@respx.mock
def test_client_retries_per_minute_limit(client, load_fixture):
    payload = load_fixture("api_football/players_super_lig_2023_p1.json")
    now = [1_800_000_000.0]
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client.guard._clock = lambda: now[0]
    client.guard._sleep = fake_sleep
    respx.get(f"{BASE}/players").mock(
        side_effect=[
            httpx.Response(200, json=load_fixture("api_football/error_rate_limit.json")),
            httpx.Response(200, json=payload),
        ]
    )
    result = client.get("players", {"league": 203, "season": 2023, "page": 1})
    assert result.payload["results"] == 2
    assert sleeps, "client should wait for the minute window"


@respx.mock
def test_client_raises_on_daily_quota_error(client, load_fixture):
    respx.get(f"{BASE}/players").mock(
        return_value=httpx.Response(200, json=load_fixture("api_football/error_quota.json"))
    )
    with pytest.raises(QuotaExceededError):
        client.get("players", {"league": 39, "season": 2023, "page": 1})


@respx.mock
def test_client_tightens_quota_from_headers(client, load_fixture):
    respx.get(f"{BASE}/leagues").mock(
        return_value=httpx.Response(
            200,
            json=load_fixture("api_football/leagues_39.json"),
            headers={"x-ratelimit-requests-remaining": "10"},
        )
    )
    client.get("leagues", {"id": 39})
    assert client.guard.used_today == 90


# Provider


@respx.mock
def test_provider_walks_all_pages(client, load_fixture, epl):
    page1 = load_fixture("api_football/players_epl_2023_p1.json")
    page2 = dict(page1, paging={"current": 2, "total": 2}, response=page1["response"][:1])
    route = respx.get(f"{BASE}/players").mock(
        side_effect=lambda request: httpx.Response(
            200, json=page1 if request.url.params["page"] == "1" else page2
        )
    )
    provider = ApiFootballStatsProvider(client)
    frame = provider.player_season_stats(epl, 2023)
    assert route.call_count == 2
    assert len(frame) == 5
    # Second call is served fully from cache.
    provider.player_season_stats(epl, 2023)
    assert route.call_count == 2
