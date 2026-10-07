from scout.storage import CacheKey, LocalRawStore, RawMeta

META = RawMeta(
    url="https://example/players",
    params={"league": "39"},
    fetched_at="2026-10-07T00:00:00+00:00",
    status_code=200,
    headers={},
)


def test_key_is_order_independent():
    a = CacheKey.of("api_football", "players", {"season": 2023, "league": 39, "page": 1})
    b = CacheKey.of("api_football", "/players", {"page": "1", "league": "39", "season": "2023"})
    assert a == b
    assert str(a.relative_path()) == "api_football/players/league=39/page=1/season=2023.json"


def test_put_get_roundtrip_is_byte_exact(tmp_path):
    store = LocalRawStore(tmp_path)
    key = CacheKey.of("api_football", "players", {"league": 203, "season": 2023, "page": 2})
    body = '{"name": "Kerem Aktürkoğlu"}'.encode()
    assert not store.exists(key)
    store.put(key, body, META)
    assert store.exists(key)
    assert store.get(key) == body
    assert store.get_meta(key) == META


def test_keys_lists_entries_and_skips_sidecars(tmp_path):
    store = LocalRawStore(tmp_path)
    k1 = CacheKey.of("api_football", "players", {"league": 39, "season": 2023, "page": 1})
    k2 = CacheKey.of("api_football", "players", {"league": 39, "season": 2023, "page": 2})
    store.put(k1, b"{}", META)
    store.put(k2, b"{}", META)
    assert store.keys("api_football", "players") == [k1, k2]


def test_no_temp_files_left_behind(tmp_path):
    store = LocalRawStore(tmp_path)
    store.put(CacheKey.of("s", "e", {"a": 1}), b"x", META)
    assert not list(tmp_path.rglob("*.tmp"))
