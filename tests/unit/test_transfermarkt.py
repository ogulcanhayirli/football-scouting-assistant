import gzip
import json
import shutil
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from scout.config import TransfermarktConfig
from scout.providers.transfermarkt import (
    TransfermarktValuationProvider,
    download_snapshot,
    refresh_snapshot,
    season_of,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "transfermarkt"


@pytest.fixture
def provider():
    return TransfermarktValuationProvider(FIXTURES)


def test_season_of_splits_on_july():
    dates = pd.Series(["2023-06-30", "2023-07-01", "2024-05-19"])
    assert season_of(dates).tolist() == [2022, 2023, 2023]


def test_league_appearances_filters_and_aggregates(provider, epl, super_lig):
    apps = provider.league_appearances([epl, super_lig])
    assert 910201 not in apps["source_player_id"].tolist()  # La Liga only
    yilmazer = apps[apps["source_player_id"] == 910102]
    assert yilmazer["minutes"].tolist() == [88]  # cup match (TRP) excluded
    ben = apps[apps["source_player_id"] == 910002].sort_values("club_id")
    assert ben["club_id"].tolist() == [9902, 9903]  # one row per club after the move


def test_players_scoped_to_leagues(provider, epl, super_lig):
    players = provider.players([epl, super_lig])
    assert sorted(players["source_player_id"].tolist()) == [
        910001,
        910002,
        910003,
        910101,
        910102,
    ]
    assert "Çağlar Şahinoğlu" in players["player_name"].tolist()
    only_tr = provider.players([super_lig])
    assert set(only_tr["source_player_id"]) == {910101, 910102}


def test_valuations_sorted_and_typed(provider, epl):
    vals = provider.valuations([epl])
    assert set(vals["source_player_id"]) == {910001, 910002, 910003}
    first = vals[vals["source_player_id"] == 910001]
    assert first["valuation_date"].is_monotonic_increasing
    assert first["market_value_eur"].tolist() == [3_000_000, 12_000_000, 25_000_000]


def test_schema_drift_is_reported(tmp_path):
    shutil.copy(FIXTURES / "appearances.csv.gz", tmp_path)
    pd.DataFrame({"player_id": [1]}).to_csv(tmp_path / "players.csv.gz", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        TransfermarktValuationProvider(tmp_path)._read("players.csv.gz")


@respx.mock
def test_download_snapshot_is_idempotent(tmp_path):
    config = TransfermarktConfig(base_url="https://data.example/data", files=["players.csv.gz"])
    route = respx.get("https://data.example/data/players.csv.gz").mock(
        return_value=httpx.Response(200, content=b"gzbytes", headers={"etag": "abc"})
    )
    download_snapshot(config, tmp_path)
    download_snapshot(config, tmp_path)
    assert route.call_count == 1
    assert (tmp_path / "transfermarkt" / "players.csv.gz").read_bytes() == b"gzbytes"
    assert (tmp_path / "transfermarkt" / "players.csv.gz.meta.json").is_file()


def test_transfers_keep_free_moves_and_blank_fees_apart(tmp_path):
    csv = (
        "player_id,transfer_date,transfer_season,from_club_id,to_club_id,transfer_fee\n"
        "1,2023-07-01,23/24,10,20,15000000.0\n"
        "2,2023-07-02,23/24,10,20,0.0\n"
        "3,2023-07-03,23/24,10,20,\n"
    )
    (tmp_path / "transfers.csv.gz").write_bytes(gzip.compress(csv.encode()))
    transfers = TransfermarktValuationProvider(tmp_path).transfers()
    assert transfers["transfer_fee_eur"].tolist()[:2] == [15_000_000, 0]
    assert pd.isna(transfers["transfer_fee_eur"].iloc[2])
    assert str(transfers["transfer_date"].dtype).startswith("datetime64")


def test_players_and_valuations_unscoped(provider, epl):
    assert 910201 in provider.players(None)["source_player_id"].tolist()  # La Liga only
    assert len(provider.valuations(None)) > len(provider.valuations([epl]))


def test_seasons_come_from_the_games_table(tmp_path, epl):
    # The 2019/20 restart ran into July 2020; a date cut off would say 2020.
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    apps = (
        "appearance_id,game_id,player_id,player_club_id,player_current_club_id,date,"
        "player_name,competition_id,yellow_cards,red_cards,goals,assists,minutes_played\n"
        "a1,77,910001,9901,9901,2020-07-26,Alex Synthetic,GB1,0,0,0,0,90\n"
        "a2,78,910001,9901,9901,2020-09-12,Alex Synthetic,GB1,0,0,0,0,90\n"
        "a3,79,910001,9901,9901,2021-02-01,Alex Synthetic,GB1,0,0,0,0,90\n"
    )
    games = "game_id,competition_id,season,date\n77,GB1,2019,2020-07-26\n78,GB1,2020,2020-09-12\n"
    (tmp_path / "appearances.csv.gz").write_bytes(gzip.compress(apps.encode()))
    (tmp_path / "games.csv.gz").write_bytes(gzip.compress(games.encode()))
    result = TransfermarktValuationProvider(tmp_path).league_appearances([epl])
    assert result.set_index("season")["matches"].to_dict() == {2019: 1, 2020: 2}  # 79: by date


def test_data_end_is_the_last_valuation(provider):
    vals = pd.read_csv(FIXTURES / "player_valuations.csv.gz")
    assert provider.data_end() == pd.Timestamp(vals["date"].max())


def test_profiles_repair_impossible_values(provider):
    profiles = provider.profiles().set_index("source_player_id")
    assert profiles.loc[910001, "contract_expiry"] == pd.Timestamp("2028-06-30")
    assert profiles.loc[910001, "agent"] == "Synthetic Sports Ltd"
    assert profiles.loc[910001, "highest_market_value_eur"] == 30_000_000
    assert pd.isna(profiles.loc[910003, "contract_expiry"])  # 1900 is an entry error
    assert pd.isna(profiles.loc[910101, "highest_market_value_eur"])  # -1 is an entry error
    assert pd.isna(profiles.loc[910002, "agent"])  # blank agent is missing, not ""


def _snapshot_routes(etag: str, body: bytes):
    url = "https://data.example/data/players.csv.gz"
    headers = {"etag": etag, "last-modified": "Sat, 05 Sep 2026 09:12:22 GMT"}
    respx.head(url).mock(return_value=httpx.Response(200, headers=headers))
    return respx.get(url).mock(return_value=httpx.Response(200, content=body, headers=headers))


@respx.mock
def test_refresh_skips_an_unchanged_release(tmp_path):
    config = TransfermarktConfig(base_url="https://data.example/data", files=["players.csv.gz"])
    get = _snapshot_routes('"v1"', b"v1 bytes")
    download_snapshot(config, tmp_path)
    results = refresh_snapshot(config, tmp_path)
    assert [r.status for r in results] == ["unchanged"]
    assert get.call_count == 1


@respx.mock
def test_refresh_archives_the_old_file_and_downloads_the_new(tmp_path):
    config = TransfermarktConfig(base_url="https://data.example/data", files=["players.csv.gz"])
    _snapshot_routes('"v1"', b"v1 bytes")
    download_snapshot(config, tmp_path)
    respx.reset()
    _snapshot_routes('"v2"', b"v2 bytes")
    (result,) = refresh_snapshot(config, tmp_path)
    folder = tmp_path / "transfermarkt"
    assert result.status == "updated"
    assert result.archived_to == folder / "archive" / "2026-09-05"
    assert (folder / "players.csv.gz").read_bytes() == b"v2 bytes"
    assert (result.archived_to / "players.csv.gz").read_bytes() == b"v1 bytes"
    assert (result.archived_to / "players.csv.gz.meta.json").is_file()
    meta = json.loads((folder / "players.csv.gz.meta.json").read_text())
    assert meta["headers"]["etag"] == '"v2"'


@respx.mock
def test_refresh_downloads_a_missing_file(tmp_path):
    config = TransfermarktConfig(base_url="https://data.example/data", files=["players.csv.gz"])
    _snapshot_routes('"v1"', b"v1 bytes")
    (result,) = refresh_snapshot(config, tmp_path)
    assert result.status == "new"
    assert result.archived_to is None
