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
