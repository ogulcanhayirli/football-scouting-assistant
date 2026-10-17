"""Data contract tests on the committed sample.

Each adapter's output on the fixtures must satisfy the canonical contracts,
and deliberately broken rows must be rejected.
"""

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from scout.contracts import PLAYER_PROFILES, PLAYER_SEASON_STATS, PLAYERS, VALUATIONS, validate
from scout.features.pipeline import build_profiles
from scout.providers.api_football.adapter import parse_players_page, rows_to_frame
from scout.providers.transfermarkt import TransfermarktValuationProvider

pytestmark = pytest.mark.contract

TM = Path(__file__).parents[1] / "fixtures" / "transfermarkt"


@pytest.fixture
def stats(load_fixture, epl, super_lig):
    rows = parse_players_page(load_fixture("api_football/players_epl_2023_p1.json"), epl, 2023)
    rows += parse_players_page(
        load_fixture("api_football/players_super_lig_2023_p1.json"), super_lig, 2023
    )
    return rows_to_frame(rows)


def test_api_football_sample_meets_contract(stats):
    validate(PLAYER_SEASON_STATS, stats)


def test_transfermarkt_sample_meets_contracts(epl, super_lig):
    provider = TransfermarktValuationProvider(TM)
    validate(PLAYERS, provider.players([epl, super_lig]))
    validate(VALUATIONS, provider.valuations([epl, super_lig]))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("shots_on", 999.0),  # more shots on target than shots
        ("minutes", -5.0),
        ("rating", 11.0),
        ("tackles_total", -1.0),
        ("position", "Sweeper"),
    ],
)
def test_contract_rejects_broken_stats(stats, column, value):
    broken = stats.copy()
    broken.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(PLAYER_SEASON_STATS, broken)


def test_contract_rejects_duplicate_player_rows(stats):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(PLAYER_SEASON_STATS, pd.concat([stats, stats.iloc[[0]]], ignore_index=True))


def test_contract_rejects_negative_valuation(epl):
    vals = TransfermarktValuationProvider(TM).valuations([epl])
    vals.loc[0, "market_value_eur"] = -1.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(VALUATIONS, vals)


def _profiles() -> pd.DataFrame:
    tm = TransfermarktValuationProvider(TM).profiles()
    links = pd.DataFrame(
        {
            "af_player_id": pd.array(range(1, len(tm) + 1), "Int64"),
            "tm_player_id": tm["source_player_id"],
        }
    )
    return build_profiles(links, tm, pd.Timestamp("2026-06-12"))


def test_transfermarkt_profiles_meet_contract():
    validate(PLAYER_PROFILES, _profiles())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("contract_expiry", pd.Timestamp("1900-06-30")),  # the unrepaired fixture value
        ("highest_market_value_eur", -1.0),
        ("agent", ""),
    ],
)
def test_contract_rejects_unrepaired_profile_values(column, value):
    broken = _profiles()
    broken.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(PLAYER_PROFILES, broken)
