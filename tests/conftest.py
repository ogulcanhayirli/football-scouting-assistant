import json
from pathlib import Path

import pytest

from scout.config import LeagueConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(relative: str) -> dict:
        return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def epl() -> LeagueConfig:
    return LeagueConfig(
        key="epl", name="Premier League", api_football_id=39, transfermarkt_id="GB1"
    )


@pytest.fixture
def super_lig() -> LeagueConfig:
    return LeagueConfig(
        key="super_lig", name="Süper Lig", api_football_id=203, transfermarkt_id="TR1"
    )
