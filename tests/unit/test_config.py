from pathlib import Path

import pytest
from pydantic import ValidationError

from scout.config import QuotaConfig, load_settings
from scout.secrets import MissingSecretError, get_secret

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("name", ["backfill.free.yaml", "backfill.pro.yaml", "backfill.ultra.yaml"])
def test_committed_configs_load(name):
    settings = load_settings(CONFIGS / name)
    keys = [lg.key for lg in settings.leagues]
    assert keys[:2] == ["epl", "super_lig"]
    assert len(keys) == len(set(keys))
    assert len({lg.api_football_id for lg in settings.leagues}) == len(keys)
    assert settings.league("super_lig").api_football_id == 203
    if name == "backfill.ultra.yaml":
        assert len(keys) == 10


def test_free_config_matches_free_tier_limits():
    quota = load_settings(CONFIGS / "backfill.free.yaml").api_football.quota
    assert (quota.daily_limit, quota.per_minute_limit) == (100, 10)


def test_safety_margin_must_be_below_limit():
    with pytest.raises(ValidationError):
        QuotaConfig(daily_limit=10, per_minute_limit=5, safety_margin=10)


def test_secret_prefers_env(monkeypatch):
    monkeypatch.setenv("X_KEY", "from-env")
    monkeypatch.setenv("X_SSM", "/param")
    assert get_secret("X_KEY", "X_SSM", ssm_fetch=lambda _: "from-ssm") == "from-env"


def test_secret_falls_back_to_ssm(monkeypatch):
    monkeypatch.delenv("X_KEY", raising=False)
    monkeypatch.setenv("X_SSM", "/scout/key")
    assert get_secret("X_KEY", "X_SSM", ssm_fetch=lambda name: f"ssm:{name}") == "ssm:/scout/key"


def test_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("X_KEY", raising=False)
    monkeypatch.delenv("X_SSM", raising=False)
    with pytest.raises(MissingSecretError):
        get_secret("X_KEY", "X_SSM")


def test_competitions_are_separate_from_leagues():
    settings = load_settings(CONFIGS / "backfill.ultra.yaml")
    competition_keys = {c.key for c in settings.competitions}
    assert {
        "champions_league",
        "europa_league",
        "conference_league",
        "world_cup",
    } <= competition_keys
    assert competition_keys.isdisjoint({lg.key for lg in settings.leagues})
    ids = [c.api_football_id for c in settings.competitions] + [
        lg.api_football_id for lg in settings.leagues
    ]
    assert len(ids) == len(set(ids))
    assert {c.kind for c in settings.competitions} == {"club", "national"}
    assert load_settings(CONFIGS / "backfill.free.yaml").competitions == []
