"""Runs scripts/feasibility.py end to end against a mocked API and the fixtures."""

import copy
import gzip
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import respx
import yaml

from scout.providers.api_football import parse_players_page
from scout.providers.api_football.adapter import rows_to_frame

ROOT = Path(__file__).parents[2]
FIX = ROOT / "tests" / "fixtures"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "feasibility_script", ROOT / "scripts" / "feasibility.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_offline(tmp_path, monkeypatch, load_fixture, extra_args=()):
    raw = tmp_path / "raw"
    shutil.copytree(FIX / "transfermarkt", raw / "transfermarkt", dirs_exist_ok=True)
    clubs = (
        "club_id,name\n9901,Test Town FC\n9903,Fixture City\n9911,Örnek Spor\n9912,Deneme Gücü SK\n"
    )
    (raw / "transfermarkt" / "clubs.csv.gz").write_bytes(gzip.compress(clubs.encode()))
    config = yaml.safe_load((ROOT / "configs" / "backfill.free.yaml").read_text())
    config["paths"] = {
        k: str(tmp_path / k) for k in ["raw_dir", "processed_dir", "state_dir", "reports_dir"]
    }
    config["paths"]["raw_dir"] = str(raw)
    config["api_football"]["quota"]["per_minute_limit"] = 1000  # no real waiting in tests
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("API_FOOTBALL_KEY", "k")
    monkeypatch.setattr(
        sys, "argv", ["feasibility", "--config", str(cfg), "--no-download", *extra_args]
    )

    pages = {
        "39": load_fixture("api_football/players_epl_2023_p1.json"),
        "203": load_fixture("api_football/players_super_lig_2023_p1.json"),
    }
    leagues = load_fixture("api_football/leagues_39.json")
    plan_error = load_fixture("api_football/error_plan.json")

    def handler(request):
        p = request.url.params
        path = request.url.path
        if path == "/status":
            return httpx.Response(
                200, json={"errors": [], "response": {"subscription": {"plan": "Free"}}}
            )
        if path == "/leagues":
            body = copy.deepcopy(leagues)
            body["response"][0]["league"]["id"] = int(p["id"])
            return httpx.Response(200, json=body)
        if path == "/fixtures":
            return httpx.Response(200, json={"errors": [], "response": [{}] * 380})
        if p["season"] not in {"2023", "2024"}:
            return httpx.Response(200, json=plan_error)
        body = copy.deepcopy(pages[p["league"]])
        body["paging"] = {"current": int(p["page"]), "total": 1}
        return httpx.Response(200, json=body)

    with respx.mock:
        respx.get(url__startswith="https://v3.football.api-sports.io").mock(side_effect=handler)
        assert load_script().main() == 0
    return json.loads((tmp_path / "reports_dir" / "feasibility.json").read_text())


def test_feasibility_script_offline(tmp_path, monkeypatch, load_fixture):
    findings = run_offline(tmp_path, monkeypatch, load_fixture)
    probes = findings["seasons"]["epl"]["free_tier_probes"]
    assert probes["2023"]["ok"]
    assert probes["2012"]["error"] == "PlanRestrictionError"
    assert findings["matching"]["super_lig"]["match_rate"] == 1.0
    assert findings["matching"]["epl"]["matched"] == 3
    assert findings["request_estimate"]
    assert (tmp_path / "reports_dir" / "match_review_super_lig.csv").is_file()


def test_feasibility_script_from_processed(tmp_path, monkeypatch, load_fixture, epl, super_lig):
    processed = tmp_path / "processed_dir" / "player_season_stats"
    for league, name in ((epl, "epl"), (super_lig, "super_lig")):
        page = load_fixture(f"api_football/players_{name}_2023_p1.json")
        frame = rows_to_frame(parse_players_page(page, league, 2023))
        target = processed / f"league={league.key}" / "season=2023.parquet"
        target.parent.mkdir(parents=True)
        frame.to_parquet(target)
    transfers = (
        "player_id,transfer_date,transfer_season,from_club_id,to_club_id,transfer_fee\n"
        "910001,2024-01-15,23/24,9999,9901,30000000.0\n"
    )
    raw_tm = tmp_path / "raw" / "transfermarkt"
    raw_tm.mkdir(parents=True)
    (raw_tm / "transfers.csv.gz").write_bytes(gzip.compress(transfers.encode()))

    findings = run_offline(tmp_path, monkeypatch, load_fixture, ["--from-processed"])
    assert findings["stats_rows"] > 0
    assert {r["league_key"] for r in findings["completeness_by_season"]} == {"epl", "super_lig"}
    assert findings["label_coverage"]
    assert findings["conflicting_matches"] == []
    assert findings["value_vs_fee"][0]["paid_transfers"] == 1
    assert findings["value_vs_fee"][0]["spearman"] is None  # too few transfers
    assert (tmp_path / "reports_dir" / "value_vs_fee.csv").is_file()
