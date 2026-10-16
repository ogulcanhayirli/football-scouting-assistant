import logging
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
import yaml

from scout.backfill.matches import (
    PROBE_MATCHES,
    match_parquet_path,
    run_match_backfill,
    write_match_report,
)
from scout.config import Settings
from scout.contracts import PLAYER_MATCH_STATS, validate
from scout.providers.api_football.adapter import parse_seasons
from scout.providers.api_football.matches import (
    finished_fixtures,
    match_rows_to_frame,
    parse_fixture_players,
)
from scout.providers.base import SeasonInfo
from scout.ratelimit import DailyQuotaExhaustedError

ROOT = Path(__file__).parents[2]


def fixture(fid, status="FT", home=(1, "Test Town"), away=(2, "Fixture City"), score=(2, 1)):
    return {
        "fixture": {"id": fid, "date": "2024-08-17T14:00:00+00:00", "status": {"short": status}},
        "league": {"id": 39, "season": 2024, "round": "Regular Season - 1"},
        "teams": {
            "home": {"id": home[0], "name": home[1]},
            "away": {"id": away[0], "name": away[1]},
        },
        "goals": {"home": score[0], "away": score[1]},
    }


def player(pid, name, minutes=90, goals=None, passes=(40, 35), position="M", **extra):
    stats = {
        "games": {
            "minutes": minutes,
            "number": pid % 99,
            "position": position,
            "rating": "7.1" if minutes else None,
            "captain": False,
            "substitute": minutes is None,
        },
        "offsides": None,
        "shots": {"total": 2, "on": 1},
        "goals": {"total": goals, "conceded": 0, "assists": None, "saves": None},
        "passes": {"total": passes[0], "key": 1, "accuracy": str(passes[1])},
        "tackles": {"total": None, "blocks": None, "interceptions": None},
        "duels": {"total": 5, "won": 3},
        "dribbles": {"attempts": None, "success": None, "past": None},
        "fouls": {"drawn": None, "committed": None},
        "cards": {"yellow": 0, "red": 0},
        "penalty": {"won": None, "commited": None, "scored": 0, "missed": 0, "saved": None},
    }
    for path, value in extra.items():
        group, field = path.split("__")
        stats[group][field] = value
    return {"player": {"id": pid, "name": name, "photo": "x"}, "statistics": [stats]}


def players_payload(home_players, away_players, home=(1, "Test Town"), away=(2, "Fixture City")):
    return {
        "errors": [],
        "response": [
            {"team": {"id": home[0], "name": home[1]}, "players": home_players},
            {"team": {"id": away[0], "name": away[1]}, "players": away_players},
        ],
    }


# Parsing


def test_finished_fixtures_drops_unplayed_matches():
    payload = {"response": [fixture(1), fixture(2, "PST"), fixture(3, "AET"), fixture(4, "AWD")]}
    assert [f["fixture"]["id"] for f in finished_fixtures(payload)] == [1, 3]


def test_parse_maps_sides_opponents_and_accurate_passes(epl):
    payload = players_payload(
        [player(10, "Alex Synthetic", goals=2), player(11, "Bench Warmer", minutes=None)],
        [player(20, "Ben Placeholder", goals=1, position="F")],
    )
    frame = match_rows_to_frame(parse_fixture_players(fixture(99), payload, epl))
    validate(PLAYER_MATCH_STATS, frame)
    alex = frame[frame["source_player_id"] == 10].iloc[0]
    assert (alex["team_name"], alex["opponent_name"], bool(alex["is_home"])) == (
        "Test Town",
        "Fixture City",
        True,
    )
    assert (alex["team_goals"], alex["opponent_goals"], alex["goals"]) == (2, 1, 2)
    assert alex["passes_accurate"] == 35  # a count here, not a percentage
    ben = frame[frame["source_player_id"] == 20].iloc[0]
    assert (ben["opponent_name"], bool(ben["is_home"]), ben["team_goals"]) == (
        "Test Town",
        False,
        1,
    )
    bench = frame[frame["source_player_id"] == 11].iloc[0]
    assert pd.isna(bench["minutes"])
    assert bool(bench["substitute"])
    assert str(frame["kickoff"].dtype) == "datetime64[ns, UTC]"


def test_negative_match_counts_become_missing(epl, caplog):
    payload = players_payload([player(10, "Alex Synthetic", shots__total=-1, shots__on=None)], [])
    with caplog.at_level(logging.WARNING):
        frame = match_rows_to_frame(parse_fixture_players(fixture(99), payload, epl))
    assert pd.isna(frame.loc[0, "shots_total"])
    assert any("fixture 99" in m for m in caplog.messages)
    validate(PLAYER_MATCH_STATS, frame)


def test_contract_rejects_more_goals_than_the_team_scored(epl):
    payload = players_payload([player(10, "Alex Synthetic", goals=1)], [])
    frame = match_rows_to_frame(parse_fixture_players(fixture(99, score=(2, 1)), payload, epl))
    frame.loc[0, "goals"] = 3  # what the adapter repairs, the contract must still catch
    with pytest.raises(pandera.errors.SchemaErrors, match="goals must not exceed team_goals"):
        validate(PLAYER_MATCH_STATS, frame)


def test_adapter_repairs_source_errors(epl, caplog):
    # Seen in real data: a player without an id, two brothers sharing one id,
    # a goal credited in a match the team lost 0-1, and position "SUB".
    home = [
        player(10, "Halil Pehlivan"),
        player(10, "Yasin Pehlivan", minutes=42),
        player(0, "No Id"),
        player(12, "Sokol Scorer", goals=1),
        player(13, "Odd Position", position="SUB"),
        player(14, "Dash Position", position="-"),
    ]
    other_match = players_payload([player(10, "Halil Pehlivan")], [])
    rows = parse_fixture_players(fixture(99, score=(0, 1)), players_payload(home, []), epl)
    rows += parse_fixture_players(fixture(98), other_match, epl)
    with caplog.at_level(logging.WARNING):
        frame = match_rows_to_frame(rows)
    validate(PLAYER_MATCH_STATS, frame)
    match = frame[frame["fixture_id"] == 99].set_index("player_name")
    assert "No Id" not in match.index
    assert "Yasin Pehlivan" not in match.index  # Halil carries id 10 most often
    assert match.loc["Halil Pehlivan", "minutes"] == 90
    assert pd.isna(match.loc["Sokol Scorer", "goals"])
    assert pd.isna(match.loc["Odd Position", "position"])
    assert pd.isna(match.loc["Dash Position", "position"])
    text = " ".join(caplog.messages)
    assert "no player id" in text
    assert "shares id 10" in text
    assert "exceed the team" in text


def test_parse_seasons_reads_match_coverage(load_fixture, epl):
    seasons = parse_seasons(load_fixture("api_football/leagues_39.json"), epl)
    assert all(isinstance(s.has_match_player_stats, bool) for s in seasons)


# Runner


class FakeGuard:
    remaining_today = 1000


class FakeClient:
    def __init__(self):
        self.requests_made = 0
        self.guard = FakeGuard()


class FakeMatchProvider:
    """Serves fixtures and player payloads from dicts, counting requests like the cache."""

    def __init__(self, fixtures, payloads, fail_after=None):
        self.client = FakeClient()
        self._fixtures, self._payloads = fixtures, payloads
        self.fetched: list[int] = []
        self.fail_after = fail_after

    def fixtures(self, league, season):
        return self._fixtures.get((league.key, season), [])

    def fixture_players(self, fixture_id):
        if fixture_id not in self.fetched:
            if self.fail_after is not None and len(self.fetched) >= self.fail_after:
                raise DailyQuotaExhaustedError("daily budget spent")
            self.fetched.append(fixture_id)
            self.client.requests_made += 1
        return self._payloads.get(fixture_id, {"response": []})


class FakeSeasons:
    def __init__(self, covered):
        self.covered = covered

    def list_seasons(self, league):
        return [
            SeasonInfo(
                league.key, s, None, None, False, True, s in self.covered.get(league.key, ())
            )
            for s in (2013, 2023, 2024)
        ]


@pytest.fixture
def settings(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "backfill.ultra.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        k: str(tmp_path / k) for k in ("raw_dir", "processed_dir", "state_dir", "reports_dir")
    }
    raw["backfill"]["stop_on_plan_error"] = False
    return Settings.model_validate(raw)


def _season(prefix, n=10):
    fixtures = [fixture(prefix + i) for i in range(n)]
    payloads = {
        prefix + i: players_payload([player(prefix + i, f"P {i}", goals=1)], []) for i in range(n)
    }
    return fixtures, payloads


def test_match_backfill_fetches_covered_seasons_and_writes_parquet(settings):
    fx23, pl23 = _season(1000)
    fx24, pl24 = _season(2000)
    provider = FakeMatchProvider({("epl", 2023): fx23, ("epl", 2024): fx24}, pl23 | pl24)
    seasons = FakeSeasons({"epl": {2023, 2024}})  # 2013 not covered: never requested
    result = run_match_backfill(settings, seasons, provider, league_keys=["epl"])
    assert result.complete
    assert [o.status for o in result.outcomes] == ["complete", "complete"]
    assert provider.client.requests_made == 20
    frame = pd.read_parquet(match_parquet_path(settings, "epl", 2024))
    assert len(frame) == 10

    again = FakeMatchProvider({("epl", 2023): fx23, ("epl", 2024): fx24}, pl23 | pl24)
    again.fetched = list(provider.fetched)  # everything cached
    rerun = run_match_backfill(settings, seasons, again, league_keys=["epl"])
    assert rerun.requests_made == 0
    assert rerun.complete


def test_season_without_player_data_costs_only_the_probe(settings):
    fixtures = [fixture(3000 + i) for i in range(40)]
    provider = FakeMatchProvider({("epl", 2023): fixtures}, {})
    result = run_match_backfill(
        settings, FakeSeasons({"epl": {2023}}), provider, league_keys=["epl"]
    )
    assert result.outcomes[0].status == "no_player_data"
    assert provider.client.requests_made == PROBE_MATCHES
    assert result.complete
    assert not match_parquet_path(settings, "epl", 2023).exists()


def test_without_probe_every_match_is_fetched(settings):
    # Only the last match has players, so the probe would have skipped the season.
    fixtures = [fixture(3500 + i) for i in range(40)]
    payloads = {
        3500 + i: players_payload([player(3500 + i, f"P {i}", goals=1)], []) for i in range(39, 40)
    }
    provider = FakeMatchProvider({("epl", 2023): fixtures}, payloads)
    result = run_match_backfill(
        settings, FakeSeasons({"epl": {2023}}), provider, league_keys=["epl"], probe_first=False
    )
    assert provider.client.requests_made == 40
    assert result.outcomes[0].status == "complete"
    assert len(pd.read_parquet(match_parquet_path(settings, "epl", 2023))) == 1


def test_without_probe_an_empty_season_writes_nothing(settings):
    fixtures = [fixture(3600 + i) for i in range(10)]
    provider = FakeMatchProvider({("epl", 2023): fixtures}, {})
    result = run_match_backfill(
        settings, FakeSeasons({"epl": {2023}}), provider, league_keys=["epl"], probe_first=False
    )
    assert provider.client.requests_made == 10
    assert result.outcomes[0].status == "no_player_data"
    assert result.complete
    assert not match_parquet_path(settings, "epl", 2023).exists()


def test_quota_stop_is_resumable(settings):
    fixtures, payloads = _season(4000, n=12)
    provider = FakeMatchProvider({("epl", 2023): fixtures}, payloads, fail_after=8)
    seasons = FakeSeasons({"epl": {2023}})
    first = run_match_backfill(settings, seasons, provider, league_keys=["epl"])
    assert first.stopped_reason.startswith("quota")
    assert not first.complete

    provider.fail_after = None
    second = run_match_backfill(settings, seasons, provider, league_keys=["epl"])
    assert second.complete
    assert second.requests_made == 4  # only the matches not cached before


def test_match_report(settings, tmp_path):
    fixtures, payloads = _season(5000, n=3)
    provider = FakeMatchProvider({("epl", 2023): fixtures}, payloads)
    result = run_match_backfill(
        settings, FakeSeasons({"epl": {2023}}), provider, league_keys=["epl"]
    )
    md = write_match_report(result, tmp_path / "reports")
    assert "| epl | 2023 | complete | 3/3 | 3 |" in md.read_text()


# Team level


def team_block(team_id, name, **stats):
    defaults = {
        "Shots on Goal": 5,
        "Shots off Goal": 4,
        "Total Shots": 12,
        "Blocked Shots": 3,
        "Shots insidebox": 8,
        "Shots outsidebox": 4,
        "Fouls": 10,
        "Corner Kicks": 6,
        "Offsides": 2,
        "Ball Possession": "58%",
        "Yellow Cards": 1,
        "Red Cards": None,
        "Goalkeeper Saves": 3,
        "Total passes": 520,
        "Passes accurate": 460,
        "Passes %": "88%",
    }
    defaults.update(stats)
    return {
        "team": {"id": team_id, "name": name},
        "statistics": [{"type": k, "value": v} for k, v in defaults.items()],
    }


def test_team_statistics_parse_percentages_sides_and_optional_xg(epl):
    from scout.contracts import TEAM_MATCH_STATS
    from scout.providers.api_football.matches import parse_fixture_statistics, team_rows_to_frame

    payload = {
        "response": [
            team_block(1, "Test Town", expected_goals="1.84"),
            team_block(2, "Fixture City", **{"Ball Possession": "42%", "Passes %": "79%"}),
        ]
    }
    frame = team_rows_to_frame(parse_fixture_statistics(fixture(99, score=(2, 1)), payload, epl))
    validate(TEAM_MATCH_STATS, frame)
    home = frame[frame["team_id"] == 1].iloc[0]
    away = frame[frame["team_id"] == 2].iloc[0]
    assert (home["possession"], away["possession"]) == (58, 42)
    assert (bool(home["is_home"]), home["goals_for"], home["goals_against"]) == (True, 2, 1)
    assert (bool(away["is_home"]), away["opponent_id"]) == (False, 1)
    assert home["xg"] == pytest.approx(1.84)
    assert pd.isna(away["xg"])  # xG only exists from 2023/24
    assert pd.isna(home["cards_red"])


def test_team_contract_rejects_impossible_shots(epl):
    from scout.contracts import TEAM_MATCH_STATS
    from scout.providers.api_football.matches import parse_fixture_statistics, team_rows_to_frame

    payload = {"response": [team_block(1, "Test Town")]}
    frame = team_rows_to_frame(parse_fixture_statistics(fixture(99), payload, epl))
    frame.loc[0, "shots_on"] = 20  # past the adapter repair
    with pytest.raises(pandera.errors.SchemaErrors, match="shots_on must not exceed"):
        validate(TEAM_MATCH_STATS, frame)


def test_team_level_backfill_writes_its_own_table(settings):
    from scout.backfill.matches import TEAM_LEVEL

    fixtures = [fixture(6000 + i) for i in range(6)]

    class TeamProvider(FakeMatchProvider):
        def fixture_statistics(self, fixture_id):
            return self.fixture_players(fixture_id)

    payloads = {
        6000 + i: {"response": [team_block(1, "Test Town"), team_block(2, "Fixture City")]}
        for i in range(6)
    }
    provider = TeamProvider({("epl", 2023): fixtures}, payloads)

    class Seasons(FakeSeasons):
        def list_seasons(self, league):
            return [
                SeasonInfo(league.key, 2023, None, None, False, True, False, True),
            ]

    result = run_match_backfill(
        settings, Seasons({}), provider, league_keys=["epl"], level=TEAM_LEVEL
    )
    assert result.outcomes[0].status == "complete"
    path = match_parquet_path(settings, "epl", 2023, "team_match_stats")
    assert len(pd.read_parquet(path)) == 12
    assert not match_parquet_path(settings, "epl", 2023).exists()  # player table untouched


def test_competition_scope_downloads_cups_not_leagues(settings):
    fixtures, payloads = _season(7000, n=6)
    provider = FakeMatchProvider(
        {("champions_league", 2023): fixtures, ("epl", 2023): fixtures}, payloads
    )
    seasons = FakeSeasons({"champions_league": {2023}, "epl": {2023}})
    result = run_match_backfill(
        settings, seasons, provider, league_keys=["champions_league", "epl"], scope="competitions"
    )
    assert [(o.league_key, o.status) for o in result.outcomes] == [("champions_league", "complete")]
    assert match_parquet_path(settings, "champions_league", 2023).exists()
    assert not match_parquet_path(settings, "epl", 2023).exists()


# Lineups and injuries


def lineup_block(team_id, players, formation="4-2-3-1", subs=()):
    return {
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "coach": {"id": 900 + team_id, "name": f"Coach {team_id}"},
        "formation": formation,
        "startXI": [{"player": p} for p in players],
        "substitutes": [{"player": p} for p in subs],
    }


def test_lineups_keep_starters_bench_and_the_grid(epl):
    from scout.contracts import MATCH_LINEUPS
    from scout.providers.api_football.matches import lineup_rows_to_frame, parse_fixture_lineups

    payload = {
        "response": [
            lineup_block(
                1,
                [
                    {"id": 10, "name": "Keeper", "number": 1, "pos": "G", "grid": "1:1"},
                    {"id": 11, "name": "Back", "number": 2, "pos": "D", "grid": "2:4"},
                    {"id": 0, "name": "Unknown", "number": 3, "pos": "D", "grid": "2:3"},
                ],
                subs=[{"id": 12, "name": "Sub", "number": 14, "pos": "M", "grid": None}],
            ),
            lineup_block(
                2,
                [{"id": 20, "name": "Old Style", "number": 9, "pos": "F", "grid": None}],
                formation=None,
            ),
        ]
    }
    frame = lineup_rows_to_frame(parse_fixture_lineups(fixture(99), payload, epl))
    validate(MATCH_LINEUPS, frame)
    by_id = frame.set_index("source_player_id")
    assert 0 not in by_id.index  # no player id: dropped
    assert (by_id.loc[11, "grid_row"], by_id.loc[11, "grid_col"]) == (2, 4)
    assert bool(by_id.loc[11, "starter"])
    assert not bool(by_id.loc[12, "starter"])
    assert pd.isna(by_id.loc[12, "grid_row"])
    assert by_id.loc[10, "formation"] == "4-2-3-1"
    assert pd.isna(by_id.loc[20, "formation"])  # older seasons have no formation


def test_injuries_parse_and_drop_repeats(epl):
    from scout.contracts import INJURIES
    from scout.providers.api_football.matches import injury_rows_to_frame, parse_injuries

    entry = {
        "player": {"id": 7, "name": "A", "type": "Missing Fixture", "reason": "Knee Injury"},
        "team": {"id": 1, "name": "Test Town"},
        "fixture": {"id": 99, "date": "2023-09-01T19:00:00+00:00"},
    }
    payload = {
        "response": [
            entry,
            entry,
            {**entry, "fixture": {"id": 100, "date": "2023-09-08T19:00:00+00:00"}},
        ]
    }
    frame = injury_rows_to_frame(parse_injuries(payload, epl, 2023))
    validate(INJURIES, frame)
    assert frame["fixture_id"].tolist() == [99, 100]
    assert frame["reason"].unique().tolist() == ["Knee Injury"]


def test_injury_backfill_makes_one_request_per_covered_season(settings):
    from scout.backfill.matches import run_injury_backfill

    class InjuryProvider(FakeMatchProvider):
        def injuries(self, league, season):
            self.client.requests_made += 1
            return {
                "response": [
                    {
                        "player": {
                            "id": 7,
                            "name": "A",
                            "type": "Questionable",
                            "reason": "Illness",
                        },
                        "team": {"id": 1, "name": "T"},
                        "fixture": {"id": season, "date": f"{season}-09-01T19:00:00+00:00"},
                    }
                ]
            }

    class Seasons(FakeSeasons):
        def list_seasons(self, league):
            return [
                SeasonInfo(league.key, s, None, None, False, True, True, True, True, s >= 2020)
                for s in (2019, 2020, 2021)
            ]

    provider = InjuryProvider({}, {})
    result = run_injury_backfill(settings, Seasons({}), provider, league_keys=["epl"])
    assert [o.season for o in result.outcomes] == [2020, 2021]  # 2019 has no injury coverage
    assert provider.client.requests_made == 2
    assert match_parquet_path(settings, "epl", 2021, "injuries").exists()


def test_inconsistent_shot_split_becomes_missing(epl):
    from scout.contracts import TEAM_MATCH_STATS
    from scout.providers.api_football.matches import parse_fixture_statistics, team_rows_to_frame

    payload = {
        "response": [team_block(1, "Test Town", **{"Shots insidebox": 13, "Shots outsidebox": 0})]
    }
    frame = team_rows_to_frame(parse_fixture_statistics(fixture(99), payload, epl))
    validate(TEAM_MATCH_STATS, frame)
    assert pd.isna(frame.loc[0, "shots_inside_box"])
    assert pd.isna(frame.loc[0, "shots_outside_box"])
    assert frame.loc[0, "shots_total"] == 12


def test_zero_total_with_shots_on_target_becomes_missing(epl):
    from scout.contracts import TEAM_MATCH_STATS
    from scout.providers.api_football.matches import parse_fixture_statistics, team_rows_to_frame

    payload = {"response": [team_block(1, "Test Town", **{"Total Shots": 0})]}
    frame = team_rows_to_frame(parse_fixture_statistics(fixture(99), payload, epl))
    validate(TEAM_MATCH_STATS, frame)
    assert pd.isna(frame.loc[0, "shots_total"])
    assert frame.loc[0, "shots_on"] > 0


def test_more_successful_dribbles_than_attempts_become_missing(epl, caplog):
    payload = players_payload(
        [player(10, "Caio Synthetic", dribbles__attempts=2, dribbles__success=3)], []
    )
    with caplog.at_level(logging.WARNING):
        frame = match_rows_to_frame(parse_fixture_players(fixture(99), payload, epl))
    validate(PLAYER_MATCH_STATS, frame)
    assert pd.isna(frame.loc[0, "dribbles_success"])
    assert pd.isna(frame.loc[0, "dribbles_attempts"])
    assert any("Dribbles success" in m for m in caplog.messages)
