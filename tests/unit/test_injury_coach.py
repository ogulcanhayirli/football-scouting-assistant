"""Youth minutes, injury history from the sidelined periods, and coach changes."""

import numpy as np
import pandas as pd
import pytest

from scout.config import load_settings
from scout.features.context import (
    club_calendar,
    coach_features,
    competition_features,
    injury_features,
    injury_source_agreement,
    sidelined_injury_features,
)
from scout.features.pipeline import _configured


def rows(*specs):
    """(league, season, player, feature date, team) tuples."""
    return pd.DataFrame(
        {
            "league_key": [s[0] for s in specs],
            "season": [s[1] for s in specs],
            "source_player_id": [s[2] for s in specs],
            "feature_date": pd.to_datetime([s[3] for s in specs]),
            "team_id": pd.array([s[4] for s in specs], dtype="Int64"),
        }
    )


def elo_matches(team, dates, league="epl", seasons=None, kind="league"):
    """Elo match rows with ``team`` at home against club 999."""
    seasons = seasons or [pd.Timestamp(d).year - (pd.Timestamp(d).month < 7) for d in dates]
    return pd.DataFrame(
        {
            "fixture_id": [hash((team, d)) % 10**8 for d in dates],
            "kickoff": pd.to_datetime(dates, utc=True),
            "league_key": league,
            "season": seasons,
            "kind": kind,
            "home_id": team,
            "away_id": 999,
        }
    )


def weekly(team, start, end, **kw):
    dates = pd.date_range(start, end, freq="7D").strftime("%Y-%m-%d").tolist()
    return elo_matches(team, dates, **kw)


def period(player, start, end, kind="Hamstring"):
    return {
        "source_player_id": player,
        "type": kind,
        "start_date": pd.Timestamp(start),
        "end_date": pd.Timestamp(end) if end else pd.NaT,
    }


# Youth


def test_youth_tournaments_have_their_own_columns():
    def match(player, league, date, minutes=90):
        return {
            "source_player_id": player,
            "league_key": league,
            "kickoff": pd.Timestamp(date, tz="UTC"),
            "minutes": minutes,
            "goals": 0,
            "assists": 0,
            "rating": 7.0,
        }

    matches = pd.DataFrame(
        [
            match(1, "uefa_u21_championship", "2021-03-24"),  # starts the youth data
            match(1, "uefa_u21_championship", "2023-07-05", minutes=60),
            match(1, "olympics_men", "2022-07-01"),  # more than 12 months before
            match(1, "world_cup", "2014-06-15"),
        ]
    )
    comps = {
        "uefa_u21_championship": ("youth", "europe"),
        "olympics_men": ("youth", "europe"),
        "world_cup": ("national", "europe"),
    }
    out = competition_features(
        rows(
            ("epl", 2023, 1, "2024-06-30", 1),
            ("epl", 2023, 2, "2024-06-30", 1),
            ("epl", 2019, 1, "2020-06-30", 1),
        ),
        matches,
        comps,
        {"epl": "europe"},
    )
    assert out.loc[0, "youth_minutes_12m"] == 60
    assert out.loc[0, "youth_matches_12m"] == 1
    assert out.loc[0, "intl_minutes_12m"] == 0  # an U21 match is not a cap
    assert out.loc[1, "youth_minutes_12m"] == 0  # covered window, no youth match
    assert np.isnan(out.loc[2, "youth_minutes_12m"])  # before the youth data starts


def test_regions_without_held_club_cups_leave_cup_columns_missing():
    matches = pd.DataFrame(
        {
            "source_player_id": [1],
            "league_key": ["champions_league"],
            "kickoff": [pd.Timestamp("2015-09-15", tz="UTC")],
            "minutes": [90],
            "goals": [0],
            "assists": [0],
            "rating": [7.0],
        }
    )
    out = competition_features(
        rows(("mls", 2023, 1, "2024-01-01", 1), ("epl", 2023, 1, "2024-06-30", 1)),
        matches,
        {"champions_league": ("club", "europe")},
        {"mls": "north_america", "epl": "europe"},
    )
    assert np.isnan(out.loc[0, "club_cup_minutes_12m"])
    assert out.loc[1, "club_cup_minutes_12m"] == 0


# Sidelined


def test_club_calendar_has_both_sides():
    cal = club_calendar(elo_matches(10, ["2019-08-10", "2019-08-17"]))
    assert set(cal["team_id"]) == {10, 999}
    assert len(cal) == 4
    assert cal["kickoff"].dt.tz is None


def test_sidelined_counts_club_matches_while_out():
    calendar = club_calendar(weekly(10, "2018-08-04", "2019-05-25"))
    sidelined = pd.DataFrame(
        [
            period(1, "2018-09-01", "2018-09-20"),  # 3 matches: 1, 8, 15 September
            period(1, "2018-10-01", "2018-10-10"),  # 6 October; within 30 days: same spell
            period(1, "2019-02-01", "2019-02-15"),  # 2 and 9 February: a new spell
            period(1, "2019-03-01", "2019-03-20", kind="Suspended"),  # not an injury
            period(1, "2019-07-15", "2019-08-01"),  # starts after the feature date
        ]
    )
    out = sidelined_injury_features(
        rows(("epl", 2018, 1, "2019-06-30", 10), ("epl", 2018, 2, "2019-06-30", 10)),
        sidelined,
        calendar,
    )
    assert out.loc[0, "injury_matches_missed_12m"] == 6
    assert out.loc[0, "injury_spells_12m"] == 2
    assert out.loc[0, "last_injury_date"] == pd.Timestamp("2019-02-09")
    assert out.loc[1, "injury_matches_missed_12m"] == 0  # asked for, never out: fit


def test_sidelined_never_reads_an_end_after_the_feature_date():
    calendar = club_calendar(weekly(10, "2018-08-04", "2019-08-31"))
    # Out from 1 June 2019. On 30 June the end (August) is not known yet; with
    # it the period would cover the August matches too.
    late_end = pd.DataFrame([period(1, "2019-06-01", "2019-08-30")])
    no_end = pd.DataFrame([period(1, "2019-06-01", None)])
    early = pd.DataFrame([period(1, "2019-06-01", "2019-06-20")])
    row = rows(("epl", 2018, 1, "2019-06-30", 10))
    a = sidelined_injury_features(row, late_end, calendar)
    b = sidelined_injury_features(row, no_end, calendar)
    c = sidelined_injury_features(row, early, calendar)
    # The five June Saturdays, nothing after the feature date.
    assert a.loc[0, "injury_matches_missed_12m"] == b.loc[0, "injury_matches_missed_12m"] == 5
    assert a.loc[0, "last_injury_date"] < pd.Timestamp("2019-06-30")
    assert c.loc[0, "injury_matches_missed_12m"] == 3  # 1, 8 and 15 June


def test_sidelined_period_from_before_the_window_counts_inside_it():
    calendar = club_calendar(weekly(10, "2018-05-05", "2019-05-25"))
    sidelined = pd.DataFrame([period(1, "2018-05-01", "2018-07-10")])
    out = sidelined_injury_features(rows(("epl", 2018, 1, "2019-06-30", 10)), sidelined, calendar)
    # Matches from 30 June 2018 on: 30 June and 7 July.
    assert out.loc[0, "injury_matches_missed_12m"] == 2


def test_sidelined_missing_outside_its_coverage_or_without_club_fixtures():
    calendar = club_calendar(weekly(10, "2009-08-01", "2026-05-30"))
    out = sidelined_injury_features(
        rows(
            ("epl", 2009, 1, "2010-06-30", 10),  # window starts before 2010/11
            ("epl", 2025, 1, "2026-06-30", 10),  # after September 2025
            ("epl", 2018, 1, "2019-06-30", 77),  # club not in the calendar
            ("epl", 2018, 1, "2019-06-30", 10),
        ),
        pd.DataFrame([period(5, "2018-09-01", "2018-09-10")]),
        calendar,
    )
    assert out["injury_spells_12m"].isna().tolist() == [True, True, True, False]


def test_injuries_table_wins_from_april_2021_and_sidelined_fills_the_rest():
    calendar = club_calendar(weekly(10, "2019-08-03", "2022-05-28"))
    injuries = pd.DataFrame(
        {
            "league_key": "epl",
            "season": [2020, 2021],
            "source_player_id": [1, 1],
            "fixture_id": [1, 2],
            "fixture_date": pd.to_datetime(["2021-04-10", "2021-10-02"], utc=True),
            "status": "Missing Fixture",
            "reason": "Knee Injury",
        }
    )
    sidelined = pd.DataFrame(
        [period(1, "2020-09-01", "2020-09-14"), period(1, "2021-09-01", "2021-12-01")]
    )
    out = injury_features(
        rows(
            ("epl", 2019, 1, "2020-07-27", 10),  # no table: sidelined
            ("epl", 2020, 1, "2021-06-30", 10),  # table starts mid window: sidelined
            ("epl", 2021, 1, "2022-06-30", 10),  # table
        ),
        injuries,
        {("epl", 2020), ("epl", 2021)},
        sidelined,
        calendar,
    )
    assert out["injury_source"].tolist() == ["sidelined", "sidelined", "injuries"]
    assert out.loc[0, "injury_matches_missed_12m"] == 0
    assert out.loc[1, "injury_matches_missed_12m"] == 2  # 5 and 12 September 2020
    assert out.loc[2, "injury_matches_missed_12m"] == 1  # the table's one row, not 13


def test_agreement_report_compares_rows_both_sources_cover():
    calendar = club_calendar(weekly(10, "2022-08-06", "2023-05-27"))
    injuries = pd.DataFrame(
        {
            "league_key": "epl",
            "season": 2022,
            "source_player_id": [1],
            "fixture_id": [1],
            "fixture_date": pd.to_datetime(["2022-10-01"], utc=True),
            "status": "Missing Fixture",
            "reason": "Knock",
        }
    )
    sidelined = pd.DataFrame([period(1, "2022-09-28", "2022-10-05")])
    table = injury_source_agreement(
        rows(("epl", 2022, 1, "2023-06-30", 10), ("epl", 2022, 2, "2023-06-30", 10)),
        injuries,
        {("epl", 2022)},
        sidelined,
        calendar,
    )
    assert table.loc[0, "rows"] == 2
    assert table.loc[0, "agree"] == 1.0


# Coaches


def coach(team, start, end=None):
    return {
        "coach_id": hash((team, start)) % 10**6 + 1,
        "team_id": team,
        "start_date": pd.Timestamp(start),
        "end_date": pd.Timestamp(end) if end else pd.NaT,
    }


def test_coach_flags_at_month_level():
    calendar = club_calendar(
        pd.concat(
            [
                weekly(10, "2017-08-12", "2018-05-12", seasons=None),
                weekly(10, "2018-08-11", "2019-05-11"),
                weekly(20, "2018-08-11", "2019-05-11"),
                weekly(30, "2018-08-11", "2019-05-11"),
            ],
            ignore_index=True,
        )
    )
    coaches = pd.DataFrame(
        [
            coach(10, "2016-01-01", "2018-06-01"),
            coach(10, "2018-06-01", "2018-12-01"),  # new for 2018/19 (June, off season)
            coach(10, "2018-12-01"),  # in season change
            coach(20, "2015-01-01"),  # same coach throughout
            coach(20, "2019-07-01"),  # after the season: not used
        ]
    )
    out = coach_features(
        rows(
            ("epl", 2018, 1, "2019-06-30", 10),
            ("epl", 2018, 2, "2019-06-30", 20),
            ("epl", 2018, 3, "2019-06-30", 30),  # no spells: unknown
            ("epl", 2017, 4, "2018-06-30", 10),
        ),
        coaches,
        calendar,
    )
    assert out.loc[0, "new_coach"] == 1
    assert out.loc[0, "coach_changed_in_season"] == 1
    assert out.loc[0, "last_coach_change_date"] == pd.Timestamp("2018-12-01")
    assert out.loc[1, "new_coach"] == 0
    assert out.loc[1, "coach_changed_in_season"] == 0
    assert pd.isna(out.loc[1, "last_coach_change_date"])
    assert np.isnan(out.loc[2, "new_coach"])
    # 2017/18: the June 2018 appointment comes after the season's last match.
    assert out.loc[3, "new_coach"] == 0
    assert out.loc[3, "coach_changed_in_season"] == 0


def test_coach_appointed_in_the_first_match_month_is_new_not_a_change():
    calendar = club_calendar(weekly(10, "2018-08-11", "2019-05-11"))
    coaches = pd.DataFrame([coach(10, "2015-01-01"), coach(10, "2018-08-01")])
    out = coach_features(rows(("epl", 2018, 1, "2019-06-30", 10)), coaches, calendar)
    assert (out.loc[0, "new_coach"], out.loc[0, "coach_changed_in_season"]) == (1, 0)


def test_coach_start_after_the_feature_date_is_unknown_even_in_the_same_month():
    calendar = club_calendar(weekly(10, "2017-08-26", "2018-05-14"))
    coaches = pd.DataFrame([coach(10, "2015-01-01"), coach(10, "2018-05-20")])
    out = coach_features(rows(("epl", 2017, 1, "2018-05-16", 10)), coaches, calendar)
    assert out.loc[0, "coach_changed_in_season"] == 0
    assert pd.isna(out.loc[0, "last_coach_change_date"])


# Pipeline


def test_configured_keeps_only_configured_leagues_and_seasons(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "api_football: {quota: {daily_limit: 100, per_minute_limit: 10}}\n"
        "leagues:\n"
        "  - {key: epl, name: EPL, api_football_id: 39, transfermarkt_id: GB1}\n"
        "  - {key: j1_league, name: J1, api_football_id: 98, transfermarkt_id: JAP1,"
        " region: asia, calendar_year: true, skip_seasons: [2026]}\n"
    )
    settings = load_settings(config)
    frame = pd.DataFrame(
        {
            "league_key": ["epl", "j1_league", "j1_league", "serie_b"],
            "season": [2025, 2025, 2026, 2025],
        }
    )
    kept = _configured(frame, settings)
    assert list(zip(kept["league_key"], kept["season"], strict=True)) == [
        ("epl", 2025),
        ("j1_league", 2025),
    ]


@pytest.mark.parametrize("region", ["north_america", "asia"])
def test_new_regions_load(region, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "api_football: {quota: {daily_limit: 100, per_minute_limit: 10}}\n"
        "leagues: [{key: x, name: X, api_football_id: 1, transfermarkt_id: X1, "
        f"region: {region}}}]\n"
    )
    assert load_settings(config).leagues[0].region == region
