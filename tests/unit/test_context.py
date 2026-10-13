import numpy as np
import pandas as pd
import pytest

from scout.features.build import feature_dates
from scout.features.context import (
    competition_features,
    injury_features,
    team_strength,
    transfer_features,
)


def rows(*specs):
    """(league, season, player, feature date) tuples."""
    return pd.DataFrame(
        {
            "league_key": [s[0] for s in specs],
            "season": [s[1] for s in specs],
            "source_player_id": [s[2] for s in specs],
            "feature_date": pd.to_datetime([s[3] for s in specs]),
        }
    )


def comp_match(player, league, date, minutes=90, goals=0, assists=0, rating=7.0):
    return {
        "source_player_id": player,
        "league_key": league,
        "kickoff": pd.Timestamp(date, tz="UTC"),
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "rating": rating,
    }


COMPS = {
    "champions_league": ("club", "europe"),
    "copa_libertadores": ("club", "south_america"),
    "world_cup": ("national", "europe"),
}
REGIONS = {"epl": "europe", "brasileirao": "south_america"}


def test_cup_and_national_minutes_count_only_the_12_months_before():
    matches = pd.DataFrame(
        [
            comp_match(1, "champions_league", "2015-09-15"),  # starts the UEFA data
            comp_match(1, "champions_league", "2016-10-01", goals=1),
            comp_match(1, "champions_league", "2017-03-01", minutes=45, assists=1),
            comp_match(1, "champions_league", "2017-07-01"),  # after the feature date
            comp_match(1, "champions_league", "2016-05-01"),  # more than 12 months before
            comp_match(1, "world_cup", "2014-06-15"),  # starts the national data
            comp_match(1, "world_cup", "2017-06-10", minutes=30),
        ]
    )
    out = competition_features(
        rows(("epl", 2016, 1, "2017-06-30"), ("epl", 2016, 2, "2017-06-30")),
        matches,
        COMPS,
        REGIONS,
    )
    first, second = out.iloc[0], out.iloc[1]
    assert first["club_cup_minutes_12m"] == 135
    assert first["club_cup_matches_12m"] == 2
    assert first["club_cup_ga_per90_12m"] == pytest.approx(2 / 135 * 90)
    assert first["ucl_minutes_12m"] == 135
    assert first["intl_minutes_12m"] == 30
    assert first["last_comp_match_date"] == pd.Timestamp("2017-06-10")
    # A player without cup matches in a covered window has 0, not missing.
    assert second["club_cup_minutes_12m"] == 0
    assert second["intl_matches_12m"] == 0
    assert pd.isna(second["club_cup_rating_12m"])


def test_cup_columns_are_missing_until_the_region_is_covered():
    matches = pd.DataFrame(
        [
            comp_match(1, "champions_league", "2015-09-15"),
            comp_match(2, "copa_libertadores", "2019-03-01"),
        ]
    )
    out = competition_features(
        rows(
            ("epl", 2015, 1, "2016-06-30"),  # window starts before the UEFA data
            ("epl", 2017, 1, "2018-06-30"),
            ("brasileirao", 2017, 2, "2018-01-01"),  # CONMEBOL data starts in 2019
            ("brasileirao", 2020, 2, "2021-03-01"),
        ),
        matches,
        COMPS,
        REGIONS,
    )
    assert out["club_cup_minutes_12m"].isna().tolist() == [True, False, True, False]
    assert out["intl_minutes_12m"].isna().all()  # no national team data at all


def team_row(fixture, team, opponent, date, shots, possession, xg=None, season=2023):
    return {
        "league_key": "epl",
        "season": season,
        "fixture_id": fixture,
        "kickoff": pd.Timestamp(date, tz="UTC"),
        "team_id": team,
        "opponent_id": opponent,
        "shots_total": shots,
        "possession": possession,
        "pass_accuracy": 0.8,
        "xg": xg,
    }


def test_team_strength_uses_opponent_shots_and_only_matches_before_the_date():
    stats = pd.DataFrame(
        [
            team_row(1, 10, 20, "2023-09-01", 15, 60.0, 2.0),
            team_row(1, 20, 10, "2023-09-01", 5, 40.0, 0.5),
            team_row(2, 10, 20, "2023-10-01", 9, 50.0, 1.0),
            team_row(2, 20, 10, "2023-10-01", 11, 50.0, 1.5),
            team_row(3, 10, 20, "2024-08-01", 99, 99.0, 9.0),  # after the feature date
            team_row(3, 20, 10, "2024-08-01", 0, 1.0, 0.0),
        ]
    )
    dates = pd.DataFrame(
        {"league_key": ["epl"], "season": [2023], "feature_date": [pd.Timestamp("2024-06-30")]}
    )
    out = team_strength(stats, dates).set_index("team_id")
    assert out.loc[10, "team_shots_pm"] == 12
    assert out.loc[10, "team_shots_against_pm"] == 8
    assert out.loc[10, "team_shot_share"] == pytest.approx(0.6)
    assert out.loc[10, "team_possession"] == 55
    assert out.loc[10, "team_xga_pm"] == pytest.approx(1.0)


def injury(player, date, reason="Knee Injury", status="Missing Fixture", fixture=None):
    return {
        "league_key": "epl",
        "season": 2023,
        "source_player_id": player,
        "fixture_id": fixture or hash((player, date)) % 10**6,
        "fixture_date": pd.Timestamp(date, tz="UTC"),
        "status": status,
        "reason": reason,
    }


def test_injuries_count_matches_and_spells_but_not_suspensions():
    injuries = pd.DataFrame(
        [
            injury(1, "2023-09-01"),
            injury(1, "2023-09-08"),  # same spell
            injury(1, "2024-02-01"),  # new spell
            injury(1, "2024-03-01", reason="Red Card"),  # a suspension
            injury(1, "2024-04-01", status="Questionable"),  # played or not, unknown
            injury(1, "2024-07-01"),  # after the feature date
        ]
    )
    out = injury_features(
        rows(
            ("epl", 2023, 1, "2024-06-30"),
            ("epl", 2023, 2, "2024-06-30"),
            ("epl", 2019, 1, "2020-07-27"),
        ),
        injuries,
        covered={("epl", 2023)},
    )
    assert out.loc[0, "injury_matches_missed_12m"] == 3
    assert out.loc[0, "injury_spells_12m"] == 2
    assert out.loc[0, "last_injury_date"] == pd.Timestamp("2024-02-01")
    assert out.loc[1, "injury_matches_missed_12m"] == 0  # covered, never injured
    assert np.isnan(out.loc[2, "injury_matches_missed_12m"])  # no injury data for 2019


def test_transfers_before_the_feature_date_only():
    transfers = pd.DataFrame(
        {
            "source_player_id": [1, 1, 1, 1],
            "transfer_date": pd.to_datetime(
                ["2019-07-01", "2022-01-15", "2023-08-01", "2024-08-01"]
            ),
            "kind": ["transfer", "loan", "transfer", "transfer"],
            "fee_eur": [2e6, None, 30e6, 80e6],
        }
    )
    out = transfer_features(
        rows(("epl", 2023, 1, "2024-06-30"), ("epl", 2023, 2, "2024-06-30")), transfers
    )
    first = out.iloc[0]
    assert first["career_moves"] == 3
    assert first["career_loans"] == 1
    assert first["career_paid_moves"] == 2
    assert first["moves_last_2y"] == 1  # the 2022 loan is more than 2 years back
    assert first["last_move_kind"] == "transfer"
    assert first["last_fee_eur"] == 30e6
    assert first["max_fee_eur_to_date"] == 30e6  # not the 80m move after the date
    assert first["on_loan"] == 0
    assert (
        first["days_since_last_move"]
        == (pd.Timestamp("2024-06-30") - pd.Timestamp("2023-08-01")).days
    )
    assert out.iloc[1].drop("last_move_kind").isna().all()  # no transfer history at all


def test_calendar_year_leagues_get_a_january_feature_date():
    seasons = pd.DataFrame({"league_key": ["epl", "brasileirao"], "season": [2023, 2023]})
    matches = pd.DataFrame(
        {
            "league_key": ["epl", "brasileirao"],
            "season": [2023, 2023],
            "kickoff": pd.to_datetime(["2024-05-19", "2023-12-06"], utc=True),
        }
    )
    out = feature_dates(seasons, matches, frozenset({"brasileirao"})).set_index("league_key")
    assert out.loc["epl", "feature_date"] == pd.Timestamp("2024-06-30")
    assert out.loc["brasileirao", "feature_date"] == pd.Timestamp("2024-01-01")
