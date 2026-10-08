import numpy as np
import pandas as pd
import pandera.errors
import pytest

from scout.contracts import PLAYER_SEASON_FEATURES, PLAYER_SEASON_LABELS, validate
from scout.features import (
    Inputs,
    build_features,
    build_links,
    feature_dates,
    labels,
    read_feature_table,
    value_asof,
)
from scout.features.build import (
    available_minutes,
    fill_recorded_zeros,
    match_tier_features,
    season_tier_features,
    team_results,
)
from scout.providers.api_football.matches import MATCH_COLUMNS

KEYS = ["league_key", "season", "source_player_id"]


def ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def match_row(fixture, day, team, opponent, player, minutes=90, goals=None, **stats):
    """One player match row; team_goals 2, opponent_goals 1 unless given."""
    row = dict.fromkeys(MATCH_COLUMNS)
    row.update(
        source="api_football",
        fixture_id=fixture,
        kickoff=ts(day),
        league_key="epl",
        league_id=39,
        season=2023,
        team_id=team,
        team_name=f"Team {team}",
        opponent_id=opponent,
        is_home=True,
        team_goals=stats.pop("team_goals", 2),
        opponent_goals=stats.pop("opponent_goals", 1),
        source_player_id=player,
        player_name=f"Player {player}",
        minutes=minutes,
        goals=goals,
        substitute=minutes is None or minutes < 60,
        rating=stats.pop("rating", 7.0 if minutes else None),
        passes_total=stats.pop("passes_total", 40 if minutes else None),
        passes_accurate=stats.pop("passes_accurate", 30 if minutes else None),
    )
    row.update(stats)
    return row


def frame(rows):
    out = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    for column in ["minutes", "goals", "rating", "passes_total", "passes_accurate", "assists"]:
        out[column] = pd.to_numeric(out[column]).astype("Float64")
    for column in [
        "shots_total",
        "shots_on",
        "passes_key",
        "tackles_total",
        "blocks",
        "interceptions",
        "duels_total",
        "duels_won",
        "dribbles_attempts",
        "dribbles_success",
        "dribbled_past",
        "fouls_drawn",
        "fouls_committed",
        "cards_yellow",
        "cards_red",
        "saves",
        "goals_conceded",
        "offsides",
        "penalty_won",
        "penalty_committed",
        "penalty_scored",
        "penalty_missed",
        "penalty_saved",
    ]:
        out[column] = pd.to_numeric(out[column]).astype("Float64")
    out["kickoff"] = pd.to_datetime(out["kickoff"], utc=True).dt.as_unit("ns")
    return out


def dates_for(season=2023, day="2024-06-30"):
    return pd.DataFrame(
        {"league_key": ["epl"], "season": [season], "feature_date": [pd.Timestamp(day)]}
    )


# Feature dates


def test_feature_date_is_30_june_or_the_day_after_the_last_match():
    season = pd.DataFrame({"league_key": ["epl", "epl"], "season": [2018, 2019]})
    matches = frame(
        [
            match_row(1, "2019-05-12", 1, 2, 10),
            match_row(2, "2020-07-26", 1, 2, 10),  # COVID restart
        ]
    )
    matches["season"] = [2018, 2019]
    dates = feature_dates(season, matches).set_index("season")["feature_date"]
    assert dates[2018] == pd.Timestamp("2019-06-30")
    assert dates[2019] == pd.Timestamp("2020-07-27")


# Match tier


def test_blank_counts_become_zero_only_where_recorded():
    rows = [match_row(i, "2023-09-01", 1, 2, 10 + i, tackles_total=2) for i in range(3)]
    rows.append(match_row(9, "2023-09-01", 1, 2, 99, tackles_total=None))  # blank: a zero
    rows.append(match_row(8, "2023-09-01", 1, 2, 98, minutes=None))  # bench: stays blank
    filled = fill_recorded_zeros(frame(rows)).set_index("source_player_id")
    assert filled.loc[99, "tackles_total"] == 0
    assert pd.isna(filled.loc[98, "tackles_total"])
    assert filled["shots_total"].isna().all()  # nobody has it: missing, not zero


def test_per90_rating_and_team_context():
    rows = [
        match_row(1, "2023-08-12", 1, 2, 10, minutes=90, goals=1, rating=8.0),
        match_row(2, "2023-08-19", 1, 3, 10, minutes=30, goals=1, rating=6.0, team_goals=1),
        match_row(3, "2023-08-26", 1, 4, 10, minutes=None, team_goals=0, opponent_goals=0),
    ]
    out = match_tier_features(frame(rows), dates_for()).set_index("source_player_id")
    row = out.loc[10]
    assert (row["minutes"], row["appearances"], row["starts"]) == (120, 2, 1)
    assert row["goals_per90"] == pytest.approx(1.5)
    assert row["rating"] == pytest.approx((8 * 90 + 6 * 30) / 120)  # weighted by minutes
    assert row["team_matches"] == 3
    assert row["team_ppg"] == pytest.approx((3 + 1 + 1) / 3)
    assert row["minutes_share"] == pytest.approx(120 / 270)
    assert row["pass_accuracy"] == pytest.approx(0.75)


def test_minutes_share_counts_only_matches_at_each_club():
    # One appearance for club 1, then a move to club 2 (Palmer, 2023/24).
    rows = [match_row(1, "2023-08-12", 1, 9, 10, minutes=20)]
    rows += [match_row(10 + i, f"2023-09-{i + 1:02d}", 2, 9, 10) for i in range(4)]
    rows += [match_row(30 + i, f"2023-09-{i + 1:02d}", 1, 9, 50) for i in range(4)]
    m = frame(rows)
    share = available_minutes(m, team_results(m), KEYS)
    # Club 1 until 12 August (1 match), club 2 from then on (4 matches).
    assert share.loc[("epl", 2023, 10)] == 5 * 90


def test_form_uses_the_last_ten_team_matches():
    rows = [match_row(i, f"2023-08-{i + 1:02d}", 1, 2, 10) for i in range(12)]
    rows += [match_row(100 + i, f"2023-08-{i + 1:02d}", 1, 2, 11, minutes=None) for i in range(10)]
    rows += [match_row(110 + i, f"2023-08-{i + 13:02d}", 1, 2, 11) for i in range(2)]
    out = match_tier_features(frame(rows), dates_for()).set_index("source_player_id")
    assert out.loc[11, "form_minutes_share"] == pytest.approx(180 / 900)


# Season tier


def test_season_tier_sums_clubs_and_reads_blank_goals_as_zero():
    stats = pd.DataFrame(
        {
            "league_key": ["epl"] * 3,
            "season": [2012] * 3,
            "source_player_id": [10, 10, 11],
            "team_id": [1, 2, 1],
            "team_name": ["A", "B", "A"],
            "minutes": [900.0, 300.0, 450.0],
            "appearances": [10.0, 4.0, 5.0],
            "lineups": [10.0, 3.0, 5.0],
            "goals": [3.0, 1.0, None],
            "cards_yellow": [1.0, 0.0, 0.0],
            "cards_red": [0.0, 0.0, 0.0],
        }
    )
    out = season_tier_features(stats).set_index("source_player_id")
    assert out.loc[10, "minutes"] == 1200
    assert out.loc[10, "team_name"] == "A"  # the club with most minutes
    assert out.loc[10, "goals_per90"] == pytest.approx(4 / 1200 * 90)
    assert out.loc[11, "goals"] == 0


# Values and labels


VALS = pd.DataFrame(
    {
        "source_player_id": [7, 7, 7, 7, 8],
        "valuation_date": pd.to_datetime(
            ["2022-01-01", "2023-12-01", "2024-07-15", "2026-05-01", "2022-01-01"]
        ),
        "market_value_eur": [1e6, 4e6, 9e6, 20e6, 5e6],
    }
)


def test_value_asof_never_looks_ahead_and_ignores_stale_values():
    rows = pd.DataFrame(
        {"tm_player_id": [7, 8, None], "feature_date": pd.to_datetime(["2024-06-30"] * 3)}
    )
    out = value_asof(rows, VALS, "feature_date")
    assert out.loc[0, "market_value_eur"] == 4e6  # not the July value
    assert pd.isna(out.loc[1, "market_value_eur"])  # 2022 value is too old
    assert pd.isna(out.loc[2, "market_value_eur"])  # not linked


def test_labels_leave_unreachable_horizons_empty():
    rows = pd.DataFrame(
        {
            "league_key": ["epl", "epl"],
            "season": [2021, 2023],
            "source_player_id": [1, 2],
            "tm_player_id": [7, 7],
            "feature_date": pd.to_datetime(["2022-06-30", "2024-06-30"]),
        }
    )
    out = labels(rows, VALS, data_end=pd.Timestamp("2026-06-12"))
    first = out.iloc[0]
    assert first["value_now"] == 1e6
    assert first["value_24m"] == 4e6  # December 2023: latest on or before 2024-06-30
    assert first["value_24m_date"] <= pd.Timestamp("2024-06-30")
    assert first["log_growth_24m"] == pytest.approx(np.log(first["value_24m"] / first["value_now"]))
    assert pd.isna(out.iloc[1]["value_24m"])  # 2026-06-30 lies past the data
    validate(PLAYER_SEASON_LABELS, out)


# End to end


def _inputs():
    matches = frame(
        [match_row(i, f"2023-08-{i + 1:02d}", 1, 2, 10, goals=1) for i in range(6)]
        + [match_row(i, f"2023-08-{i + 1:02d}", 1, 2, 11) for i in range(6)]
    )
    season_stats = pd.DataFrame(
        {
            "league_key": ["epl"] * 3,
            "season": [2023, 2023, 2014],
            "source_player_id": [10, 11, 10],
            "player_name": ["Alex Synthetic", "Ben Placeholder", "Alex Synthetic"],
            "birth_date": pd.to_datetime(["2000-01-01", "1995-01-01", "2000-01-01"]),
            "nationality": ["England"] * 3,
            "height_cm": [180.0, None, 180.0],
            "position": ["Attacker", "Midfielder", "Attacker"],
            "team_id": [1, 1, 1],
            "team_name": ["Team 1"] * 3,
            "minutes": [540.0, 540.0, 600.0],
            "appearances": [6.0, 6.0, 7.0],
            "lineups": [6.0, 6.0, 7.0],
            "goals": [6.0, None, 2.0],
            "cards_yellow": [0.0, 0.0, 0.0],
            "cards_red": [0.0, 0.0, 0.0],
        }
    )
    tm_players = pd.DataFrame(
        {
            "source_player_id": pd.array([7, 8], dtype="Int64"),
            "birth_date": pd.to_datetime(["2000-01-01", "1995-01-01"]),
            "sub_position": ["Centre-Forward", "Central Midfield"],
            "foot": ["right", "left"],
            "height_cm": [181.0, 175.0],
        }
    )
    links = pd.DataFrame(
        {
            "af_player_id": pd.array([10, 11], dtype="Int64"),
            "tm_player_id": pd.array([7, 8], dtype="Int64"),
        }
    )
    vals = VALS.copy()
    vals.loc[len(vals)] = [8, pd.Timestamp("2024-03-01"), 6e6]
    return Inputs(season_stats, matches, links, tm_players, vals, pd.Timestamp("2026-06-12"))


def test_build_features_end_to_end():
    features, label_table = build_features(_inputs())
    validate(PLAYER_SEASON_FEATURES, features)
    validate(PLAYER_SEASON_LABELS, label_table)
    by = features.set_index(["season", "source_player_id"])
    assert by.loc[(2023, 10), "tier"] == "match"
    assert by.loc[(2014, 10), "tier"] == "season"
    assert pd.isna(by.loc[(2014, 10), "rating"])  # no detailed stats before 2015
    assert by.loc[(2023, 10), "goals_per90"] == pytest.approx(1.0)
    assert by.loc[(2023, 11), "height_cm"] == 175  # Transfermarkt fills the gap
    assert by.loc[(2023, 10), "age"] == pytest.approx(24.5, abs=0.01)
    assert by.loc[(2023, 10), "market_value"] == 4e6
    assert by.loc[(2023, 10), "value_date"] <= by.loc[(2023, 10), "feature_date"]


def test_contract_catches_a_value_from_after_the_feature_date():
    features, _ = build_features(_inputs())
    features.loc[0, "value_date"] = features.loc[0, "feature_date"] + pd.Timedelta(days=1)
    with pytest.raises(pandera.errors.SchemaErrors, match="leak"):
        validate(PLAYER_SEASON_FEATURES, features)


def test_build_links_drops_players_linked_differently_per_league(monkeypatch):
    stats = pd.DataFrame(
        {
            "league_key": ["epl", "super_lig", "epl"],
            "source_player_id": [1, 1, 2],
            "minutes": [100.0, 100.0, 100.0],
        }
    )
    answers = {"epl": {1: 11, 2: 22}, "super_lig": {1: 12}}

    def fake_match(players, _tm):
        league = "epl" if 2 in set(players["source_player_id"]) else "super_lig"
        ids = players["source_player_id"].tolist()
        return pd.DataFrame(
            {
                "af_player_id": ids,
                "tm_player_id": [answers[league][i] for i in ids],
                "reason": "matched",
                "rule": "name_score",
            }
        )

    monkeypatch.setattr("scout.features.pipeline.match_players", fake_match)
    links = build_links(stats, pd.DataFrame())
    assert links["af_player_id"].tolist() == [2]


def test_partitioned_write_round_trips(tmp_path):
    from scout.features.pipeline import _write_partitioned

    features, _ = build_features(_inputs())
    _write_partitioned(features, tmp_path / "t")
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert files == [
        "t/league=epl/season=2014/data.parquet",
        "t/league=epl/season=2023/data.parquet",
    ]
    assert "season" not in pd.read_parquet(tmp_path / files[0]).columns
    back = read_feature_table(tmp_path / "t")
    assert sorted(back["season"].tolist()) == sorted(features["season"].tolist())
    assert str(back["feature_date"].dtype) == "datetime64[ms]"
