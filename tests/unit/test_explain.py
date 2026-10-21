import numpy as np
import pandas as pd

from scout.explain import (
    MIN_PEERS,
    explain,
    flags_and_explanation,
    for_rows,
    peer_table,
    population,
    rank_text,
)


def _peers(n=100, seed=3, **overrides):
    """n regular centre forwards aged 22 in one season, every stat in the
    middle of a wide spread; ``overrides`` set player 1's numbers."""
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(
        {
            "league_key": "epl",
            "season": 2025,
            "source_player_id": np.arange(1, n + 1),
            "sub_position": "Centre-Forward",
            "position": "Attacker",
            "age": 22.0,
            "regular": True,
            "minutes": 2000.0,
            "goals_per90": rng.uniform(0.2, 0.6, n),
            "assists_per90": rng.uniform(0.1, 0.3, n),
            "shots_total": rng.uniform(20, 60, n),
            "team_shots_pm": 12.0,
            "team_matches": 38.0,
            "pass_accuracy": rng.uniform(0.65, 0.85, n),
            "minutes_share": rng.uniform(0.6, 0.9, n),
            "injury_matches_missed_12m": rng.integers(0, 4, n).astype(float),
            "value_change_1y": rng.uniform(1.0, 1.4, n),
        }
    )
    middle = {c: rows[c].median() for c in rows.columns if rows[c].dtype == float}
    middle.pop("age"), middle.pop("minutes")
    middle.pop("team_shots_pm"), middle.pop("team_matches")
    for column, value in {**middle, **overrides}.items():
        rows.loc[0, column] = value
    return rows


def _explained(rows):
    table = peer_table(rows)
    return table, flags_and_explanation(table)


def test_a_standout_reads_as_a_plain_sentence_with_number_and_percentile():
    rows = _peers(shots_total=128.0, goals_per90=0.95)  # 128 of 456 team shots: 28%
    _, out = _explained(rows)
    text = out.loc[0, "explanation"]
    assert "Takes 28% of his team's shots, top 1% of centre forwards aged 21 to 23." in text
    assert "Scores 0.95 goals per 90 minutes, top 1% of centre forwards aged 21 to 23." in text


def test_at_most_three_sentences_furthest_from_the_middle_first():
    rows = _peers(
        goals_per90=0.95,
        assists_per90=0.9,
        shots_total=130.0,
        pass_accuracy=0.99,
    )
    rows.loc[1, "pass_accuracy"] = 0.995  # pass accuracy is only second best
    _, out = _explained(rows)
    sentences = out.loc[0, "explanation"].split(". ")
    assert len(sentences) == 3
    assert "pass" not in out.loc[0, "explanation"]


def test_less_is_better_stats_rank_from_the_bottom():
    rows = _peers(injury_matches_missed_12m=15.0)
    _, out = _explained(rows)
    assert "Missed 15 matches injured in the last 12 months, bottom 1%" in out.loc[0, "explanation"]
    assert rank_text(97.3) == "top 3%"
    assert rank_text(99.9) == "top 1%"
    assert rank_text(4.2) == "bottom 5%"


def test_ties_count_against_the_claim_and_big_rises_read_as_multiples():
    rows = _peers(value_change_1y=15.0)
    rows["cards_yellow_per90"] = 0.2
    rows.loc[:29, "cards_yellow_per90"] = 0.0  # player 1 shares 0.00 with 29 others
    _, out = _explained(rows)
    text = out.loc[0, "explanation"]
    assert "yellow" not in text  # 30% are at least as good: not the top 10%
    assert "Market value 15 times what it was a year before, top 1%" in text


def test_no_notable_deviation_gives_a_neutral_sentence_not_an_empty_cell():
    rows = _peers()
    _, out = _explained(rows)
    text = out.loc[0, "explanation"]
    assert text.startswith("No stat stands out against other centre forwards aged 21 to 23")
    assert out["explanation"].str.len().gt(0).all()
    bare = rows[["league_key", "season", "source_player_id", "sub_position", "age"]].assign(
        regular=True, minutes=1000.0
    )
    empty = explain(peer_table(bare))
    assert empty.str.startswith("Too few stats recorded").all()


def test_explanation_and_flags_quote_the_same_number_and_never_disagree():
    # flagged: low minutes, and the bottom of his peers; both say 30%
    rows = _peers(minutes_share=0.30, value_change_1y=0.70)
    _, out = _explained(rows)
    assert "Plays 30% of team minutes, bottom 1%" in out.loc[0, "explanation"]
    assert "(30% of team minutes)" in out.loc[0, "flags"]
    assert "Market value -30% on a year before" in out.loc[0, "explanation"]
    assert "(value -30% on a year before)" in out.loc[0, "flags"]
    # not flagged (50% is above the 47% line) though the bottom of his peers:
    # the explanation stays silent on minutes rather than contradict the flags
    rows = _peers(minutes_share=0.50)
    _, out = _explained(rows)
    assert out.loc[0, "flag_count"] == 0
    assert "minutes" not in out.loc[0, "explanation"]


def test_small_age_bands_fall_back_to_the_position_group():
    rows = _peers(goals_per90=0.95)
    rows.loc[: MIN_PEERS - 10, "age"] = 19.0  # player 1 among 21 players under 21
    _, out = _explained(rows)
    assert "top 1% of centre forwards." in out.loc[0, "explanation"]


def test_population_and_rows_line_up_on_keys():
    rows = _peers(goals_per90=0.95)
    second = rows.iloc[[0]].assign(league_key="ucl_like", minutes=10.0)  # fewer minutes
    pop = population(pd.concat([rows, second]))
    assert len(pop) == len(rows)
    assert (pop["league_key"] == "epl").all()
    table = peer_table(pop)
    picked = for_rows(table, rows.iloc[[5, 0]])
    assert picked.index.tolist() == [5, 0]
    stray = for_rows(table, second)  # not in the population: neutral, flags from its row
    out = flags_and_explanation(stray)
    assert out["explanation"].iloc[0].startswith("Too few stats")
