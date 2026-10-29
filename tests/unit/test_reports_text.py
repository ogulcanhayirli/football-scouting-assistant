"""Generated scout reports (prompts 19 and 19c): fact sheets, the
faithfulness check (typed numbers and the claims), the judge's checkable
verdicts, the batches with a fake client (slow ones too), the token cap, the
cost and the page. No test calls the API."""

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from scout import reports_text as rt
from scout.config import (
    ApiFootballConfig,
    LeagueConfig,
    PathsConfig,
    QuotaConfig,
    ReportsConfig,
    Settings,
)
from scout.lists import DICTIONARY

CFG = ReportsConfig()


def _candidates() -> pd.DataFrame:
    frame = pd.DataFrame({c: [None, None] for c in DICTIONARY["candidates"]})
    return frame.assign(
        season=2025,
        player_id=[1, 2],
        player=["Alex Moreno", "Ben Okafor"],
        age=[20.6, 22.1],
        position=["Left Winger", "Centre-Back"],
        club_at_forecast=["Como", "Hellas Verona"],
        league=["Serie A", "Serie A"],
        current_club=["Como", "Torino"],
        current_league=["Serie A", "Serie A"],
        moved_since_forecast=[False, True],
        value_rank=[305, 1500],
        elite_decile=[10, None],
        elite_chance_pct=[30, None],
        elite_chance_players=[142, None],
        step_up_decile=[10, 4],
        step_up_chance_pct=[25, 0],
        step_up_chance_players=[878, 870],
        big_move_decile=[10, 3],
        big_move_chance_pct=[25, 5],
        big_move_chance_players=[739, 741],
        fall_back_risk_pct=[15, None],
        fall_back_risk_players=[691, None],
        flag_count=[0, 2],
        flags=[
            "none",
            "under 47% of team minutes (23% of team minutes); 5 or more matches missed injured "
            "(9 matches missed injured)",
        ],
        explanation=[
            "Gives 0.47 assists per 90 minutes, top 4% of wingers aged under 21.",
            "Wins 71% of his duels, top 3% of centre backs aged 21 to 23.",
        ],
        rising=[False, True],
        value_eur=[30e6, 1.2e6],
        fair_value_eur=[21_632_000.0, 3_000_000.0],
        ability_pct=[92.0, 81.0],
        so_far_date=pd.Timestamp("2026-10-24"),
        so_far_league=["Serie A", "Serie A"],
        so_far_matches=[4, 0],
        so_far_minutes=[175.0, 0.0],
        so_far_minutes_share=[0.24, 0.0],
        so_far_flags=["under 47% of team minutes (24% of team minutes)", "none"],
    )


def _bargains() -> pd.DataFrame:
    frame = pd.DataFrame({c: [None] for c in DICTIONARY["bargains"]})
    return frame.assign(
        season=2025,
        list="prime",
        list_rank=1,
        player_id=2,
        player="Ben Okafor",
        age=22.1,
        position="Centre-Back",
        club_at_forecast="Hellas Verona",
        league="Serie A",
        current_club="Torino",
        current_league="Serie A",
        moved_since_forecast=True,
        ability_pct=81.0,
        minutes_share=0.69,
        value_eur=1.2e6,
        fair_value_eur=3_000_000.0,
        value_to_fair=0.4,
        value_to_fair_pct=0.3,
        contract_expiry=pd.to_datetime(["2028-12-31"]),
        flag_count=2,
        flags="under 47% of team minutes (23% of team minutes); 5 or more matches missed "
        "injured (9 matches missed injured)",
        explanation="Wins 71% of his duels, top 3% of centre backs aged 21 to 23.",
        on_loan=False,
        stale_value=False,
    )


def _neighbours() -> pd.DataFrame:
    rows = []
    for pid, names in (
        (1, ["M. Jaber", "S. Vegas", "Luis Ortega"]),
        (2, ["J. Tah", "A. Bah", "K. Ito"]),
    ):
        for rank, name in enumerate(names, start=1):
            rows.append(
                {
                    "season": 2025,
                    "player_id": pid,
                    "player": "x",
                    "league_key": "serie_a",
                    "group": "winger",
                    "rank": rank,
                    "neighbour_id": 100 + rank,
                    "neighbour": name,
                    "neighbour_age": 24.5 + rank,
                    "neighbour_club": ["Saint Etienne", "Leon", "Granada"][rank - 1],
                    "neighbour_league_key": ["ligue_2", "liga_mx", "segunda"][rank - 1],
                    "neighbour_ability_pct": [93.4, 19.0, 55.0][rank - 1],
                    "similarity": 0.9 - rank / 10,
                }
            )
    return pd.DataFrame(rows)


LEAGUES = {
    "serie_a": "Serie A",
    "ligue_2": "Ligue 2",
    "liga_mx": "Liga MX",
    "segunda": "Segunda División",
}


@pytest.fixture
def sheets():
    return {
        s.player_id: s for s in rt.fact_sheets(_candidates(), _bargains(), _neighbours(), LEAGUES)
    }


GOOD = (
    "Alex Moreno is a 20-year-old left winger at Como in Serie A, still with the same club "
    "today. His ability sits in the 92nd percentile of regular players in his position. "
    "The market values him at €30m, 139% of the "
    "€21.6m our model calls fair, so he is priced dearly rather than cheaply. Of 142 past "
    "players in his elite band, 30% became one of the 100 most valuable players within 2 "
    "seasons. Of 878 past players in his step up band, 25% saw their value rank become at "
    "least 3 times better, and of 739 in his big move band, 25% joined a top 20 club within "
    "24 months. He raises no warning flags. He gives 0.47 assists per 90 minutes, top 4% of "
    "wingers aged under 21, and M. Jaber is the closest in style, though not in level."
)


# The fact sheet


def test_one_sheet_a_player_with_both_tables_merged(sheets):
    assert set(sheets) == {1, 2}
    ben = sheets[2]
    assert ben.tables == "candidates and bargains"
    assert "Bargain list: prime list (29 and under), #1" in ben.text
    assert "Club today: Torino, Serie A (he moved after the forecast)" in ben.text
    assert "Contract until: 31 December 2028" in ben.text
    assert "Elite chance: not rated" in ben.text
    assert "Step up chance: of 870 past players in the same band (decile 4 of 10), under 5%" in (
        ben.text
    )
    assert "Warning flags: 2: under 47% of team minutes (23% of team minutes)" in ben.text
    assert "Rising: yes" in ben.text
    alex = sheets[1]
    assert "Market value: €30m" in alex.text
    assert "Fair value (our model): €22m; his market value is 139% of it (above our fair" in (
        alex.text
    )
    assert "Similar players today: nearest first: M. Jaber (25, Saint Etienne, Ligue 2, " in (
        alex.text
    )
    assert alex.similar() == ["M. Jaber", "S. Vegas", "Luis Ortega"]
    assert "Ability: 92nd percentile among regular players of his position in 2025/26" in alex.text
    assert alex.template.startswith("Gives 0.47 assists")
    # Prompt 19c: the season so far is never on the sheet the report is written from.
    assert "so far" not in alex.text.lower()
    assert "175" not in alex.text
    assert "among the cheapest 0.3%" in ben.text


def test_past_the_middle_the_bargain_place_counts_from_the_dear_end():
    dear = _bargains().assign(value_to_fair=2.65, value_to_fair_pct=97.2)
    ben = rt.fact_sheets(_candidates(), dear, _neighbours(), LEAGUES)[1]
    assert "his market value is 265% of it (above our fair value), among the dearest 2.8%" in (
        ben.text
    )


def test_the_hash_follows_the_cited_facts_the_prompt_and_the_model(sheets):
    alex = sheets[1]
    again = rt.fact_sheets(_candidates(), _bargains(), _neighbours(), LEAGUES)[0]
    assert again.hash("claude-haiku-5-5") == alex.hash("claude-haiku-5-5")
    assert alex.hash("claude-haiku-5-5") != alex.hash("claude-sonnet-5-5")
    # The season so far changes with every match and is not cited (prompt 19c):
    # a new match keeps the report; a new club or a new forecast does not.
    played = _candidates().assign(so_far_minutes=[260.0, 90.0], so_far_matches=[5, 1])
    same = rt.fact_sheets(played, _bargains(), _neighbours(), LEAGUES)[0]
    assert same.hash("claude-haiku-5-5") == alex.hash("claude-haiku-5-5")
    for column, value in (("current_club", "Inter"), ("value_eur", 32e6)):
        changed = _candidates()
        changed.loc[0, column] = value
        other = rt.fact_sheets(changed, _bargains(), _neighbours(), LEAGUES)[0]
        assert other.hash("claude-haiku-5-5") != alex.hash("claude-haiku-5-5")


def test_the_prefix_is_fixed_and_cached_and_the_sheet_is_the_only_variable_part(sheets):
    a, b = rt.writer_params(CFG, sheets[1]), rt.writer_params(CFG, sheets[2])
    assert a["system"] == b["system"]
    assert a["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert a["model"] == "claude-haiku-5-5"
    assert a["messages"] == [{"role": "user", "content": sheets[1].text}]
    assert {k for k in a if a[k] != b[k]} == {"messages"}


# The check


def test_a_faithful_report_passes(sheets):
    result = rt.check(GOOD, sheets[1], gazetteer={"barcelona"})
    assert result.violations == []
    assert result.passed
    assert 120 <= result.words <= rt.WORDS_ASKED


def test_a_report_that_invents_a_club_fails(sheets):
    report = GOOD.replace("at Como in Serie A", "at Como in Serie A, on loan from Juventus")
    result = rt.check(report, sheets[1])
    assert not result.passed
    assert result.violations == ['the name "Juventus" is not in the fact sheet']


@pytest.mark.parametrize(
    ("wrong", "right"),
    [("€21.7m", "€21.6m"), ("140%", "139%"), ("0.4 assists", "0.47 assists"), ("91st", "92nd")],
)
def test_a_report_that_rounds_a_number_wrongly_fails(sheets, wrong, right):
    result = rt.check(GOOD.replace(right, wrong), sheets[1])
    assert result.violations == [
        f'the number "{wrong.split()[0]}" is not in the fact sheet, or not rounded correctly'
    ]


@pytest.mark.parametrize(
    ("rounded", "exact"), [("€22m", "€21.6m"), ("0.5 assists", "0.47 assists")]
)
def test_a_correctly_rounded_number_passes(sheets, rounded, exact):
    assert rt.check(GOOD.replace(exact, rounded), sheets[1]).passed


def test_a_similar_players_number_needs_his_name_in_the_sentence(sheets):
    # 93rd is M. Jaber's ability, not Alex Moreno's (prompt 19: left to the
    # judge; prompt 19c: a similar player's number only where he is named).
    found = rt.check(GOOD.replace("92nd", "93rd"), sheets[1]).violations
    assert found == [
        'the number "93rd" is on the fact sheet only for a similar player, and this sentence '
        "names none"
    ]
    named = GOOD.replace(
        "M. Jaber is the closest in style", "M. Jaber, ability 93rd percentile, is the closest"
    )
    assert rt.check(named, sheets[1]).violations == []


def test_units_and_counts_must_match(sheets):
    # 30% exists, but not as euros; 142 players exist, but not as minutes
    report = GOOD.replace("€30m", "€30").replace("Of 142 past players", "Of 142 minutes")
    found = rt.check(report, sheets[1]).violations
    assert 'the number "€30" is not in the fact sheet, or not rounded correctly' in found
    assert 'the number "142" is on the fact sheet, but not as a count of minutes' in found


def test_a_percentage_must_be_of_the_kind_its_words_say(sheets):
    # 25% is a chance on the sheet, not his market value against fair value.
    found = rt.check(GOOD.replace("139% of the", "25% of the"), sheets[1]).violations
    assert found == [
        'the number "25%" is on the fact sheet, but not as market value as a share of fair value'
    ]


def test_a_rate_must_go_with_the_count_of_its_own_chance(sheets):
    swapped = GOOD.replace("players in his elite band, 30% became", "players in his elite band, "
                           "25% became")  # fmt: skip
    assert rt.check(swapped, sheets[1]).violations == [
        'the rate "25%" and the count "142" are not the rate and the count of one chance on the '
        "fact sheet"
    ]
    # The step up rate and count, told as the elite chance.
    told = GOOD.replace(
        "Of 142 past players in his elite band, 30% became",
        "Of 878 past players in his elite band, 25% became",
    )
    assert rt.check(told, sheets[1]).violations == [
        'the rate "25%" of 878 past players is not the elite chance'
    ]


def test_only_the_first_similar_player_is_the_closest(sheets):
    wrong = GOOD.replace("M. Jaber is the closest", "S. Vegas is the closest")
    assert rt.check(wrong, sheets[1]).violations == [
        "the report calls S. Vegas the closest in playing style, but the sheet lists the similar "
        "players nearest first and M. Jaber is the first"
    ]
    both = GOOD.replace("M. Jaber is the closest", "M. Jaber and S. Vegas are the closest")
    assert rt.check(both, sheets[1]).passed


def test_a_first_name_added_to_an_initial_fails(sheets):
    # Run 9: "Harry Armstrong" for H. Armstrong passed, as a possible ordinary
    # word opening a sentence; 334 of 2,733 passed reports had one.
    report = GOOD.replace(
        "and M. Jaber is the closest in style, though not in level.",
        "and nobody else is listed. Mohamed Jaber is the closest in style, though not in level.",
    )
    assert rt.unknown_names(report, sheets[1]) == ["Mohamed Jaber"]
    assert rt.given_names("He is like M. Jaber. The Jaber style.", sheets[1]) == []


def test_cheap_is_below_fair_value_and_dear_above(sheets):
    report = GOOD.replace("priced dearly rather than cheaply", "priced cheaply")
    assert rt.check(report, sheets[1]).violations == [
        'the report says "cheaply", but his market value is 139% of his fair value (below 100% is '
        "cheap, above it dear)"
    ]
    assert rt.check(GOOD.replace("dearly rather than cheaply", "dearly, not cheaply"),
                    sheets[1]).passed  # fmt: skip


def test_a_player_above_fair_value_is_on_the_ranking_not_a_bargain_list():
    dear = _bargains().assign(value_to_fair=1.5, value_to_fair_pct=48.0, list_rank=550)
    ben = rt.fact_sheets(_candidates(), dear, _neighbours(), LEAGUES)[1]
    assert "Bargain list" not in ben.text
    assert (
        "Value ranking: prime list (29 and under) of the ranking by value against fair value, "
        "#550; priced above his fair value, so not a bargain"
    ) in ben.text
    assert ben.fact("bargain_list").label == "Value ranking"
    flagged = dear.assign(below_fair_value=False)
    assert rt.fact_sheets(_candidates(), flagged, _neighbours(), LEAGUES)[1].text == ben.text
    assert "never call him one" in rt.WRITER_INSTRUCTIONS


def test_the_word_bargain_only_for_a_player_below_fair_value(sheets):
    said = GOOD.replace(
        "He raises no warning flags.", "He raises no warning flags and is a bargain."
    )
    assert rt.check(said, sheets[1]).violations == [
        'the report says "bargain", but he has a market value 139% of his fair value: only a '
        "player priced below his fair value is called a bargain"
    ]
    for fine in ("is not a bargain", "is no bargain"):
        report = GOOD.replace(
            "He raises no warning flags.", f"He raises no warning flags and {fine}."
        )
        assert rt.check(report, sheets[1]).passed
    assert rt.bargain_violations("He is on the prime bargain list at #1.", sheets[2]) == []


def test_a_kept_report_that_calls_a_dear_player_a_bargain_is_written_again(sheets):
    sheet = sheets[1]
    same = _stored("passed", 0, fact_hash=sheet.hash(CFG.writer_model))
    assert len(rt.plan([sheet], same.assign(report=GOOD), CFG.writer_model)[0]) == 0
    bad = same.assign(report=GOOD + " He is a bargain on the prime list.")
    assert len(rt.plan([sheet], bad, CFG.writer_model)[0]) == 1


def test_the_season_so_far_and_other_seasons_stay_out_of_the_report(sheets):
    report = GOOD.replace(
        "He raises no warning flags.", "He raises no warning flags. So far in 2026/27 he is fit."
    )
    assert rt.check(report, sheets[1]).violations == [
        "the report mentions the season so far, which is not in the fact sheet (the page shows "
        "it next to the report); describe only the season of the sheet",
        'the season "2026/27" is not the season of the fact sheet',
    ]
    assert rt.check(GOOD.replace("players in his position", "players in 2025/26"),
                    sheets[1]).passed  # fmt: skip


@pytest.mark.parametrize(
    "written", ["N'Guessan", "N\u2019Guessan", "N'Guessan's", "N\u02bcGuessan"]
)
def test_apostrophes_in_names_are_one_mark_and_never_split_a_name(written):
    # Run 9 sent A. N'Guessan's own report to the template: "N'Guessan" folded
    # to "n guessan", which no name of his sheet was.
    cand = _candidates()
    cand.loc[0, "player"] = "A. N'Guessan"
    sheet = rt.fact_sheets(cand, _bargains(), _neighbours(), LEAGUES)[0]
    assert rt.fold(written) == "nguessan"
    assert rt.unknown_names(f"Then {written} scored. A. {written} is a winger.", sheet) == []
    spans = rt.capital_spans(f"He says A. {written} plays.")
    assert spans[-1] == f"A. {written}".removesuffix("'s")


def test_a_number_the_report_computed_fails(sheets):
    report = GOOD.replace("so he is priced dearly", "about €8.4m more than fair, so priced dearly")
    found = rt.check(report, sheets[1]).violations
    assert 'the number "€8.4m" is not in the fact sheet, or not rounded correctly' in found


def test_the_length_is_checked_between_90_and_170_words(sheets):
    # A faithful report a few words over the 150 asked for passes (prompt 19b).
    over = GOOD + " He is quick." * 7
    assert rt.check(over, sheets[1]).words == 162
    assert rt.check(over, sheets[1]).passed
    found = rt.check(GOOD + " He is quick." * 10, sheets[1]).violations
    assert found == ["the report has 171 words; it must have 90 to 170"]
    assert rt.check("Alex Moreno plays for Como.", sheets[1]).violations == [
        "the report has 5 words; it must have 90 to 170"
    ]


def test_the_word_count_is_never_above_a_plain_whitespace_count():
    # Diagnosis of the first real run: the reports were long, not miscounted.
    for text in [GOOD, "He is 20 — a winger at Como (€30m) - top 4%.", "Of 142, 30% did."]:
        assert rt.words(text) <= len(text.split())
    assert rt.words(GOOD) == len(GOOD.split())


def test_the_prompt_asks_for_about_130_words_with_an_example_of_that_length():
    text = rt.WRITER_INSTRUCTIONS
    assert "about 130 words, never more than 150" in text
    example = text.split("use only your own fact sheet.\n\n")[1].split("\n\nThe fact sheet")[0]
    assert rt.WORDS_TARGET <= rt.words(example) <= 140
    assert f"{rt.words(example)} words" in text


def test_the_length_note_gives_the_count_and_the_cut():
    assert rt.length_note(177) == (
        "your report has 177 words; the limit is 150: cut at least 27 words, and aim for about "
        "130 (cut about 47) by dropping the least telling facts, not by squeezing sentences"
    )
    assert rt.length_note(140) is None
    assert rt.length_note(60) == "your report has 60 words; write about 130 (add about 70)"


def test_the_writer_has_room_to_think_without_truncating():
    assert CFG.writer_max_tokens >= 6000


def test_ordinary_words_opening_a_sentence_are_not_names(sheets):
    report = GOOD.replace(
        "He raises no warning flags.", "Strong numbers. Despite that, he raises none."
    )
    assert rt.unknown_names(report, sheets[1]) == []


def test_a_known_name_opening_a_sentence_is_checked(sheets):
    report = GOOD.replace("He raises no warning flags.", "Barcelona noticed him.")
    assert rt.unknown_names(report, sheets[1], gazetteer={"barcelona"}) == ["Barcelona"]
    assert rt.unknown_names(report, sheets[1]) == []  # unknown: read as an ordinary word


def test_initials_numbers_in_names_and_possessives(sheets):
    alex = sheets[1]
    ok = "M. Jaber of Saint Etienne in Ligue 2 is alike. Como's winger. Serie A. He is fine."
    assert rt.unknown_names(ok, alex) == []
    assert rt.unknown_names("He is alike to M. Jaber of Ligue 1 in style.", alex) == ["Ligue 1"]
    assert rt.unknown_names("A Como-based winger, unlike Real Madrid players.", alex) == [
        "Real Madrid"
    ]


def test_numbers_are_read_with_units_tags_and_precision():
    found = {
        n.text: n for n in rt.extract_numbers("a 20-year-old, €21.6m, -25%, 1,811 minutes, 92nd")
    }
    assert found["€21.6m"].value == pytest.approx(21.6e6)
    assert found["€21.6m"].step == pytest.approx(1e5)
    assert found["20"].tag == "age"
    assert (found["-25%"].unit, found["-25%"].value) == ("pct", -25.0)
    assert (found["1,811"].value, found["1,811"].tag) == (1811.0, "minutes")
    assert found["92nd"].unit == "ordinal"


# The batches, with a fake client


def _message(text: str, usage: dict | None = None, stop: str = "end_turn"):
    u = {"input_tokens": 500, "output_tokens": 300, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 1200, **(usage or {})}  # fmt: skip
    created = SimpleNamespace(
        ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=u["cache_creation_input_tokens"]
    )
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        usage=SimpleNamespace(**u, cache_creation=created),
    )


class FakeBatches:
    """The Message Batches surface the module uses; ``respond(custom_id,
    params)`` gives a reply's text (or None for an errored request).
    ``slow``: seconds a batch takes, by its order of creation, on ``clock``;
    the others end at once."""

    def __init__(self, respond, slow=None, clock=None):
        self.respond = respond
        self.sent: list[list[dict]] = []
        self.results_by_id = {}
        self.slow = slow or {}
        self.clock = clock or (lambda: 0.0)
        self.ends = {}

    def create(self, requests):
        batch_id = f"msgbatch_{len(self.sent)}"
        self.sent.append(requests)
        items = []
        for r in requests:
            text = self.respond(r["custom_id"], r["params"])
            if text is None:
                result = SimpleNamespace(type="errored", error=SimpleNamespace(type="api_error"))
            else:
                result = SimpleNamespace(type="succeeded", message=_message(text))
            items.append(SimpleNamespace(custom_id=r["custom_id"], result=result))
        self.results_by_id[batch_id] = items
        self.ends[batch_id] = self.clock() + self.slow.get(len(self.sent) - 1, 0)
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        ended = self.clock() >= self.ends[batch_id]
        return SimpleNamespace(id=batch_id, processing_status="ended" if ended else "in_progress")

    def results(self, batch_id):
        assert self.clock() >= self.ends[batch_id], "results of a batch still running"
        return iter(self.results_by_id[batch_id])

    def cancel(self, batch_id):
        raise AssertionError("no batch should be cancelled here")


def _client(respond, slow=None, clock=None):
    batches = FakeBatches(respond, slow, clock)
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


class Clock:
    """A fake monotonic clock that the runner's sleep moves on."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 1.0)


def _runner(client):
    return rt.Batches(client, poll_seconds=0, deadline=float("inf"), sleep=lambda s: None)


def test_a_failing_report_goes_back_with_its_violations_then_passes(sheets):
    def respond(custom_id, params):
        if custom_id.startswith("w0-"):
            return GOOD.replace("at Como in Serie A", "at Como in Serie A, on loan from Juventus")
        return GOOD

    client, batches = _client(respond)
    usage = {}
    out = rt.write_reports([sheets[1]], CFG, _runner(client), rt.Budget(10**9), usage=usage)
    o = out[1]
    assert (o.status, o.source, o.attempts, o.first_try) == ("passed", "model", 2, False)
    retry = batches.sent[1][0]["params"]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert 'the name "Juventus" is not in the fact sheet' in retry[2]["content"]
    assert usage["output_tokens"] == 600


def test_after_two_retries_the_template_is_shown_and_the_failure_logged(sheets, caplog):
    client, batches = _client(lambda cid, p: GOOD.replace("€30m", "€35m"))
    with caplog.at_level(logging.WARNING):
        out = rt.write_reports([sheets[1]], CFG, _runner(client), rt.Budget(10**9))
    o = out[1]
    assert (o.status, o.source, o.attempts) == ("failed", "template", 3)
    assert o.report == sheets[1].template
    assert o.violations == ['the number "€35m" is not in the fact sheet, or not rounded correctly']
    assert len(batches.sent) == 3  # the first try and two retries, no more
    assert "failed the check 3 times" in caplog.text


def test_an_errored_request_is_retried_from_the_sheet(sheets):
    client, batches = _client(lambda cid, p: None if cid.startswith("w0-") else GOOD)
    out = rt.write_reports([sheets[1]], CFG, _runner(client), rt.Budget(10**9))
    assert out[1].status == "passed"
    assert len(batches.sent[1][0]["params"]["messages"]) == 1


def test_the_token_cap_holds_requests_back_and_is_never_passed(sheets):
    one = rt.worst_case(rt.writer_params(CFG, sheets[1]))
    budget = rt.Budget(cap=one + 10)
    client, batches = _client(lambda cid, p: GOOD)
    out = rt.write_reports([sheets[1], sheets[2]], CFG, _runner(client), budget)
    assert [len(b) for b in batches.sent] == [1]
    assert (out[2].status, out[2].source) == ("held", "template")
    assert budget.used <= budget.cap
    assert (
        rt.Budget(cap=one - 1).take(
            [{"custom_id": "x", "params": rt.writer_params(CFG, sheets[1])}]
        )[0]
        == []
    )


def test_the_worst_case_bounds_the_real_input(sheets):
    params = rt.writer_params(CFG, sheets[2])
    text = json.dumps([params["system"], params["messages"]], ensure_ascii=False)
    assert rt.estimate_tokens(params) >= len(text) / 2.5  # far above a real tokenizer's count


def test_cost_comes_from_the_usage_fields():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "cache_read_input_tokens": 1_000_000, "cache_write_1h": 1_000_000}  # fmt: skip
    # input 0.10, cache read 0.01, one hour cache write 0.20, output 0.50; half price in a batch
    assert rt.cost_usd(usage, "claude-haiku-5-5", CFG) == pytest.approx(0.405)
    assert rt.cost_usd(usage, "claude-haiku-5-5", CFG, batch=False) == pytest.approx(0.81)
    assert rt.cost_usd({"input_tokens": 10**6}, "claude-sonnet-5-5", CFG) == pytest.approx(1.0)
    got = rt.usage_of(_message("x", {"cache_creation_input_tokens": 70}))
    assert got["cache_write_1h"] == 70
    assert rt.billed_tokens(got) == 500 + 300 + 70 + 1200


def test_the_judge_reads_field_names_and_its_verdicts_are_checked(sheets):
    verdict = {
        "faithful": False,
        "issues": [
            {"span": "on loan from Juventus", "field": "none", "problem": "unsupported",
             "explanation": "No loan is on the sheet."},
        ],
        "hedging_adequate": True,
        "readability": 4,
        "reason": "Juventus is invented.",
    }  # fmt: skip
    report = GOOD.replace("at Como in Serie A", "at Como in Serie A, on loan from Juventus")
    client, batches = _client(lambda cid, p: json.dumps(verdict))
    got = rt.judge_reports([(sheets[1], report)], CFG, _runner(client), rt.Budget(10**9))
    params = batches.sent[0][0]["params"]
    assert params["model"] == "claude-sonnet-5-5"
    assert params["output_config"]["format"]["schema"] == rt.VERDICT_SCHEMA
    content = params["messages"][0]["content"]
    assert "[market_value] Market value: €30m" in content
    assert "[similar_players] Similar players today: nearest first: M. Jaber" in content
    assert "Report:\n" + report in content
    v = got[1]
    assert (v.faithful, v.offending_span, v.checkable) == (False, "on loan from Juventus", True)
    assert v.issues[0]["field"] == "none"
    assert rt.parse_verdict('{"faithful": true}') is None
    assert rt.parse_verdict("not json") is None


def test_a_verdict_whose_span_or_field_does_not_hold_is_set_aside(sheets):
    def verdict(faithful, issues):
        return rt.Verdict(faithful, "", True, 4, "", issues)

    def issue(span, field, problem="contradicts"):
        return {"span": span, "field": field, "problem": problem, "explanation": "x"}

    text = sheets[1].text
    ok = verdict(False, [issue("S. Vegas is the closest", "similar_players")])
    assert rt.verify(ok, "Then S. Vegas  is the closest.", text) == []
    assert rt.verify(verdict(False, [issue("Harry Kane", "player")]), GOOD, text) == [
        'the span "Harry Kane" is not in the report'
    ]
    assert rt.verify(verdict(False, [issue("Alex Moreno", "minutes")]), GOOD, text) == [
        'the field "minutes" is not a line of the sheet'
    ]
    assert rt.verify(verdict(False, []), GOOD, text) == ["not faithful, yet it lists no issue"]
    assert rt.verify(verdict(True, [issue("Alex Moreno", "player")]), GOOD, text) == [
        "faithful, yet it lists issues"
    ]
    assert rt.verify(verdict(False, [issue("Alex Moreno", "none")]), GOOD, text) == [
        "it contradicts no field"
    ]
    assert set(rt.ISSUE_SCHEMA["properties"]["field"]["enum"]) == {*rt.FIELD_KEYS, "none"}
    assert rt.sheet_keys(text) <= set(rt.FIELD_KEYS)


def test_the_judge_reads_the_labels_as_the_writer_was_told():
    # Run 9's judge faulted "elite band", "dearly" above fair value and the
    # first similar player called the closest: the writer's own rules.
    for text in (rt.WRITER_INSTRUCTIONS, rt.JUDGE_INSTRUCTIONS):
        assert rt.LABEL_MEANINGS in text
        assert "{labels}" not in text
    assert "the first one is the closest" in rt.LABEL_MEANINGS
    assert "never Harry Armstrong" in rt.WRITER_INSTRUCTIONS


# The run, the storage and the page


def _settings(tmp_path) -> Settings:
    settings = Settings(
        api_football=ApiFootballConfig(quota=QuotaConfig(daily_limit=100, per_minute_limit=10)),
        leagues=[
            LeagueConfig(key="serie_a", name="Serie A", api_football_id=135, transfermarkt_id="IT1")
        ],
        paths=PathsConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            state_dir=tmp_path / "state",
            reports_dir=tmp_path / "reports",
        ),
    )
    lists = settings.paths.processed_dir / "lists"
    lists.mkdir(parents=True)
    _candidates().to_parquet(lists / "candidates.parquet")
    _bargains().to_parquet(lists / "bargains.parquet")
    emb = settings.paths.processed_dir / "embeddings"
    emb.mkdir()
    _neighbours().to_parquet(emb / "neighbours.parquet")
    return settings


VERDICT = {"faithful": True, "issues": [], "hedging_adequate": True,
           "readability": 5, "reason": "Clear and supported."}  # fmt: skip


def _respond(cid, params):
    if cid.startswith("j-"):
        return json.dumps(VERDICT)
    sheet = params["messages"][0]["content"]
    return GOOD if "Alex Moreno" in sheet else "Ben Okafor is a centre-back at Torino."


def test_the_run_writes_checks_judges_stores_and_reuses(tmp_path):
    settings = _settings(tmp_path)
    client, batches = _client(_respond)
    now = pd.Timestamp("2026-10-26 05:30", tz="UTC")
    summary = rt.run_reports(settings, client=client, now=now)
    assert (summary["written"], summary["reused"]) == (2, 0)
    assert (summary["passed_first_try"], summary["failed_after_retries"]) == (1, 1)
    assert (summary["judged"], summary["judge_faithful"]) == (1, 1)
    assert summary["cost_usd"] > 0
    assert summary["tokens_used"] <= summary["token_cap"]
    folder = settings.paths.processed_dir / "reports_text"
    table = pd.read_parquet(folder / "reports.parquet")
    assert set(table.columns) == set(rt.REPORT_COLUMNS)
    alex, ben = table.set_index("player_id").loc[1], table.set_index("player_id").loc[2]
    assert (alex["source"], alex["status"], bool(alex["judge_faithful"])) == (
        "model",
        "passed",
        True,
    )
    assert (ben["source"], ben["status"], ben["report"]) == ("template", "failed", ben["report"])
    assert ben["report"].startswith("Wins 71% of his duels")
    assert "the name" in ben["violations"] or "words" in ben["violations"]
    assert json.loads((folder / "summary.json").read_text())["judged"] == 1
    assert json.loads((folder / rt.PENDING).read_text()) == {"batches": []}
    assert bool(alex["judge_checkable"])
    assert json.loads(alex["judge_issues"]) == []
    kept = json.loads((folder / "verdicts.jsonl").read_text().splitlines()[0])
    assert kept["fact_sheet"].startswith("Player: Alex Moreno")
    assert len((folder / "failures.jsonl").read_text().splitlines()) == 1
    assert len((folder / "verdicts.jsonl").read_text().splitlines()) == 1
    md = (settings.paths.reports_dir / "reports_text.md").read_text()
    assert "## For the owner: 1 judged reports" in md
    assert "Alex Moreno" in md
    sample = pd.read_csv(settings.paths.reports_dir / "judge_sample.csv")
    assert list(sample.columns) == rt.AGREEMENT_COLUMNS
    assert len(sample) == 1

    assert (ben["prompt_version"], ben["template_runs"]) == (rt.PROMPT_VERSION, 1)
    assert (alex["prompt_version"], alex["template_runs"]) == (rt.PROMPT_VERSION, 0)
    # The retry was told its word count and what to add.
    retry = batches.sent[1][0]["params"]["messages"][2]["content"]
    assert "your report has 7 words; write about 130 (add about 123)" in retry
    assert "it must have 90 to 170" not in retry
    first = json.loads((folder / "summary.json").read_text())
    assert first["words_first_try"]["max"] == 141
    assert (first["over_asked_first_try"], first["outside_check_first_try"]) == (0, 1)

    # A week on with the same facts: the passed report is reused; the one that
    # ended on the template is asked for once more, and fails again.
    sent = len(batches.sent)
    again = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=7))
    assert (again["written"], again["reused"], again["judged"]) == (1, 1, 0)
    assert len(batches.sent) == sent + 3
    ben = pd.read_parquet(folder / "reports.parquet").set_index("player_id").loc[2]
    assert (ben["status"], ben["template_runs"]) == ("failed", 2)

    # After two runs on the template it is not asked for again under this prompt.
    sent = len(batches.sent)
    third = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=14))
    assert (third["written"], third["reused"], third["judged"]) == (0, 2, 0)
    assert len(batches.sent) == sent
    assert third["cost_usd"] == 0


COMPARED = ["player_id", "report", "source", "status", "attempts", "first_try", "violations",
            "template_runs", "fact_hash", "judge_faithful", "judge_checkable", "judge_issues",
            "judge_readability"]  # fmt: skip


def _table(settings) -> pd.DataFrame:
    path = settings.paths.processed_dir / "reports_text" / "reports.parquet"
    return pd.read_parquet(path)[COMPARED].sort_values("player_id").reset_index(drop=True)


def test_a_slow_writer_batch_is_collected_next_run_and_ends_as_a_fast_one(tmp_path):
    # Run 12 waited its full 100 minutes on one batch (prompt 19c). Now the
    # step waits 20; a batch still running is kept and the next run collects it.
    now = pd.Timestamp("2026-11-06 05:30", tz="UTC")
    fast = _settings(tmp_path / "fast")
    client, _ = _client(_respond)
    rt.run_reports(fast, client=client, now=now)

    slow = _settings(tmp_path / "slow")
    clock = Clock()
    client, batches = _client(_respond, slow={0: 30 * 60}, clock=clock)
    first = rt.run_reports(slow, client=client, now=now, sleep=clock.sleep, clock=clock)
    assert 20 * 60 <= clock.t <= 21 * 60  # waited 20 minutes, not 100
    assert (first["pending_players"], first["batches_pending"]) == (2, ["msgbatch_0"])
    assert (first["written"], first["judged"]) == (0, 0)
    folder = slow.paths.processed_dir / "reports_text"
    waiting = json.loads((folder / rt.PENDING).read_text())["batches"]
    assert [(b["id"], b["kind"], len(b["requests"])) for b in waiting] == [
        ("msgbatch_0", "writer", 2)
    ]
    rows = _table(slow)
    assert list(rows["status"]) == ["pending", "pending"]
    assert list(rows["source"]) == ["template", "template"]
    assert len(batches.sent) == 1  # nothing more was sent while it ran

    # The next day the batch has ended: collected first, checked, the failing
    # report retried in this run's rounds, the sample judged.
    clock.t += 24 * 3600
    second = rt.run_reports(slow, client=client, now=now + pd.Timedelta(days=1),
                            sleep=clock.sleep, clock=clock)  # fmt: skip
    assert (second["collected"], second["pending_players"], second["batches_pending"]) == (2, 0, [])
    assert batches.sent[1][0]["custom_id"] == "w1-2"  # Ben's first retry: no new first try
    assert (second["judged"], second["judge_faithful"]) == (1, 1)
    pd.testing.assert_frame_equal(_table(slow), _table(fast))
    assert json.loads((folder / rt.PENDING).read_text()) == {"batches": []}

    # Reused as a fast one is: no request for Alex Moreno, his row unchanged.
    sent = len(batches.sent)
    rt.run_reports(slow, client=client, now=now + pd.Timedelta(days=8), sleep=clock.sleep,
                   clock=clock)  # fmt: skip
    asked = [r["custom_id"] for batch in batches.sent[sent:] for r in batch]
    assert not any(cid.endswith("-1") for cid in asked)
    assert _table(slow).iloc[0]["report"] == GOOD


def test_a_batch_still_running_at_the_next_run_is_left_alone(tmp_path):
    settings = _settings(tmp_path)
    clock = Clock()
    client, batches = _client(_respond, slow={0: 50 * 3600}, clock=clock)
    now = pd.Timestamp("2026-11-06 05:30", tz="UTC")
    rt.run_reports(settings, client=client, now=now, sleep=clock.sleep, clock=clock)
    clock.t += 3600
    again = rt.run_reports(settings, client=client, now=now + pd.Timedelta(hours=1),
                           sleep=clock.sleep, clock=clock)  # fmt: skip
    assert len(batches.sent) == 1  # its players are not asked for twice
    assert (again["pending_players"], again["batches_pending"]) == (2, ["msgbatch_0"])
    assert list(_table(settings)["status"]) == ["pending", "pending"]


def test_a_collected_reply_whose_facts_changed_is_dropped(tmp_path):
    settings = _settings(tmp_path)
    clock = Clock()
    client, batches = _client(_respond, slow={0: 30 * 60}, clock=clock)
    now = pd.Timestamp("2026-11-06 05:30", tz="UTC")
    rt.run_reports(settings, client=client, now=now, sleep=clock.sleep, clock=clock)
    lists = settings.paths.processed_dir / "lists"
    _candidates().assign(current_club=["Inter", "Torino"]).to_parquet(lists / "candidates.parquet")
    clock.t += 24 * 3600
    summary = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=1),
                             sleep=clock.sleep, clock=clock)  # fmt: skip
    assert (summary["collected"], summary["collected_stale"]) == (1, 1)
    # Alex Moreno written again from his new sheet, next to Ben Okafor's retry
    assert [r["custom_id"] for r in batches.sent[1]] == ["w1-2", "w0-1"]


def test_a_slow_judge_batch_gives_its_verdicts_to_the_next_run(tmp_path):
    settings = _settings(tmp_path)
    clock = Clock()
    # batch 0: the writer, batch 1 and 2: Ben's retries, batch 3: the judge
    client, batches = _client(_respond, slow={3: 30 * 60}, clock=clock)
    now = pd.Timestamp("2026-11-06 05:30", tz="UTC")
    first = rt.run_reports(settings, client=client, now=now, sleep=clock.sleep, clock=clock)
    assert (first["judged"], first["batches_pending"]) == (0, ["msgbatch_3"])
    clock.t += 24 * 3600
    second = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=1),
                            sleep=clock.sleep, clock=clock)  # fmt: skip
    assert (second["judged"], second["judge_faithful"], second["judge_due"]) == (1, 1, False)
    table = pd.read_parquet(settings.paths.processed_dir / "reports_text" / "reports.parquet")
    alex = table.set_index("player_id").loc[1]
    assert pd.Timestamp(alex["judged_at"]) == now  # judged when the sample was drawn
    # Ben Okafor asked for once more (three attempts), no second sample
    assert len(batches.sent) == 7
    assert not any(r["custom_id"].startswith("j-") for b in batches.sent[4:] for r in b)


def test_the_judge_samples_once_a_week_however_often_the_run(tmp_path):
    settings = _settings(tmp_path)
    client, _ = _client(_respond)
    now = pd.Timestamp("2026-11-06 05:30", tz="UTC")
    assert rt.run_reports(settings, client=client, now=now)["judge_due"]
    lists = settings.paths.processed_dir / "lists"
    _candidates().assign(current_club=["Inter", "Torino"]).to_parquet(lists / "candidates.parquet")
    daily = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=1))
    assert (daily["written"], daily["judge_due"], daily["judged"]) == (2, False, 0)
    weekly = rt.run_reports(settings, client=client, now=now + pd.Timedelta(days=6, hours=-0.5))
    assert (weekly["judge_due"], weekly["judged"]) == (True, 1)


def _stored(status, runs, version=rt.PROMPT_VERSION, fact_hash="old"):
    return pd.DataFrame([{"player_id": 1, "status": status, "template_runs": runs,
                          "prompt_version": version, "model": CFG.writer_model,
                          "fact_hash": fact_hash}])  # fmt: skip


def test_the_reuse_rule(sheets):
    sheet = sheets[1]
    same = sheet.hash(CFG.writer_model)

    def todo(stored):
        return len(rt.plan([sheet], stored, CFG.writer_model, 2)[0])

    assert todo(_stored("passed", 0, fact_hash=same)) == 0  # unchanged: reused
    assert todo(_stored("passed", 0)) == 1  # facts changed: written again
    assert todo(_stored("failed", 1, fact_hash=same)) == 1  # one more try, facts unchanged
    assert todo(_stored("failed", 2)) == 0  # given up, even with changed facts
    assert todo(_stored("failed", 2, version="19.1")) == 1  # a new prompt starts over
    assert todo(_stored("failed", None, version=None)) == 1  # run 8's rows had no count
    assert todo(_stored("held", 0, fact_hash=same)) == 1
    other_model = _stored("failed", 2).assign(model="claude-sonnet-5-5")
    assert todo(other_model) == 1  # a new writer model starts over too


def test_without_the_key_one_line_is_logged_and_nothing_runs(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.INFO, logger="scout.reports_text"):
        assert rt.run_reports(_settings(tmp_path)) == {}
    assert len(caplog.records) == 1
    assert "ANTHROPIC_API_KEY is not set" in caplog.records[0].message
    assert rt.cost_line({}) == "Reports: skipped (no ANTHROPIC_API_KEY)."


def test_the_page_shows_a_report_only_while_its_facts_are_unchanged(tmp_path):
    settings = _settings(tmp_path)
    client, _ = _client(_respond)
    rt.run_reports(settings, client=client, now=pd.Timestamp("2026-10-26", tz="UTC"))
    agreement = tmp_path / "agreement.csv"
    block = rt.page_reports(settings, _candidates(), _bargains(), agreement)
    assert block["rows"]["1"] == ["report", GOOD]
    assert block["rows"]["2"][0] == "template"
    assert block["label"] == (
        "written by Claude Haiku 5.5 from the facts above and checked automatically; "
        "odds are historical rates"
    )
    assert block["agreement"] is None
    changed = _candidates()
    changed.loc[0, "value_eur"] = 32e6
    assert rt.page_reports(settings, changed, _bargains(), agreement)["rows"]["1"][0] == "template"


def test_the_owners_agreement_counts_yes_and_no_only(tmp_path):
    path = tmp_path / "agreement.csv"
    assert rt.agreement(path) is None
    rows = pd.DataFrame(
        [
            {"player_id": 1, "owner_agrees": "yes"},
            {"player_id": 2, "owner_agrees": "No"},
            {"player_id": 3, "owner_agrees": None},
        ],
        columns=rt.AGREEMENT_COLUMNS,
    )
    rows.to_csv(path, index=False)
    assert rt.agreement(path) == {"read": 2, "agreed": 1}


def test_the_committed_agreement_file_has_the_columns_the_page_reads():
    table = rt.load_agreement()
    assert list(table.columns) == rt.AGREEMENT_COLUMNS


def test_the_dictionary_names_every_stored_column():
    words = rt.dictionary()
    assert list(words["column"]) == list(rt.REPORT_COLUMNS)
    assert set(words["table"]) == {"reports"}
    assert words["meaning"].str.len().gt(20).all()


def test_the_judge_test_scores_the_hand_sorting(tmp_path):
    settings = _settings(tmp_path)
    client, _ = _client(_respond)
    rt.run_reports(settings, client=client, now=pd.Timestamp("2026-11-06", tz="UTC"))
    folder = settings.paths.processed_dir / "reports_text"
    table = pd.read_parquet(folder / "reports.parquet").set_index("player_id")
    rows = [
        {"player_id": 1, "player": "Alex Moreno", "fact_hash": table.at[1, "fact_hash"][:12],
         "category": "real_error", "field": "similar_players", "span": "M. Jaber is the closest",
         "note": ""},
        {"player_id": 2, "player": "Ben Okafor", "fact_hash": "nothere", "category": "real_error",
         "field": "player", "span": "x", "note": ""},
    ]  # fmt: skip
    sorting = tmp_path / "sorting.csv"
    pd.DataFrame(rows).to_csv(sorting, index=False)
    found = {"faithful": False, "hedging_adequate": True, "readability": 4, "reason": "r",
             "issues": [{"span": "M. Jaber is the closest in style", "field": "similar_players",
                         "problem": "contradicts", "explanation": "e"}]}  # fmt: skip
    judge, batches = _client(lambda cid, p: json.dumps(found))
    summary = rt.judge_test(settings, client=judge, sorting=sorting)
    assert batches.sent[0][0]["custom_id"] == "t-1"
    assert "[player] Player: Alex Moreno" in batches.sent[0][0]["params"]["messages"][0]["content"]
    assert (summary["scored"], summary["agreed"], summary["not_found"]) == (1, 1, ["Ben Okafor"])
    md = (settings.paths.reports_dir / "judge_test.md").read_text()
    assert "| Alex Moreno | real_error | similar_players | not faithful |" in md


def test_the_committed_sorting_has_the_columns_the_judge_test_reads():
    table = pd.read_csv(rt.SORTING)
    assert set(table["category"]) == {"real_error", "judge_mistake", "sheet_gap"}
    assert (table.loc[table["category"] == "real_error", "span"].str.len() > 0).all()
    assert set(table["field"]) <= set(rt.FIELD_KEYS)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (999_999.9999999997, "€1m"),  # float noise in 52 bargains' values (prompt 20i)
        (999_499.0, "€999k"),
        (1_000_000.0, "€1m"),
        (1_300_000.0, "€1.3m"),
        (21_600_000.0, "€22m"),
        (750_000.0, "€750k"),
    ],
)
def test_money_follows_the_rounded_number(value, text):
    assert rt.money(value) == text
