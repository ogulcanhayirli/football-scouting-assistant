"""The static page and the weekly publish."""

import gzip
import json
import re
import shutil
import subprocess

import pandas as pd
import pytest

from scout.lists import DICTIONARY, dictionary
from scout.publish import (
    LISTS_PREFIX,
    REPORTS_PREFIX,
    SITE_FILES,
    check_lists,
    keep_reports,
    publish,
    publish_site,
    website_url,
)
from scout.site import BARGAIN_COLUMNS, CANDIDATE_COLUMNS, flag_keys, page_data, render


def _candidates():
    frame = pd.DataFrame({c: [None, None] for c in DICTIONARY["candidates"]})
    frame = frame.assign(
        season=2025,
        player_id=[1, 2],
        player=["Alex </script><b>x</b>", "Ben"],
        age=[20.6, 22.1],
        position=["Left Winger", "Centre-Back"],
        club=["Como", "Club"],
        league=["Serie A", "Süper Lig"],
        value_rank=[305.0, 1500.0],
        elite_decile=[10.0, None],
        elite_chance_pct=[30.0, None],
        elite_chance_players=[142.0, None],
        fall_back_risk_pct=[20.0, None],
        fall_back_risk_players=[418.0, None],
        flag_count=[1, 0],
        flags=["under 47% of team minutes (17% of team minutes)", "none"],
        explanation=["Gives 0.47 assists per 90 minutes.", "No stat stands out."],
        rising=[False, True],
        value_eur=[30e6, 2e6],
        fair_value_eur=[22e6, None],
        ability_pct=[92.0, 60.0],
    )
    return frame


def _bargains():
    frame = pd.DataFrame({c: [None] for c in DICTIONARY["bargains"]})
    return frame.assign(
        season=2025,
        list="prime",
        list_rank=1,
        player_id=2,
        player="Ben",
        value_eur=1e6,
        fair_value_eur=2e6,
        value_to_fair=0.5,
        value_to_fair_pct=0.3,
        minutes_share=0.6,
        flag_count=0,
        flags="none",
        explanation="No stat stands out.",
        on_loan=False,
        stale_value=False,
        contract_expiry=pd.to_datetime(["2028-06-30"]),
        value_date=pd.to_datetime(["2026-06-01"]),
    )


def _data():
    return page_data(
        _candidates(),
        _bargains(),
        dictionary(),
        rising={2},
        feature_dates={"2026-01-01": ["J1 League"], "2026-06-30": ["Serie A"]},
        valuation_end=pd.Timestamp("2026-06-12"),
        latest_value_date=pd.Timestamp("2026-06-11"),
        built=pd.Timestamp("2026-10-21"),
        forecasts=[{"date": "2026-10-21", "season": 2025, "revisions": 2}],
        tracked=37,
    )


def test_flag_keys_name_the_raised_flags():
    keys = flag_keys(
        pd.Series(
            [
                "under 47% of team minutes (17% of team minutes); "
                "value lower than a year before (value -25% on a year before)",
                "none",
                None,
            ]
        )
    )
    assert keys == [["low_minutes", "value_falling"], [], []]


def test_page_data_carries_the_window_dates_and_both_tables():
    data = _data()
    assert data["window"] == {"seasons": 5, "first": 2019, "last": 2023}
    assert data["season_label"] == "2025/26"
    assert data["valuation_end"] == "2026-06-12"
    assert data["candidates"]["columns"] == [*CANDIDATE_COLUMNS, "flag_keys"]
    assert data["bargains"]["columns"] == [*BARGAIN_COLUMNS, "flag_keys"]
    bargain = dict(zip(data["bargains"]["columns"], data["bargains"]["rows"][0], strict=True))
    assert bargain["rising"] is True  # the riser top 50, read from the scouting rows
    assert bargain["contract_expiry"] == "2028-06-30"
    candidate = dict(zip(data["candidates"]["columns"], data["candidates"]["rows"][1], strict=True))
    assert candidate["fair_value_eur"] is None  # missing is null, never NaN
    assert data["fall_back"] == [{"flags": "1", "pct": 20, "players": 418}]
    assert bargain["below_fair_value"] is True  # 0.5 of fair: a bargain (prompt 20g)
    assert data["below_fair"] == {
        "prime": {"below": 1, "ranked": 1},
        "experienced": {"below": 0, "ranked": 0},
    }
    json.dumps(data, allow_nan=False)


def test_the_bargains_open_on_the_players_below_fair_value():
    dear = _bargains().assign(value_to_fair=1.4, below_fair_value=False)
    data = page_data(
        _candidates(),
        pd.concat([_bargains().assign(below_fair_value=True), dear.assign(list_rank=2)]),
        dictionary(),
        rising=set(),
        feature_dates={"2026-06-30": ["epl"]},
        valuation_end=pd.Timestamp("2026-06-12"),
        latest_value_date=None,
        built=pd.Timestamp("2026-10-28"),
        forecasts=[],
        tracked=37,
    )
    assert data["below_fair"]["prime"] == {"below": 1, "ranked": 2}
    html = render(_data())
    assert "allBargains: false" in html
    assert (
        'if (state.tab === "bargains" && !state.allBargains && !o.below_fair_value) return false;'
        in html
    )
    assert '"show all " + n.ranked.toLocaleString("en")' in html
    assert "above fair value, not a bargain" in html
    assert "only a player priced below his fair value is called a bargain" in html  # how to read
    assert "A bargain is a player priced below his fair value" in html  # what it knows


def test_the_page_embeds_the_data_so_no_text_can_end_the_script():
    html = render(_data())
    block = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert block is not None
    data = json.loads(block.group(1))  # the escaped text still parses
    assert data["candidates"]["rows"][0][0] == "Alex </script><b>x</b>"
    assert "__DATA__" not in html
    assert "<title>Football scouting assistant</title>" in html
    assert "cdn" not in html.lower()  # no outside script or stylesheet


class FakeS3:
    def __init__(self):
        self.objects, self.args = {}, {}

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803 (boto3 names)
        body = Body.read() if hasattr(Body, "read") else Body
        self.objects[(Bucket, Key)] = body
        self.args[(Bucket, Key)] = kw

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):  # noqa: N803
        yield {"Contents": []}


def _site(tmp_path):
    site = tmp_path / "site"
    for key in SITE_FILES:
        path = site / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"<html>page</html>" if key == "index.html" else b"a,b\n1,2\n")
    return site


def _lists(tmp_path):
    lists = tmp_path / "lists"
    lists.mkdir()
    for name in ("candidates", "bargains", "players", "dictionary"):
        for suffix in ("csv", "parquet"):
            (lists / f"{name}.{suffix}").write_bytes(b"x")
    return lists


def test_publish_sends_the_lists_and_the_page_and_nothing_else(tmp_path):
    s3 = FakeS3()
    lists, sent = publish(s3, "data-bucket", "site-bucket", _lists(tmp_path), _site(tmp_path))
    assert len(lists.transferred) == 8
    data_keys = {k for b, k in s3.objects if b == "data-bucket"}
    assert data_keys
    assert all(k.startswith(LISTS_PREFIX) for k in data_keys)
    assert sent[-1] == "index.html"  # the page goes last, after its downloads
    page = s3.objects[("site-bucket", "index.html")]
    assert gzip.decompress(page) == b"<html>page</html>"
    assert s3.args[("site-bucket", "index.html")]["ContentEncoding"] == "gzip"
    assert s3.args[("site-bucket", "index.html")]["ContentType"].startswith("text/html")
    assert "ContentEncoding" not in s3.args[("site-bucket", "data/bargains.parquet")]
    similar = s3.args[("site-bucket", "data/similar.json")]  # prompt 18, loaded on demand
    assert similar["ContentEncoding"] == "gzip"
    assert similar["ContentType"].startswith("application/json")
    # Prompt 20f: the players table as CSV, Parquet and the compact JSON.
    for key in ("data/players.csv", "data/players.parquet", "data/players.json"):
        assert ("site-bucket", key) in s3.objects
    assert s3.args[("site-bucket", "data/players.json")]["ContentEncoding"] == "gzip"
    assert {"processed/lists/players.csv", "processed/lists/players.parquet"} <= data_keys


def test_the_page_loads_the_similar_file_and_links_each_player():
    html = render(_data())
    assert 'fetch("data/similar.json")' in html
    assert 'href="#similar" data-tab="similar"' in html
    assert "sim-open" in html
    data = _data()
    assert "player_id" in data["candidates"]["columns"]  # the panel's key
    assert "player_id" in data["bargains"]["columns"]


def test_publish_refuses_any_other_file(tmp_path):
    site = _site(tmp_path)
    (site / "data" / "player_match_stats.parquet").write_bytes(b"x")
    with pytest.raises(ValueError, match="outside the allowed set"):
        publish_site(FakeS3(), "site-bucket", site)
    lists = _lists(tmp_path)
    (lists / "scouting.parquet").write_bytes(b"x")
    with pytest.raises(ValueError, match="not published"):
        check_lists(lists)


def test_website_url_uses_the_regional_endpoint():
    assert website_url("scout-site-1", "eu-central-1") == (
        "http://scout-site-1.s3-website.eu-central-1.amazonaws.com/"
    )


def test_the_panel_shows_the_written_report_under_its_label():
    reports = {
        "available": True,
        "label": "written by Claude Haiku 5.5 from the facts above and checked automatically; "
        "odds are historical rates",
        "writer": "Claude Haiku 5.5",
        "judge": "Claude Sonnet 5.5",
        "written": 1,
        "template": 1,
        "judged": 1,
        "judge_faithful": 1,
        "agreement": {"read": 2, "agreed": 1},
        "rows": {"1": ["report", "text"], "2": ["template", "No stat stands out."]},
    }
    data = page_data(
        _candidates(),
        _bargains(),
        dictionary(),
        rising={2},
        feature_dates={"2026-06-30": ["Serie A"]},
        valuation_end=pd.Timestamp("2026-06-12"),
        latest_value_date=None,
        built=pd.Timestamp("2026-10-26"),
        forecasts=[],
        tracked=37,
        reports=reports,
    )
    assert "rows" not in data["reports"]  # the reports load from data/reports.json
    assert data["reports"]["agreement"] == {"read": 2, "agreed": 1}
    html = render(data)
    assert 'fetch("data/reports.json")' in html
    assert "Report and similar players" in html
    assert "Written reports." in html
    assert _data()["reports"] == {"available": False}


def test_an_unreviewed_legend_reads_to_watch():
    assert "Status: to watch: not yet reviewed by the owner." in render(_data())


class RefusingS3(FakeS3):
    def __init__(self, refuse: str):
        super().__init__()
        self.refuse = refuse

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803 (boto3 names)
        if Key.startswith(self.refuse):
            raise PermissionError("AccessDenied")
        super().put_object(Bucket, Key, Body, **kw)


def test_publish_keeps_the_reports_best_effort(tmp_path):
    reports = tmp_path / "reports_text"
    reports.mkdir()
    (reports / "reports.parquet").write_bytes(b"x")
    (reports / "summary.json").write_bytes(b"{}")
    (reports / "pending.json").write_bytes(b'{"batches": []}')
    s3 = FakeS3()
    publish(
        s3, "data-bucket", "site-bucket", _lists(tmp_path), _site(tmp_path), reports_dir=reports
    )
    assert ("data-bucket", REPORTS_PREFIX + "reports.parquet") in s3.objects
    # the batches a run left running reach the next run (prompt 19c)
    assert ("data-bucket", REPORTS_PREFIX + "pending.json") in s3.objects
    # a policy without processed/reports_text/ never stops the page
    s3 = RefusingS3(REPORTS_PREFIX)
    other = tmp_path / "again"
    other.mkdir()
    _, sent = publish(
        s3, "data-bucket", "site-bucket", _lists(other), _site(other), reports_dir=reports
    )
    assert sent[-1] == "index.html"
    assert not [k for b, k in s3.objects if k.startswith(REPORTS_PREFIX)]
    (reports / "secrets.txt").write_bytes(b"x")  # only the known files are ever kept
    assert keep_reports(FakeS3(), "data-bucket", reports) == []


def test_the_ask_tab_is_off_without_a_chat_url():
    data = _data()
    assert data["chat"] == {"url": None}
    html = render(data)
    assert 'id="ask-form"' in html
    assert "The chat is not switched on yet." in html


def test_the_ask_tab_gets_the_url_and_the_caps_from_the_config():
    from types import SimpleNamespace

    from scout.chat.prompt import SUGGESTED
    from scout.config import ChatConfig
    from scout.site import chat_block

    url = "https://abc.lambda-url.eu-central-1.on.aws/"
    block = chat_block(SimpleNamespace(chat=ChatConfig(url=url)))
    assert block == {
        "url": url,
        "model": "Claude Sonnet 5.5",
        "max_chars": 500,
        "per_hour": 20,
        "suggested": list(SUGGESTED),
    }
    html = render({**_data(), "chat": block})
    payload = json.loads(
        re.search(r'id="data">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</")
    )
    assert payload["chat"]["url"] == url
    # The page posts the question with the last few turns and shows the tool results.
    assert "fetch(D.chat.url" in html
    assert "From the tables (" in html
    # Prompt 20b: the tab wakes the chat when it opens, and a slow answer says why.
    assert "JSON.stringify({ warm: true })" in html
    assert "The first answer can take up to 30 seconds." in html
    assert "var SLOW_MS = 5000;" in html


def test_the_page_opens_on_the_chat_with_suggestions_and_follow_ups():
    from scout.chat.prompt import SUGGESTED

    html = render({**_data(), "seasons": {"first": 2015, "count": 11}})
    # Prompt 20d: the page opens on the chat when the address has no data section.
    assert '["#candidates", "#bargains", "#similar", "#greats", "#archetypes", "#players"]' in html
    # What it knows and does not; questions send themselves; follow ups.
    assert "What it knows." in html
    assert "What it does not know." in html
    assert "function followUps(results, asked)" in html
    assert "ask(b.dataset.q)" in html
    assert SUGGESTED == (
        "Who plays most like R9 today?",
        "Who looks like Messi did at 19?",
        "Which 20 year olds have the best chance of becoming elite?",
    )


def _function(html, name):
    """One function of the page's script, by its name, up to its closing brace."""
    start = html.index(f"  function {name}(")
    return html[start : html.index("\n  }\n", start) + 4]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_an_answer_renders_lists_and_paragraphs_and_never_html():
    html = render(_data())
    script = _function(html, "esc") + _function(html, "answerHtml")
    answer = (
        "Rodri is in the data.\nHe is on <b>neither</b> list.\n\n"
        "- Ability: 99th percentile.\n- Minutes: 1,513 & 44%.\n\n"
        "1. A. Tetteh, 0.79\n2. M. Daku, 0.78\n\n3. Igor Neves, 0.73\n"
        "Value -12% on a year before. <img src=x onerror=alert(1)>"
    )
    out = subprocess.run(
        ["node", "-e", script + "process.stdout.write(answerHtml(process.argv[1]));", answer],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out == (
        "<p>Rodri is in the data.<br>He is on &lt;b&gt;neither&lt;/b&gt; list.</p>"
        "<ul><li>Ability: 99th percentile.</li><li>Minutes: 1,513 &amp; 44%.</li></ul>"
        '<ol start="1"><li>A. Tetteh, 0.79</li><li>M. Daku, 0.78</li></ol>'
        '<ol start="3"><li>Igor Neves, 0.73</li></ol>'
        "<p>Value -12% on a year before. &lt;img src=x onerror=alert(1)&gt;</p>"
    )
    assert "say(answerHtml(b.answer" in html


def _body(html):
    return html[html.index("<body>") : html.index('<script type="application/json"')]


def test_the_page_is_the_chat_and_everything_else_is_one_click_away():
    html = render(_data())
    body = _body(html)
    # Prompt 20e: the title, the subtitle, the box, then the rest under it.
    order = [
        "<h1>Football scouting assistant</h1>",
        'id="subtitle"',
        'id="ask-form"',
        'id="ask-suggest"',
        'id="ask-fine"',
        'id="ask-about"',
        'id="explore"',
        'id="data-area"',
        'id="footer"',
    ]
    places = [body.index(x) for x in order]
    assert places == sorted(places)
    # The subtitle's counts come from the build: both tables, players and leagues.
    data = _data()
    assert (data["players"], data["player_leagues"]) == (2, 2)
    assert '"Ask about " + int(D.players) + " players across " + D.player_leagues' in html
    assert "Every answer comes from the tables on this page." in html
    # What it knows sits in a native details element, collapsed.
    about = re.search(r'<details class="about" id="ask-about">', body)
    assert about is not None
    assert "open" not in about.group(0)
    assert "<summary>What it knows and what it does not</summary>" in body
    assert "answers with four lookup tools" in html
    # The yellow banner and the grey paragraph are gone; so is the Ask tab button.
    assert "Answers are built only from the tables shown on this page." not in html
    for gone in ("ask-line", "tab-ask", 'role="tab"'):
        assert gone not in html
    # The data dates and the caps in one muted line.
    assert '"Players as of " + day(' in html
    assert '", clubs and this season to " + day(D.so_far_date)' in html
    assert '" characters, " + D.chat.per_hour + " questions an hour"' in html
    # The counter shows only from 400 characters on.
    assert '<span id="ask-count" hidden></span>' in body
    assert "var COUNT_FROM = 400;" in html


def test_the_data_sections_open_from_one_row_of_links_with_their_counts():
    html = render(_data())
    body = _body(html)
    nav = re.search(r'<nav class="explore".*?</nav>', body, re.S).group(0)
    labels = re.findall(r'<a href="#(\w+)" data-tab="\1">([^<]+)</a>', nav)
    assert labels == [
        ("candidates", "Candidates"),
        ("bargains", "Bargains"),
        ("similar", "Similar players"),
        ("greats", "Greats"),
        ("archetypes", "Archetypes"),
        ("players", "Players"),
    ]
    assert nav.startswith(
        '<nav class="explore" id="explore" aria-label="Explore the data">Explore the data:'
    )
    assert 'class="n"' not in nav  # no counts in the links
    # The counts are in the section headings.
    assert 'DATA.candidates.length.toLocaleString("en") + " players</span>"' in html
    assert "' players below fair value, <span class=\"n\">of '" in html  # prompt 20g
    for heading in ("Similar players", "Measured greats", "Archetypes", "All players"):
        assert f"<h2>{heading} <span" in body
    # Hidden until a link is clicked, and the how to read box starts collapsed.
    assert '<div class="wide" id="data-area" hidden>' in body
    assert '<details class="howto" id="howto">' in body


def test_the_footer_keeps_every_download_and_credit():
    html = render({**_data(), "greats": True})
    for line in (
        '<a href=\\"data/candidates.csv\\">candidates.csv</a>',
        '<a href=\\"data/bargains.csv\\">bargains.csv</a>',
        '<a href=\\"data/dictionary.csv\\">dictionary.csv</a>',
        '<a href=\\"data/candidates.parquet\\">candidates</a>',
        '<a href=\\"data/bargains.parquet\\">bargains</a>',
        '<a href=\\"data/players.csv\\">players.csv</a>',
        '<a href=\\"data/players.parquet\\">players</a>',
        "Sources: player statistics from API Football (api-football.com), used under the "
        "owner's subscription; raw responses are not published. ",
        "Market values and transfers from transfermarkt-datasets by dcaribou "
        "(github.com/dcaribou/transfermarkt-datasets), derived from Transfermarkt "
        "(transfermarkt.com), used for non commercial purposes only and credited here. "
        "No club logos or photos are stored or shown.",
        "StatsBomb open data (github.com/statsbomb/open-data)",
        "A non commercial project.",
    ):
        assert line in html
    assert '<footer class="col" id="footer">' in html


def test_the_subtitle_counts_the_players_table_when_there_is_one():
    from scout.players import COLUMNS

    players = pd.DataFrame({c: [None, None, None] for c in COLUMNS}).assign(
        player_id=[1, 2, 3], player=["Alex", "Ben", "Carl"], league=["Serie A", "Süper Lig", "J1"]
    )
    data = page_data(
        _candidates(),
        _bargains(),
        dictionary(),
        players=players,
        rising=set(),
        feature_dates={"2026-06-30": ["Serie A"]},
        valuation_end=pd.Timestamp("2026-06-12"),
        latest_value_date=None,
        built=pd.Timestamp("2026-10-28"),
        forecasts=[],
        tracked=37,
    )
    assert (data["players"], data["player_leagues"], data["listed"]) == (3, 3, 2)


def test_the_players_section_loads_its_file_on_open_and_shows_fifty():
    html = render(_data())
    body = _body(html)
    # Prompt 20f: loaded only when the section opens or a name is typed.
    assert 'fetch("data/players.json")' in html
    assert html.count('fetch("data/players.json")') == 1
    assert '<input type="search" id="pl-q"' in body
    assert "PL_SHOWN = 50" in html
    heads = re.search(r'<table id="pl-table">.*?</thead>', body, re.S).group(0)
    assert re.findall(r"<th>([^<]+)</th>", heads) == [
        "Player",
        "Age",
        "Club",
        "League",
        "Position",
        "Minutes",
        "Ability",
        "Value",
        "Lists",
    ]
    # A player on neither list shows the code's reasons, as the chat does.
    assert "[o.why_not_candidate, o.why_not_bargain]" in html
    # The what it knows text says the lists are a part of all players.
    assert '"The two lists hold " + int(D.listed) + " players' in html


def test_the_style_is_one_system_font_and_follows_the_system_theme():
    html = render(_data())
    style = html[html.index("<style>") : html.index("</style>")]
    assert "@import" not in style
    assert "@font-face" not in style
    assert "font: 16px/1.5 system-ui" in style
    assert "body { font-size: 17px; }" in style
    assert ".col { max-width: 720px;" in style
    assert ".wide { max-width: 1100px;" in style
    assert "@media (prefers-color-scheme: dark)" in style
    assert ":focus-visible { outline: 2px solid var(--accent);" in style
