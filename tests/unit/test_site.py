"""The static page and the weekly publish."""

import gzip
import json
import re

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
    json.dumps(data, allow_nan=False)


def test_the_page_embeds_the_data_so_no_text_can_end_the_script():
    html = render(_data())
    block = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert block is not None
    data = json.loads(block.group(1))  # the escaped text still parses
    assert data["candidates"]["rows"][0][0] == "Alex </script><b>x</b>"
    assert "__DATA__" not in html
    assert "<title>Scouting lists 2025/26</title>" in html
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
    for name in ("candidates", "bargains", "dictionary"):
        for suffix in ("csv", "parquet"):
            (lists / f"{name}.{suffix}").write_bytes(b"x")
    return lists


def test_publish_sends_the_lists_and_the_page_and_nothing_else(tmp_path):
    s3 = FakeS3()
    lists, sent = publish(s3, "data-bucket", "site-bucket", _lists(tmp_path), _site(tmp_path))
    assert len(lists.transferred) == 6
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


def test_the_page_loads_the_similar_file_and_links_each_player():
    html = render(_data())
    assert 'fetch("data/similar.json")' in html
    assert 'data-tab="similar"' in html
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
    s3 = FakeS3()
    publish(
        s3, "data-bucket", "site-bucket", _lists(tmp_path), _site(tmp_path), reports_dir=reports
    )
    assert ("data-bucket", REPORTS_PREFIX + "reports.parquet") in s3.objects
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
    assert 'id="tab-ask"' in html
    assert "Answers are built only from the tables shown on this page." in html
    assert "The chat is not switched on yet." in html


def test_the_ask_tab_gets_the_url_and_the_caps_from_the_config():
    from types import SimpleNamespace

    from scout.config import ChatConfig
    from scout.site import chat_block

    url = "https://abc.lambda-url.eu-central-1.on.aws/"
    block = chat_block(SimpleNamespace(chat=ChatConfig(url=url)))
    assert block == {"url": url, "model": "Claude Sonnet 5.5", "max_chars": 500, "per_hour": 20}
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
