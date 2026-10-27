"""The published files the chat answers from, read once per Lambda instance.

The weekly run's ``scout publish`` stores these in the site bucket, the
page, the CSV and the JSON files gzip encoded (``Content-Encoding: gzip``,
which S3 does not undo for an SDK reader, so ``read`` unpacks them):

* ``data/candidates.csv`` and ``data/bargains.csv``: the two tables;
* ``data/dictionary.csv``: one sentence per column;
* ``data/similar.json``: today's players, the earlier seasons of players
  once in the top 500 and the measured greats, with their vectors;
* ``data/reports.json``: each player's written report, or the note that the
  explanation written by rule is shown;
* ``index.html``: only its data block is read, for what the page says around
  the tables (the dates, the five season window, the elite pool, the flag
  rules, the fall back buckets) and the bargains' rising tag, which the CSV
  does not carry.

The source is either an S3 bucket or a local folder laid out the same way
(``data/site/`` after ``scout site``), so tests and the evaluation read the
very files the Lambda reads.
"""

from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import re
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PAGE = "index.html"
FILES = {
    "candidates": "data/candidates.csv",
    "bargains": "data/bargains.csv",
    "dictionary": "data/dictionary.csv",
    "similar": "data/similar.json",
    "reports": "data/reports.json",
}
DATA_BLOCK = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)
TEXT_COLUMNS = {
    "player",
    "position",
    "club_at_forecast",
    "league",
    "current_club",
    "current_league",
    "flags",
    "explanation",
    "list",
    "contract_expiry",
    "value_date",
    "so_far_date",
    "so_far_league",
    "so_far_flags",
    "table",
    "column",
    "meaning",
}
INT_COLUMNS = {"season", "player_id", "list_rank", "flag_count"}


class Source:
    """Where the published files are: ``read(key)`` returns their bytes."""

    def read(self, key: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def version(self) -> str:
        """Changes when the page is republished (the S3 ETag of the page)."""
        return ""


class FolderSource(Source):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def version(self) -> str:
        page = self.root / PAGE
        return str(page.stat().st_mtime_ns) if page.is_file() else ""


class S3Source(Source):
    def __init__(self, client: Any, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    def read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def version(self) -> str:
        return self.client.head_object(Bucket=self.bucket, Key=PAGE)["ETag"]


def unpack(body: bytes) -> bytes:
    """The bytes as written, whether or not they were stored gzip encoded."""
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


def _value(column: str, text: str):
    if text == "":
        return None
    if column in TEXT_COLUMNS:
        return text
    if text in ("True", "False"):
        return text == "True"
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if column in INT_COLUMNS else number


def parse_csv(text: str) -> list[dict]:
    """Rows as dicts: empty cells None, True and False as booleans, numbers
    as floats (ids, ranks and counts as integers), everything else as text."""
    reader = csv.DictReader(io.StringIO(text))
    return [{k: _value(k, v) for k, v in row.items()} for row in reader]


def page_meta(html: str) -> dict:
    """The page's data block (``scout.site.page_data``), as the page reads it."""
    found = DATA_BLOCK.search(html)
    if not found:
        raise ValueError("the page has no data block")
    payload = found.group(1).replace("<\\/", "</").replace("<\\!--", "<!--")
    return json.loads(payload)


def int8(text: str) -> array:
    """The base64 vectors of ``similar.json`` as signed bytes."""
    out = array("b")
    out.frombytes(base64.b64decode(text))
    return out


@dataclass
class Similar:
    """``similar.json`` with every row as a dict and the vectors decoded."""

    raw: dict
    today: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    greats: list[dict] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.raw.get("available"))

    @property
    def dim(self) -> int:
        return int(self.raw.get("dim", 0))

    @property
    def scale(self) -> int:
        return int(self.raw.get("scale", 127))


def _rows(table: dict, d: dict, vectors: array) -> list[dict]:
    dim = d["dim"]
    out = []
    for i, values in enumerate(table["rows"]):
        o = dict(zip(table["columns"], values, strict=True))
        o["_v"] = vectors[i * dim : (i + 1) * dim]
        if "club" in o and isinstance(o["club"], int):
            o["club_name"] = d["clubs"][o["club"]]
        if "club_now" in o:
            o["club_now_name"] = d["clubs"][o["club_now"]] if o["club_now"] is not None else None
        if "league" in o and isinstance(o["league"], int):
            o["league_name"] = d["leagues"][o["league"]]
        if isinstance(o.get("group"), int):
            o["group_name"] = d["groups"][o["group"]]
        out.append(o)
    return out


def parse_similar(d: dict) -> Similar:
    if not d.get("available"):
        return Similar(raw=d)
    sim = Similar(
        raw=d,
        today=_rows(d["today"], d, int8(d["today"]["vectors"])),
        history=_rows(d["history"], d, int8(d["history"]["vectors"])),
    )
    greats = d.get("greats") or {}
    if greats.get("available"):
        sim.greats = _rows(greats, d, int8(greats["vectors"]))
        for o in sim.greats:
            o["group_name"] = o["group"]
    return sim


@dataclass
class Tables:
    """Everything the tools read."""

    meta: dict
    candidates: list[dict]
    bargains: list[dict]
    dictionary: list[dict]
    similar: Similar
    reports: dict
    version: str = ""
    people: dict | None = None  # scout.chat.tools.people, built on first use

    @property
    def built(self) -> str:
        return str(self.meta.get("built", ""))


def load(source: Source) -> Tables:
    """Read and parse every published file the chat uses; the six reads run
    at once (from S3 each is mostly waiting)."""
    version = source.version()
    keys = {"page": PAGE, **FILES}
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        bodies = dict(zip(keys, pool.map(source.read, keys.values()), strict=True))
    text = {k: unpack(body).decode("utf-8") for k, body in bodies.items()}
    meta = page_meta(text.pop("page"))
    bargains = parse_csv(text["bargains"])
    rising = _rising_bargains(meta)
    for row in bargains:
        row["rising"] = row["player_id"] in rising
    for key in ("candidates", "bargains", "dictionary"):
        meta.pop(key, None)  # the CSV files are the tables
    return Tables(
        meta=meta,
        candidates=parse_csv(text["candidates"]),
        bargains=bargains,
        dictionary=parse_csv(text["dictionary"]),
        similar=parse_similar(json.loads(text["similar"])),
        reports=json.loads(text["reports"]),
        version=version,
    )


def _rising_bargains(meta: dict) -> set[int]:
    """The bargains' rising tag as the page shows it (the riser top 50)."""
    table = meta.get("bargains") or {}
    columns = table.get("columns", [])
    if "rising" not in columns:
        return set()
    pid, rising = columns.index("player_id"), columns.index("rising")
    return {int(r[pid]) for r in table.get("rows", []) if r[rising]}
