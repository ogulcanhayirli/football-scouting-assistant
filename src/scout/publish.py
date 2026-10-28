"""``scout publish``: the weekly run's only writes to AWS.

Two destinations, nothing else (CLAUDE.md, the one exception to rule 8):

* ``processed/lists/`` in the private data bucket: the two tables, the
  players table (prompt 20f) and the dictionary of ``scout lists``, skipped
  when the bytes are already there (``scout.sync``);
* the public site bucket: ``index.html`` and ``data/`` from ``scout site``
  (the two tables, the players table as CSV, Parquet and the page's JSON,
  the dictionary, the similar players file, the written reports, and the
  StatsBomb logo when the measured greats are in it).

Since prompt 19 it also keeps ``processed/reports_text/`` (``scout
reports``) in the data bucket, so next week's run reuses every report whose
facts did not change. That copy is best effort: when it fails (the IAM
user's policy predates prompt 19, docs/DEPLOY.md) a warning is logged and
the page is published all the same.
  Only the files in ``SITE_FILES`` are ever sent, so no other table can be
  published by mistake. The page and the CSV files are stored gzip encoded
  (S3 serves them with ``Content-Encoding: gzip``, which browsers unpack),
  the page in about a fifth of its size.

The S3 client is passed in, so tests use an in memory fake.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any

from scout.sync import SyncResult, sync_dir

log = logging.getLogger(__name__)

LISTS_PREFIX = "processed/lists/"
REPORTS_PREFIX = "processed/reports_text/"
REPORT_FILES = {
    "reports.parquet",
    "dictionary.csv",
    "summary.json",
    "runs.jsonl",
    "verdicts.jsonl",
    "failures.jsonl",
    "pending.json",  # prompt 19c: the batches a run left running, for the next run to collect
}
LIST_FILES = {
    f"{name}.{suffix}"
    for name in ("candidates", "bargains", "players", "dictionary")
    for suffix in ("csv", "parquet")
}
SITE_FILES = {
    "index.html": "text/html; charset=utf-8",
    "data/candidates.csv": "text/csv; charset=utf-8",
    "data/bargains.csv": "text/csv; charset=utf-8",
    "data/dictionary.csv": "text/csv; charset=utf-8",
    "data/candidates.parquet": "application/vnd.apache.parquet",
    "data/bargains.parquet": "application/vnd.apache.parquet",
    "data/players.csv": "text/csv; charset=utf-8",
    "data/players.parquet": "application/vnd.apache.parquet",
    "data/players.json": "application/json; charset=utf-8",  # prompt 20f
    "data/similar.json": "application/json; charset=utf-8",
    "data/reports.json": "application/json; charset=utf-8",
}
# Sent when ``scout site`` wrote it: the StatsBomb logo comes only with the
# measured greats (prompt 18b), which are never shown without it.
OPTIONAL_SITE_FILES = {"statsbomb-logo.png": "image/png"}
GZIPPED = {".html", ".csv", ".json"}
CACHE = "public, max-age=300"  # a weekly page: five minutes of browser cache is plenty


def check_lists(lists_dir: Path) -> None:
    """The lists folder holds only the published tables."""
    found = {p.name for p in Path(lists_dir).iterdir() if p.is_file()}
    extra = found - LIST_FILES
    if extra:
        raise ValueError(f"unexpected files in {lists_dir}, not published: {sorted(extra)}")
    missing = LIST_FILES - found
    if missing:
        raise FileNotFoundError(f"{lists_dir} lacks {sorted(missing)}: run `scout lists` first")


def publish_site(client: Any, bucket: str, site_dir: Path, dry_run: bool = False) -> list[str]:
    """Upload the page and the downloads; refuses anything outside ``SITE_FILES``."""
    site_dir = Path(site_dir)
    found = {p.relative_to(site_dir).as_posix() for p in site_dir.rglob("*") if p.is_file()}
    allowed = {**SITE_FILES, **OPTIONAL_SITE_FILES}
    extra = found - set(allowed)
    if extra:
        raise ValueError(f"refusing to publish files outside the allowed set: {sorted(extra)}")
    missing = set(SITE_FILES) - found
    if missing:
        raise FileNotFoundError(f"{site_dir} lacks {sorted(missing)}: run `scout site` first")
    sent = []
    keys = [*SITE_FILES, *(k for k in OPTIONAL_SITE_FILES if k in found)]
    # The downloads first and the page last, so the page never links a file
    # that is not there yet.
    for key in sorted(keys, key=lambda k: k == "index.html"):
        path = site_dir / key
        body = path.read_bytes()
        extra_args = {"ContentType": allowed[key], "CacheControl": CACHE}
        if path.suffix in GZIPPED:
            body = gzip.compress(body, compresslevel=9, mtime=0)
            extra_args["ContentEncoding"] = "gzip"
        if not dry_run:
            client.put_object(Bucket=bucket, Key=key, Body=body, **extra_args)
        sent.append(key)
        log.info("site: %s (%.0f KB sent)%s", key, len(body) / 1e3, " (dry run)" if dry_run else "")
    return sent


def keep_reports(client: Any, bucket: str, reports_dir: Path, dry_run: bool = False) -> list[str]:
    """``processed/reports_text/`` up to the data bucket, the files in
    ``REPORT_FILES`` only; never raises, so the page never waits on it."""
    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        return []
    extra = {p.name for p in reports_dir.iterdir() if p.is_file()} - REPORT_FILES
    if extra:
        log.warning("reports: not kept, unexpected files in %s: %s", reports_dir, sorted(extra))
        return []
    try:
        result = sync_dir(client, bucket, reports_dir, REPORTS_PREFIX, "up", dry_run)
    except Exception as exc:  # the page is published either way
        log.warning("reports: %s not kept in s3://%s/ (%s)", reports_dir, bucket, exc)
        return []
    return result.transferred


def publish(
    client: Any,
    data_bucket: str,
    site_bucket: str,
    lists_dir: Path,
    site_dir: Path,
    dry_run: bool = False,
    site_client: Any = None,
    reports_dir: Path | None = None,
) -> tuple[SyncResult, list[str]]:
    """``site_client``: another client for the site bucket (a session publishing
    by hand assumes ``scout-site-publisher``); the weekly run's user has both.
    ``reports_dir``: ``processed/reports_text/``, kept best effort."""
    check_lists(lists_dir)
    lists = sync_dir(client, data_bucket, Path(lists_dir), LISTS_PREFIX, "up", dry_run)
    sent = publish_site(site_client or client, site_bucket, site_dir, dry_run)
    if reports_dir is not None:
        kept = keep_reports(client, data_bucket, reports_dir, dry_run)
        log.info("reports: %d files kept under %s", len(kept), REPORTS_PREFIX)
    return lists, sent


def website_url(bucket: str, region: str) -> str:
    """The S3 website endpoint of a bucket (eu-central-1 uses the dot form)."""
    return f"http://{bucket}.s3-website.{region}.amazonaws.com/"
