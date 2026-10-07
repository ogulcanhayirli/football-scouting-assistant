"""Command line entry point: ``uv run scout <command> --config <file>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scout.config import Settings, load_settings
from scout.providers.api_football import ApiFootballClient, ApiFootballStatsProvider
from scout.ratelimit import QuotaGuard
from scout.secrets import api_football_key
from scout.storage import LocalRawStore


def _setup_logging(settings: Settings) -> None:
    settings.paths.state_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(settings.paths.state_dir / "scout.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def build_client(settings: Settings) -> ApiFootballClient:
    af = settings.api_football
    guard = QuotaGuard(
        daily_limit=af.quota.daily_limit,
        per_minute_limit=af.quota.per_minute_limit,
        safety_margin=af.quota.safety_margin,
        state_path=settings.paths.state_dir / "api_football_quota.json",
    )
    return ApiFootballClient(
        api_key=api_football_key(),
        store=LocalRawStore(settings.paths.raw_dir),
        guard=guard,
        base_url=af.base_url,
        timeout=af.timeout_seconds,
        max_retries=af.max_retries,
    )


def cmd_backfill(settings: Settings, args: argparse.Namespace) -> int:
    from scout.backfill import build_coverage, run_backfill, write_coverage_report

    with build_client(settings) as client:
        provider = ApiFootballStatsProvider(client)
        result = run_backfill(settings, provider, league_keys=args.league, seasons=args.season)
        report = build_coverage(settings, client.store, result)
    json_path, md_path = write_coverage_report(report, settings.paths.reports_dir)
    logging.getLogger(__name__).info("Coverage report: %s and %s", md_path, json_path)
    return 0 if result.complete else 2


def cmd_coverage(settings: Settings, args: argparse.Namespace) -> int:
    from scout.backfill import build_coverage, write_coverage_report

    report = build_coverage(settings, LocalRawStore(settings.paths.raw_dir))
    write_coverage_report(report, settings.paths.reports_dir)
    return 0


def cmd_transfermarkt(settings: Settings, args: argparse.Namespace) -> int:
    from scout.providers.transfermarkt import download_snapshot

    download_snapshot(settings.transfermarkt, settings.paths.raw_dir)
    return 0


def cmd_sync(settings: Settings, args: argparse.Namespace) -> int:
    import os

    from scout.sync import sync_data

    bucket = args.bucket or os.environ.get("SCOUT_S3_BUCKET")
    if not bucket:
        logging.getLogger(__name__).error("Set SCOUT_S3_BUCKET or pass --bucket")
        return 1
    import boto3  # optional extra: uv sync --extra aws

    folders = {"raw": settings.paths.raw_dir, "processed": settings.paths.processed_dir}
    if args.only:
        folders = {name: folders[name] for name in args.only}
    results = sync_data(
        boto3.client("s3"), bucket, folders, args.direction, args.prefix, args.dry_run
    )
    for name, result in results.items():
        print(f"{name}: {result.summary()}{' (dry run)' if args.dry_run else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout")
    parser.add_argument("--config", type=Path, default=Path("configs/backfill.free.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="Download player season stats (resumable)")
    backfill.add_argument("--league", action="append", help="League key; repeatable")
    backfill.add_argument("--season", action="append", type=int, help="Season; repeatable")
    backfill.set_defaults(func=cmd_backfill)

    coverage = sub.add_parser("coverage", help="Rebuild the coverage report from the cache")
    coverage.set_defaults(func=cmd_coverage)

    tm = sub.add_parser("transfermarkt", help="Download the transfermarkt-datasets snapshot")
    tm.set_defaults(func=cmd_transfermarkt)

    sync = sub.add_parser("sync", help="Copy data/raw and data/processed to or from S3")
    sync.add_argument("direction", choices=["up", "down"])
    sync.add_argument("--bucket", help="Defaults to $SCOUT_S3_BUCKET")
    sync.add_argument("--prefix", default="", help="Key prefix inside the bucket")
    sync.add_argument("--only", action="append", choices=["raw", "processed"])
    sync.add_argument("--dry-run", action="store_true", help="List what would be copied")
    sync.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    _setup_logging(settings)
    return args.func(settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
