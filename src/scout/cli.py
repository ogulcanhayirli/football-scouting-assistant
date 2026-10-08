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
        if args.level == "match":
            from scout.backfill.matches import run_match_backfill, write_match_report
            from scout.providers.api_football.matches import ApiFootballMatchProvider

            result = run_match_backfill(
                settings,
                provider,
                ApiFootballMatchProvider(client),
                league_keys=args.league,
                seasons=args.season,
            )
            md_path = write_match_report(result, settings.paths.reports_dir)
            logging.getLogger(__name__).info("Match report: %s", md_path)
            return 0 if result.complete else 2
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


def cmd_features(settings: Settings, args: argparse.Namespace) -> int:
    from scout.features import run_features

    run_features(settings)
    return 0


def cmd_athena(settings: Settings, args: argparse.Namespace) -> int:
    import os

    from scout.athena import AthenaQueryError, analyst_session, run_query

    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    client = analyst_session(region).client("athena")
    try:
        frame, stats = run_query(client, args.sql)
    except AthenaQueryError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(frame.to_markdown(index=False) if not frame.empty else "(no rows)")
    print(
        f"\n{len(frame)} rows, {stats['scanned_bytes'] / 1e6:.2f} MB scanned, "
        f"{stats['runtime_ms']} ms"
    )
    return 0


def cmd_featurestore(settings: Settings, args: argparse.Namespace) -> int:
    import json
    import os

    import pandas as pd

    from scout.athena import AthenaQueryError, analyst_session, run_query
    from scout.features import read_feature_table
    from scout.featurestore import (
        CHECK_COLUMNS,
        FEATURE_GROUP,
        INGEST_ROLE,
        compare_point_in_time,
        expected_point_in_time,
        ingest,
        point_in_time_sql,
        to_records,
    )

    log = logging.getLogger(__name__)
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    features = read_feature_table(
        settings.paths.processed_dir / "features" / "player_season_features"
    )
    if args.action == "ingest":
        records = to_records(features if args.limit is None else features.head(args.limit))
        client = analyst_session(region, INGEST_ROLE).client("sagemaker-featurestore-runtime")
        written, errors = ingest(client, records, workers=args.workers)
        log.info("Put %d of %d records into %s", written, len(records), FEATURE_GROUP)
        for error in errors[:10]:
            log.error("%s", error)
        return 0 if not errors else 2

    session = analyst_session(region)
    described = session.client("sagemaker").describe_feature_group(FeatureGroupName=FEATURE_GROUP)
    catalog = described["OfflineStoreConfig"]["DataCatalogConfig"]
    as_of = pd.Timestamp(args.as_of)
    sql = point_in_time_sql(catalog["Database"], catalog["TableName"], as_of, CHECK_COLUMNS)
    try:
        found, stats = run_query(session.client("athena"), sql)
    except AthenaQueryError as exc:
        log.error("%s", exc)
        return 1
    report = compare_point_in_time(expected_point_in_time(features, as_of), found)
    print(json.dumps(report | {"scanned_mb": round(stats["scanned_bytes"] / 1e6, 2)}, indent=2))
    return 0 if report["identical"] else 3


def cmd_sync(settings: Settings, args: argparse.Namespace) -> int:
    import os

    from scout.sync import sync_data

    bucket = args.bucket or os.environ.get("SCOUT_S3_BUCKET")
    if not bucket:
        logging.getLogger(__name__).error("Set SCOUT_S3_BUCKET or pass --bucket")
        return 1
    import boto3  # optional extra: uv sync --extra aws
    from botocore.config import Config

    folders = {"raw": settings.paths.raw_dir, "processed": settings.paths.processed_dir}
    if args.only:
        folders = {name: folders[name] for name in args.only}
    client = boto3.client("s3", config=Config(max_pool_connections=args.workers + 2))
    results = sync_data(
        client, bucket, folders, args.direction, args.prefix, args.dry_run, args.workers
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
    backfill.add_argument(
        "--level",
        choices=["season", "match"],
        default="season",
        help="season: /players totals (default); match: /fixtures/players per match",
    )
    backfill.set_defaults(func=cmd_backfill)

    coverage = sub.add_parser("coverage", help="Rebuild the coverage report from the cache")
    coverage.set_defaults(func=cmd_coverage)

    tm = sub.add_parser("transfermarkt", help="Download the transfermarkt-datasets snapshot")
    tm.set_defaults(func=cmd_transfermarkt)

    features = sub.add_parser("features", help="Build point in time feature and label tables")
    features.set_defaults(func=cmd_features)

    athena_cmd = sub.add_parser("athena", help="Run a SQL query on the feature tables")
    athena_cmd.add_argument("sql", help="Query, for example SELECT * FROM player_season_features")
    athena_cmd.add_argument("--region", help="Defaults to $AWS_DEFAULT_REGION")
    athena_cmd.set_defaults(func=cmd_athena)

    fs = sub.add_parser("featurestore", help="SageMaker Feature Store: ingest or check")
    fs_sub = fs.add_subparsers(dest="action", required=True)
    fs_ingest = fs_sub.add_parser("ingest", help="Put every feature row into the feature group")
    fs_ingest.add_argument("--limit", type=int, help="Only the first N rows (a trial run)")
    fs_ingest.add_argument("--workers", type=int, default=16)
    fs_pit = fs_sub.add_parser("pit", help="Compare a point in time query with our tables")
    fs_pit.add_argument("--as-of", required=True, help="Date, for example 2023-01-01")
    fs.set_defaults(func=cmd_featurestore)

    sync = sub.add_parser("sync", help="Copy data/raw and data/processed to or from S3")
    sync.add_argument("direction", choices=["up", "down"])
    sync.add_argument("--bucket", help="Defaults to $SCOUT_S3_BUCKET")
    sync.add_argument("--prefix", default="", help="Key prefix inside the bucket")
    sync.add_argument("--only", action="append", choices=["raw", "processed"])
    sync.add_argument("--dry-run", action="store_true", help="List what would be copied")
    sync.add_argument("--workers", type=int, default=8, help="Parallel transfers (default 8)")
    sync.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    _setup_logging(settings)
    return args.func(settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
