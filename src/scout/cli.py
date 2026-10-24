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

# scout.scouting.STRENGTH_SOURCES, listed here so the parser does not import
# LightGBM; a test keeps the two equal.
SCOUTING_STRENGTHS = ["transfer", "elo", "transfer_elo", "v2"]


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
        if args.level == "transfers":
            from scout.backfill.transfers import run_transfer_backfill
            from scout.providers.api_football.transfers import ApiFootballTransferProvider

            result = run_transfer_backfill(
                settings, ApiFootballTransferProvider(client), league_keys=args.league
            )
            logging.getLogger(__name__).info(
                "Transfers: %s, %d/%d clubs, %d rows, %d requests %s",
                result.status,
                result.clubs_done,
                result.clubs_total,
                result.rows,
                result.requests_made,
                result.detail,
            )
            return 0 if result.complete else 2
        if args.level == "injuries":
            from scout.backfill.matches import run_injury_backfill, write_match_report
            from scout.providers.api_football.matches import ApiFootballMatchProvider

            result = run_injury_backfill(
                settings,
                provider,
                ApiFootballMatchProvider(client),
                league_keys=args.league,
                scope=args.scope,
            )
            write_match_report(result, settings.paths.reports_dir, f"injuries_{args.scope}")
            return 0 if result.complete else 2
        if args.level in ("match", "team", "lineup"):
            from scout.backfill.matches import LEVELS, run_match_backfill, write_match_report
            from scout.providers.api_football.matches import ApiFootballMatchProvider

            result = run_match_backfill(
                settings,
                provider,
                ApiFootballMatchProvider(client),
                league_keys=args.league,
                seasons=args.season,
                level=LEVELS["player" if args.level == "match" else args.level],
                scope=args.scope,
            )
            name = "player" if args.level == "match" else args.level
            if args.scope == "competitions":
                name = f"{name}_competitions"
            md_path = write_match_report(result, settings.paths.reports_dir, name)
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


def cmd_current(settings: Settings, args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from scout.backfill.current import run_current
    from scout.providers.api_football.current import ApiFootballCurrentProvider

    as_of = args.as_of or datetime.now(UTC).date().isoformat()
    with build_client(settings) as client:
        result = run_current(
            settings,
            ApiFootballStatsProvider(client),
            ApiFootballCurrentProvider(client, as_of),
            max_requests=args.max_requests,
            steps=tuple(args.step or ("clubs", "squads", "transfers", "matches", "injuries")),
        )
    logging.getLogger(__name__).info(
        "Current %s: %d clubs, %d squads, %d transfer lists, %d finished matches, "
        "%d requests, tables %s, stopped=%s",
        result.as_of,
        result.clubs,
        result.squads_done,
        result.transfers_done,
        result.matches,
        result.requests_made,
        result.tables,
        result.stopped_reason,
    )
    return 0 if result.complete else 2


def cmd_transfermarkt(settings: Settings, args: argparse.Namespace) -> int:
    from scout.providers.transfermarkt import (
        TransfermarktValuationProvider,
        download_snapshot,
        refresh_snapshot,
    )

    if not args.refresh:
        download_snapshot(settings.transfermarkt, settings.paths.raw_dir)
        return 0
    results = refresh_snapshot(settings.transfermarkt, settings.paths.raw_dir)
    for r in results:
        where = f", old copy in {r.archived_to}" if r.archived_to else ""
        print(f"{r.name}: {r.status} (published {r.last_modified}){where}")
    tm = TransfermarktValuationProvider(settings.paths.raw_dir / "transfermarkt")
    print(f"Latest valuation date: {tm.data_end().date()}")
    if any(r.status != "unchanged" for r in results):
        print("Snapshot changed: run `scout features` to rebuild links, profiles and features.")
    return 0


def cmd_features(settings: Settings, args: argparse.Namespace) -> int:
    if args.current or args.season is not None:
        from scout.features import run_current_features

        run_current_features(settings, args.season)
        return 0
    from scout.features import run_features

    run_features(settings)
    return 0


def cmd_score(settings: Settings, args: argparse.Namespace) -> int:
    from scout.models.moves import score_stored_moves
    from scout.models.run import score_stored_fair_value
    from scout.scouting import score_stored

    score_stored_fair_value(settings)
    score_stored(settings)
    score_stored_moves(settings)
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
        from botocore.config import Config

        client = analyst_session(region, INGEST_ROLE).client(
            "sagemaker-featurestore-runtime",
            config=Config(max_pool_connections=args.workers + 2),
        )
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


def cmd_moves(settings: Settings, args: argparse.Namespace) -> int:
    from scout.models.moves import run_moves

    run_moves(settings)
    return 0


def cmd_scouting(settings: Settings, args: argparse.Namespace) -> int:
    from scout.scouting import DEFAULT_STRENGTH, run_scouting

    run_scouting(settings, args.strength or DEFAULT_STRENGTH)
    return 0


def cmd_embed(settings: Settings, args: argparse.Namespace) -> int:
    from scout.embeddings import embed_stored, train_embeddings

    if args.train:
        print(train_embeddings(settings))
    else:
        for path in embed_stored(settings).values():
            print(path)
    return 0


def cmd_similar(settings: Settings, args: argparse.Namespace) -> int:
    from scout.embeddings import read_index, similar_text
    from scout.lists import league_names

    index = read_index(settings.paths.processed_dir)
    try:
        text = similar_text(
            index,
            args.player,
            league_names(settings),
            at_age=args.at_age,
            k=args.k,
            age=(args.age_min, args.age_max),
            leagues=args.league,
            ability=(args.ability_min, args.ability_max),
        )
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(text)
    return 0


def cmd_lists(settings: Settings, args: argparse.Namespace) -> int:
    from scout.lists import run_lists

    print(run_lists(settings))
    return 0


def cmd_forecast(settings: Settings, args: argparse.Namespace) -> int:
    from scout import forecast

    if args.action == "freeze":
        import pandas as pd

        from scout.lists import league_names

        date = args.date or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        scouting = pd.read_parquet(settings.paths.processed_dir / "models" / "scouting.parquet")
        folder = forecast.freeze(
            settings.paths.processed_dir / "lists",
            scouting,
            league_names(settings),
            Path(args.root) / date,
        )
        print(folder)
    elif args.action == "revise":
        import subprocess

        import pandas as pd

        if not args.folder or not args.reason:
            raise SystemExit("forecast revise needs a forecast folder and --reason")
        folder = Path(args.folder)
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(folder / "candidates.csv")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print(
            forecast.revise(
                settings.paths.processed_dir / "lists",
                folder,
                args.reason,
                pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
                commit,
            )
        )
    elif args.action == "window-freeze":
        import pandas as pd

        from scout.window import FOLDER, freeze_window

        date = args.date or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        folder = Path(args.folder) if args.folder else Path(args.root) / FOLDER
        print(freeze_window(settings, folder, date, refreeze=args.refreeze))
    elif args.action == "window-score":
        from scout.window import FOLDER, run_window_score

        print(run_window_score(settings, Path(args.folder or Path(args.root) / FOLDER)))
    elif args.action == "score":
        if not args.folder:
            raise SystemExit("forecast score needs a forecast folder")
        print(forecast.run_score(settings, Path(args.folder), args.as_of))
    else:
        print(forecast.run_history(settings))
    return 0


def cmd_elo(settings: Settings, args: argparse.Namespace) -> int:
    from scout.elo_compare import run_elo_comparison

    print(run_elo_comparison(settings))
    return 0


def cmd_ability(settings: Settings, args: argparse.Namespace) -> int:
    from scout.ability_compare import run_ability_comparison

    print(run_ability_comparison(settings))
    return 0


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from scout.models.run import run_all

    names = [args.target]
    tuned = None
    if args.tuning_job:
        import os

        from scout.athena import analyst_session
        from scout.training import RUNNER_ROLE, best_parameters

        region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
        client = analyst_session(region, RUNNER_ROLE).client("sagemaker")
        tuned = best_parameters(client, args.tuning_job)
        print(f"Best settings from {args.tuning_job}: {tuned}")
    summary = run_all(settings, names, tuned)
    for name, models in summary.items():
        print(f"\n{name}")
        for model, m in models.items():
            print(
                f"  {model:13s} mae_log {m['mae_log']:.3f}  median_ape {m['median_ape']:.1%}"
                f"  spearman {m['spearman']:.3f}  rows {m['rows']}"
            )
    return 0


def _training_context(args: argparse.Namespace) -> dict:
    import os

    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    return {
        "account": account,
        "region": region,
        "data_bucket": os.environ["SCOUT_S3_BUCKET"],
        "ml_bucket": f"scout-ml-{account}",
    }


def cmd_train(settings: Settings, args: argparse.Namespace) -> int:
    import json

    from scout.athena import analyst_session
    from scout.training import (
        RUNNER_ROLE,
        build_source_package,
        job_name,
        training_job_request,
        tuning_job_request,
        tuning_results,
        upload_package,
        wait_for_training,
        wait_for_tuning,
    )

    log = logging.getLogger(__name__)
    ctx = _training_context(args)
    tuning = args.command == "tune"
    name = job_name("tune" if tuning else "train", args.target)
    if args.dry_run:
        _, digest = build_source_package()
        code_uri = f"s3://{ctx['ml_bucket']}/code/sourcedir-{digest}.tar.gz"
    else:
        session = analyst_session(ctx["region"], RUNNER_ROLE)
        code_uri = upload_package(session.client("s3"), ctx["ml_bucket"])
    if tuning:
        request = tuning_job_request(
            name, args.target, code_uri, max_jobs=args.max_jobs, max_parallel=args.parallel, **ctx
        )
    else:
        request = training_job_request(name, args.target, code_uri, **ctx)
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0

    client = session.client("sagemaker")
    if not tuning:
        client.create_training_job(**request)
        log.info("Started training job %s", name)
        job = wait_for_training(client, name)
        print(f"{name}: {job['TrainingJobStatus']} {job.get('FailureReason', '')}")
        for metric in job.get("FinalMetricDataList", []):
            print(f"  {metric['MetricName']}: {metric['Value']:.4f}")
        print(f"  billable seconds: {job.get('BillableTimeInSeconds')}")
        return 0 if job["TrainingJobStatus"] == "Completed" else 1

    client.create_hyper_parameter_tuning_job(**request)
    log.info("Started tuning job %s (%d jobs, %d at a time)", name, args.max_jobs, args.parallel)
    job = wait_for_tuning(client, name)
    rows = tuning_results(client, name)
    import pandas as pd

    table = pd.DataFrame(rows)
    print(f"{name}: {job['HyperParameterTuningJobStatus']}")
    print(table.to_markdown(index=False))
    path = settings.paths.reports_dir / f"tuning_{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Tuning job {name}\n\n" + table.to_markdown(index=False) + "\n")
    return 0 if job["HyperParameterTuningJobStatus"] == "Completed" else 1


def cmd_site(settings: Settings, args: argparse.Namespace) -> int:
    from scout.site import build_site

    for path in build_site(settings, args.out):
        print(path)
    return 0


def cmd_publish(settings: Settings, args: argparse.Namespace) -> int:
    import os

    from scout.publish import publish, website_url

    log = logging.getLogger(__name__)
    data_bucket = args.bucket or os.environ.get("SCOUT_S3_BUCKET")
    site_bucket = args.site_bucket or os.environ.get("SCOUT_SITE_BUCKET")
    if not data_bucket or not site_bucket:
        log.error("Set SCOUT_S3_BUCKET and SCOUT_SITE_BUCKET (or pass --bucket, --site-bucket)")
        return 1
    import boto3  # optional extra: uv sync --extra aws

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    site_dir = Path(args.site) if args.site else settings.paths.processed_dir.parent / "site"
    site_client = None
    if args.role:
        from scout.athena import analyst_session

        site_client = analyst_session(region, args.role).client("s3")
    lists, sent = publish(
        boto3.client("s3", region_name=region),
        data_bucket,
        site_bucket,
        settings.paths.processed_dir / "lists",
        site_dir,
        args.dry_run,
        site_client,
    )
    suffix = " (dry run)" if args.dry_run else ""
    print(f"processed/lists: {lists.summary()}{suffix}")
    print(f"site: {len(sent)} files to {site_bucket}{suffix}")
    print(f"page: {website_url(site_bucket, region)}")
    return 0


def cmd_sync(settings: Settings, args: argparse.Namespace) -> int:
    import os

    from scout.sync import sync_data

    bucket = args.bucket or os.environ.get("SCOUT_S3_BUCKET")
    if not bucket:
        logging.getLogger(__name__).error("Set SCOUT_S3_BUCKET or pass --bucket")
        return 1
    import boto3  # optional extra: uv sync --extra aws
    from botocore.config import Config

    folders = {
        "raw": settings.paths.raw_dir,
        "processed": settings.paths.processed_dir,
        "reports": settings.paths.reports_dir,
    }
    # Narrower than raw: just the Transfermarkt snapshot (the weekly run).
    extra = {"transfermarkt": ("raw/transfermarkt", settings.paths.raw_dir / "transfermarkt")}
    if args.only:
        folders = dict(extra.get(name, (name, folders.get(name))) for name in args.only)
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
        choices=["season", "match", "team", "lineup", "injuries", "transfers"],
        default="season",
        help="season: /players totals (default); match: /fixtures/players per match; "
        "team: /fixtures/statistics per match; lineup: /fixtures/lineups per match; "
        "injuries: /injuries per league season; "
        "transfers: /transfers once per club in the downloaded leagues",
    )
    backfill.add_argument(
        "--scope",
        choices=["leagues", "competitions"],
        default="leagues",
        help="match and team levels: the leagues, or the cups and national team competitions",
    )
    backfill.set_defaults(func=cmd_backfill)

    coverage = sub.add_parser("coverage", help="Rebuild the coverage report from the cache")
    coverage.set_defaults(func=cmd_coverage)

    cur = sub.add_parser(
        "current",
        help="Today's squads, transfers and the season in progress (display only; spends requests)",
    )
    cur.add_argument("--as-of", help="Snapshot day, YYYY-MM-DD (default: today, UTC)")
    cur.add_argument("--max-requests", type=int, help="Stop after this many requests")
    cur.add_argument(
        "--step",
        action="append",
        choices=["clubs", "squads", "transfers", "matches", "injuries"],
        help="repeatable; default all (clubs always run)",
    )
    cur.set_defaults(func=cmd_current)

    tm = sub.add_parser("transfermarkt", help="Download the transfermarkt-datasets snapshot")
    tm.add_argument(
        "--refresh",
        action="store_true",
        help="check the published release and download only the files that changed; "
        "replaced files are moved to archive/<date>/",
    )
    tm.set_defaults(func=cmd_transfermarkt)

    features = sub.add_parser("features", help="Build point in time feature and label tables")
    features.add_argument(
        "--current",
        action="store_true",
        help="rebuild only the newest season and keep every earlier one as stored (the "
        "weekly run); needs one full build first",
    )
    features.add_argument("--season", type=int, help="with --current: this season instead")
    features.set_defaults(func=cmd_features)

    score = sub.add_parser(
        "score",
        help="Score the newest season with the stored models, no training: fair value, "
        "elite, step up, riser, big move (the weekly run; run after features --current)",
    )
    score.set_defaults(func=cmd_score)

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

    bt = sub.add_parser("backtest", help="Expanding window backtest of the fair value model")
    bt.add_argument("--target", choices=["fair_value"], default="fair_value")
    bt.add_argument("--tuning-job", help="Also run the best settings of this tuning job")
    bt.set_defaults(func=cmd_backtest)

    for command, help_text in (
        ("train", "Run the backtest as one SageMaker training job (default settings)"),
        ("tune", "Automatic model tuning: many training jobs, scored on the backtest"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--target", choices=["fair_value"], default="fair_value")
        p.add_argument("--dry-run", action="store_true", help="Print the request, start nothing")
        if command == "tune":
            p.add_argument("--max-jobs", type=int, default=20)
            p.add_argument("--parallel", type=int, default=2)
        p.set_defaults(func=cmd_train)

    moves = sub.add_parser(
        "moves", help="Big move model: backtest, current scores, report (reports/moves.md)"
    )
    moves.set_defaults(func=cmd_moves)

    scouting = sub.add_parser(
        "scouting",
        help="League strength, ability, elite, step up and riser models (reports/scouting.md)",
    )
    scouting.add_argument(
        "--strength",
        choices=SCOUTING_STRENGTHS,
        default=None,
        help="Ability score version: transfer (version 1), elo, transfer_elo, or v2 (per 90 "
        "composite); the default is in scout.scouting.DEFAULT_STRENGTH. All but transfer "
        "need the Elo tables (run `elo` first)",
    )
    scouting.set_defaults(func=cmd_scouting)

    embed = sub.add_parser(
        "embed",
        help="Style vectors and similar players: embed every row with the stored model and "
        "write data/processed/embeddings/ (the weekly run; after score), or --train",
    )
    embed.add_argument(
        "--train",
        action="store_true",
        help="test the learned embedding against the baseline on 2023 to 2025, refit it on "
        "every season, store it and write reports/embeddings.md",
    )
    embed.set_defaults(func=cmd_embed)

    similar = sub.add_parser(
        "similar",
        help="Players who play like a player now, or like he did at an age (run embed first)",
    )
    similar.add_argument("player", help="name (accents ignored) or API Football id")
    similar.add_argument(
        "--at-age",
        type=int,
        help="his season at this age against today's players a year either side",
    )
    similar.add_argument("-k", type=int, default=10, help="how many (default 10)")
    similar.add_argument("--age-min", type=int, help="now: youngest age")
    similar.add_argument("--age-max", type=int, help="now: oldest age")
    similar.add_argument(
        "--league", action="append", help="now: league key, repeatable (for example la_liga)"
    )
    similar.add_argument("--ability-min", type=float, help="now: lowest ability percentile")
    similar.add_argument("--ability-max", type=float, help="now: highest ability percentile")
    similar.set_defaults(func=cmd_similar)

    lists = sub.add_parser(
        "lists",
        help="The candidates and bargains tables, their dictionary and reports/candidates.md; "
        "run after backtest, moves, scouting and forecast history",
    )
    lists.set_defaults(func=cmd_lists)

    fc = sub.add_parser(
        "forecast",
        help="Registered forecasts: freeze today's lists into forecasts/<date>, score a "
        "folder at a later feature date, rebuild and score 2015 to 2023 (history; run before "
        "lists), revise a folder from today's lists with a recorded reason, or freeze and "
        "score the January window forecast",
    )
    fc.add_argument(
        "action",
        choices=["freeze", "score", "history", "revise", "window-freeze", "window-score"],
    )
    fc.add_argument(
        "folder",
        nargs="?",
        help="score, revise: the forecast folder; window-*: default forecasts/window-2027-01",
    )
    fc.add_argument(
        "--refreeze",
        action="store_true",
        help="window-freeze: rewrite a frozen window folder (the 15 December refresh)",
    )
    fc.add_argument(
        "--reason", help="revise: why the folder is rewritten (kept in forecast.json and README)"
    )
    fc.add_argument("--as-of", help="score: the later feature date (default: the newest)")
    fc.add_argument("--date", help="freeze: the folder's date (default: today, UTC)")
    fc.add_argument("--root", default="forecasts", help="freeze: where the folder goes")
    fc.set_defaults(func=cmd_forecast)

    elo = sub.add_parser(
        "elo",
        help="Team Elo from the cached fixtures, league strength, comparison (reports/elo.md)",
    )
    elo.set_defaults(func=cmd_elo)

    ability = sub.add_parser(
        "ability",
        help="Ability score versions: stability and scouting backtest (reports/ability.md)",
    )
    ability.set_defaults(func=cmd_ability)

    site = sub.add_parser(
        "site",
        help="The static page: index.html with both tables embedded, and the downloads "
        "(data/site/); run after lists",
    )
    site.add_argument("--out", type=Path, help="Folder to write (default: data/site)")
    site.set_defaults(func=cmd_site)

    pub = sub.add_parser(
        "publish",
        help="Upload the lists to processed/lists/ and the page to the site bucket, nothing else "
        "(the weekly run)",
    )
    pub.add_argument("--bucket", help="Data bucket; defaults to $SCOUT_S3_BUCKET")
    pub.add_argument("--site-bucket", help="Site bucket; defaults to $SCOUT_SITE_BUCKET")
    pub.add_argument(
        "--site", type=Path, help="Folder written by `scout site` (default: data/site)"
    )
    pub.add_argument(
        "--role",
        help="Assume this role for the site bucket, for example scout-site-publisher (a "
        "session publishing by hand; the weekly run's user needs none)",
    )
    pub.add_argument("--dry-run", action="store_true", help="List what would be sent")
    pub.set_defaults(func=cmd_publish)

    sync = sub.add_parser("sync", help="Copy data/raw, data/processed and reports to or from S3")
    sync.add_argument("direction", choices=["up", "down"])
    sync.add_argument("--bucket", help="Defaults to $SCOUT_S3_BUCKET")
    sync.add_argument("--prefix", default="", help="Key prefix inside the bucket")
    sync.add_argument(
        "--only",
        action="append",
        choices=["raw", "processed", "reports", "transfermarkt"],
        help="repeatable; transfermarkt is raw/transfermarkt alone",
    )
    sync.add_argument("--dry-run", action="store_true", help="List what would be copied")
    sync.add_argument("--workers", type=int, default=8, help="Parallel transfers (default 8)")
    sync.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    _setup_logging(settings)
    return args.func(settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
