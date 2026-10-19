"""SageMaker training entry point: one expanding window backtest of LightGBM.

SageMaker runs ``python train.py --target fair_value --learning_rate 0.05 ...``
with the feature tables mounted under ``SM_CHANNEL_FEATURES``. The script
runs the same backtest as ``scout backtest`` (same folds, gap, purge and leak
check) with the given settings and prints its metrics as
``backtest_mae_log=0.5612``, which automatic model tuning reads from the log.
So every candidate is scored out of time over all test seasons, not on one
split.

Outputs: the summary and every out of time prediction go to
``SM_OUTPUT_DATA_DIR`` (uploaded as output.tar.gz).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # the packaged scout modules

from scout.models.backtest import metrics, run_backtest
from scout.models.estimators import DEFAULT_PARAMS, NUM_BOOST_ROUND, make_lightgbm
from scout.models.targets import TARGETS, build_dataset
from scout.tables import read_feature_table

TUNED = {
    "learning_rate": float,
    "num_leaves": int,
    "min_data_in_leaf": int,
    "feature_fraction": float,
    "bagging_fraction": float,
    "lambda_l2": float,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), default="fair_value")
    parser.add_argument(
        "--objective", choices=["regression", "regression_l1", "huber"], default="regression"
    )
    parser.add_argument("--num_boost_round", type=int, default=NUM_BOOST_ROUND)
    for name, kind in TUNED.items():
        parser.add_argument(f"--{name}", type=kind, default=DEFAULT_PARAMS[name])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SM_CHANNEL_FEATURES", "data/processed/features")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SM_OUTPUT_DATA_DIR", "reports/train")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    started = time.time()
    target = TARGETS[args.target]
    params = {"objective": args.objective, **{n: getattr(args, n) for n in TUNED}}
    features = read_feature_table(args.data_dir / "player_season_features")
    labels = read_feature_table(args.data_dir / "player_season_labels")
    data, columns = build_dataset(features, labels, target)
    model = make_lightgbm(params, rounds=args.num_boost_round)
    predictions = run_backtest(data, columns, target, {"lightgbm": model})
    result = metrics(predictions["y"], predictions["pred_lightgbm"])
    by_league = {
        league: metrics(group["y"], group["pred_lightgbm"])["mae_log"]
        for league, group in predictions.groupby("league_key")
    }
    summary = {
        "target": args.target,
        "params": params | {"num_boost_round": args.num_boost_round},
        "metrics": result,
        "mae_log_by_league": by_league,
        "seconds": round(time.time() - started, 1),
    }
    # The lines automatic model tuning reads (see MetricDefinitions).
    print(f"backtest_mae_log={result['mae_log']:.6f}")
    print(f"backtest_spearman={result['spearman']:.6f}")
    print(f"backtest_median_ape={result['median_ape']:.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
