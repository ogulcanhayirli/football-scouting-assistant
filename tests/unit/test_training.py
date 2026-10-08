import importlib.util
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from test_models import _dataset

from scout.tables import PARTITION_FILE
from scout.training import (
    METRICS,
    PACKAGE_FILES,
    build_source_package,
    job_name,
    training_job_request,
    tuning_job_request,
    tuning_results,
)

ROOT = Path(__file__).parents[2]
CTX = {
    "account": "123456789012",
    "region": "eu-central-1",
    "data_bucket": "data-bucket",
    "ml_bucket": "scout-ml-123456789012",
}


def test_package_is_deterministic_and_holds_only_the_lean_modules():
    body, digest = build_source_package()
    again, digest_again = build_source_package()
    assert (body, digest) == (again, digest_again)
    with tarfile.open(fileobj=io.BytesIO(body)) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(PACKAGE_FILES)
    assert "scout/features/pipeline.py" not in names  # no scraping or contract code


def test_job_names_fit_the_tuning_limit():
    name = job_name("tune", "growth_24m", datetime(2026, 10, 8, 17, 3, 0, tzinfo=UTC))
    assert name == "scout-tune-g24-1008-170300"
    assert len(name) <= 32


def test_training_request_points_at_the_features_and_the_runner_bucket():
    request = training_job_request("scout-train-x", "growth_24m", "s3://b/code.tar.gz", **CTX)
    source = request["InputDataConfig"][0]["DataSource"]["S3DataSource"]
    assert source["S3Uri"] == "s3://data-bucket/processed/features/"
    assert request["RoleArn"].endswith(":role/scout-sagemaker-training")
    assert request["OutputDataConfig"]["S3OutputPath"] == "s3://scout-ml-123456789012/output/"
    assert request["HyperParameters"]["sagemaker_program"] == "train.py"
    assert request["HyperParameters"]["target"] == "growth_24m"
    assert "pytorch-training:2.8.0-cpu-py312" in request["AlgorithmSpecification"]["TrainingImage"]


def test_tuning_request_minimises_the_backtest_error():
    request = tuning_job_request(
        "scout-tune-x", "growth_24m", "s3://b/c.tar.gz", max_jobs=20, max_parallel=2, **CTX
    )
    config = request["HyperParameterTuningJobConfig"]
    assert config["HyperParameterTuningJobObjective"] == {
        "Type": "Minimize",
        "MetricName": "backtest:mae_log",
    }
    assert config["ResourceLimits"] == {"MaxNumberOfTrainingJobs": 20, "MaxParallelTrainingJobs": 2}
    names = {r["Name"] for ranges in config["ParameterRanges"].values() for r in ranges}
    assert {"learning_rate", "num_leaves", "objective", "num_boost_round"} <= names
    static = request["TrainingJobDefinition"]["StaticHyperParameters"]
    assert not names & set(static)  # a tuned setting is never also fixed


def test_metric_regexes_read_what_train_py_prints():
    import re

    line = {
        "backtest:mae_log": "backtest_mae_log=0.572921",
        "backtest:spearman": "backtest_spearman=-0.1",
        "backtest:median_ape": "backtest_median_ape=0.43",
    }
    for metric in METRICS:
        assert re.search(metric["Regex"], line[metric["Name"]]).group(1)


def test_tuning_results_sort_by_error_and_keep_failures_last():
    class Client:
        def list_training_jobs_for_hyper_parameter_tuning_job(self, **_):
            return {
                "TrainingJobSummaries": [
                    {
                        "TrainingJobName": "a",
                        "TrainingJobStatus": "Completed",
                        "TunedHyperParameters": {"num_leaves": "8"},
                        "FinalHyperParameterTuningJobObjectiveMetric": {"Value": 0.56},
                    },
                    {
                        "TrainingJobName": "b",
                        "TrainingJobStatus": "Failed",
                        "TunedHyperParameters": {"num_leaves": "64"},
                    },
                    {
                        "TrainingJobName": "c",
                        "TrainingJobStatus": "Completed",
                        "TunedHyperParameters": {"num_leaves": "16"},
                        "FinalHyperParameterTuningJobObjectiveMetric": {"Value": 0.55},
                    },
                ]
            }

    assert [r["job"] for r in tuning_results(Client(), "t")] == ["c", "a", "b"]


def _write(frame: pd.DataFrame, root: Path) -> None:
    for (league, season), part in frame.groupby(["league_key", "season"], observed=True):
        folder = root / f"league={league}" / f"season={season}"
        folder.mkdir(parents=True)
        part.drop(columns="season").to_parquet(folder / PARTITION_FILE, index=False)


def test_train_py_runs_the_backtest_and_prints_the_metric(tmp_path, capsys):
    data = _dataset(per_season=60).rename(columns={"y": "log_growth_24m"})
    data["regular"] = True
    data["value_24m_date"] = data["feature_date"] + pd.DateOffset(months=23)
    features = data.drop(columns=["log_growth_24m", "value_24m_date"])
    labels = data[["league_key", "season", "source_player_id", "log_growth_24m", "value_24m_date"]]
    _write(features, tmp_path / "in" / "player_season_features")
    _write(labels, tmp_path / "in" / "player_season_labels")

    spec = importlib.util.spec_from_file_location("train", ROOT / "training" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    summary = train.main(
        [
            "--target",
            "growth_24m",
            "--num_boost_round",
            "50",
            "--data-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    printed = capsys.readouterr().out
    assert f"backtest_mae_log={summary['metrics']['mae_log']:.6f}" in printed
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["target"] == "growth_24m"
    assert (tmp_path / "out" / "predictions.parquet").is_file()
