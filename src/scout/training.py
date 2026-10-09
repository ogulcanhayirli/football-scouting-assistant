"""Run the backtest as SageMaker training jobs, alone or under automatic model tuning.

Everything goes through the ``scout-ml-runner`` role (infra ``ScoutTraining``
stack). The flow:

1. ``build_source_package``: ``training/train.py``, ``training/requirements.txt``
   and the few ``scout`` modules the backtest needs, as ``sourcedir.tar.gz``,
   named by a hash of its contents so an unchanged package is uploaded once.
2. A training job or a tuning job on the AWS PyTorch 2.8 CPU image
   (Python 3.12), reading the feature tables from the data bucket.
3. ``train.py`` prints ``backtest_mae_log=...``; the tuner minimises it.

The definitions are built by pure functions, so tests and ``--dry-run`` can
show exactly what would be sent without calling AWS.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.8.0-cpu-py312"
INSTANCE_TYPE = "ml.m5.large"
RUNNER_ROLE = "scout-ml-runner"
TRAINING_ROLE = "scout-sagemaker-training"
FEATURES_PREFIX = "processed/features/"
MAX_RUNTIME_SECONDS = 1800

# Files that go into the package, relative to the repository root.
PACKAGE_FILES = {
    "train.py": "training/train.py",
    "requirements.txt": "training/requirements.txt",
    "scout/__init__.py": "src/scout/__init__.py",
    "scout/tables.py": "src/scout/tables.py",
    "scout/models/__init__.py": "src/scout/models/__init__.py",
    "scout/models/targets.py": "src/scout/models/targets.py",
    "scout/models/backtest.py": "src/scout/models/backtest.py",
    "scout/models/estimators.py": "src/scout/models/estimators.py",
}

METRICS = [
    {"Name": "backtest:mae_log", "Regex": r"backtest_mae_log=([0-9.]+)"},
    {"Name": "backtest:spearman", "Regex": r"backtest_spearman=([-0-9.]+)"},
    {"Name": "backtest:median_ape", "Regex": r"backtest_median_ape=([0-9.]+)"},
]

# The search space for automatic model tuning (LightGBM's own names).
TUNING_RANGES = {
    "ContinuousParameterRanges": [
        {
            "Name": "learning_rate",
            "MinValue": "0.005",
            "MaxValue": "0.2",
            "ScalingType": "Logarithmic",
        },
        {"Name": "feature_fraction", "MinValue": "0.3", "MaxValue": "1.0", "ScalingType": "Linear"},
        {"Name": "bagging_fraction", "MinValue": "0.5", "MaxValue": "1.0", "ScalingType": "Linear"},
        {"Name": "lambda_l2", "MinValue": "0.001", "MaxValue": "30", "ScalingType": "Logarithmic"},
    ],
    "IntegerParameterRanges": [
        {"Name": "num_leaves", "MinValue": "4", "MaxValue": "128", "ScalingType": "Logarithmic"},
        {
            "Name": "min_data_in_leaf",
            "MinValue": "5",
            "MaxValue": "300",
            "ScalingType": "Logarithmic",
        },
        {
            "Name": "num_boost_round",
            "MinValue": "100",
            "MaxValue": "1500",
            "ScalingType": "Logarithmic",
        },
    ],
    "CategoricalParameterRanges": [
        {"Name": "objective", "Values": ["regression", "regression_l1", "huber"]},
    ],
}

SHORT = {"fair_value": "fv", "growth_24m": "g24", "growth_36m": "g36"}


def build_source_package(root: Path = ROOT) -> tuple[bytes, str]:
    """The ``sourcedir.tar.gz`` bytes and a short content hash. Deterministic:
    fixed order, timestamps and owners, so the same files give the same hash."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as tar:
        for name, source in sorted(PACKAGE_FILES.items()):
            data = (root / source).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    body = buffer.getvalue()
    digest = hashlib.sha256(
        b"".join((root / s).read_bytes() for _, s in sorted(PACKAGE_FILES.items()))
    ).hexdigest()[:12]
    return body, digest


def job_name(kind: str, target: str, now: datetime | None = None) -> str:
    """``scout-<kind>-<target>-<timestamp>``; tuning job names allow 32 characters."""
    stamp = (now or datetime.now(UTC)).strftime("%m%d-%H%M%S")
    return f"scout-{kind}-{SHORT[target]}-{stamp}"


def _static_hyperparameters(target: str, code_uri: str, region: str) -> dict[str, str]:
    return {
        "target": target,
        "sagemaker_program": "train.py",
        "sagemaker_submit_directory": code_uri,
        "sagemaker_region": region,
        "sagemaker_container_log_level": "20",
    }


def _shared(account: str, region: str, data_bucket: str, ml_bucket: str) -> dict[str, Any]:
    return {
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE.format(region=region),
            "TrainingInputMode": "File",
            "MetricDefinitions": METRICS,
        },
        "RoleArn": f"arn:aws:iam::{account}:role/{TRAINING_ROLE}",
        "InputDataConfig": [
            {
                "ChannelName": "features",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{data_bucket}/{FEATURES_PREFIX}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{ml_bucket}/output/"},
        "ResourceConfig": {"InstanceType": INSTANCE_TYPE, "InstanceCount": 1, "VolumeSizeInGB": 30},
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
    }


def training_job_request(
    name: str,
    target: str,
    code_uri: str,
    *,
    account: str,
    region: str,
    data_bucket: str,
    ml_bucket: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hyper = _static_hyperparameters(target, code_uri, region)
    hyper.update({k: str(v) for k, v in (params or {}).items()})
    return {
        "TrainingJobName": name,
        "HyperParameters": hyper,
        **_shared(account, region, data_bucket, ml_bucket),
        "Tags": [{"Key": "project", "Value": "football-scouting-assistant"}],
    }


def tuning_job_request(
    name: str,
    target: str,
    code_uri: str,
    *,
    account: str,
    region: str,
    data_bucket: str,
    ml_bucket: str,
    max_jobs: int,
    max_parallel: int,
) -> dict[str, Any]:
    definition = _shared(account, region, data_bucket, ml_bucket)
    definition["StaticHyperParameters"] = _static_hyperparameters(target, code_uri, region)
    return {
        "HyperParameterTuningJobName": name,
        "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {
                "Type": "Minimize",
                "MetricName": "backtest:mae_log",
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": max_jobs,
                "MaxParallelTrainingJobs": max_parallel,
            },
            "ParameterRanges": TUNING_RANGES,
            "TrainingJobEarlyStoppingType": "Off",
        },
        "TrainingJobDefinition": definition,
        "Tags": [{"Key": "project", "Value": "football-scouting-assistant"}],
    }


def upload_package(s3: Any, ml_bucket: str) -> str:
    body, digest = build_source_package()
    key = f"code/sourcedir-{digest}.tar.gz"
    s3.put_object(Bucket=ml_bucket, Key=key, Body=body)
    return f"s3://{ml_bucket}/{key}"


def wait_for_training(client: Any, name: str, poll: float = 20, sleep=time.sleep) -> dict:
    while True:
        job = client.describe_training_job(TrainingJobName=name)
        status = job["TrainingJobStatus"]
        if status in {"Completed", "Failed", "Stopped"}:
            return job
        log.info("%s: %s (%s)", name, status, job.get("SecondaryStatus"))
        sleep(poll)


def wait_for_tuning(client: Any, name: str, poll: float = 60, sleep=time.sleep) -> dict:
    while True:
        job = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        status = job["HyperParameterTuningJobStatus"]
        counts = job.get("TrainingJobStatusCounters", {})
        if status in {"Completed", "Failed", "Stopped"}:
            return job
        log.info(
            "%s: %s, %d done, %d running, %d failed",
            name,
            status,
            counts.get("Completed", 0),
            counts.get("InProgress", 0),
            counts.get("NonRetryableError", 0) + counts.get("RetryableError", 0),
        )
        sleep(poll)


def tuning_results(client: Any, name: str) -> list[dict[str, Any]]:
    """Every training job of a tuning job: its settings and final objective."""
    rows, token = [], None
    while True:
        kwargs = {"HyperParameterTuningJobName": name, "MaxResults": 100}
        if token:
            kwargs["NextToken"] = token
        page = client.list_training_jobs_for_hyper_parameter_tuning_job(**kwargs)
        for job in page["TrainingJobSummaries"]:
            metric = job.get("FinalHyperParameterTuningJobObjectiveMetric", {})
            rows.append(
                {
                    "job": job["TrainingJobName"],
                    "status": job["TrainingJobStatus"],
                    "mae_log": metric.get("Value"),
                    **{k: v for k, v in job["TunedHyperParameters"].items()},
                }
            )
        token = page.get("NextToken")
        if not token:
            break
    return sorted(rows, key=lambda r: (r["mae_log"] is None, r["mae_log"] or 0))


INTEGER_PARAMS = {"num_leaves", "min_data_in_leaf", "num_boost_round"}


def best_parameters(client: Any, name: str) -> dict[str, Any]:
    """The tuned settings of the best training job, typed for LightGBM."""
    job = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
    tuned = job["BestTrainingJob"]["TunedHyperParameters"]
    out: dict[str, Any] = {}
    for key, value in tuned.items():
        value = value.strip('"')
        if key in INTEGER_PARAMS:
            out[key] = int(float(value))
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value  # categorical, for example the objective
    return out
