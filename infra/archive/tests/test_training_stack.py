import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template
from scout_infra.training_stack import ScoutTrainingStack

ACCOUNT = "123456789012"
BUCKET = "example-data-bucket"


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = ScoutTrainingStack(
        app,
        "ScoutTraining",
        data_bucket_name=BUCKET,
        session_user_name="scout-session-sync",
        env=cdk.Environment(account=ACCOUNT, region="eu-central-1"),
    )
    return Template.from_stack(stack)


def _statements(template):
    return [
        s
        for p in template.find_resources("AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]


def _actions(statement):
    return statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]


def test_only_allowed_resources_with_scout_names(template):
    kinds = {r["Type"] for r in template.to_json()["Resources"].values()}
    assert kinds <= {
        "AWS::S3::Bucket",
        "AWS::S3::BucketPolicy",
        "AWS::IAM::Role",
        "AWS::IAM::Policy",
        "AWS::CDK::Metadata",
    }
    for bucket in template.find_resources("AWS::S3::Bucket").values():
        assert bucket["Properties"]["BucketName"].startswith("scout-")
    for role in template.find_resources("AWS::IAM::Role").values():
        assert role["Properties"]["RoleName"].startswith("scout-")
        assert "policy/scout-boundary" in str(role["Properties"]["PermissionsBoundary"])


def test_runner_may_only_pass_the_training_role_to_sagemaker(template):
    passes = [s for s in _statements(template) if "iam:PassRole" in _actions(s)]
    assert len(passes) == 1
    assert passes[0]["Condition"] == {
        "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
    }
    assert "TrainingRole" in json.dumps(passes[0]["Resource"])


def test_jobs_are_limited_to_scout_names_and_nothing_deletes(template):
    for statement in _statements(template):
        actions = _actions(statement)
        assert not [a for a in actions if "Delete" in a or a.endswith(":*")], statement["Sid"]
        if any(a.startswith("sagemaker:Create") for a in actions):
            assert "/scout-*" in json.dumps(statement["Resource"])


def test_training_reads_but_never_writes_the_data_bucket(template):
    for statement in _statements(template):
        if BUCKET in json.dumps(statement["Resource"]):
            assert set(_actions(statement)) <= {"s3:GetObject", "s3:ListBucket"}


def test_unscoped_resources_are_only_where_aws_offers_no_scope(template):
    unscoped = {a for s in _statements(template) if s["Resource"] == "*" for a in _actions(s)}
    assert unscoped <= {
        "cloudwatch:PutMetricData",
        "ecr:GetAuthorizationToken",
        "sagemaker:ListTrainingJobs",
        "sagemaker:ListHyperParameterTuningJobs",
    }
