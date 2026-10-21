import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template
from scout_infra.feature_store_stack import (
    FEATURE_GROUP,
    SCHEMA_DIR,
    ScoutFeatureStoreStack,
    feature_definitions,
)

ACCOUNT = "123456789012"


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = ScoutFeatureStoreStack(
        app,
        "ScoutFeatureStore",
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


def test_only_resources_the_deployment_policy_allows(template):
    kinds = {r["Type"] for r in template.to_json()["Resources"].values()}
    assert kinds <= {
        "AWS::S3::Bucket",
        "AWS::S3::BucketPolicy",
        "AWS::SageMaker::FeatureGroup",
        "AWS::IAM::Role",
        "AWS::IAM::Policy",
        "AWS::CDK::Metadata",
    }
    for bucket in template.find_resources("AWS::S3::Bucket").values():
        assert bucket["Properties"]["BucketName"].startswith("scout-")
    for role in template.find_resources("AWS::IAM::Role").values():
        assert role["Properties"]["RoleName"].startswith("scout-")


def test_feature_group_is_offline_only_iceberg(template):
    template.has_resource_properties(
        "AWS::SageMaker::FeatureGroup",
        {
            "FeatureGroupName": FEATURE_GROUP,
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "OfflineStoreConfig": Match.object_like(
                {"TableFormat": "Iceberg", "DisableGlueTableCreation": False}
            ),
        },
    )
    group = next(iter(template.find_resources("AWS::SageMaker::FeatureGroup").values()))
    assert "OnlineStoreConfig" not in group["Properties"]


def test_every_feature_column_is_defined_once():
    columns = json.loads((SCHEMA_DIR / "player_season_features.json").read_text())
    definitions = feature_definitions()
    names = [d["name"] for d in definitions]
    assert len(names) == len(set(names))
    assert names[:3] == ["record_id", "event_time", "season"]
    assert names[3:] == [c["name"] for c in columns]
    assert {d["type"] for d in definitions} <= {"String", "Integral", "Fractional"}


def test_store_role_is_for_sagemaker_and_scoped_to_its_bucket(template):
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "RoleName": "scout-feature-store",
            "AssumeRolePolicyDocument": Match.object_like(
                {
                    "Statement": [
                        Match.object_like({"Principal": {"Service": "sagemaker.amazonaws.com"}})
                    ]
                }
            ),
        },
    )
    for statement in _statements(template):
        assert statement["Resource"] != "*"
        actions = (
            statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        )
        assert not [a for a in actions if "Delete" in a or a.endswith(":*")]


def test_ingest_role_may_only_put_records_into_the_group(template):
    puts = [s for s in _statements(template) if "sagemaker:PutRecord" in s["Action"]]
    assert len(puts) == 1
    assert FEATURE_GROUP in json.dumps(puts[0]["Resource"])
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "RoleName": "scout-feature-ingest",
            "AssumeRolePolicyDocument": Match.object_like(
                {"Statement": [Match.object_like({"Principal": {"AWS": Match.any_value()}})]}
            ),
        },
    )


def test_analyst_gets_read_only_access_to_the_offline_store(template):
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {"PolicyName": "scout-read-offline-store", "Roles": ["scout-analyst"]},
    )


def test_every_role_carries_the_permissions_boundary(template):
    roles = template.find_resources("AWS::IAM::Role")
    assert roles
    for role in roles.values():
        boundary = role["Properties"]["PermissionsBoundary"]
        assert "policy/scout-boundary" in str(boundary)
