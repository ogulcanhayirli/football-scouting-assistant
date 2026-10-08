import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template
from scout_infra.data_stack import (
    BYTES_SCANNED_CUTOFF,
    SCHEMA_DIR,
    TABLES,
    ScoutDataStack,
)

ACCOUNT = "123456789012"
BUCKET = "example-data-bucket"


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = ScoutDataStack(
        app,
        "ScoutData",
        data_bucket_name=BUCKET,
        session_user_name="scout-session-sync",
        env=cdk.Environment(account=ACCOUNT, region="eu-central-1"),
    )
    return Template.from_stack(stack)


def _resources(template, kind):
    return template.find_resources(kind)


def test_only_resources_the_deployment_policy_allows(template):
    kinds = {r["Type"] for r in template.to_json()["Resources"].values()}
    allowed = {
        "AWS::S3::Bucket",
        "AWS::S3::BucketPolicy",
        "AWS::Glue::Database",
        "AWS::Glue::Table",
        "AWS::Athena::WorkGroup",
        "AWS::IAM::Role",
        "AWS::IAM::Policy",
        "AWS::CDK::Metadata",
    }
    assert kinds <= allowed, kinds - allowed


def test_every_name_is_inside_the_allowed_prefixes(template):
    for bucket in _resources(template, "AWS::S3::Bucket").values():
        assert bucket["Properties"]["BucketName"].startswith("scout-")
    for role in _resources(template, "AWS::IAM::Role").values():
        assert role["Properties"]["RoleName"].startswith("scout-")


def test_results_bucket_is_private_encrypted_and_expiring(template):
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketName": f"scout-athena-results-{ACCOUNT}",
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
            "BucketEncryption": Match.object_like(
                {"ServerSideEncryptionConfiguration": [Match.any_value()]}
            ),
            "LifecycleConfiguration": {
                "Rules": [Match.object_like({"ExpirationInDays": 30, "Status": "Enabled"})]
            },
        },
    )
    template.has_resource("AWS::S3::Bucket", {"DeletionPolicy": "Retain"})


def test_workgroup_enforces_results_location_and_scan_limit(template):
    template.has_resource_properties(
        "AWS::Athena::WorkGroup",
        {
            "Name": "scout",
            "WorkGroupConfiguration": Match.object_like(
                {
                    "EnforceWorkGroupConfiguration": True,
                    "BytesScannedCutoffPerQuery": BYTES_SCANNED_CUTOFF,
                }
            ),
        },
    )


@pytest.mark.parametrize("table", list(TABLES))
def test_tables_match_the_exported_schema_and_use_projection(template, table):
    declared = json.loads((SCHEMA_DIR / f"{table}.json").read_text())
    found = [
        t["Properties"]["TableInput"]
        for t in _resources(template, "AWS::Glue::Table").values()
        if t["Properties"]["TableInput"]["Name"] == table
    ]
    assert len(found) == 1
    table_input = found[0]
    columns = table_input["StorageDescriptor"]["Columns"]
    assert [{"name": c["Name"], "type": c["Type"]} for c in columns] == declared
    names = {c["Name"] for c in columns}
    assert {"league", "season"}.isdisjoint(names)  # partition keys can't repeat columns
    params = table_input["Parameters"]
    assert params["projection.enabled"] == "true"
    assert params["storage.location.template"] == (
        f"s3://{BUCKET}/processed/features/{table}/league=${{league}}/season=${{season}}/"
    )


def test_analyst_role_is_assumable_only_by_the_session_user(template):
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "RoleName": "scout-analyst",
            "AssumeRolePolicyDocument": {
                "Statement": [
                    {
                        "Action": "sts:AssumeRole",
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": {
                                "Fn::Join": [
                                    "",
                                    [
                                        "arn:",
                                        {"Ref": "AWS::Partition"},
                                        f":iam::{ACCOUNT}:user/scout-session-sync",
                                    ],
                                ]
                            }
                        },
                    }
                ],
                "Version": "2012-10-17",
            },
        },
    )


def test_analyst_can_read_features_but_never_write_or_delete_data(template):
    statements = [
        s
        for p in _resources(template, "AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]
    actions = [
        a
        for s in statements
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    ]
    assert not [a for a in actions if "Delete" in a or a.endswith(":*")]
    data_writes = [
        s
        for s in statements
        if "s3:PutObject" in s["Action"] and BUCKET in json.dumps(s["Resource"])
    ]
    assert data_writes == []
    for s in statements:
        assert s["Resource"] != "*", s["Sid"]
