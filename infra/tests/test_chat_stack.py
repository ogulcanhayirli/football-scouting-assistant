import json

import aws_cdk as cdk
import pytest
from aws_cdk import aws_lambda as lambda_
from aws_cdk.assertions import Match, Template
from scout_infra import chat_bundle
from scout_infra.chat_stack import (
    FUNCTION,
    KEY_PARAMETER,
    ROLE,
    ScoutChatStack,
    chat_bucket_name,
    site_origin,
)
from scout_infra.site_stack import site_bucket_name

ACCOUNT = "123456789012"
REGION = "eu-central-1"


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Template:
    code = tmp_path_factory.mktemp("code")
    (code / "index.py").write_text("def handler(event, context):\n    return {}\n")
    app = cdk.App()
    stack = ScoutChatStack(
        app,
        "ScoutChat",
        code=lambda_.Code.from_asset(str(code)),
        env=cdk.Environment(account=ACCOUNT, region=REGION),
    )
    return Template.from_stack(stack)


def test_one_function_its_url_role_logs_and_the_state_bucket(template):
    kinds = {r["Type"] for r in template.to_json()["Resources"].values()}
    allowed = {
        "AWS::Lambda::Function",
        "AWS::Lambda::Url",
        "AWS::Lambda::Permission",
        "AWS::IAM::Role",
        "AWS::IAM::Policy",
        "AWS::S3::Bucket",
        "AWS::S3::BucketPolicy",
        "AWS::Logs::LogGroup",
        "AWS::CDK::Metadata",
    }
    assert kinds <= allowed, kinds - allowed
    # One function: no custom resource Lambdas (log retention, auto delete).
    template.resource_count_is("AWS::Lambda::Function", 1)
    template.resource_count_is("AWS::IAM::Role", 1)


def test_the_function(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "FunctionName": FUNCTION,
            "Runtime": "python3.12",
            "Architectures": ["arm64"],
            "Handler": "scout.chat.handler.handler",
            "MemorySize": 512,
            "Timeout": 180,
        },
    )
    # No reserved concurrency while the account's Lambda limit is 10 (the
    # owner's call, 26 October): reserving 2 would leave fewer than 10 free.
    fn = next(iter(template.find_resources("AWS::Lambda::Function").values()))
    assert "ReservedConcurrentExecutions" not in fn["Properties"]


def test_the_key_is_a_noecho_parameter_never_a_value(template):
    params = template.to_json()["Parameters"]
    assert params[KEY_PARAMETER]["NoEcho"] is True
    fn = next(iter(template.find_resources("AWS::Lambda::Function").values()))
    env = fn["Properties"]["Environment"]["Variables"]
    assert env["ANTHROPIC_API_KEY"] == {"Ref": KEY_PARAMETER}
    assert env["SITE_BUCKET"] == site_bucket_name(ACCOUNT)
    assert "sk-ant" not in json.dumps(template.to_json())


def test_a_public_url_open_to_the_site_only(template):
    template.has_resource_properties(
        "AWS::Lambda::Url",
        {
            "AuthType": "NONE",
            "Cors": {
                "AllowOrigins": [site_origin(ACCOUNT, REGION)],
                "AllowMethods": ["POST"],
            },
        },
    )
    assert site_origin(ACCOUNT, REGION) == (
        f"http://scout-site-{ACCOUNT}.s3-website.eu-central-1.amazonaws.com"
    )
    template.has_resource_properties(
        "AWS::Lambda::Permission",
        {"Action": "lambda:InvokeFunctionUrl", "Principal": "*", "FunctionUrlAuthType": "NONE"},
    )


def test_the_role_reads_the_page_keeps_state_and_carries_the_boundary(template):
    role = next(iter(template.find_resources("AWS::IAM::Role").values()))["Properties"]
    assert role["RoleName"] == ROLE
    assert role["RoleName"].startswith("scout-")
    assert "scout-boundary" in json.dumps(role["PermissionsBoundary"])
    policy = next(iter(template.find_resources("AWS::IAM::Policy").values()))
    statements = policy["Properties"]["PolicyDocument"]["Statement"]
    actions = sorted(
        a
        for s in statements
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    )
    assert actions == [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "s3:GetObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject",
    ]
    text = json.dumps(statements)
    assert "index.html" in text
    assert "/data/*" in text
    assert "/state/*" in text
    assert "s3:DeleteObject" not in text
    assert '"*"' not in text


def test_the_state_bucket_is_private_and_retained(template):
    template.has_resource(
        "AWS::S3::Bucket",
        {
            "Properties": Match.object_like(
                {
                    "BucketName": chat_bucket_name(ACCOUNT),
                    "PublicAccessBlockConfiguration": {
                        "BlockPublicAcls": True,
                        "BlockPublicPolicy": True,
                        "IgnorePublicAcls": True,
                        "RestrictPublicBuckets": True,
                    },
                }
            ),
            "DeletionPolicy": "Retain",
        },
    )
    assert chat_bucket_name(ACCOUNT).startswith("scout-")


def test_logs_are_kept_a_month(template):
    template.has_resource_properties(
        "AWS::Logs::LogGroup", {"LogGroupName": f"/aws/lambda/{FUNCTION}", "RetentionInDays": 30}
    )


def test_the_bundle_holds_only_the_chat_package(tmp_path):
    copied = chat_bundle.copy_code(tmp_path)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in copied)
    assert names[0] == "scout/__init__.py"
    assert "scout/chat/handler.py" in names
    assert all(n.startswith("scout/chat/") for n in names[1:])
    assert not (tmp_path / "scout" / "lists.py").exists()
    pins = chat_bundle.REQUIREMENTS.read_text().splitlines()
    assert "anthropic==2.3.0" in pins
