import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template
from scout_infra.site_stack import PUBLISHER_ROLE, ScoutSiteStack, site_bucket_name

ACCOUNT = "123456789012"


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = ScoutSiteStack(
        app,
        "ScoutSite",
        session_user_name="scout-session-sync",
        env=cdk.Environment(account=ACCOUNT, region="eu-central-1"),
    )
    return Template.from_stack(stack)


def test_only_the_bucket_its_policy_and_the_publisher_role(template):
    kinds = {r["Type"] for r in template.to_json()["Resources"].values()}
    # No CloudFront, no Lambda (an auto delete custom resource would add one).
    allowed = {
        "AWS::S3::Bucket",
        "AWS::S3::BucketPolicy",
        "AWS::IAM::Role",
        "AWS::IAM::Policy",
        "AWS::CDK::Metadata",
    }
    assert kinds <= allowed, kinds - allowed


def test_the_publisher_may_only_put_site_objects_and_carries_the_boundary(template):
    roles = template.find_resources("AWS::IAM::Role")
    assert len(roles) == 1
    role = next(iter(roles.values()))["Properties"]
    assert role["RoleName"] == PUBLISHER_ROLE
    assert "scout-boundary" in str(role["PermissionsBoundary"])
    trust = role["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["AWS"]
    assert "user/scout-session-sync" in str(trust)
    policy = next(iter(template.find_resources("AWS::IAM::Policy").values()))
    statements = policy["Properties"]["PolicyDocument"]["Statement"]
    assert [s["Action"] for s in statements] == ["s3:PutObject"]


def test_the_bucket_is_a_website_inside_the_allowed_prefix(template):
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketName": site_bucket_name(ACCOUNT),
            "WebsiteConfiguration": {"IndexDocument": "index.html"},
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": False,
                "RestrictPublicBuckets": False,
            },
        },
    )
    assert site_bucket_name(ACCOUNT).startswith("scout-")


def test_the_public_may_only_read_objects(template):
    policies = template.find_resources("AWS::S3::BucketPolicy")
    assert len(policies) == 1
    statements = next(iter(policies.values()))["Properties"]["PolicyDocument"]["Statement"]
    public = [s for s in statements if s.get("Principal") == {"AWS": "*"}]
    assert len(public) == 1
    assert public[0]["Effect"] == "Allow"
    assert public[0]["Action"] == "s3:GetObject"
    resource = public[0]["Resource"]["Fn::Join"][1]
    assert resource[-1] == "/*"  # objects only: listing the bucket stays closed


def test_the_bucket_outlives_the_stack(template):
    bucket = next(iter(template.find_resources("AWS::S3::Bucket").values()))
    assert bucket["DeletionPolicy"] == "Retain"
