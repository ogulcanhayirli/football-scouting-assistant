"""The public site bucket: S3 static website hosting, public read on this
bucket only (decision 9).

No CloudFront, no Lambda: the weekly GitHub Actions run (``scout publish``)
puts ``index.html`` and ``data/`` here, and the S3 website endpoint serves
them over plain HTTP. Anyone may read an object; nobody may list the bucket
or write to it without credentials. The data bucket is not touched and stays
private.

The website endpoint has no HTTPS, so the bucket policy cannot deny plain
HTTP the way the private buckets do; nothing secret is ever stored here.

``scout-site-publisher`` lets a session republish by hand (``scout publish``
after a manual retrain): the session user may assume it, and it may only put
objects into this bucket. The weekly run uses its own IAM user instead
(docs/DEPLOY.md).
"""

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

from scout_infra.boundary import apply_boundary

PUBLISHER_ROLE = "scout-site-publisher"


def site_bucket_name(account: str) -> str:
    return f"scout-site-{account}"


class ScoutSiteStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, session_user_name: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.bucket = s3.Bucket(
            self,
            "Site",
            bucket_name=site_bucket_name(self.account),
            website_index_document="index.html",
            public_read_access=True,  # a bucket policy: s3:GetObject on objects, for anyone
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=True,
                ignore_public_acls=True,
                block_public_policy=False,
                restrict_public_buckets=False,
            ),
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
        )
        session_user = iam.User.from_user_name(self, "SessionUser", session_user_name)
        publisher = iam.Role(
            self,
            "PublisherRole",
            role_name=PUBLISHER_ROLE,
            description="Put the weekly page and its downloads into the site bucket",
            assumed_by=iam.ArnPrincipal(session_user.user_arn),
            max_session_duration=Duration.hours(1),
        )
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="PutSiteObjects",
                actions=["s3:PutObject"],
                resources=[self.bucket.arn_for_objects("*")],
            )
        )
        apply_boundary(self)
        CfnOutput(self, "SiteBucket", value=self.bucket.bucket_name)
        CfnOutput(self, "PublisherRoleArn", value=publisher.role_arn)
        CfnOutput(self, "WebsiteUrl", value=self.bucket.bucket_website_url)
