"""The chat behind the page's Ask tab (prompt 20, decision 11).

Decision 9 is revised narrowly: one Lambda for the chat and nothing else;
the page stays static and the weekly run is unchanged.

* ``scout-chat``: Python 3.12 on arm64, 1,024 MB (512 until prompt 20b: twice
  the CPU halves the cold start), 180 seconds, behind a public Function URL
  (no API Gateway) whose CORS allows only the site's origin. The code is
  ``src/scout/chat`` with its pinned dependencies
  (``scout_infra.chat_bundle``). No reserved concurrency for now: the
  account's Lambda limit is 10 (10 must stay unreserved), so the function
  shares those 10 and they are its cap until the quota increase is granted.
* Its role, ``scout-chat-lambda`` under ``scout-boundary``, may read the
  page and ``data/*`` in the site bucket, read and write ``state/*`` in the
  chat bucket, and write its own log group. Nothing else.
* ``scout-chat-<account>``: a private bucket for the caps' state (the
  site bucket is public, so the state cannot live there), retained on
  delete.
* With ``warm_ping`` (the context value ``chatWarmPing``, prompt 20b), the
  EventBridge rule ``scout-chat-warm`` invokes the function every 5 minutes
  with ``{"warm": true}``: it loads the tables and calls no model, so a
  visitor rarely meets a cold start. It needs EventBridge rights in
  ``scout-cfn-exec`` (docs/DEPLOY.md); the owner added them and the app
  switched it on on 27 October.
* The Claude key is a NoEcho stack parameter, given at deploy time
  (``--parameters ScoutChat:AnthropicApiKey=...`` from the environment), so
  it is never in the synthesized template, ``cdk.out`` or any other file;
  later deploys keep the previous value unless a new one is given.
"""

from aws_cdk import CfnOutput, CfnParameter, Duration, RemovalPolicy, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

from scout_infra.boundary import apply_boundary
from scout_infra.site_stack import site_bucket_name

FUNCTION = "scout-chat"
ROLE = "scout-chat-lambda"
KEY_PARAMETER = "AnthropicApiKey"
WARM_RULE = "scout-chat-warm"
WARM_MINUTES = 5
MEMORY_MB = 1024


def chat_bucket_name(account: str) -> str:
    return f"scout-chat-{account}"


def site_origin(account: str, region: str) -> str:
    """The page's origin: the S3 website endpoint of the site bucket."""
    return f"http://{site_bucket_name(account)}.s3-website.{region}.amazonaws.com"


class ScoutChatStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        code: lambda_.Code,
        warm_ping: bool = False,
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)
        key = CfnParameter(
            self,
            KEY_PARAMETER,
            type="String",
            no_echo=True,
            min_length=1,
            description="Claude API key for the chat, from the deploying environment",
        )
        site = s3.Bucket.from_bucket_name(self, "Site", site_bucket_name(self.account))
        state = s3.Bucket(
            self,
            "State",
            bucket_name=chat_bucket_name(self.account),
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/aws/lambda/{FUNCTION}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )
        role = iam.Role(
            self,
            "Role",
            role_name=ROLE,
            description="The chat Lambda: read the published page, keep the caps' state",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadThePublishedPage",
                actions=["s3:GetObject"],
                resources=[site.arn_for_objects("index.html"), site.arn_for_objects("data/*")],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ListTheStateBucket", actions=["s3:ListBucket"], resources=[state.bucket_arn]
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="KeepTheCapsState",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[state.arn_for_objects("state/*")],
            )
        )
        log_group.grant_write(role)
        fn = lambda_.Function(
            self,
            "Chat",
            function_name=FUNCTION,
            description="The Ask tab's chat over the published tables (prompt 20)",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="scout.chat.handler.handler",
            code=code,
            memory_size=MEMORY_MB,
            timeout=Duration.seconds(180),
            # TODO(owner): reserved_concurrent_executions=2 once the Lambda quota passes 10.
            role=role,
            log_group=log_group,
            environment={
                "SITE_BUCKET": site.bucket_name,
                "STATE_BUCKET": state.bucket_name,
                "ANTHROPIC_API_KEY": key.value_as_string,
            },
        )
        url = fn.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors=lambda_.FunctionUrlCorsOptions(
                allowed_origins=[site_origin(self.account, self.region)],
                allowed_methods=[lambda_.HttpMethod.POST],
                allowed_headers=["content-type"],
                max_age=Duration.hours(1),
            ),
        )
        if warm_ping:
            events.Rule(
                self,
                "Warm",
                rule_name=WARM_RULE,
                description="Keeps the chat warm: loads the tables, calls no model (prompt 20b)",
                schedule=events.Schedule.rate(Duration.minutes(WARM_MINUTES)),
                targets=[
                    targets.LambdaFunction(
                        fn,
                        event=events.RuleTargetInput.from_object({"warm": True}),
                        retry_attempts=0,
                    )
                ],
            )
        apply_boundary(self)
        CfnOutput(self, "ChatUrl", value=url.url)
        CfnOutput(self, "StateBucket", value=state.bucket_name)
