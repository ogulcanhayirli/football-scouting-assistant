"""CDK entry point: ``npx aws-cdk@2 diff`` and ``deploy`` from this folder.

Two stacks: ``ScoutSite``, the public bucket the weekly run publishes the
page to (decision 9), and ``ScoutChat``, the one Lambda behind the page's
Ask tab (prompt 20, decision 11; its code folder is built first by
``scout_infra.chat_bundle``; ``chatWarmPing`` in ``cdk.json`` adds its warm
ping once the deploy policy allows EventBridge rules). The retired Glue,
Feature Store and training stacks are in ``archive/`` (its README says
why). The account comes from the
credentials the CDK CLI runs with, so no account specific value is committed.
"""

import os

import aws_cdk as cdk
from aws_cdk import aws_lambda as lambda_
from scout_infra import chat_bundle
from scout_infra.chat_stack import ScoutChatStack
from scout_infra.site_stack import ScoutSiteStack

app = cdk.App()
env = cdk.Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=app.node.try_get_context("region")
)
ScoutSiteStack(
    app,
    "ScoutSite",
    session_user_name=app.node.try_get_context("sessionUser"),
    env=env,
    description="Football scouting assistant: public S3 website bucket for the weekly page",
)
ScoutChatStack(
    app,
    "ScoutChat",
    code=lambda_.Code.from_asset(str(chat_bundle.build())),
    warm_ping=str(app.node.try_get_context("chatWarmPing")).lower() == "true",
    env=env,
    description="Football scouting assistant: the Ask tab's chat Lambda and its caps' state",
)
cdk.Tags.of(app).add("project", "football-scouting-assistant")
app.synth()
