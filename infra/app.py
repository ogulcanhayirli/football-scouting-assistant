"""CDK entry point: ``npx aws-cdk@2 diff`` and ``deploy`` from this folder.

One stack since decision 9: ``ScoutSite``, the public bucket the weekly run
publishes the page to. The retired Glue, Feature Store and training stacks
are in ``archive/`` (its README says why). The account comes from the
credentials the CDK CLI runs with, so no account specific value is committed.
"""

import os

import aws_cdk as cdk
from scout_infra.site_stack import ScoutSiteStack

app = cdk.App()
ScoutSiteStack(
    app,
    "ScoutSite",
    session_user_name=app.node.try_get_context("sessionUser"),
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"], region=app.node.try_get_context("region")
    ),
    description="Football scouting assistant: public S3 website bucket for the weekly page",
)
cdk.Tags.of(app).add("project", "football-scouting-assistant")
app.synth()
