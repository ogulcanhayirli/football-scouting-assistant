"""CDK entry point: ``npx aws-cdk@2 diff`` and ``deploy`` from this folder.

The account comes from the credentials the CDK CLI runs with, and the data
bucket from ``SCOUT_S3_BUCKET``, so no account specific value is committed.
"""

import os

import aws_cdk as cdk
from scout_infra.data_stack import ScoutDataStack
from scout_infra.feature_store_stack import ScoutFeatureStoreStack

app = cdk.App()
region = app.node.try_get_context("region")
data = ScoutDataStack(
    app,
    "ScoutData",
    data_bucket_name=app.node.try_get_context("dataBucket") or os.environ["SCOUT_S3_BUCKET"],
    session_user_name=app.node.try_get_context("sessionUser"),
    env=cdk.Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=region),
    description="Football scouting assistant: Glue catalog and Athena over the feature tables",
)
feature_store = ScoutFeatureStoreStack(
    app,
    "ScoutFeatureStore",
    session_user_name=app.node.try_get_context("sessionUser"),
    env=cdk.Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=region),
    description="Football scouting assistant: SageMaker Feature Store, offline store only",
)
feature_store.add_dependency(data)  # it extends the scout-analyst role from ScoutData
cdk.Tags.of(app).add("project", "football-scouting-assistant")
app.synth()
