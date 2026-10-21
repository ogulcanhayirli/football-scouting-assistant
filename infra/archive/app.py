"""Archived CDK app of the retired stacks (README.md in this folder): run
``npx aws-cdk@2 diff`` or ``destroy`` from this folder only.

The account comes from the credentials the CDK CLI runs with, and the data
bucket from ``SCOUT_S3_BUCKET``, so no account specific value is committed.
"""

import os

import aws_cdk as cdk
from scout_infra.data_stack import ScoutDataStack
from scout_infra.feature_store_stack import ScoutFeatureStoreStack
from scout_infra.training_stack import ScoutTrainingStack

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
feature_store.add_stack_dependency(data)  # it extends the scout-analyst role from ScoutData
ScoutTrainingStack(
    app,
    "ScoutTraining",
    data_bucket_name=app.node.try_get_context("dataBucket") or os.environ["SCOUT_S3_BUCKET"],
    session_user_name=app.node.try_get_context("sessionUser"),
    env=cdk.Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=region),
    description="Football scouting assistant: SageMaker training and automatic model tuning",
)
cdk.Tags.of(app).add("project", "football-scouting-assistant")
app.synth()
