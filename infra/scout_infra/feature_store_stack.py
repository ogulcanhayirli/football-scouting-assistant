"""SageMaker Feature Store over the player season features (offline store only).

What it creates:

* Feature group ``scout-player-season-features``. Record identifier
  ``record_id`` (``<league>#<player id>``), event time ``event_time`` (the
  row's feature date). Offline store only, as Iceberg tables in Glue, so
  Athena reads it without partition repairs; no online store, which would
  bill for storage around the clock and which nothing needs yet.
* Bucket ``scout-feature-store-<account>`` for the offline store files.
* Role ``scout-feature-store``: SageMaker writes the offline store with it.
  Same actions as the AWS managed AmazonSageMakerFeatureStoreAccess policy,
  but limited to this bucket instead of every bucket named "sagemaker".
* Role ``scout-feature-ingest``: the session user assumes it to put records,
  into this one feature group only.
* Read access for the existing ``scout-analyst`` role to the offline store
  table, so ``scout athena`` can run point in time queries on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

SCHEMA_DIR = Path(__file__).parent / "schemas"
FEATURE_GROUP = "scout-player-season-features"
RECORD_ID = "record_id"
EVENT_TIME = "event_time"
OFFLINE_DATABASE = "sagemaker_featurestore"  # where Feature Store registers its tables
RESERVED = {"write_time", "api_invocation_time", "is_deleted"}  # added by the offline store

# Glue type of a feature table column -> Feature Store feature type. Feature
# Store has three types; booleans become 0 or 1, timestamps ISO 8601 strings.
FEATURE_TYPES = {
    "bigint": "Integral",
    "boolean": "Integral",
    "double": "Fractional",
    "string": "String",
    "timestamp": "String",
}


def feature_definitions() -> list[dict[str, str]]:
    """Record id and event time first, then ``season`` (a partition key in the
    Parquet layout, so not in the schema file), then every feature column."""
    columns = json.loads((SCHEMA_DIR / "player_season_features.json").read_text("utf-8"))
    names = {c["name"] for c in columns}
    clash = names & (RESERVED | {RECORD_ID, EVENT_TIME, "season"})
    if clash:
        raise ValueError(f"Feature names reserved by Feature Store or this stack: {clash}")
    return [
        {"name": RECORD_ID, "type": "String"},
        {"name": EVENT_TIME, "type": "String"},
        {"name": "season", "type": "Integral"},
        *({"name": c["name"], "type": FEATURE_TYPES[c["type"]]} for c in columns),
    ]


class ScoutFeatureStoreStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        session_user_name: str,
        analyst_role_name: str = "scout-analyst",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        account = self.account

        offline = s3.Bucket(
            self,
            "OfflineStore",
            bucket_name=f"scout-feature-store-{account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )

        glue_resources = [
            self.format_arn(service="glue", resource="catalog"),
            self.format_arn(service="glue", resource="database", resource_name=OFFLINE_DATABASE),
            self.format_arn(
                service="glue", resource="table", resource_name=f"{OFFLINE_DATABASE}/*"
            ),
        ]

        store_role = iam.Role(
            self,
            "FeatureStoreRole",
            role_name="scout-feature-store",
            description="SageMaker Feature Store writes the scout offline store with this role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        store_role.add_to_policy(
            iam.PolicyStatement(
                sid="WriteOfflineStore",
                actions=["s3:PutObject", "s3:PutObjectAcl", "s3:GetObject", "s3:GetBucketAcl"],
                resources=[offline.bucket_arn, offline.arn_for_objects("*")],
            )
        )
        store_role.add_to_policy(
            iam.PolicyStatement(
                sid="ListOfflineStore",  # Iceberg reads its own metadata files back
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[offline.bucket_arn],
            )
        )
        store_role.add_to_policy(
            iam.PolicyStatement(
                sid="UpdateOfflineTable",
                actions=["glue:GetTable", "glue:UpdateTable", "glue:GetDatabase"],
                resources=glue_resources,
            )
        )

        group = sagemaker.CfnFeatureGroup(
            self,
            "FeatureGroup",
            feature_group_name=FEATURE_GROUP,
            description="Player, league and season features as of each feature date",
            record_identifier_feature_name=RECORD_ID,
            event_time_feature_name=EVENT_TIME,
            feature_definitions=[
                sagemaker.CfnFeatureGroup.FeatureDefinitionProperty(
                    feature_name=f["name"], feature_type=f["type"]
                )
                for f in feature_definitions()
            ],
            offline_store_config={
                "S3StorageConfig": {"S3Uri": f"s3://{offline.bucket_name}/offline-store"},
                "TableFormat": "Iceberg",
                "DisableGlueTableCreation": False,
            },
            role_arn=store_role.role_arn,
        )
        group.node.add_dependency(store_role)
        group_arn = self.format_arn(
            service="sagemaker", resource="feature-group", resource_name=FEATURE_GROUP
        )

        session_user = iam.User.from_user_name(self, "SessionUser", session_user_name)
        ingest = iam.Role(
            self,
            "IngestRole",
            role_name="scout-feature-ingest",
            description="Put records into the scout feature group",
            assumed_by=iam.ArnPrincipal(session_user.user_arn),
            max_session_duration=cdk.Duration.hours(1),
        )
        ingest.add_to_policy(
            iam.PolicyStatement(
                sid="PutRecords",
                actions=["sagemaker:PutRecord", "sagemaker:DescribeFeatureGroup"],
                resources=[group_arn],
            )
        )

        analyst = iam.Role.from_role_name(self, "AnalystRole", analyst_role_name)
        iam.Policy(
            self,
            "AnalystReadsOfflineStore",
            policy_name="scout-read-offline-store",
            roles=[analyst],
            statements=[
                iam.PolicyStatement(
                    sid="DescribeGroup",
                    actions=["sagemaker:DescribeFeatureGroup"],
                    resources=[group_arn],
                ),
                iam.PolicyStatement(
                    sid="ReadOfflineCatalog",
                    actions=[
                        "glue:GetDatabase",
                        "glue:GetTable",
                        "glue:GetTables",
                        "glue:GetPartitions",
                    ],
                    resources=glue_resources,
                ),
                iam.PolicyStatement(
                    sid="ReadOfflineFiles",
                    actions=["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
                    resources=[offline.bucket_arn, offline.arn_for_objects("*")],
                ),
            ],
        )

        cdk.CfnOutput(self, "FeatureGroupName", value=FEATURE_GROUP)
        cdk.CfnOutput(self, "IngestRoleArn", value=ingest.role_arn)
        cdk.CfnOutput(self, "OfflineStoreBucket", value=offline.bucket_name)
