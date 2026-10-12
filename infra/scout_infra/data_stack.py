"""Glue catalog, Athena workgroup and a query role over the feature tables.

What it creates:

* Glue database ``scout`` with the tables ``player_season_features`` and
  ``player_season_labels``. They point at the Parquet files ``scout sync``
  uploads to the existing data bucket, and use partition projection, so new
  seasons are visible without crawlers or ``MSCK REPAIR``.
* Athena workgroup ``scout``: query results go to its own bucket, every query
  must use that location, and a query stops after scanning 1 GB.
* Bucket ``scout-athena-results-<account>`` for query results, private,
  encrypted, results expire after 30 days.
* Role ``scout-analyst``: the session user may assume it to run Athena
  queries. It can read only the feature tables and write only query results.

Every name starts with ``scout-`` or lives in Glue and Athena, which is all
the deployment policy (docs/AWS_SETUP.md) allows.
"""

from __future__ import annotations

import json
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_athena as athena
from aws_cdk import aws_glue as glue
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

from scout_infra.boundary import apply_boundary

SCHEMA_DIR = Path(__file__).parent / "schemas"
DATABASE = "scout"
WORKGROUP = "scout"
FEATURES_PREFIX = "processed/features"
TABLES = {
    "player_season_features": "One row per player, league and season, as of its feature date",
    "player_season_labels": "Fair value and growth targets for each feature row",
}
# Partition values of the league key; must match configs/backfill.ultra.yaml
# (tests/unit/test_infra_config.py checks it).
LEAGUES = [
    "epl",
    "super_lig",
    "la_liga",
    "bundesliga",
    "serie_a",
    "ligue_1",
    "eredivisie",
    "primeira_liga",
    "belgian_pro_league",
    "championship",
    "scottish_premiership",
    "danish_superliga",
    "greek_super_league",
    "russian_premier_league",
    "ukrainian_premier_league",
    "brasileirao",
    "liga_profesional",
]
SEASON_RANGE = (2010, 2030)
BYTES_SCANNED_CUTOFF = 1024**3  # 1 GB per query; the whole feature set is about 6 MB


def load_columns(table: str) -> list[dict[str, str]]:
    return json.loads((SCHEMA_DIR / f"{table}.json").read_text(encoding="utf-8"))


class ScoutDataStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        data_bucket_name: str,
        session_user_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        account = self.account
        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", data_bucket_name)

        results = s3.Bucket(
            self,
            "AthenaResults",
            bucket_name=f"scout-athena-results-{account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(expiration=cdk.Duration.days(30))],
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )

        database = glue.CfnDatabase(
            self,
            "Database",
            catalog_id=account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=DATABASE,
                description="Football scouting assistant: point in time features and labels",
            ),
        )

        tables = {}
        for name, description in TABLES.items():
            location = f"s3://{data_bucket_name}/{FEATURES_PREFIX}/{name}/"
            table = glue.CfnTable(
                self,
                f"Table{''.join(p.title() for p in name.split('_'))}",
                catalog_id=account,
                database_name=DATABASE,
                table_input=glue.CfnTable.TableInputProperty(
                    name=name,
                    description=description,
                    table_type="EXTERNAL_TABLE",
                    partition_keys=[
                        glue.CfnTable.ColumnProperty(name="league", type="string"),
                        glue.CfnTable.ColumnProperty(name="season", type="int"),
                    ],
                    parameters={
                        "classification": "parquet",
                        "EXTERNAL": "TRUE",
                        "projection.enabled": "true",
                        "projection.league.type": "enum",
                        "projection.league.values": ",".join(LEAGUES),
                        "projection.season.type": "integer",
                        "projection.season.range": ",".join(map(str, SEASON_RANGE)),
                        "storage.location.template": location
                        + "league=${league}/season=${season}/",
                    },
                    storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                        location=location,
                        columns=[
                            glue.CfnTable.ColumnProperty(name=c["name"], type=c["type"])
                            for c in load_columns(name)
                        ],
                        input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                        output_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                        serde_info=glue.CfnTable.SerdeInfoProperty(
                            serialization_library="org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
                        ),
                    ),
                ),
            )
            table.node.add_dependency(database)
            tables[name] = table

        workgroup = athena.CfnWorkGroup(
            self,
            "WorkGroup",
            name=WORKGROUP,
            description="Queries over the scouting feature tables",
            recursive_delete_option=True,
            state="ENABLED",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                enforce_work_group_configuration=True,
                publish_cloud_watch_metrics_enabled=True,
                bytes_scanned_cutoff_per_query=BYTES_SCANNED_CUTOFF,
                engine_version=athena.CfnWorkGroup.EngineVersionProperty(
                    selected_engine_version="Athena engine version 3"
                ),
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{results.bucket_name}/results/",
                    encryption_configuration=athena.CfnWorkGroup.EncryptionConfigurationProperty(
                        encryption_option="SSE_S3"
                    ),
                ),
            ),
        )

        session_user = iam.User.from_user_name(self, "SessionUser", session_user_name)
        analyst = iam.Role(
            self,
            "AnalystRole",
            role_name="scout-analyst",
            description="Run Athena queries over the scouting feature tables",
            assumed_by=iam.ArnPrincipal(session_user.user_arn),
            max_session_duration=cdk.Duration.hours(1),
        )
        workgroup_arn = self.format_arn(
            service="athena", resource="workgroup", resource_name=WORKGROUP
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="RunQueries",
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution",
                    "athena:GetWorkGroup",
                ],
                resources=[workgroup_arn],
            )
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="ReadCatalog",
                actions=[
                    "glue:GetDatabase",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:GetPartitions",
                ],
                resources=[
                    self.format_arn(service="glue", resource="catalog"),
                    self.format_arn(service="glue", resource="database", resource_name=DATABASE),
                    self.format_arn(
                        service="glue", resource="table", resource_name=f"{DATABASE}/*"
                    ),
                ],
            )
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="ReadFeatureFiles",
                actions=["s3:GetObject"],
                resources=[data_bucket.arn_for_objects(f"{FEATURES_PREFIX}/*")],
            )
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="ListFeatureFiles",
                actions=["s3:ListBucket"],
                resources=[data_bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [f"{FEATURES_PREFIX}/*"]}},
            )
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="LocateDataBucket",  # carries no prefix, so it can't share the condition
                actions=["s3:GetBucketLocation"],
                resources=[data_bucket.bucket_arn],
            )
        )
        analyst.add_to_policy(
            iam.PolicyStatement(
                sid="QueryResults",
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[results.bucket_arn, results.arn_for_objects("*")],
            )
        )

        apply_boundary(self)

        cdk.CfnOutput(self, "AnalystRoleArn", value=analyst.role_arn)
        cdk.CfnOutput(self, "WorkGroupName", value=workgroup.name)
        cdk.CfnOutput(self, "ResultsBucket", value=results.bucket_name)
        cdk.CfnOutput(self, "GlueDatabase", value=DATABASE)
