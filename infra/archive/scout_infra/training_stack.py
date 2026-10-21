"""SageMaker training and automatic model tuning for the backtest.

What it creates:

* Bucket ``scout-ml-<account>``: the code package (``code/``) and job outputs
  (``output/``). Outputs expire after 90 days; code packages are small and kept.
* Role ``scout-sagemaker-training``: SageMaker runs the jobs with it. Reads the
  feature tables, writes to the bucket, writes its logs and metrics, pulls the
  AWS training image.
* Role ``scout-ml-runner``: the session user assumes it to upload the code
  package and start, watch and stop training and tuning jobs named
  ``scout-*``. It may pass only the training role, and only to SageMaker.

Both roles carry the account's ``scout-boundary``, which already allows
exactly these actions (docs/AWS_SETUP.md, step 5).
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

from scout_infra.boundary import apply_boundary

FEATURES_PREFIX = "processed/features"
JOB_PREFIX = "scout-"


class ScoutTrainingStack(cdk.Stack):
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
        account, region = self.account, self.region
        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", data_bucket_name)

        ml = s3.Bucket(
            self,
            "MlBucket",
            bucket_name=f"scout-ml-{account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(prefix="output/", expiration=cdk.Duration.days(90))],
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )
        log_group = f"arn:aws:logs:{region}:{account}:log-group:/aws/sagemaker/TrainingJobs"

        training = iam.Role(
            self,
            "TrainingRole",
            role_name="scout-sagemaker-training",
            description="SageMaker runs the scout training and tuning jobs with this role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="ReadFeatures",
                actions=["s3:GetObject"],
                resources=[data_bucket.arn_for_objects(f"{FEATURES_PREFIX}/*")],
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="ListFeatures",
                actions=["s3:ListBucket"],
                resources=[data_bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [f"{FEATURES_PREFIX}/*"]}},
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="CodeAndOutputs",
                actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                resources=[ml.bucket_arn, ml.arn_for_objects("*")],
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="Logs",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[log_group, f"{log_group}:*"],
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="Metrics",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],  # PutMetricData has no resource level permissions
                conditions={
                    "StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/TrainingJobs"}
                },
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="PullTrainingImage",
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[f"arn:aws:ecr:{region}:763104351884:repository/pytorch-training"],
            )
        )
        training.add_to_policy(
            iam.PolicyStatement(
                sid="EcrToken",
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],  # the token call has no resource
            )
        )

        session_user = iam.User.from_user_name(self, "SessionUser", session_user_name)
        runner = iam.Role(
            self,
            "RunnerRole",
            role_name="scout-ml-runner",
            description="Start and watch the scout training and tuning jobs",
            assumed_by=iam.ArnPrincipal(session_user.user_arn),
            max_session_duration=cdk.Duration.hours(4),
        )
        jobs = [
            self.format_arn(
                service="sagemaker", resource="training-job", resource_name=f"{JOB_PREFIX}*"
            ),
            self.format_arn(
                service="sagemaker",
                resource="hyper-parameter-tuning-job",
                resource_name=f"{JOB_PREFIX}*",
            ),
        ]
        runner.add_to_policy(
            iam.PolicyStatement(
                sid="RunJobs",
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:CreateHyperParameterTuningJob",
                    "sagemaker:DescribeHyperParameterTuningJob",
                    "sagemaker:StopHyperParameterTuningJob",
                    "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                resources=jobs,
            )
        )
        runner.add_to_policy(
            iam.PolicyStatement(
                sid="ListJobs",
                actions=["sagemaker:ListTrainingJobs", "sagemaker:ListHyperParameterTuningJobs"],
                resources=["*"],  # list calls have no resource
            )
        )
        runner.add_to_policy(
            iam.PolicyStatement(
                sid="PassTrainingRole",
                actions=["iam:PassRole"],
                resources=[training.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            )
        )
        runner.add_to_policy(
            iam.PolicyStatement(
                sid="CodeAndOutputs",
                actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                resources=[ml.bucket_arn, ml.arn_for_objects("*")],
            )
        )
        runner.add_to_policy(
            iam.PolicyStatement(
                sid="ReadLogs",
                actions=["logs:DescribeLogStreams", "logs:GetLogEvents", "logs:FilterLogEvents"],
                resources=[log_group, f"{log_group}:*"],
            )
        )

        apply_boundary(self)

        cdk.CfnOutput(self, "MlBucketName", value=ml.bucket_name)
        cdk.CfnOutput(self, "RunnerRoleArn", value=runner.role_arn)
        cdk.CfnOutput(self, "TrainingRoleArn", value=training.role_arn)
