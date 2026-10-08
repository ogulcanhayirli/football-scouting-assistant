# AWS setup for deployments

Run once, by the account owner, with an administrator login. After this, the
session's IAM user (`scout-session-sync`) can deploy the project's CDK stacks
but nothing else: it may only assume CDK's bootstrap roles, and those roles
may only create what the policy below allows.

How it fits together:

```
scout-session-sync (access keys in the cloud session)
   │  sts:AssumeRole, nothing more
   ▼
cdk-hnb659fds-deploy-role ──► CloudFormation ──► cdk-hnb659fds-cfn-exec-role
                                                     │  limited by scout-cfn-exec
                                                     ▼
                                    Glue, Athena, scout-* buckets, scout-* roles,
                                    SageMaker feature groups
```

Everything below runs in **AWS CloudShell** (the `>_` icon in the console's top
bar), with the region set to **eu-central-1 (Frankfurt)**. CloudShell already
has the AWS CLI, Node.js and npm.

## 1. The policy that limits what deployments may create

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

cat > scout-cfn-exec.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Catalog",
      "Effect": "Allow",
      "Action": ["glue:*", "athena:*"],
      "Resource": "*"
    },
    {
      "Sid": "ProjectBuckets",
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::scout-*", "arn:aws:s3:::scout-*/*"]
    },
    {
      "Sid": "ProjectRoles",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole", "iam:DeleteRole", "iam:GetRole", "iam:UpdateRole",
        "iam:TagRole", "iam:UntagRole", "iam:UpdateAssumeRolePolicy",
        "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:GetRolePolicy",
        "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::${ACCOUNT}:role/scout-*"
    },
    {
      "Sid": "BootstrapVersionCheck",
      "Effect": "Allow",
      "Action": "ssm:GetParameters",
      "Resource": "arn:aws:ssm:eu-central-1:${ACCOUNT}:parameter/cdk-bootstrap/hnb659fds/version"
    },
    {
      "Sid": "FeatureStore",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateFeatureGroup", "sagemaker:DeleteFeatureGroup",
        "sagemaker:DescribeFeatureGroup", "sagemaker:UpdateFeatureGroup",
        "sagemaker:AddTags", "sagemaker:DeleteTags", "sagemaker:ListTags"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam create-policy --policy-name scout-cfn-exec --policy-document file://scout-cfn-exec.json
```

`BootstrapVersionCheck` is needed because every CDK template reads the
bootstrap version from SSM while CloudFormation runs it as the execution role.

To change the policy later, write the new JSON to the same file and add it as
the default version (a policy keeps at most five versions):

```bash
aws iam create-policy-version \
  --policy-arn "arn:aws:iam::${ACCOUNT}:policy/scout-cfn-exec" \
  --policy-document file://scout-cfn-exec.json --set-as-default
```

## 2. Bootstrap CDK in eu-central-1 with that policy

```bash
npx --yes aws-cdk@2 bootstrap "aws://${ACCOUNT}/eu-central-1" \
  --cloudformation-execution-policies "arn:aws:iam::${ACCOUNT}:policy/scout-cfn-exec"
```

This creates a `CDKToolkit` stack: a staging bucket and four roles named
`cdk-hnb659fds-*`. Without `--cloudformation-execution-policies`, CDK would
give deployments full administrator rights; the flag is what keeps them
narrow.

## 3. Let the session user assume the CDK roles

```bash
cat > scout-session-deploy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssumeCdkRoles",
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::${ACCOUNT}:role/cdk-hnb659fds-*-${ACCOUNT}-eu-central-1"
    },
    {
      "Sid": "AssumeProjectRoles",
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::${ACCOUNT}:role/scout-*"
    }
  ]
}
EOF

aws iam put-user-policy --user-name scout-session-sync \
  --policy-name scout-deploy --policy-document file://scout-session-deploy.json
```

The second statement lets the session assume roles the stacks create, such
as the role that runs Athena queries. Those roles still need to trust the
user, and their trust policies are written in the CDK code.

## 5. A permissions boundary on every project role

Steps 1 to 3 let deployments create roles named `scout-*` with **any**
policy inside, and the session may assume `scout-*` roles. So a stack could
in principle create a `scout-*` role with administrator rights. This step
closes that: `scout-boundary` is the most any `scout-*` role can ever do,
whatever its own policies say, and deployments may only create or change
roles that carry it. It covers what the deployed roles need today plus
SageMaker training and tuning, and contains no IAM rights except passing the
training role to SageMaker.

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=football-scout-data-ogulcanhayirli   # the data bucket
R=eu-central-1

cat > scout-boundary.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ProjectBuckets", "Effect": "Allow", "Action": "s3:*",
     "Resource": ["arn:aws:s3:::scout-*", "arn:aws:s3:::scout-*/*"]},
    {"Sid": "ReadDataBucket", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
     "Resource": ["arn:aws:s3:::${BUCKET}", "arn:aws:s3:::${BUCKET}/processed/*"]},
    {"Sid": "Catalog", "Effect": "Allow",
     "Action": ["glue:GetDatabase", "glue:GetDatabases", "glue:GetTable", "glue:GetTables",
                "glue:GetPartition", "glue:GetPartitions", "glue:BatchGetPartition",
                "glue:UpdateTable"],
     "Resource": ["arn:aws:glue:${R}:${ACCOUNT}:catalog",
                  "arn:aws:glue:${R}:${ACCOUNT}:database/scout",
                  "arn:aws:glue:${R}:${ACCOUNT}:table/scout/*",
                  "arn:aws:glue:${R}:${ACCOUNT}:database/sagemaker_featurestore",
                  "arn:aws:glue:${R}:${ACCOUNT}:table/sagemaker_featurestore/*"]},
    {"Sid": "Athena", "Effect": "Allow", "Action": "athena:*",
     "Resource": "arn:aws:athena:${R}:${ACCOUNT}:workgroup/scout"},
    {"Sid": "SageMaker", "Effect": "Allow",
     "Action": ["sagemaker:PutRecord", "sagemaker:DescribeFeatureGroup",
                "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob", "sagemaker:ListTrainingJobs",
                "sagemaker:CreateHyperParameterTuningJob",
                "sagemaker:DescribeHyperParameterTuningJob",
                "sagemaker:StopHyperParameterTuningJob",
                "sagemaker:ListHyperParameterTuningJobs",
                "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
                "sagemaker:AddTags", "sagemaker:ListTags"],
     "Resource": "*"},
    {"Sid": "PassTrainingRole", "Effect": "Allow", "Action": "iam:PassRole",
     "Resource": "arn:aws:iam::${ACCOUNT}:role/scout-sagemaker-training",
     "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
    {"Sid": "TrainingLogs", "Effect": "Allow",
     "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                "logs:DescribeLogStreams", "logs:GetLogEvents", "logs:FilterLogEvents"],
     "Resource": "arn:aws:logs:${R}:${ACCOUNT}:log-group:/aws/sagemaker/*"},
    {"Sid": "TrainingMetrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData",
     "Resource": "*"},
    {"Sid": "PullAwsTrainingImages", "Effect": "Allow",
     "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
     "Resource": "*"}
  ]
}
EOF

aws iam create-policy --policy-name scout-boundary --policy-document file://scout-boundary.json
```

Then a new version of the deployment policy: the role actions that change
what a role may do now require the boundary.

```bash
BOUNDARY="arn:aws:iam::${ACCOUNT}:policy/scout-boundary"

cat > scout-cfn-exec.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "Catalog", "Effect": "Allow", "Action": ["glue:*", "athena:*"], "Resource": "*"},
    {"Sid": "ProjectBuckets", "Effect": "Allow", "Action": "s3:*",
     "Resource": ["arn:aws:s3:::scout-*", "arn:aws:s3:::scout-*/*"]},
    {"Sid": "ProjectRolesWithBoundary", "Effect": "Allow",
     "Action": ["iam:CreateRole", "iam:PutRolePolicy", "iam:DeleteRolePolicy",
                "iam:AttachRolePolicy", "iam:DetachRolePolicy",
                "iam:PutRolePermissionsBoundary"],
     "Resource": "arn:aws:iam::${ACCOUNT}:role/scout-*",
     "Condition": {"StringEquals": {"iam:PermissionsBoundary": "${BOUNDARY}"}}},
    {"Sid": "ProjectRolesOther", "Effect": "Allow",
     "Action": ["iam:GetRole", "iam:DeleteRole", "iam:UpdateRole", "iam:TagRole",
                "iam:UntagRole", "iam:UpdateAssumeRolePolicy", "iam:GetRolePolicy",
                "iam:PassRole"],
     "Resource": "arn:aws:iam::${ACCOUNT}:role/scout-*"},
    {"Sid": "BootstrapVersionCheck", "Effect": "Allow", "Action": "ssm:GetParameters",
     "Resource": "arn:aws:ssm:${R}:${ACCOUNT}:parameter/cdk-bootstrap/hnb659fds/version"},
    {"Sid": "FeatureStore", "Effect": "Allow",
     "Action": ["sagemaker:CreateFeatureGroup", "sagemaker:DeleteFeatureGroup",
                "sagemaker:DescribeFeatureGroup", "sagemaker:UpdateFeatureGroup",
                "sagemaker:AddTags", "sagemaker:DeleteTags", "sagemaker:ListTags"],
     "Resource": "*"}
  ]
}
EOF

aws iam create-policy-version \
  --policy-arn "arn:aws:iam::${ACCOUNT}:policy/scout-cfn-exec" \
  --policy-document file://scout-cfn-exec.json --set-as-default
```

Why it holds: removing a boundary (`iam:DeleteRolePermissionsBoundary`) is
not allowed at all, setting one is only allowed to `scout-boundary`, and
deployments can't edit the boundary policy itself, because they have no
rights on IAM policies. The CDK app attaches the boundary to every role it
creates; the existing roles get it on their next deploy.

## 4. Recommended: a budget alarm

Billing, then Budgets, then Create budget: a monthly cost budget of 20 USD
with an email alert at 80% of actual and 100% of forecast.

## Undoing it

```bash
aws iam delete-user-policy --user-name scout-session-sync --policy-name scout-deploy
aws cloudformation delete-stack --stack-name CDKToolkit --region eu-central-1
aws iam delete-policy --policy-arn "arn:aws:iam::${ACCOUNT}:policy/scout-cfn-exec"
```

Delete the project's own stacks (`cdk destroy`) before the toolkit stack.
The CDK staging bucket is kept on delete and has to be emptied and removed by
hand.
