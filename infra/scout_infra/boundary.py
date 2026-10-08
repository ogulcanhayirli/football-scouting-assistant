"""Attach the account's ``scout-boundary`` permissions boundary to every role
a stack creates.

The boundary is created once by the account owner (docs/AWS_SETUP.md, step 5)
and caps what any ``scout-*`` role can do. The deployment policy only lets
CloudFormation create or change roles that carry it, so a role without it
would fail to deploy rather than slip through.
"""

from aws_cdk import Stack
from aws_cdk import aws_iam as iam

BOUNDARY_POLICY = "scout-boundary"


def apply_boundary(stack: Stack) -> None:
    policy = iam.ManagedPolicy.from_managed_policy_name(stack, "ScoutBoundary", BOUNDARY_POLICY)
    iam.PermissionsBoundary.of(stack).apply(policy)
