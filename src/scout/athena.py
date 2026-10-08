"""Run Athena queries over the feature tables as the ``scout-analyst`` role.

``uv run scout athena "SELECT ..."`` assumes the role the ScoutData stack
creates (infra/), runs the query in the ``scout`` workgroup, waits for it and
prints the rows. The workgroup decides where results go and stops any query
that scans more than 1 GB.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

ROLE_NAME = "scout-analyst"
WORKGROUP = "scout"
DATABASE = "scout"
DONE = {"SUCCEEDED", "FAILED", "CANCELLED"}


class AthenaQueryError(RuntimeError):
    pass


def analyst_session(region: str, role_name: str = ROLE_NAME) -> Any:
    """A boto3 session holding temporary credentials of the analyst role."""
    import boto3

    sts = boto3.client("sts", region_name=region)
    account = sts.get_caller_identity()["Account"]
    creds = sts.assume_role(
        RoleArn=f"arn:aws:iam::{account}:role/{role_name}", RoleSessionName="scout-athena"
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=region,
    )


def run_query(
    client: Any,
    sql: str,
    *,
    workgroup: str = WORKGROUP,
    database: str = DATABASE,
    poll_seconds: float = 0.5,
    timeout_seconds: float = 120,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run ``sql`` and return the rows as strings plus the execution statistics."""
    query_id = client.start_query_execution(
        QueryString=sql,
        WorkGroup=workgroup,
        QueryExecutionContext={"Database": database},
    )["QueryExecutionId"]
    waited = 0.0
    while True:
        execution = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state in DONE:
            break
        if waited >= timeout_seconds:
            client.stop_query_execution(QueryExecutionId=query_id)
            raise AthenaQueryError(f"query {query_id} still {state} after {timeout_seconds}s")
        sleep(poll_seconds)
        waited += poll_seconds
    if state != "SUCCEEDED":
        reason = execution["Status"].get("StateChangeReason", "no reason given")
        raise AthenaQueryError(f"query {query_id} {state}: {reason}")

    rows: list[list[str | None]] = []
    columns: list[str] = []
    token = None
    while True:
        kwargs = {"QueryExecutionId": query_id, "MaxResults": 1000}
        if token:
            kwargs["NextToken"] = token
        page = client.get_query_results(**kwargs)
        if not columns:
            columns = [c["Name"] for c in page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
            data = page["ResultSet"]["Rows"][1:]  # the first row repeats the header
        else:
            data = page["ResultSet"]["Rows"]
        rows += [[cell.get("VarCharValue") for cell in row["Data"]] for row in data]
        token = page.get("NextToken")
        if not token:
            break
    stats = execution.get("Statistics", {})
    return pd.DataFrame(rows, columns=columns), {
        "query_id": query_id,
        "scanned_bytes": stats.get("DataScannedInBytes", 0),
        "runtime_ms": stats.get("TotalExecutionTimeInMillis", 0),
    }
