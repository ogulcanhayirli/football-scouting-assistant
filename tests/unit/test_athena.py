import pytest

from scout.athena import AthenaQueryError, run_query


class FakeAthena:
    def __init__(self, states, pages, reason=None):
        self.states = list(states)
        self.pages = pages
        self.reason = reason
        self.started = []
        self.stopped = []

    def start_query_execution(self, QueryString, WorkGroup, QueryExecutionContext):  # noqa: N803
        self.started.append((QueryString, WorkGroup, QueryExecutionContext["Database"]))
        return {"QueryExecutionId": "q1"}

    def get_query_execution(self, QueryExecutionId):  # noqa: N803
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        status = {"State": state}
        if self.reason:
            status["StateChangeReason"] = self.reason
        return {
            "QueryExecution": {
                "Status": status,
                "Statistics": {"DataScannedInBytes": 2_000_000, "TotalExecutionTimeInMillis": 900},
            }
        }

    def get_query_results(self, QueryExecutionId, MaxResults, NextToken=None):  # noqa: N803
        return self.pages[NextToken]

    def stop_query_execution(self, QueryExecutionId):  # noqa: N803
        self.stopped.append(QueryExecutionId)


def _row(*values):
    return {"Data": [{"VarCharValue": v} if v is not None else {} for v in values]}


META = {"ColumnInfo": [{"Name": "player"}, {"Name": "goals"}]}
PAGES = {
    None: {
        "ResultSet": {"ResultSetMetadata": META, "Rows": [_row("player", "goals"), _row("A", "3")]},
        "NextToken": "t2",
    },
    "t2": {"ResultSet": {"ResultSetMetadata": META, "Rows": [_row("B", None)]}},
}


def test_run_query_waits_pages_and_drops_the_header_row():
    client = FakeAthena(["QUEUED", "RUNNING", "SUCCEEDED"], PAGES)
    frame, stats = run_query(client, "SELECT 1", sleep=lambda _: None)
    assert client.started == [("SELECT 1", "scout", "scout")]
    assert frame["player"].tolist() == ["A", "B"]
    assert frame.loc[0, "goals"] == "3"
    assert frame["goals"].isna().tolist() == [False, True]  # Athena's NULL
    assert stats["scanned_bytes"] == 2_000_000


def test_failed_query_raises_with_athenas_reason():
    client = FakeAthena(["FAILED"], PAGES, reason="COLUMN_NOT_FOUND: x")
    with pytest.raises(AthenaQueryError, match="COLUMN_NOT_FOUND"):
        run_query(client, "SELECT x", sleep=lambda _: None)


def test_slow_query_is_stopped_at_the_timeout():
    client = FakeAthena(["RUNNING"], PAGES)
    with pytest.raises(AthenaQueryError, match="still RUNNING"):
        run_query(client, "SELECT 1", timeout_seconds=1, poll_seconds=0.5, sleep=lambda _: None)
    assert client.stopped == ["q1"]
