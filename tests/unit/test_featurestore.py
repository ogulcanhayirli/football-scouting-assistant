import numpy as np
import pandas as pd

from scout.featurestore import (
    compare_point_in_time,
    expected_point_in_time,
    ingest,
    point_in_time_sql,
    to_records,
)


def _features():
    return pd.DataFrame(
        {
            "league_key": ["epl", "epl", "epl", "super_lig"],
            "season": pd.array([2021, 2022, 2022, 2022], dtype="Int64"),
            "source_player_id": pd.array([10, 10, 11, 10], dtype="Int64"),
            "feature_date": pd.to_datetime(
                ["2022-06-30", "2023-06-30", "2023-06-30", "2023-06-30"]
            ),
            "regular": [True, False, True, True],
            "minutes": pd.array([900.0, 300.0, None, 1200.0], dtype="Float64"),
            "goals_per90": [0.1, 0.30000000000000004, np.nan, 0.5],
            "rating": pd.array([7.0, 6.5, None, 7.2], dtype="Float64"),
            "market_value": [1e6, 2e6, np.nan, 3e6],
            "player_name": ["A", "A", "B", "A"],
        }
    )


def test_records_carry_id_event_time_and_skip_missing_values():
    records = to_records(_features())
    first = {f["FeatureName"]: f["ValueAsString"] for f in records[0]}
    assert first["record_id"] == "epl#10"
    assert first["event_time"] == "2022-06-30T00:00:00Z"
    assert first["regular"] == "1"
    assert first["season"] == "2021"
    second = {f["FeatureName"]: f["ValueAsString"] for f in records[1]}
    assert float(second["goals_per90"]) == 0.30000000000000004  # exact round trip
    third = {f["FeatureName"] for f in records[2]}
    assert {"minutes", "goals_per90", "rating", "market_value"}.isdisjoint(third)
    assert records[3][0]["ValueAsString"] == "super_lig#10"  # same player, other league


def test_ingest_reports_failures_and_keeps_going():
    class Client:
        def __init__(self):
            self.put = []

        def put_record(self, FeatureGroupName, Record):  # noqa: N803
            rid = Record[0]["ValueAsString"]
            if rid == "epl#11":
                raise RuntimeError("ValidationException")
            self.put.append(rid)

    client = Client()
    written, errors = ingest(client, to_records(_features()), workers=2)
    assert written == 3
    assert errors == ["epl#11: ValidationException"]


def test_expected_point_in_time_takes_the_latest_row_on_or_before_the_date():
    before = expected_point_in_time(_features(), pd.Timestamp("2023-01-01"))
    assert before.index.tolist() == ["epl#10"]
    assert before.loc["epl#10", "season"] == 2021
    after = expected_point_in_time(_features(), pd.Timestamp("2023-06-30"))
    assert after.index.tolist() == ["epl#10", "epl#11", "super_lig#10"]
    assert after.loc["epl#10", "season"] == 2022


def test_point_in_time_sql_ranks_before_dropping_deletes():
    sql = point_in_time_sql("sagemaker_featurestore", "t", pd.Timestamp("2023-01-01"), ["season"])
    assert "from_iso8601_timestamp('2023-01-01T00:00:00Z')" in sql
    assert sql.index("row_number()") < sql.index("NOT is_deleted")
    assert "write_time DESC" in sql


def test_compare_finds_identical_and_different_answers():
    expected = expected_point_in_time(_features(), pd.Timestamp("2023-06-30"))
    as_strings = expected.reset_index()[
        ["record_id", "event_time", "season", "minutes", "goals_per90", "rating", "market_value"]
    ].astype(object)
    as_strings = as_strings.map(lambda v: None if pd.isna(v) else str(v))
    assert compare_point_in_time(expected, as_strings)["identical"]

    wrong = as_strings.copy()
    wrong.loc[0, "season"] = "2021"  # an older version leaked through
    wrong = wrong[wrong["record_id"] != "epl#11"]
    report = compare_point_in_time(expected, wrong)
    assert not report["identical"]
    assert report["mismatches"]["season"] == 1
    assert report["missing"] == ["epl#11"]
