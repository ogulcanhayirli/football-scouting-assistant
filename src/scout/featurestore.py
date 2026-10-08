"""Load the feature table into SageMaker Feature Store and check its point in time view.

Feature Store's model: a record is identified by ``record_id`` (here
``<league>#<player id>``) and stamped with ``event_time`` (here the feature
date). The offline store keeps every version, so "the features of every
player as of date D" is: per record, the latest event time on or before D.

``compare_point_in_time`` checks that the offline store answers that question
exactly as our own tables do. If both agree, Feature Store's point in time
semantics and ours are the same, and either can feed training.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

FEATURE_GROUP = "scout-player-season-features"
INGEST_ROLE = "scout-feature-ingest"
CHECK_COLUMNS = ["season", "minutes", "goals_per90", "rating", "market_value"]


def record_id(league_key: str, player_id: int) -> str:
    return f"{league_key}#{int(player_id)}"


def iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _value(value: Any) -> str | None:
    """Feature Store takes strings; missing values are left out of the record."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, pd.Timestamp):
        return iso(value)
    if isinstance(value, float):
        return repr(value)  # shortest string that round trips exactly
    return str(value)


def to_records(features: pd.DataFrame) -> list[list[dict[str, str]]]:
    """One PutRecord payload per feature row."""
    records = []
    for row in features.to_dict(orient="records"):
        record = [
            {
                "FeatureName": "record_id",
                "ValueAsString": record_id(row["league_key"], row["source_player_id"]),
            },
            {"FeatureName": "event_time", "ValueAsString": iso(row["feature_date"])},
        ]
        for name, value in row.items():
            text = _value(value)
            if text is not None:
                record.append({"FeatureName": name, "ValueAsString": text})
        records.append(record)
    return records


def ingest(
    client: Any,
    records: Iterable[list[dict[str, str]]],
    *,
    feature_group: str = FEATURE_GROUP,
    workers: int = 16,
) -> tuple[int, list[str]]:
    """Put every record; returns the number written and the errors, if any.

    Putting the same record again adds a new version with a later write time;
    point in time queries keep the latest write per record and event time.
    """
    errors: list[str] = []

    def put(record: list[dict[str, str]]) -> bool:
        try:
            client.put_record(FeatureGroupName=feature_group, Record=record)
            return True
        except Exception as exc:  # report every failure, keep going
            rid = next(f["ValueAsString"] for f in record if f["FeatureName"] == "record_id")
            errors.append(f"{rid}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(put, records))
    return written, errors


def point_in_time_sql(database: str, table: str, as_of: pd.Timestamp, columns: list[str]) -> str:
    """Latest version of every record with an event time on or before ``as_of``.

    Ranking happens before the ``is_deleted`` filter: if the latest version is
    a delete, the record is gone as of that date, not shown in an older state.
    """
    selected = ", ".join(["record_id", "event_time", *columns])
    return f"""
SELECT {selected}
FROM (
  SELECT *, row_number() OVER (
    PARTITION BY record_id
    ORDER BY from_iso8601_timestamp(event_time) DESC, api_invocation_time DESC, write_time DESC
  ) AS rn
  FROM "{database}"."{table}"
  WHERE from_iso8601_timestamp(event_time) <= from_iso8601_timestamp('{iso(as_of)}')
)
WHERE rn = 1 AND NOT is_deleted
""".strip()


def expected_point_in_time(features: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The same question answered from our own table."""
    known = features[features["feature_date"] <= as_of]
    latest = (
        known.sort_values("feature_date")
        .groupby(["league_key", "source_player_id"], as_index=False)
        .tail(1)
    )
    out = latest.assign(
        record_id=[
            record_id(lg, p)
            for lg, p in zip(latest["league_key"], latest["source_player_id"], strict=True)
        ],
        event_time=latest["feature_date"].map(iso),
    )
    return out.set_index("record_id").sort_index()


def compare_point_in_time(
    expected: pd.DataFrame, found: pd.DataFrame, columns: list[str] = CHECK_COLUMNS
) -> dict[str, Any]:
    """Row by row comparison; Athena returns strings, so values are compared as numbers."""
    found = found.set_index("record_id").sort_index()
    missing = sorted(set(expected.index) - set(found.index))
    extra = sorted(set(found.index) - set(expected.index))
    both = expected.index.intersection(found.index)
    mismatches: dict[str, int] = {}
    wrong_event = int((expected.loc[both, "event_time"] != found.loc[both, "event_time"]).sum())
    for column in columns:
        want = pd.to_numeric(expected.loc[both, column], errors="coerce").astype(float)
        got = pd.to_numeric(found.loc[both, column], errors="coerce").astype(float)
        same = (want.isna() & got.isna()) | ((want - got).abs() <= 1e-9 * want.abs().clip(lower=1))
        mismatches[column] = int((~same).sum())
    return {
        "expected_rows": len(expected),
        "found_rows": len(found),
        "missing": missing[:10],
        "missing_count": len(missing),
        "extra": extra[:10],
        "extra_count": len(extra),
        "wrong_event_time": wrong_event,
        "mismatches": mismatches,
        "identical": not missing
        and not extra
        and wrong_event == 0
        and not any(mismatches.values()),
    }
