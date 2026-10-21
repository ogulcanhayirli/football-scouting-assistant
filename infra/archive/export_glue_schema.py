"""Write the Glue column lists for the feature tables, from their Parquet schema.

Usage (from the repository root):
    uv run python infra/archive/export_glue_schema.py [--root data/processed/features]

Archived with the ScoutData stack (infra/archive/README.md). The JSON files
under infra/archive/scout_infra/schemas/ are what that stack declares in
Glue; infra/archive/tests/test_glue_schema.py compared them with the
pipeline's output and is no longer run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_DIR = Path(__file__).parent / "scout_infra" / "schemas"
TABLES = ["player_season_features", "player_season_labels"]


def glue_type(arrow_type: pa.DataType) -> str:
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_int64(arrow_type):
        return "bigint"
    if pa.types.is_floating(arrow_type):
        return "double"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    raise ValueError(f"No Glue type for {arrow_type}")


def glue_columns(parquet_file: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(parquet_file)
    return [
        {"name": field.name, "type": glue_type(field.type)}
        for field in schema
        if not field.name.startswith("__")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/features"))
    args = parser.parse_args()
    for table in TABLES:
        sample = next(iter(sorted((args.root / table).glob("league=*/season=*/data.parquet"))))
        columns = glue_columns(sample)
        path = SCHEMA_DIR / f"{table}.json"
        path.write_text(json.dumps(columns, indent=1) + "\n", encoding="utf-8")
        print(f"{path}: {len(columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
