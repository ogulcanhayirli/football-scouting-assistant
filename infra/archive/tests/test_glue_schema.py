"""The Glue tables in infra/ must declare exactly what the feature pipeline writes.

Archived, not run: it needs the main package and tests/unit/test_features.py
on the path (it lived in tests/unit until the stack was archived)."""

import importlib.util
import json
from pathlib import Path

from test_features import _inputs

from scout.features import build_features
from scout.features.pipeline import _write_partitioned

ROOT = Path(__file__).parents[3]


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_glue_schema", ROOT / "infra" / "archive" / "export_glue_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glue_schemas_match_the_pipeline_output(tmp_path):
    exporter = load_exporter()
    features, label_table = build_features(_inputs())
    _write_partitioned(features, tmp_path / "player_season_features")
    _write_partitioned(label_table, tmp_path / "player_season_labels")
    for table in exporter.TABLES:
        written = next((tmp_path / table).glob("league=*/season=*/data.parquet"))
        declared = json.loads((exporter.SCHEMA_DIR / f"{table}.json").read_text())
        assert exporter.glue_columns(written) == declared, (
            f"{table}: run scripts/export_glue_schema.py and redeploy the data stack"
        )
