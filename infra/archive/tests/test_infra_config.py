"""The Glue partition projection must list every league the pipeline writes.

Archived, not run (it needs the main package)."""

import ast
from pathlib import Path

from scout.config import load_settings

ROOT = Path(__file__).parents[3]


def test_glue_leagues_match_the_ultra_config():
    tree = ast.parse((ROOT / "infra" / "archive" / "scout_infra" / "data_stack.py").read_text())
    leagues = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LEAGUES"
    )
    configured = [lg.key for lg in load_settings(ROOT / "configs" / "backfill.ultra.yaml").leagues]
    assert leagues == configured
