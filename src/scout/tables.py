"""Read the partitioned Parquet tables (``league=<key>/season=<year>/data.parquet``).

Kept free of heavy imports so the SageMaker training package can ship it
without the scraping, matching and contract dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARTITION_FILE = "data.parquet"


def read_feature_table(root: Path) -> pd.DataFrame:
    """Read a partitioned feature table back, restoring ``season`` from the folders."""
    parts = []
    for path in sorted(Path(root).glob(f"league=*/season=*/{PARTITION_FILE}")):
        season = int(path.parent.name.split("=", 1)[1])
        parts.append(pd.read_parquet(path).assign(season=season))
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["season"] = frame["season"].astype("Int64")
    return frame
