"""Apply the adapter's name repair to processed tables built before it (prompt 18b).

Usage:
    uv run python scripts/repair_names.py --config configs/backfill.ultra.yaml [--dry-run]

The API Football adapters repair garbled names since prompt 18b
(``scout.providers.api_football.names``, rule 7). The repair is a pure
function of the name string, so applying it to a processed partition gives
what parsing the cached responses again would give (``rebuild_match_seasons.py``),
without the raw cache, which a fresh container does not hold. The raw cache
is not touched. Only partitions with a changed name are rewritten; each is
checked by its contract first. Features and the lists take the repaired
names at their next build (the weekly run rebuilds the newest season).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scout.config import load_settings
from scout.contracts import schemas
from scout.providers.api_football.names import NAME_COLUMNS, clean_names, garbled

log = logging.getLogger("repair_names")

# Processed tables written by the API Football adapters, with their contracts.
TABLES = {
    "player_season_stats": schemas.PLAYER_SEASON_STATS,
    "player_match_stats": schemas.PLAYER_MATCH_STATS,
    "match_lineups": schemas.MATCH_LINEUPS,
    "injuries": schemas.INJURIES,
    "fixture_events": schemas.FIXTURE_EVENTS,
    "transfers": schemas.TRANSFERS,
    "coaches": schemas.COACH_SPELLS,
    "current": None,  # squads, transfers and the season in progress: display only
}


def repair(processed: Path, dry_run: bool) -> tuple[int, int, set]:
    files = changed = 0
    examples: set = set()
    for table, schema in TABLES.items():
        for path in sorted((processed / table).rglob("*.parquet")):
            files += 1
            frame = pd.read_parquet(path)
            columns = [c for c in NAME_COLUMNS if c in frame]
            bad = {c: garbled(frame[c]) for c in columns}
            if not any(b.any() for b in bad.values()):
                continue
            before = {c: frame.loc[b, c].copy() for c, b in bad.items() if b.any()}
            clean_names(frame)
            for c, old in before.items():
                examples |= set(zip(old, frame.loc[old.index, c], strict=True))
            if schema is not None and table != "current":
                schemas.validate(schema, frame)
            changed += 1
            log.info("%s: %s", path, ", ".join(f"{c} {len(v)}" for c, v in before.items()))
            if not dry_run:
                frame.to_parquet(path, index=False)
    return files, changed, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/backfill.ultra.yaml")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings(args.config)
    files, changed, examples = repair(settings.paths.processed_dir, args.dry_run)
    print(
        f"{changed} of {files} partitions had garbled names; {len(examples)} distinct names "
        f"repaired{' (dry run, nothing written)' if args.dry_run else ''}"
    )
    for old, new in sorted(examples)[:10]:
        print(f"  {old!r} -> {new!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
