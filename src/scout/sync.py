"""Copy the local data folders to and from a private S3 bucket.

``data/raw`` maps to ``<prefix>raw/`` and ``data/processed`` to
``<prefix>processed/``. A file is skipped when the other side already holds
the same bytes, judged by size and MD5 against the S3 ETag (objects are
uploaded in one part, and with SSE-S3 encryption the ETag is then the MD5;
under SSE-KMS it is not, and every file would be copied again). Nothing is ever deleted on
either side, and the bucket keeps old versions when a file changes.

The S3 client is passed in, so tests use an in memory fake and the module
does not import boto3.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

Direction = Literal["up", "down"]

# put_object sends one part (S3 allows up to 5 GiB), which keeps the ETag
# equal to the MD5. Every file here is far below this limit.
SINGLE_PART_LIMIT = 5 * 1024**3


@dataclass
class SyncResult:
    direction: Direction
    transferred: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"sync {self.direction}: {len(self.transferred)} transferred, "
            f"{len(self.skipped)} already there"
        )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(path: Path, size: int, etag: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return _md5(path) == etag.strip('"')


def list_remote(client: Any, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    """Map key -> (size, etag) for every object under ``prefix``."""
    objects: dict[str, tuple[int, str]] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            objects[obj["Key"]] = (int(obj["Size"]), str(obj["ETag"]))
    return objects


def _local_path(local_dir: Path, key: str, prefix: str) -> Path:
    """Local path for a key, refusing keys that would land outside ``local_dir``."""
    relative = key[len(prefix) :]
    target = (local_dir / relative).resolve()
    if not relative or not target.is_relative_to(local_dir.resolve()):
        raise ValueError(f"Refusing S3 key outside the target folder: {key!r}")
    return target


def sync_dir(
    client: Any,
    bucket: str,
    local_dir: Path,
    prefix: str,
    direction: Direction,
    dry_run: bool = False,
) -> SyncResult:
    """Copy one folder in one direction, skipping files that are already there."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    remote = list_remote(client, bucket, prefix)
    result = SyncResult(direction)

    if direction == "up":
        files = sorted(p for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else []
        for path in files:
            key = prefix + path.relative_to(local_dir).as_posix()
            if key in remote and _same(path, *remote[key]):
                result.skipped.append(key)
                continue
            if path.stat().st_size > SINGLE_PART_LIMIT:
                raise ValueError(f"{path} is too large for a single part upload")
            if not dry_run:
                with path.open("rb") as body:
                    client.put_object(Bucket=bucket, Key=key, Body=body)
            result.transferred.append(key)
    else:
        for key, (size, etag) in sorted(remote.items()):
            if key.endswith("/"):
                continue  # folder placeholder objects
            path = _local_path(local_dir, key, prefix)
            if _same(path, size, etag):
                result.skipped.append(key)
                continue
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                client.download_file(bucket, key, str(tmp))
                tmp.replace(path)
            result.transferred.append(key)

    log.info("%s (%s -> s3://%s/%s)", result.summary(), local_dir, bucket, prefix)
    return result


def sync_data(
    client: Any,
    bucket: str,
    folders: dict[str, Path],
    direction: Direction,
    prefix: str = "",
    dry_run: bool = False,
) -> dict[str, SyncResult]:
    """Sync several named folders, for example ``{"raw": data/raw}``."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return {
        name: sync_dir(client, bucket, local, f"{prefix}{name}/", direction, dry_run)
        for name, local in folders.items()
    }
