"""Raw response cache.

Every API response is stored byte for byte, with a sidecar JSON file holding
the request and fetch metadata. Processed tables are rebuilt from this cache
only, so re-running a transformation never spends API requests.

The cache is also the backfill's resume state: a request whose body file
exists is never made again. The body is written last, through an atomic
rename, so its presence means the response is complete.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, unquote

from scout.fsutil import atomic_write_bytes, atomic_write_text


@dataclass(frozen=True)
class CacheKey:
    """Identifies one request: source, endpoint and sorted query parameters."""

    source: str
    endpoint: str
    params: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, source: str, endpoint: str, params: dict[str, object]) -> CacheKey:
        items = tuple(sorted((str(k), str(v)) for k, v in params.items()))
        return cls(source, endpoint.strip("/"), items)

    def relative_path(self, suffix: str = ".json") -> Path:
        # One directory per parameter keeps listings small and paths readable:
        # api_football/players/league=39/season=2023/page=1.json
        parts = [self.source, self.endpoint]
        parts += [f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in self.params]
        return Path(*parts[:-1]) / f"{parts[-1]}{suffix}"


@dataclass(frozen=True)
class RawMeta:
    url: str
    params: dict[str, str]
    fetched_at: str
    status_code: int
    headers: dict[str, str]


class RawStore(Protocol):
    def exists(self, key: CacheKey) -> bool: ...
    def get(self, key: CacheKey) -> bytes: ...
    def get_meta(self, key: CacheKey) -> RawMeta | None: ...
    def put(self, key: CacheKey, body: bytes, meta: RawMeta) -> None: ...
    def keys(self, source: str, endpoint: str) -> list[CacheKey]: ...


class LocalRawStore:
    """Raw store on the local filesystem. An S3 version can follow the same protocol."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _body(self, key: CacheKey) -> Path:
        return self.root / key.relative_path(".json")

    def _meta(self, key: CacheKey) -> Path:
        return self.root / key.relative_path(".meta.json")

    def exists(self, key: CacheKey) -> bool:
        return self._body(key).is_file()

    def get(self, key: CacheKey) -> bytes:
        return self._body(key).read_bytes()

    def get_meta(self, key: CacheKey) -> RawMeta | None:
        path = self._meta(key)
        if not path.is_file():
            return None
        return RawMeta(**json.loads(path.read_text(encoding="utf-8")))

    def put(self, key: CacheKey, body: bytes, meta: RawMeta) -> None:
        atomic_write_text(self._meta(key), json.dumps(asdict(meta), indent=2, sort_keys=True))
        atomic_write_bytes(self._body(key), body)  # written last: marks the entry complete

    def keys(self, source: str, endpoint: str) -> list[CacheKey]:
        base = self.root / source / endpoint
        if not base.is_dir():
            return []
        found = []
        for path in sorted(base.rglob("*.json")):
            if path.name.endswith(".meta.json"):
                continue
            rel = path.relative_to(base).with_suffix("")
            params = []
            for part in rel.parts:
                k, _, v = part.partition("=")
                params.append((unquote(k), unquote(v)))
            found.append(CacheKey(source, endpoint, tuple(sorted(params))))
        return found
