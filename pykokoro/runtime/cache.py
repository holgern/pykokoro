from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...

    @staticmethod
    def make_key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            if isinstance(p, (bytes, bytearray)):
                b = bytes(p)
            else:
                b = json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            h.update(b)
            h.update(b"\x1f")
        return h.hexdigest()

    @classmethod
    def from_dir(cls, cache_dir: str | None) -> "Cache":
        if not cache_dir:
            return NullCache()
        return DiskCache(Path(cache_dir))


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
