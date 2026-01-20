"""Two-layer caching interface: memory + disk."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TwoLayerCache(Generic[T]):
    """Memory + disk cache with LRU eviction."""

    def __init__(self, disk_path: Path | None = None, memory_limit: int = 1000) -> None:
        self._memory: dict[str, T] = {}
        self._access_order: list[str] = []
        self._memory_limit = memory_limit
        self._disk_path = disk_path
        if disk_path:
            disk_path.mkdir(parents=True, exist_ok=True)

    def _make_key(self, data: dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _disk_file(self, key: str) -> Path | None:
        return self._disk_path / f"{key}.pkl" if self._disk_path else None

    def get(self, key_data: dict[str, Any]) -> T | None:
        key = self._make_key(key_data)
        if key in self._memory:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory[key]
        disk_file = self._disk_file(key)
        if disk_file and disk_file.exists():
            with open(disk_file, "rb") as f:
                value = pickle.load(f)
            self._put_memory(key, value)
            return value
        return None

    def put(self, key_data: dict[str, Any], value: T) -> None:
        key = self._make_key(key_data)
        self._put_memory(key, value)
        disk_file = self._disk_file(key)
        if disk_file:
            with open(disk_file, "wb") as f:
                pickle.dump(value, f)

    def _put_memory(self, key: str, value: T) -> None:
        if key in self._memory:
            self._access_order.remove(key)
        elif len(self._memory) >= self._memory_limit:
            evict = self._access_order.pop(0)
            del self._memory[evict]
        self._memory[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        self._memory.clear()
        self._access_order.clear()
