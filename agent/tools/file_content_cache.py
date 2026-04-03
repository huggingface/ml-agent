"""Cache for detecting unchanged local file re-reads."""

from __future__ import annotations

import hashlib


def _short_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _resolve(path: str) -> str:
    try:
        from pathlib import Path
        return str(Path(path).resolve())
    except Exception:
        return path


class FileContentCache:
    """Tracks file content hashes to skip re-reading unchanged files."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, int]] = {}

    def record_read(self, path: str, content: str, turn: int) -> None:
        key = _resolve(path)
        self._cache[key] = (_short_hash(content), turn)

    def check_unchanged(self, path: str, content: str) -> tuple[bool, int | None]:
        key = _resolve(path)
        cached = self._cache.get(key)
        if cached is None:
            return False, None
        cached_hash, turn = cached
        return _short_hash(content) == cached_hash, turn

    def clear_path(self, path: str) -> None:
        key = _resolve(path)
        self._cache.pop(key, None)
