"""Caching helpers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import logfire
from pydantic_core import from_json, to_json


class CacheManager(ABC):
    """Interface for cache storage operations.

    Methods should perform atomic writes and avoid partial file updates. Reads
    and writes are expected to complete within tens of milliseconds for small
    payloads to keep cache interactions negligible compared to model calls.
    """

    @abstractmethod
    def write_json_atomic(self, path: Path, content: Any) -> None:
        """Persist ``content`` to ``path`` atomically as pretty JSON."""


class JSONCacheManager(CacheManager):
    """Cache manager writing JSON objects to disk."""

    def write_json_atomic(self, path: Path, content: Any) -> None:  # noqa: D401
        with logfire.span("cache.write_json_atomic", attributes={"path": str(path)}):
            data = (
                content
                if not isinstance(content, (str, bytes, bytearray))
                else from_json(
                    content
                    if isinstance(content, (bytes, bytearray))
                    else content.encode("utf-8")
                )
            )
            if not isinstance(data, dict):
                logfire.error("cache content must be a JSON object", path=str(path))
                raise TypeError("cache content must be a JSON object")

            logfire.debug(
                "Preparing cache file write",
                path=str(path),
                keys=len(data),
            )

            tmp_path = path.with_suffix(".tmp")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "wb") as fh:
                    try:
                        fh.write(cast(Any, to_json)(data, sort_keys=True, indent=2))
                    except TypeError:  # pragma: no cover - legacy pydantic-core
                        fh.write(cast(Any, to_json)(data, indent=2))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
                size = path.stat().st_size
                logfire.debug(
                    "Wrote cache file",
                    path=str(path),
                    bytes=size,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
