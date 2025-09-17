# SPDX-License-Identifier: MIT
"""Shared cache helpers for core modules.

This module centralises cache manager configuration so that components such as
``core.conversation`` and ``core.mapping`` can write JSON results without
importing each other. The helpers expose a simple override hook used by tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import CacheManager, JSONCacheManager

_cache_manager: CacheManager = JSONCacheManager()


def configure_cache_manager(manager: CacheManager) -> None:
    """Override the cache manager used for JSON writes."""
    global _cache_manager
    _cache_manager = manager


def cache_write_json_atomic(path: Path, content: Any) -> None:
    """Atomically write ``content`` as JSON using the configured manager."""
    _cache_manager.write_json_atomic(path, content)


__all__ = ["cache_write_json_atomic", "configure_cache_manager"]
