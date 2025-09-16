# SPDX-License-Identifier: MIT
"""Cache path helpers."""

from __future__ import annotations

from pathlib import Path

from constants import DEFAULT_CACHE_DIR  # noqa: F401  (kept for backward imports)
from runtime.environment import RuntimeEnv  # noqa: F401

# Deprecated: fixed-name feature cache has been removed in favour of hashed
# prompt-cache entries under stage-specific subdirectories. This module is kept
# to avoid breaking imports from older tooling.

__all__: list[str] = []
