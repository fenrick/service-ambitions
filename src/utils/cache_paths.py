# SPDX-License-Identifier: MIT
"""Cache path helpers."""

from __future__ import annotations

from pathlib import Path

from runtime.environment import RuntimeEnv


def feature_cache(service_id: str, plateau: int) -> Path:
    """Return canonical cache path for features for ``service_id`` at ``plateau``.

    Ensures the parent directory exists before returning the final path.
    """

    try:
        settings = RuntimeEnv.instance().settings
        cache_root = settings.cache_dir
        context = settings.context_id
    except RuntimeError:  # pragma: no cover - settings unavailable
        cache_root = Path(".cache")
        context = "unknown"

    path = cache_root / context / service_id / str(plateau) / "features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["feature_cache"]
