"""Dry-run support types.

This module isolates the DryRunInvocation exception so it can be imported
from multiple packages without creating circular dependencies.
"""

from __future__ import annotations

from pathlib import Path


class DryRunInvocation(RuntimeError):
    """Raised during dry-run when an agent call would be made.

    Attributes:
        stage: Logical stage name (e.g. "descriptions", "features_2", "mapping_apps").
        model: Model identifier associated with the agent call.
        cache_file: Expected cache file path where a payload would be read/written.
        service_id: The ID of the service being processed (when available).
    """

    def __init__(
        self,
        *,
        stage: str,
        model: str,
        cache_file: Path | None,
        service_id: str | None,
    ) -> None:
        self.stage = stage
        self.model = model
        self.cache_file = cache_file
        self.service_id = service_id
        msg = (
            "dry-run: cache miss would invoke agent"
            f" (stage={stage}, model={model}, service={service_id or 'unknown'}"
            f", cache_file={str(cache_file) if cache_file else 'n/a'})"
        )
        super().__init__(msg)


__all__ = ["DryRunInvocation"]
