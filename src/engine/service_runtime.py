"""Per-service runtime container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from engine.plateau_runtime import PlateauRuntime
from models import ServiceInput


@dataclass
class ServiceRuntime:
    """Hold execution artefacts for a single service.

    Attributes:
        service: The service definition being processed.
        plateaus: Runtimes for each plateau.
        line: JSONL output line produced after successful execution.
        success: Flag indicating whether execution completed without errors.
    """

    service: ServiceInput
    plateaus: List[PlateauRuntime] = field(default_factory=list)
    line: str | None = None
    success: bool = False

    def status(self) -> bool:
        """Return ``True`` when execution succeeded."""
        return self.success


__all__ = ["ServiceRuntime"]
