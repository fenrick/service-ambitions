"""Runtime container for plateau generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from models import MappingFeatureGroup, PlateauFeature


@dataclass
class PlateauRuntime:
    """Hold generation artefacts for a single plateau."""

    plateau: int
    plateau_name: str
    description: str
    features: list[PlateauFeature] = field(default_factory=list)
    mappings: dict[str, list[MappingFeatureGroup]] = field(default_factory=dict)
    _success: bool = False

    def set_results(
        self,
        *,
        features: list[PlateauFeature],
        mappings: dict[str, list[MappingFeatureGroup]],
    ) -> None:
        """Store ``features`` and ``mappings`` for this plateau."""

        self.features = list(features)
        self.mappings = mappings
        self._success = True

    def status(self) -> bool:
        """Return ``True`` when generation succeeded."""

        return self._success
