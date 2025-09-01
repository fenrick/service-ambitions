"""Runtime container for plateau generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import logfire

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

        with logfire.span(
            "plateau_runtime.set_results",
            attributes={"plateau": self.plateau_name},
        ):
            self.features = list(features)
            self.mappings = mappings
            self._success = True
            logfire.debug(
                "Stored plateau results",
                plateau=self.plateau_name,
                feature_count=len(self.features),
                mapping_sets=len(self.mappings),
            )

    def status(self) -> bool:
        """Return ``True`` when generation succeeded."""

        logfire.debug(
            "Plateau status checked",
            plateau=self.plateau_name,
            success=self._success,
        )
        return self._success
