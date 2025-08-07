"""Domain models for service feature evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServiceInput(BaseModel):
    """Basic description of a service under consideration."""

    name: str = Field(..., description="Human readable service name.")
    description: str = Field(..., description="Short explanation of the service.")


class PlateauFeature(BaseModel):
    """Feature assessed during a service plateau."""

    feature_id: str = Field(..., description="Unique identifier for the feature.")
    name: str = Field(..., description="Feature name.")
    description: str = Field(..., description="Explanation of the feature.")


class PlateauResult(BaseModel):
    """Result of evaluating a particular plateau feature."""

    feature: PlateauFeature = Field(..., description="The assessed feature.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised performance score between 0 and 1."
    )


class ServiceEvolution(BaseModel):
    """Summary of a service's progress across plateau features."""

    service: ServiceInput = Field(..., description="Service being evaluated.")
    results: list[PlateauResult] = Field(
        default_factory=list, description="Outcomes for evaluated plateau features."
    )


__all__ = [
    "ServiceInput",
    "PlateauFeature",
    "PlateauResult",
    "ServiceEvolution",
]
