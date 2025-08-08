"""Domain models for service feature evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Contribution(BaseModel):
    """Item contributing to a feature's assessment."""

    item: str = Field(..., description="Name of the mapped element.")
    contribution: str = Field(
        ..., description="Explanation of how the item supports the feature."
    )


class ServiceInput(BaseModel):
    """Basic description of a service under consideration."""

    service_id: str = Field(..., description="Unique identifier for the service.")
    name: str = Field(..., description="Human readable service name.")
    customer_type: str | None = Field(
        None, description="Primary customer segment for the service."
    )
    description: str = Field(..., description="Short explanation of the service.")
    jobs_to_be_done: list[str] = Field(
        ..., description="Customer jobs the service seeks to address."
    )


class ServiceFeaturePlateau(BaseModel):
    """Definition of a service feature plateau."""

    id: str = Field(..., description="Unique plateau identifier.")
    name: str = Field(..., description="Human readable plateau name.")
    description: str = Field(..., description="Explanation of plateau characteristics.")


class MappingItem(BaseModel):
    """Reference item used for feature mapping."""

    id: str = Field(..., description="Unique identifier for the reference item.")
    name: str = Field(..., description="Human readable item name.")
    description: str = Field(..., description="Explanation of the item.")


class PlateauFeature(BaseModel):
    """Feature assessed during a service plateau."""

    feature_id: str = Field(..., description="Unique identifier for the feature.")
    name: str = Field(..., description="Feature name.")
    description: str = Field(..., description="Explanation of the feature.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised performance score between 0 and 1."
    )
    customer_type: str = Field(
        ..., description="Audience that benefits from the feature."
    )
    data: list[Contribution] = Field(
        default_factory=list,
        description="Conceptual data types related to the feature.",
    )
    applications: list[Contribution] = Field(
        default_factory=list, description="Applications relevant to the feature."
    )
    technology: list[Contribution] = Field(
        default_factory=list,
        description="Supporting technologies for the feature.",
    )


class PlateauResult(BaseModel):
    """Collection of features describing a service at a plateau level."""

    plateau: int = Field(..., ge=1, description="Plateau level evaluated.")
    service_description: str = Field(
        ..., description="Description of the service at this plateau."
    )
    features: list[PlateauFeature] = Field(
        default_factory=list,
        description="Features identified for this plateau level.",
    )


class ServiceEvolution(BaseModel):
    """Summary of a service's progress across plateaus."""

    service: ServiceInput = Field(..., description="Service being evaluated.")
    plateaus: list[PlateauResult] = Field(
        default_factory=list, description="Evaluated plateaus for the service."
    )


class DescriptionResponse(BaseModel):
    """Schema for service description responses."""

    description: str = Field(
        ..., description="Explanation of the service at a plateau."
    )


class FeatureItem(BaseModel):
    """Schema for individual plateau features."""

    feature_id: str = Field(
        ..., description="Unique string identifier for the feature."
    )
    name: str = Field(..., description="Short feature title.")
    description: str = Field(..., description="Explanation of the feature.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Maturity score between 0 and 1."
    )


class PlateauFeaturesResponse(BaseModel):
    """Schema for plateau feature generation responses."""

    learners: list[FeatureItem] = Field(..., description="Features for learners.")
    staff: list[FeatureItem] = Field(..., description="Features for staff.")
    community: list[FeatureItem] = Field(
        ..., description="Features for the wider community."
    )


class MappingFeature(BaseModel):
    """Schema for mapped features."""

    feature_id: str = Field(..., description="Feature identifier.")
    data: list[Contribution] = Field(..., description="Related data items.")
    applications: list[Contribution] = Field(..., description="Relevant applications.")
    technology: list[Contribution] = Field(..., description="Supporting technologies.")


class MappingResponse(BaseModel):
    """Schema for feature mapping responses."""

    features: list[MappingFeature] = Field(
        ..., description="Collection of features with mapping details."
    )


__all__ = [
    "ServiceInput",
    "ServiceFeaturePlateau",
    "PlateauFeature",
    "Contribution",
    "PlateauResult",
    "ServiceEvolution",
    "DescriptionResponse",
    "FeatureItem",
    "PlateauFeaturesResponse",
    "MappingFeature",
    "MappingItem",
    "MappingResponse",
]
