"""Pydantic models describing service inputs, evaluation outputs, and configuration.

These definitions act as the contract between the command-line interface, the
generation pipeline, and any downstream tooling that consumes service ambition
results. Each class documents the structure and semantics of the data exchanged
throughout the system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model with strict settings to prevent shape drift."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Contribution(StrictModel):
    """Item contributing to a feature's assessment.

    A ``Contribution`` explains how a particular reference item supports or
    detracts from a candidate feature when calculating plateau performance.
    """

    item: Annotated[str, Field(min_length=1, description="Name of the mapped element.")]
    contribution: str = Field(
        ..., description="Explanation of how the item supports the feature."
    )


class ServiceFeature(StrictModel):
    """Feature already delivered by the service.

    Seed features provide additional context about existing capabilities prior to
    plateau analysis.
    """

    feature_id: Annotated[
        str, Field(min_length=1, description="Unique feature identifier.")
    ]
    name: Annotated[
        str, Field(min_length=1, description="Human readable feature name.")
    ]
    description: str = Field(..., description="Explanation of the feature.")


class ServiceInput(StrictModel):
    """Basic description of a service under consideration.

    This model is supplied to the generator to describe the baseline service
    before any plateau analysis occurs.
    """

    service_id: Annotated[
        str, Field(min_length=1, description="Unique identifier for the service.")
    ]
    name: Annotated[
        str, Field(min_length=1, description="Human readable service name.")
    ]
    customer_type: (
        Annotated[
            str,
            Field(
                min_length=1, description="Primary customer segment for the service."
            ),
        ]
        | None
    ) = None
    description: str = Field(..., description="Short explanation of the service.")
    jobs_to_be_done: list[str] = Field(
        ..., description="Customer jobs the service seeks to address."
    )
    features: list[ServiceFeature] = Field(
        default_factory=list,
        description="Existing features currently offered by the service.",
    )


class ServiceFeaturePlateau(StrictModel):
    """Definition of a service feature plateau.

    Plateaus describe maturity stages that a service feature can achieve. They
    provide reference points when comparing services against the model.
    """

    id: Annotated[str, Field(min_length=1, description="Unique plateau identifier.")]
    name: Annotated[
        str, Field(min_length=1, description="Human readable plateau name.")
    ]
    description: str = Field(..., description="Explanation of plateau characteristics.")


class MappingItem(StrictModel):
    """Reference item used for feature mapping.

    Items originate from datasets defined in :class:`MappingTypeConfig` and are
    used to justify a feature's presence during mapping.
    """

    id: Annotated[
        str,
        Field(min_length=1, description="Unique identifier for the reference item."),
    ]
    name: Annotated[str, Field(min_length=1, description="Human readable item name.")]
    description: str = Field(..., description="Explanation of the item.")


class MappingTypeConfig(StrictModel):
    """Configuration for a feature mapping type.

    The ``dataset`` field indicates a JSONL file located in ``data/`` that
    provides the catalogue of :class:`MappingItem` records for the given type.
    """

    dataset: Annotated[
        str,
        Field(min_length=1, description="Mapping dataset name without file extension."),
    ]
    label: Annotated[
        str,
        Field(min_length=1, description="Human readable label for the mapping type."),
    ]


class AppConfig(StrictModel):
    """Top-level application configuration controlling generation behaviour."""

    model: Annotated[
        str,
        Field(min_length=1, description="Chat model in '<provider>:<model>' format."),
    ] = "openai:gpt-4o-mini"
    log_level: Annotated[
        str, Field(min_length=1, description="Logging verbosity level.")
    ] = "INFO"
    prompt_dir: Annotated[
        Path,
        Field(min_length=1, description="Directory containing prompt components."),
    ] = Path("prompts")
    context_id: Annotated[
        str, Field(min_length=1, description="Situational context identifier.")
    ] = "university"
    inspiration: Annotated[
        str, Field(min_length=1, description="Inspirations identifier.")
    ] = "general"
    concurrency: int = Field(
        5,
        ge=1,
        description="Number of services to process concurrently.",
    )
    request_timeout: Annotated[
        int,
        Field(gt=0, description="Per-request timeout in seconds."),
    ] = 60
    retries: Annotated[
        int,
        Field(ge=1, description="Number of retry attempts."),
    ] = 5
    retry_base_delay: Annotated[
        float,
        Field(gt=0, description="Initial backoff delay in seconds."),
    ] = 0.5
    mapping_types: dict[str, MappingTypeConfig] = Field(
        default_factory=dict,
        description="Mapping type definitions keyed by field name.",
    )
    plateau_map: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping from plateau names to numeric identifiers.",
    )


class PlateauFeature(StrictModel):
    """Feature assessed during a service plateau.

    Each feature includes a normalised ``score`` and optional mapping
    contributions that reference external catalogues.
    """

    feature_id: Annotated[
        str, Field(min_length=1, description="Unique identifier for the feature.")
    ]
    name: Annotated[str, Field(min_length=1, description="Feature name.")]
    description: str = Field(..., description="Explanation of the feature.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised performance score between 0 and 1."
    )
    customer_type: Annotated[
        str, Field(min_length=1, description="Audience that benefits from the feature.")
    ]
    mappings: dict[str, list[Contribution]] = Field(
        default_factory=dict,
        description="Mapping contributions keyed by mapping type.",
    )


class PlateauResult(StrictModel):
    """Collection of features describing a service at a plateau level.

    Instances are appended to :class:`ServiceEvolution` as the generator evaluates
    successive plateaus for a service.
    """

    plateau: int = Field(..., ge=1, description="Plateau level evaluated.")
    plateau_name: Annotated[
        str, Field(min_length=1, description="Human readable name for the plateau.")
    ]
    service_description: str = Field(
        ..., description="Description of the service at this plateau."
    )
    features: list[PlateauFeature] = Field(
        default_factory=list,
        description="Features identified for this plateau level.",
    )


class ServiceEvolution(StrictModel):
    """Summary of a service's progress across plateaus.

    The ``plateaus`` list is ordered from lowest to highest maturity and
    represents the full evolution output for a given :class:`ServiceInput`.
    """

    service: ServiceInput = Field(..., description="Service being evaluated.")
    plateaus: list[PlateauResult] = Field(
        default_factory=list, description="Evaluated plateaus for the service."
    )


class DescriptionResponse(StrictModel):
    """Schema for intermediate service description responses."""

    description: str = Field(
        ..., description="Explanation of the service at a plateau."
    )


class FeatureItem(StrictModel):
    """Schema for individual plateau features returned by generation APIs."""

    feature_id: Annotated[
        str,
        Field(min_length=1, description="Unique string identifier for the feature."),
    ]
    name: Annotated[str, Field(min_length=1, description="Short feature title.")]
    description: str = Field(..., description="Explanation of the feature.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Maturity score between 0 and 1."
    )


class PlateauFeaturesResponse(StrictModel):
    """Schema for plateau feature generation responses.

    Features are grouped by audience segment to simplify downstream rendering.
    """

    learners: list[FeatureItem] = Field(..., description="Features for learners.")
    staff: list[FeatureItem] = Field(..., description="Features for staff.")
    community: list[FeatureItem] = Field(
        ..., description="Features for the wider community."
    )


class MappingFeature(StrictModel):
    """Schema for mapped features with dynamic mapping types.

    Any additional properties on the input are interpreted as mapping lists and
    consolidated into the ``mappings`` dictionary by ``_collect_mappings``.
    """

    feature_id: Annotated[str, Field(min_length=1, description="Feature identifier.")]
    mappings: dict[str, list[Contribution]] = Field(
        default_factory=dict, description="Mapping contributions by type."
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _collect_mappings(cls, data: dict[str, object]) -> dict[str, object]:
        """Collect arbitrary mapping lists into ``mappings``.

        The validator mutates ``data`` by removing any keys other than
        ``feature_id`` and storing them under ``mappings``. This allows callers to
        submit flexible mapping types without defining explicit model fields.
        """
        mapping: dict[str, object] = {}
        for key in list(data.keys()):
            if key != "feature_id":
                mapping[key] = data.pop(key)
        data["mappings"] = mapping
        return data


class MappingResponse(StrictModel):
    """Schema for feature mapping responses returned by the mapping worker."""

    features: list[MappingFeature] = Field(
        ..., description="Collection of features with mapping details."
    )


__all__ = [
    "ServiceInput",
    "ServiceFeature",
    "ServiceFeaturePlateau",
    "PlateauFeature",
    "Contribution",
    "PlateauResult",
    "ServiceEvolution",
    "DescriptionResponse",
    "FeatureItem",
    "PlateauFeaturesResponse",
    "MappingFeature",
    "AppConfig",
    "MappingItem",
    "MappingTypeConfig",
    "MappingResponse",
    "StrictModel",
]
