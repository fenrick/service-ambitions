"""Pydantic models describing service inputs, evaluation outputs, and configuration.

These definitions act as the contract between the command-line interface, the
generation pipeline, and any downstream tooling that consumes service ambition
results. Each class documents the structure and semantics of the data exchanged
throughout the system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    """Base model with strict settings to prevent shape drift."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


CMMI_LABELS = {
    1: "Initial",
    2: "Managed",
    3: "Defined",
    4: "Quantitatively Managed",
    5: "Optimizing",
}


class MaturityScore(BaseModel):
    level: Annotated[int, Field(ge=1, le=5, description="CMMI level (1–5).")]
    label: Annotated[str, Field(min_length=1, description="CMMI label matching level.")]
    justification: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def label_matches_level(self):
        if self.label != CMMI_LABELS.get(self.level):
            raise ValueError(
                f"label must match level ({self.level} → {CMMI_LABELS[self.level]})"
            )
        return self


class Contribution(StrictModel):
    """Weighted reference item supporting a feature.

    The ``contribution`` value measures how strongly the referenced item
    influences a feature's assessment. Larger numbers indicate greater
    importance.
    """

    item: Annotated[
        str, Field(min_length=1, description="Identifier of the mapped element.")
    ]
    contribution: float = Field(
        ..., ge=0.1, le=1.0, description="Importance weight from 0.1 to 1.0."
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


class JobToBeDone(StrictModel):
    """Customer job that a service aims to satisfy."""

    name: Annotated[str, Field(min_length=1, description="Human readable job name.")]

    # Allow unknown fields so additional metadata is preserved.
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class Role(StrictModel):
    """Role that benefits from generated service features."""

    role_id: Annotated[str, Field(min_length=1, description="Unique role identifier.")]
    name: Annotated[str, Field(min_length=1, description="Human readable role name.")]
    description: Annotated[
        str, Field(min_length=1, description="Explanation of the role.")
    ]


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
    jobs_to_be_done: list[JobToBeDone] = Field(
        ..., description="Customer jobs the service seeks to address."
    )
    features: list[ServiceFeature] = Field(
        default_factory=list,
        description="Existing features currently offered by the service.",
    )

    @field_validator("jobs_to_be_done", mode="before")
    @classmethod
    def _coerce_job_objects(cls, value: list[object]) -> list[object]:
        """Coerce string job entries into objects while preserving details."""

        if not isinstance(value, list):
            raise TypeError("jobs_to_be_done must be a list")

        objects: list[object] = []
        for job in value:
            # Convert legacy string entries into object form
            if isinstance(job, str):
                objects.append({"name": job})
            elif isinstance(job, dict) and "name" in job:
                # Retain full object to preserve additional fields
                objects.append(job)
            else:
                raise TypeError(
                    "jobs_to_be_done entries must be strings or objects with a 'name'"
                    " field"
                )
        return objects


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


class ReasoningConfig(StrictModel):
    """Optional reasoning parameters for OpenAI models.

    Fields are mapped to the ``openai_reasoning_*`` settings when constructing
    the model. Unknown keys are allowed so additional parameters can be
    provided without code changes.
    """

    effort: Literal["minimal", "low", "medium", "high"] | None = Field(
        None, description="Effort level for OpenAI reasoning tasks."
    )
    summary: Literal["detailed", "concise"] | None = Field(
        None, description="Summary style for reasoning traces."
    )

    # Permit other reasoning settings that may be added by OpenAI.
    model_config = ConfigDict(extra="allow")


class StageModels(StrictModel):
    """Optional per-stage model configuration."""

    descriptions: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Model used for plateau descriptions in '<provider>:<model>' format."
            ),
        ),
    ]
    features: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Model used for feature generation in '<provider>:<model>' format."
            ),
        ),
    ]
    mapping: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Model used for feature mapping in '<provider>:<model>' format."
            ),
        ),
    ]
    search: Annotated[
        str | None,
        Field(
            default=None,
            description="Model used for web search in '<provider>:<model>' format.",
        ),
    ]


class AppConfig(StrictModel):
    """Top-level application configuration controlling generation behaviour."""

    model: Annotated[
        str,
        Field(min_length=1, description="Chat model in '<provider>:<model>' format."),
    ] = "openai:gpt-5"
    models: StageModels | None = Field(None, description="Per-stage model overrides.")
    reasoning: ReasoningConfig | None = Field(
        None, description="Optional reasoning configuration for the model."
    )
    log_level: Annotated[
        str, Field(min_length=1, description="Logging verbosity level.")
    ] = "INFO"
    prompt_dir: Annotated[
        Path,
        Field(description="Directory containing prompt components."),
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
    batch_size: int | None = Field(
        None,
        ge=1,
        description="Number of services to schedule per batch.",
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
    features_per_role: Annotated[
        int,
        Field(ge=1, description="Required number of features per role."),
    ] = 5
    mapping_types: dict[str, MappingTypeConfig] = Field(
        default_factory=dict,
        description="Mapping type definitions keyed by field name.",
    )


class PlateauFeature(StrictModel):
    """Feature assessed during a service plateau.

    Each feature includes a CMMI ``score`` and optional mapping contributions
    that reference external catalogues.
    """

    feature_id: Annotated[
        str, Field(min_length=1, description="Unique identifier for the feature.")
    ]
    name: Annotated[str, Field(min_length=1, description="Feature name.")]
    description: str = Field(..., description="Explanation of the feature.")
    score: MaturityScore
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

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Version of the ServiceEvolution schema.",
    )
    service: ServiceInput = Field(..., description="Service being evaluated.")
    plateaus: list[PlateauResult] = Field(
        default_factory=list, description="Evaluated plateaus for the service."
    )


class DescriptionResponse(StrictModel):
    """Schema for intermediate service description responses."""

    description: str = Field(
        ..., description="Explanation of the service at a plateau."
    )


class PlateauDescription(StrictModel):
    """Single plateau description entry."""

    plateau: Annotated[int, Field(ge=1, description="Plateau level (1-indexed).")]
    plateau_name: Annotated[str, Field(min_length=1, description="Plateau label.")]
    description: Annotated[
        str,
        Field(
            min_length=1,
            description="Narrative explaining the service at this plateau.",
        ),
    ]


class PlateauDescriptionsResponse(StrictModel):
    """Schema for batch plateau description responses."""

    descriptions: Annotated[
        List[PlateauDescription],
        Field(min_length=1, description="Ordered plateau descriptions."),
    ]


class FeatureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_id: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"^FEAT-\d+-(learners|academics|professional_staff)-[a-z0-9]+(?:-[a-z0-9]+)*$",
            description="Format: FEAT-{plateau}-{role}-{kebab-case-short-name}",
        ),
    ]
    name: Annotated[str, Field(min_length=1)]
    description: Annotated[str, Field(min_length=1)]
    score: MaturityScore


class FeaturesBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Feature lists for each role; lists may be empty when responses are partial.
    learners: Annotated[List[FeatureItem], Field(default_factory=list)]
    academics: Annotated[List[FeatureItem], Field(default_factory=list)]
    professional_staff: Annotated[List[FeatureItem], Field(default_factory=list)]


class PlateauFeaturesResponse(BaseModel):
    """Schema for plateau feature generation responses.
    Features are grouped by role identifier to simplify downstream rendering."""

    model_config = ConfigDict(extra="forbid")
    features: FeaturesBlock

    @model_validator(mode="after")
    def unique_feature_ids(self):
        all_items = (
            self.features.learners
            + self.features.academics
            + self.features.professional_staff
        )
        ids = [f.feature_id for f in all_items]
        if len(ids) != len(set(ids)):
            raise ValueError("feature_id values must be unique across all roles")
        return self


class RoleFeaturesResponse(BaseModel):
    """Schema used when repairing missing role features."""

    model_config = ConfigDict(extra="forbid")
    features: list[FeatureItem]


def _normalize_mapping_values(
    mapping: dict[str, object],
) -> dict[str, list[Contribution]]:
    """Return mapping dictionary with nested structures flattened.

    Agents occasionally wrap mapping lists in redundant dictionaries, for example
    ``{"applications": {"applications": [...]}}`` or nest them under an extra
    ``"mappings"`` key. This helper extracts the underlying lists and replaces any
    unrecognised structures with empty lists so model validation succeeds.
    """

    normalised: dict[str, list[Contribution]] = {}
    for key, value in mapping.items():
        if isinstance(value, list):
            normalised[key] = value
            continue
        if isinstance(value, dict):
            direct = value.get(key)
            if isinstance(direct, list):
                normalised[key] = direct
                continue
            nested = value.get("mappings")
            if isinstance(nested, dict):
                inner = nested.get(key)
                if isinstance(inner, list):
                    normalised[key] = inner
                    continue
            elif isinstance(nested, list):
                normalised[key] = nested
                continue
        # Default to an empty list when the structure is unrecognised.
        normalised[key] = []
    return normalised


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
            if key == "feature_id":
                # Preserve the identifying field untouched.
                continue
            value = data.pop(key)
            if key == "mappings" and isinstance(value, dict):
                # Agents may wrap mapping types in a top-level "mappings" block.
                mapping.update(value)
                continue
            mapping[key] = value
        # Normalise nested mapping structures and ensure list values.
        data["mappings"] = _normalize_mapping_values(mapping)
        return data

    @field_validator("mappings")
    @classmethod
    def _limit_mapping_items(
        cls, value: dict[str, list[Contribution]]
    ) -> dict[str, list[Contribution]]:
        """Ensure each mapping list contains at most five items."""

        for key, items in value.items():
            if len(items) > 5:
                raise ValueError(
                    f"Too many entries for {key}: {len(items)} (maximum 5)"
                )
        return value


class MappingResponse(StrictModel):
    """Schema for feature mapping responses returned by the mapping worker."""

    features: list[MappingFeature] = Field(
        ..., description="Collection of features with mapping details."
    )


__all__ = [
    "AppConfig",
    "Contribution",
    "DescriptionResponse",
    "PlateauDescription",
    "PlateauDescriptionsResponse",
    "FeatureItem",
    "FeaturesBlock",
    "JobToBeDone",
    "MappingFeature",
    "MappingItem",
    "MappingResponse",
    "MappingTypeConfig",
    "MaturityScore",
    "PlateauFeature",
    "PlateauFeaturesResponse",
    "RoleFeaturesResponse",
    "PlateauResult",
    "StageModels",
    "ReasoningConfig",
    "Role",
    "SCHEMA_VERSION",
    "ServiceEvolution",
    "ServiceFeature",
    "ServiceFeaturePlateau",
    "ServiceInput",
    "StrictModel",
]
