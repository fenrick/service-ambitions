# SPDX-License-Identifier: MIT
"""Pydantic models describing service inputs, evaluation outputs, and configuration.

These definitions act as the contract between the command-line interface, the
generation pipeline, and any downstream tooling that consumes service ambition
results. Each class documents the structure and semantics of the data exchanged
throughout the system.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


class ServiceMeta(StrictModel):
    """Metadata describing a single generation run."""

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Version of the ServiceEvolution schema.",
    )
    run_id: Annotated[
        str,
        Field(min_length=1, description="Unique identifier for this run."),
    ]
    seed: int | None = Field(
        default=None, description="Seed used for deterministic generation."
    )
    models: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of generation stages to model names.",
    )
    web_search: bool = Field(
        default=False, description="Whether web search was enabled."
    )
    mapping_types: list[str] = Field(
        default_factory=list,
        description="Mapping categories included during feature mapping.",
    )
    context_window: int = Field(
        0,
        ge=0,
        description="Context window of the primary generation model in tokens.",
    )
    diagnostics: bool = Field(
        False, description="Whether diagnostics mode was enabled during the run."
    )
    catalogue_hash: str | None = Field(
        default=None,
        description="SHA256 hash of the compiled mapping catalogue used for this run.",
    )
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO-8601 timestamp when the run metadata was created.",
    )

    @field_validator("catalogue_hash")
    @classmethod
    def _validate_hash(cls, value: str | None) -> str | None:
        """Ensure ``catalogue_hash`` is a 64 character hex string when provided."""

        if value is None:
            return value
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("catalogue_hash must be 64 hex characters")
        return value


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
    """Reference item identifier supporting a feature."""

    item: Annotated[
        str, Field(min_length=1, description="Identifier of the mapped element.")
    ]


class DiagnosticContribution(Contribution):
    """Contribution with an additional rationale used during diagnostics."""

    rationale: Annotated[
        str, Field(min_length=1, description="Reason the item applies to the feature.")
    ]


class DefinitionItem(StrictModel):
    """Named definition entry used when rendering reference sections."""

    name: Annotated[str, Field(min_length=1, description="Definition term.")]
    description: Annotated[
        str, Field(min_length=1, description="Explanation of the term.")
    ]


class DefinitionBlock(StrictModel):
    """Container for a definition title and its bullet entries."""

    title: Annotated[
        str, Field(min_length=1, description="Definitions section heading.")
    ] = "Definitions"
    bullets: list[DefinitionItem] = Field(
        default_factory=list, description="Collection of definition items."
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
    parent_id: (
        Annotated[
            str,
            Field(
                min_length=1, description="Unique identifier for the parent service."
            ),
        ]
        | None
    ) = None
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


class ServiceFeaturePlateauDescription(StrictModel):
    """Structured description for a service feature plateau."""

    core_idea: Annotated[
        str, Field(min_length=1, description="Fundamental principle of the plateau.")
    ]
    key_characteristics: Annotated[
        list[str],
        Field(min_length=1, description="Distinctive traits defining the plateau."),
    ]
    what_it_feels_like: Annotated[
        str,
        Field(
            min_length=1,
            description="Typical experience when operating at this plateau.",
        ),
    ]


class ServiceFeaturePlateau(StrictModel):
    """Definition of a service feature plateau.

    Plateaus describe maturity stages that a service feature can achieve. They
    provide reference points when comparing services against the model.
    """

    id: Annotated[str, Field(min_length=1, description="Unique plateau identifier.")]
    name: Annotated[
        str, Field(min_length=1, description="Human readable plateau name.")
    ]
    description: ServiceFeaturePlateauDescription = Field(
        ..., description="Explanation of plateau characteristics."
    )


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


class MappingSet(StrictModel):
    """Configuration for a mapping dataset.

    Each set links a feature mapping field to a JSON file on disk that
    provides the corresponding :class:`MappingItem` catalogue.
    """

    name: Annotated[
        str,
        Field(min_length=1, description="Human readable mapping set name."),
    ]
    file: Annotated[
        str,
        Field(min_length=1, description="Filename containing mapping items."),
    ]
    field: Annotated[
        str,
        Field(min_length=1, description="Feature mapping field name."),
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
    web_search: bool = Field(
        False, description="Enable OpenAI web search tooling for model browsing."
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
    mapping_sets: list[MappingSet] = Field(
        default_factory=list,
        description="Mapping dataset configurations.",
    )
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
    """Summary of a service's progress across plateaus."""

    meta: ServiceMeta = Field(..., description="Metadata for this run.")
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


class RoleFeaturesResponse(BaseModel):
    """Schema used when repairing missing role features."""

    model_config = ConfigDict(extra="forbid")
    features: list[FeatureItem]


def _extract_mapping_list(value: object, key: str) -> list[object]:
    """Return mapping list from ``value`` or an empty list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []

    direct = value.get(key)
    if isinstance(direct, list):
        return direct

    nested = value.get("mappings")
    if isinstance(nested, dict):  # Prefer nested dict over list for precision
        inner = nested.get(key)
        return inner if isinstance(inner, list) else []
    if isinstance(nested, list):  # Fallback when nested is already a list
        return nested
    return []


def _normalize_mapping_values(
    mapping: dict[str, object],
) -> dict[str, list[object]]:
    """Return mapping dictionary with nested structures flattened."""

    return {key: _extract_mapping_list(value, key) for key, value in mapping.items()}


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
        for key in tuple(data):
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


class MappingResponse(StrictModel):
    """Schema for feature mapping responses returned by the mapping worker."""

    features: list[MappingFeature] = Field(
        ..., description="Collection of features with mapping details."
    )


class MappingDiagnosticsFeature(StrictModel):
    """Schema for mapped features with rationales during diagnostics."""

    feature_id: Annotated[str, Field(min_length=1, description="Feature identifier.")]
    mappings: dict[str, list[DiagnosticContribution]] = Field(
        default_factory=dict,
        description="Mapping contributions with rationales by type.",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _collect_mappings(cls, data: dict[str, object]) -> dict[str, object]:
        """Collect arbitrary mapping lists into ``mappings``."""

        mapping: dict[str, object] = {}
        for key in tuple(data):
            if key == "feature_id":
                continue
            value = data.pop(key)
            if key == "mappings" and isinstance(value, dict):
                mapping.update(value)
                continue
            mapping[key] = value
        data["mappings"] = _normalize_mapping_values(mapping)
        return data


class MappingDiagnosticsResponse(StrictModel):
    """Schema for diagnostic mapping responses including rationales."""

    features: list[MappingDiagnosticsFeature] = Field(
        ..., description="Collection of features with mapping details and rationales."
    )


class FeatureMappingRef(StrictModel):
    """Reference to a feature associated with a mapping item."""

    feature_id: Annotated[str, Field(min_length=1, description="Feature identifier.")]
    description: Annotated[
        str, Field(min_length=1, description="Explanation of the feature.")
    ]


class MappingFeatureGroup(StrictModel):
    """Grouping of features keyed by mapping item identifier."""

    id: Annotated[
        str, Field(min_length=1, description="Identifier of the mapping item.")
    ]
    mappings: list[FeatureMappingRef] = Field(
        default_factory=list,
        description="Features linked to the mapping item.",
    )


__all__ = [
    "AppConfig",
    "Contribution",
    "DiagnosticContribution",
    "DefinitionBlock",
    "DefinitionItem",
    "DescriptionResponse",
    "PlateauDescription",
    "PlateauDescriptionsResponse",
    "FeatureItem",
    "FeaturesBlock",
    "JobToBeDone",
    "MappingFeature",
    "MappingItem",
    "MappingResponse",
    "MappingDiagnosticsFeature",
    "MappingDiagnosticsResponse",
    "FeatureMappingRef",
    "MappingFeatureGroup",
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
    "ServiceMeta",
    "ServiceEvolution",
    "ServiceFeature",
    "ServiceFeaturePlateau",
    "ServiceInput",
    "StrictModel",
]
