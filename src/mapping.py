"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Mapping, Sequence

from loader import (
    load_mapping_items,
    load_mapping_type_config,
    load_prompt_text,
)
from models import (
    Contribution,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from conversation import ConversationSession

logger = logging.getLogger(__name__)


def _render_items(items: list[MappingItem]) -> str:
    """Return a bullet list representation of mapping reference items.

    Each entry is formatted as ``<id>: <name> - <description>`` so the string
    can be embedded directly within agent prompts.
    """

    # Present each mapping reference item on a separate line so it can be
    # directly inserted into the agent prompt as a bullet list.
    return "\n".join(
        f"- {entry.id}: {entry.name} - {entry.description}" for entry in items
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return a bullet list of feature details for prompt construction.

    Features are presented using their ID, name and description to provide the
    agent with enough context for mapping decisions while keeping prompts
    compact.
    """

    # The agent prompt expects a concise summary of each feature, therefore we
    # format the feature ID, name and description on a single line.
    return "\n".join(
        f"- {feat.feature_id}: {feat.name} - {feat.description}" for feat in features
    )


def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> PlateauFeature:
    """Return ``feature`` augmented with mapping information.

    This is a convenience wrapper around :func:`map_features` for mapping a
    single feature while preserving the interface of the bulk function.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.
        mapping_types: Mapping configuration to apply. Defaults to
            :func:`loader.load_mapping_type_config`.

    Returns:
        A :class:`PlateauFeature` with mapping information applied.
    """

    return map_features(session, [feature], mapping_types)[0]


def _build_mapping_prompt(
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig],
) -> str:
    """Return a prompt requesting mappings for ``features``."""

    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    items = load_mapping_items([cfg.dataset for cfg in mapping_types.values()])
    sections = []
    for cfg in mapping_types.values():
        # Include a bullet list of reference items for each mapping type.
        sections.append(
            f"## Available {cfg.label}\n\n{_render_items(items[cfg.dataset])}\n"
        )
    mapping_sections = "\n".join(sections)
    mapping_labels = ", ".join(cfg.label for cfg in mapping_types.values())
    mapping_fields = ", ".join(mapping_types.keys())
    return template.format(
        mapping_labels=mapping_labels,
        mapping_sections=mapping_sections,
        mapping_fields=mapping_fields,
        features=_render_features(features),
        schema=str(schema),
    )


def _parse_mapping_response(response: str) -> MappingResponse:
    """Return a validated mapping response."""

    try:
        return MappingResponse.model_validate_json(response)
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Invalid JSON from mapping response: %s", exc)
        raise ValueError("Agent returned invalid JSON") from exc


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
) -> list[PlateauFeature]:
    """Return ``features`` merged with mapping ``payload``."""

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            # Each feature must appear in the response; fail fast when absent.
            raise ValueError(f"Missing mappings for feature {feature.feature_id}")
        update_data: dict[str, list[Contribution]] = {}
        for key in mapping_types.keys():
            values = mapped.get(key)
            if not values:
                # Every mapping type requires at least one contribution.
                raise ValueError(
                    f"'{key}' key missing or empty for feature {feature.feature_id}"
                )
            update_data[key] = values
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    return results


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> list[PlateauFeature]:
    """Return ``features`` augmented with mapping information.

    A single prompt is sent to the agent requesting mappings for all supplied
    features. Missing or empty mapping lists raise :class:`ValueError`.
    """
    # Use configured mappings when none are explicitly supplied.
    mapping_types = mapping_types or load_mapping_type_config()
    prompt = _build_mapping_prompt(features, mapping_types)
    logger.debug("Requesting mappings for %s features", len(features))
    response = session.ask(prompt)
    logger.debug("Raw multi-feature mapping response: %s", response)
    payload = _parse_mapping_response(response)
    return _merge_mapping_results(features, payload, mapping_types)


__all__ = ["map_feature", "map_features"]
