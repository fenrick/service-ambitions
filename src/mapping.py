"""Utilities for enriching plateau features with mapping data."""

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
    """Return a bullet list representation of mapping reference items."""

    # Present each mapping reference item on a separate line so it can be
    # directly inserted into the agent prompt as a bullet list.
    return "\n".join(
        f"- {entry.id}: {entry.name} - {entry.description}" for entry in items
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return a bullet list of feature details for prompt construction."""

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
    single feature.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.
        mapping_types: Mapping configuration to apply. Defaults to
            :func:`loader.load_mapping_type_config`.

    Returns:
        A :class:`PlateauFeature` with mapping information applied.
    """

    return map_features(session, [feature], mapping_types)[0]


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> list[PlateauFeature]:
    """Return ``features`` augmented with mapping information.

    The function sends a single prompt containing all features and mapping
    reference lists. The agent must respond with JSON containing a ``features``
    array. Each feature in the array must provide non-empty lists for each
    configured mapping type. A :class:`ValueError` is raised if the response
    cannot be parsed, a feature is missing or any list is empty.

    Args:
        session: Active conversation session used to query the agent.
        features: Plateau features to map.
        mapping_types: Mapping configuration to apply. Defaults to
            :func:`loader.load_mapping_type_config`.

    Returns:
        List of :class:`PlateauFeature` objects with mapping information applied.

    Raises:
        ValueError: If the agent response is invalid or any mapping list is
        missing.
    """

    mapping_types = mapping_types or load_mapping_type_config()
    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    items = load_mapping_items([cfg.dataset for cfg in mapping_types.values()])
    sections = []
    for field, cfg in mapping_types.items():
        sections.append(
            f"## Available {cfg.label}\n\n{_render_items(items[cfg.dataset])}\n"
        )
    mapping_sections = "\n".join(sections)
    mapping_labels = ", ".join(cfg.label for cfg in mapping_types.values())
    mapping_fields = ", ".join(mapping_types.keys())
    prompt = template.format(
        mapping_labels=mapping_labels,
        mapping_sections=mapping_sections,
        mapping_fields=mapping_fields,
        features=_render_features(features),
        schema=str(schema),
    )
    logger.debug("Requesting mappings for %s features", len(features))
    response = session.ask(prompt)
    logger.debug("Raw multi-feature mapping response: %s", response)
    try:
        payload = MappingResponse.model_validate_json(response)
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Invalid JSON from mapping response: %s", exc)
        raise ValueError("Agent returned invalid JSON") from exc

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}

    results: list[PlateauFeature] = []
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            raise ValueError(f"Missing mappings for feature {feature.feature_id}")
        update_data: dict[str, list[Contribution]] = {}
        for key in mapping_types.keys():
            values = mapped.get(key)
            if not values:
                raise ValueError(
                    f"'{key}' key missing or empty for feature {feature.feature_id}"
                )
            update_data[key] = values
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    return results


__all__ = ["map_feature", "map_features"]
