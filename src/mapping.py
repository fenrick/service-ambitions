"""Utilities for enriching plateau features with mapping data."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Sequence

from loader import load_mapping_items, load_prompt_text
from models import MappingResponse, PlateauFeature

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from conversation import ConversationSession

logger = logging.getLogger(__name__)


def _render_items(items: list[dict[str, str]]) -> str:
    """Return a bullet list representation of mapping reference items."""

    # Present each mapping reference item on a separate line so it can be
    # directly inserted into the agent prompt as a bullet list.
    return "\n".join(
        f"- {entry['id']}: {entry['name']} - {entry['description']}" for entry in items
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
) -> PlateauFeature:
    """Return ``feature`` augmented with mapping information.

    This is a convenience wrapper around :func:`map_features` for mapping a
    single feature.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.

    Returns:
        A :class:`PlateauFeature` with mapping information applied.
    """

    return map_features(session, [feature])[0]


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
) -> list[PlateauFeature]:
    """Return ``features`` augmented with data, application and technology mappings.

    The function sends a single prompt containing all features and mapping
    reference lists. The agent must respond with JSON containing a ``features``
    array. Each feature in the array must provide non-empty ``data``,
    ``applications`` and ``technology`` lists. A :class:`ValueError` is raised
    if the response cannot be parsed, a feature is missing or any list is
    empty.

    Args:
        session: Active conversation session used to query the agent.
        features: Plateau features to map.

    Returns:
        List of :class:`PlateauFeature` objects with mapping information applied.

    Raises:
        ValueError: If the agent response is invalid or any mapping list is
        missing.
    """

    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    mapping_items = load_mapping_items()
    prompt = template.format(
        data_items=_render_items(mapping_items["information"]),
        application_items=_render_items(mapping_items["applications"]),
        technology_items=_render_items(mapping_items["technologies"]),
        features=_render_features(features),
        schema=str(schema),
    )
    # The schema is appended verbatim to ensure the agent adheres exactly to the
    # expected JSON structure.
    prompt = f"{prompt}\n\nJSON schema:\n{schema}"
    logger.debug("Requesting mappings for %s features", len(features))
    response = session.ask(prompt)
    logger.debug("Raw multi-feature mapping response: %s", response)
    try:
        payload = MappingResponse.model_validate_json(response)
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Invalid JSON from mapping response: %s", exc)
        raise ValueError("Agent returned invalid JSON") from exc

    mapped_lookup = {item.feature_id: item for item in payload.features}

    results: list[PlateauFeature] = []
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            raise ValueError(f"Missing mappings for feature {feature.feature_id}")
        for key in ("data", "applications", "technology"):
            if not getattr(mapped, key):
                raise ValueError(
                    f"'{key}' key missing or empty for feature {feature.feature_id}"
                )
        merged = feature.model_copy(
            update={
                "data": mapped.data,
                "applications": mapped.applications,
                "technology": mapped.technology,
            }
        )
        results.append(merged)
    return results


__all__ = ["map_feature", "map_features"]
