"""Utilities for enriching plateau features with mapping data."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Sequence

from loader import load_mapping_items, load_mapping_prompt
from models import Contribution, PlateauFeature

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from conversation import ConversationSession

logger = logging.getLogger(__name__)


def _render_items(items: list[dict[str, str]]) -> str:
    """Return a bullet list representation of mapping reference items."""

    return "\n".join(
        f"- {entry['id']}: {entry['name']} - {entry['description']}" for entry in items
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return a bullet list of feature details for prompt construction."""

    return "\n".join(
        f"- {feat.feature_id}: {feat.name} - {feat.description}" for feat in features
    )


def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    prompt_dir: str = "prompts",
) -> PlateauFeature:
    """Return ``feature`` augmented with data, application and technology mappings.

    The function prompts ``session`` three times using a mapping template: once
    each for data, applications and technology. The agent must respond with JSON
    containing a list for the requested category. Each element of the list must
    provide ``item`` and ``contribution`` fields. If any list is missing or
    empty, a :class:`ValueError` is raised.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.
        prompt_dir: Directory containing prompt templates.

    Returns:
        A :class:`PlateauFeature` with mapping information applied.

    Raises:
        ValueError: If a response cannot be parsed or a required list is empty.
    """

    template = load_mapping_prompt(prompt_dir)
    mapping_items = load_mapping_items()

    categories = (
        ("data", "Information", "information"),
        ("applications", "Applications", "applications"),
        ("technology", "Technologies", "technologies"),
    )

    mapped: dict[str, list[Contribution]] = {}
    for key, label, item_key in categories:
        prompt = template.format(
            feature_name=feature.name,
            feature_description=feature.description,
            category_label=label,
            category_items=_render_items(mapping_items[item_key]),
            category_key=key,
        )
        logger.debug("Requesting %s mappings for feature %s", key, feature.feature_id)
        response = session.ask(prompt)
        logger.debug("Raw %s mapping response: %s", key, response)

        try:
            payload: dict[str, Any] = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from mapping response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

        raw_items = payload.get(key)
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"'{key}' key missing or empty")

        mapped[key] = [Contribution(**item) for item in raw_items]

    merged = {**feature.model_dump(), **mapped}
    return PlateauFeature(**merged)


def map_features(
    session: ConversationSession, features: Sequence[PlateauFeature]
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

    mapping_items = load_mapping_items()
    prompt = (
        "Map each feature to relevant Data, Applications and Technologies from"
        " the lists below.\n\n"
        "## Available Data\n"
        f"{_render_items(mapping_items['information'])}\n\n"
        "## Available Applications\n"
        f"{_render_items(mapping_items['applications'])}\n\n"
        "## Available Technologies\n"
        f"{_render_items(mapping_items['technologies'])}\n\n"
        "## Features\n"
        f"{_render_features(features)}\n\n"
        "Return JSON with a 'features' array. Each element must include\n"
        "'feature_id', 'data', 'applications' and 'technology' arrays. Each\n"
        "array must contain at least one object with 'item' and 'contribution'.\n"
        "Use only items from the provided lists. Do not include any text"
        " outside the JSON object."
    )
    logger.debug("Requesting mappings for %s features", len(features))
    response = session.ask(prompt)
    logger.debug("Raw multi-feature mapping response: %s", response)
    try:
        payload: dict[str, Any] = json.loads(response)
    except json.JSONDecodeError as exc:  # pragma: no cover - logging
        logger.error("Invalid JSON from mapping response: %s", exc)
        raise ValueError("Agent returned invalid JSON") from exc

    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("'features' key missing or invalid")
    mapped_lookup = {item.get("feature_id"): item for item in raw_features}

    results: list[PlateauFeature] = []
    for feature in features:
        data = mapped_lookup.get(feature.feature_id)
        if not isinstance(data, dict):
            raise ValueError(f"Missing mappings for feature {feature.feature_id}")
        mapped: dict[str, list[Contribution]] = {}
        for key in ("data", "applications", "technology"):
            raw_list = data.get(key)
            if not isinstance(raw_list, list) or not raw_list:
                raise ValueError(
                    f"'{key}' key missing or empty for feature {feature.feature_id}"
                )
            mapped[key] = [Contribution(**item) for item in raw_list]
        merged = {**feature.model_dump(), **mapped}
        results.append(PlateauFeature(**merged))
    return results


__all__ = ["map_feature", "map_features"]
