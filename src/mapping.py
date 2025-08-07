"""Utilities for enriching plateau features with mapping data."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

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


__all__ = ["map_feature"]
