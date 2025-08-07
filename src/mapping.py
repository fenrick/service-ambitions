"""Feature mapping utilities via Pydantic AI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from loader import load_mapping_items, load_mapping_prompt
from models import PlateauFeature

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from conversation import ConversationSession

logger = logging.getLogger(__name__)


class MappingItem(BaseModel):
    """Single mapping entry describing a contribution."""

    item: str = Field(..., description="Name of the mapped element.")
    contribution: str = Field(
        ..., description="Explanation of how the item contributes to the feature."
    )


class MappedPlateauFeature(PlateauFeature):
    """Extension of :class:`PlateauFeature` including mapping information."""

    mappings: list[MappingItem] = Field(
        default_factory=list,
        description="List of related items and their contributions.",
    )


def _render_items(items: list[dict[str, str]]) -> str:
    """Return a bullet list representation of mapping items."""

    return "\n".join(
        f"- {entry['id']}: {entry['name']} - {entry['description']}" for entry in items
    )


async def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    prompt_dir: str = "prompts",
) -> MappedPlateauFeature:
    """Return ``feature`` augmented with contribution mappings.

    The function prompts the provided ``session`` three times to generate mapping
    data for the given ``feature``—once each for information, applications and
    technologies. Each agent response must contain a ``mappings`` key whose value
    is a list of objects with ``item`` and ``contribution`` fields.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.
        prompt_dir: Directory containing prompt templates.

    Returns:
        A new :class:`MappedPlateauFeature` including any mappings returned by the
        agent.

    Raises:
        ValueError: If the agent response cannot be parsed or lacks mapping data.
    """

    template = load_mapping_prompt(prompt_dir)
    mapping_items = load_mapping_items()
    mappings: list[MappingItem] = []
    categories = (
        ("information", "Information"),
        ("applications", "Applications"),
        ("technologies", "Technologies"),
    )

    for key, label in categories:
        prompt = template.format(
            feature_name=feature.name,
            feature_description=feature.description,
            category_label=label,
            category_items=_render_items(mapping_items[key]),
        )

        logger.debug("Requesting %s mappings for feature %s", key, feature.feature_id)
        response = await asyncio.to_thread(session.ask, prompt)
        logger.debug("Raw %s mapping response: %s", key, response)

        try:
            payload: dict[str, Any] = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from mapping response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

        raw_mappings = payload.get("mappings", [])
        if not isinstance(raw_mappings, list):
            raise ValueError("'mappings' key missing or not a list")

        mappings.extend(MappingItem(**item) for item in raw_mappings)

    return MappedPlateauFeature(**feature.model_dump(), mappings=mappings)


__all__ = [
    "MappingItem",
    "MappedPlateauFeature",
    "map_feature",
]
