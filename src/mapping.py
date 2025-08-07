"""Feature mapping utilities via Pydantic AI."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

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


async def map_feature(
    session: ConversationSession, feature: PlateauFeature
) -> MappedPlateauFeature:
    """Return ``feature`` augmented with contribution mappings.

    The function prompts the provided ``session`` to generate mapping data for the
    given ``feature``. The agent is expected to respond with JSON containing a
    ``mappings`` key whose value is a list of objects with ``item`` and
    ``contribution`` fields.

    Args:
        session: Active conversation session used to query the agent.
        feature: Plateau feature to map.

    Returns:
        A new :class:`MappedPlateauFeature` including any mappings returned by the
        agent.

    Raises:
        ValueError: If the agent response cannot be parsed or lacks mapping data.
    """

    prompt = (
        "Provide mapping items for the following feature. "
        "Respond in JSON with a 'mappings' key where each item has 'item' and "
        "'contribution' fields.\n"
        f"Feature name: {feature.name}\nDescription: {feature.description}"
    )

    logger.debug("Requesting mappings for feature %s", feature.feature_id)
    response = await session.ask(prompt)
    logger.debug("Raw mapping response: %s", response)

    try:
        payload: dict[str, Any] = json.loads(response)
    except json.JSONDecodeError as exc:  # pragma: no cover - logging
        logger.error("Invalid JSON from mapping response: %s", exc)
        raise ValueError("Agent returned invalid JSON") from exc

    raw_mappings = payload.get("mappings", [])
    if not isinstance(raw_mappings, list):
        raise ValueError("'mappings' key missing or not a list")

    mappings = [MappingItem(**item) for item in raw_mappings]
    return MappedPlateauFeature(**feature.model_dump(), mappings=mappings)


__all__ = [
    "MappingItem",
    "MappedPlateauFeature",
    "map_feature",
]
