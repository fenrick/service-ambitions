"""Prompt rendering utilities for feature mappings.

This module provides deterministic formatting for mapping prompts to ensure
consistent interaction with the language model."""

from __future__ import annotations

import json
from typing import Sequence

from loader import load_prompt_text
from models import MappingItem, MappingResponse, PlateauFeature


def _render_items(items: Sequence[MappingItem]) -> str:
    """Return bullet list representation of ``items`` sorted by identifier."""

    return "\n".join(
        f"- {entry.id}: {entry.name} - {entry.description}"
        for entry in sorted(items, key=lambda i: i.id)
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return bullet list representation of ``features`` sorted by feature ID."""

    return "\n".join(
        f"- {feat.feature_id}: {feat.name} - {feat.description}"
        for feat in sorted(features, key=lambda f: f.feature_id)
    )


def render_set_prompt(
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
) -> str:
    """Return a prompt requesting mappings for ``features`` against ``set_name``.

    Args:
        set_name: Name of the mapping catalogue.
        items: Available catalogue items.
        features: Features requiring mapping enrichment.

    Returns:
        Fully rendered prompt string.
    """

    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    mapping_section = f"## Available {set_name}\n\n{_render_items(items)}\n"
    prompt = template.format(
        mapping_labels=set_name,
        mapping_sections=mapping_section,
        mapping_fields=set_name,
        features=_render_features(features),
        schema=str(schema),
    )
    return prompt


__all__ = ["render_set_prompt"]
