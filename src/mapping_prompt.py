"""Prompt rendering utilities for feature mappings.

This module provides deterministic formatting for mapping prompts to ensure
consistent interaction with the language model."""

from __future__ import annotations

import json
from typing import Sequence

from loader import load_prompt_text
from models import (
    MappingDiagnosticsResponse,
    MappingItem,
    MappingResponse,
    PlateauFeature,
)

MAPPING_SCHEMA = json.dumps(MappingResponse.model_json_schema(), indent=2)
MAPPING_DIAGNOSTICS_SCHEMA = json.dumps(
    MappingDiagnosticsResponse.model_json_schema(), indent=2
)


def _sanitize(value: str) -> str:
    """Replace newlines and tabs with spaces."""

    return value.replace("\n", " ").replace("\t", " ")


def _render_items(items: Sequence[MappingItem]) -> str:
    """Return tab-separated ``items`` sorted by identifier.

    The compact format ``ID\tname\tdescription`` is used for each line. Any
    embedded newline or tab characters are replaced with spaces to ensure
    deterministic rendering for the language model.
    """

    return "\n".join(
        "\t".join(_sanitize(part) for part in (entry.id, entry.name, entry.description))
        for entry in sorted(items, key=lambda i: i.id)
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return tab-separated ``features`` sorted by feature ID.

    Lines are formatted as ``ID\tname\tdescription`` and sanitised to replace
    any internal newlines or tabs with spaces.
    """

    return "\n".join(
        "\t".join(
            _sanitize(part) for part in (feat.feature_id, feat.name, feat.description)
        )
        for feat in sorted(features, key=lambda f: f.feature_id)
    )


def render_set_prompt(
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    *,
    diagnostics: bool = False,
) -> str:
    """Return a prompt requesting mappings for ``features`` against ``set_name``.

    Args:
        set_name: Name of the mapping catalogue.
        items: Available catalogue items.
        features: Features requiring mapping enrichment.

    Returns:
        Fully rendered prompt string.
    """
    # Instruction template defines sections for catalogue items and feature
    # descriptions. Rendering is deterministic as both helpers sort inputs.
    instruction = load_prompt_text(
        "mapping_prompt_diagnostics" if diagnostics else "mapping_prompt"
    )
    catalogue_lines = _render_items(items)
    feature_lines = _render_features(features)
    mapping_section = f"## Available {set_name}\n\n{catalogue_lines}\n"
    return instruction.format(
        mapping_labels=set_name,
        mapping_sections=mapping_section,
        mapping_fields=set_name,
        features=feature_lines,
        schema=MAPPING_DIAGNOSTICS_SCHEMA if diagnostics else MAPPING_SCHEMA,
    )


__all__ = ["render_set_prompt"]
