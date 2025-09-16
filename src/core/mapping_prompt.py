# SPDX-License-Identifier: MIT
"""Prompt rendering utilities for feature mappings.

This module provides deterministic formatting for mapping prompts to ensure
consistent interaction with the language model.
"""

from __future__ import annotations

from typing import Sequence

from pydantic_core import to_json

from io_utils.loader import load_prompt_text
from models import MappingItem, PlateauFeature


def _sanitize(value: str) -> str:
    """Replace newlines and tabs with spaces."""
    return value.replace("\n", " ").replace("\t", " ")


def _render_items(items: Sequence[MappingItem]) -> str:
    """Return JSON formatted ``items`` sorted by identifier.

    Each catalogue entry is rendered as an object containing ``id``, ``name``
    and ``description`` fields. Newline and tab characters are normalised to
    spaces to provide deterministic prompts for the language model.
    """
    data = [
        {
            "id": _sanitize(entry.id),
            "name": _sanitize(entry.name),
            "description": _sanitize(entry.description),
        }
        for entry in sorted(items, key=lambda i: i.id)
    ]
    # ``to_json`` emits ``bytes`` so decode before returning.
    return to_json(data, indent=2).decode()


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return JSON formatted ``features`` sorted by feature ID.

    Feature identifiers, names and descriptions are normalised to remove
    embedded newlines or tabs before serialisation to ensure deterministic
    prompts.
    """
    data = [
        {
            "id": _sanitize(feat.feature_id),
            "name": _sanitize(feat.name),
            "description": _sanitize(feat.description),
        }
        for feat in sorted(features, key=lambda f: f.feature_id)
    ]
    # ``to_json`` emits ``bytes`` so decode before returning.
    return to_json(data, indent=2).decode()


def render_set_prompt(
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    *,
    service_name: str,
    service_description: str,
    plateau: int,
    diagnostics: bool = False,
    facets_meta: Sequence[dict[str, object]] | None = None,
) -> str:
    """Return a prompt requesting mappings for ``features`` against ``set_name``.

    Args:
        set_name: Name of the mapping catalogue.
        items: Available catalogue items.
        features: Features requiring mapping enrichment.
        service_name: Human readable name of the service.
        service_description: Description of the service at ``plateau``.
        plateau: Numeric plateau level being evaluated.
        diagnostics: When ``True``, use the diagnostics template variant which
            logs additional context for debugging.
        facets_meta: Optional facet schema for the dataset (list of objects with
            ``id``, ``label``, ``type``, ``required`` and optional ``options``).
            When provided, it is embedded in the prompt to instruct the model to
            include all required facet values in each mapping contribution.

    Returns:
        Fully rendered prompt string.

    Notes:
        The optional ``facets_meta`` provides a dataset-specific facet schema
        (list of dicts with id/label/type/required/options) that is rendered
        into the prompt when present so the model can populate required facet
        values for each mapping contribution.
    """
    # Instruction template defines sections for catalogue items, feature
    # descriptions and service metadata. Rendering is deterministic as both
    # helpers sort inputs and the plateau description is sanitised.
    instruction = load_prompt_text(
        "mapping_prompt_diagnostics" if diagnostics else "mapping_prompt"
    )
    catalogue_lines = _render_items(items)
    feature_lines = _render_features(features)
    mapping_section = f"## Available {set_name}\n\n```json\n{catalogue_lines}\n```"

    # Optional: include facet schema instructions when provided by the dataset
    # metadata. The templates include a placeholder that expands to an empty
    # string when no facets are configured to preserve backwards compatibility.
    facet_lines = ""
    if facets_meta:
        # Reduce to a stable, minimal schema the model can follow.
        facet_lines = to_json(facets_meta, indent=2).decode()
        facet_lines = (
            "\n## Facets (optional per mapping entry)\n\n"
            "When provided, each mapping object may include a 'facets' object "
            "with keys from the schema below. Only include keys defined here.\n\n"
            f"```json\n{facet_lines}\n```\n"
        )

    # Manual placeholder substitution avoids ``str.format`` interpreting JSON
    # braces within the template as formatting fields. This ensures the prompt
    # retains literal brace characters required for example JSON structures and
    # narrative instructions.
    replacements = {
        "{mapping_labels}": set_name,
        "{mapping_sections}": mapping_section,
        "{mapping_fields}": set_name,
        "{facet_instructions}": facet_lines,
        "{features}": f"```json\n{feature_lines}\n```",
        "{service_name}": _sanitize(service_name),
        "{service_description}": _sanitize(service_description),
        "{plateau}": str(plateau),
    }

    for placeholder, value in replacements.items():
        instruction = instruction.replace(placeholder, value)

    return instruction


__all__ = ["render_set_prompt"]
