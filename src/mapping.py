"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Mapping, Sequence

import logfire

from loader import load_mapping_items, load_mapping_type_config, load_prompt_text
from models import (
    Contribution,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
)

if TYPE_CHECKING:
    from conversation import ConversationSession


class MappingError(RuntimeError):
    """Raised when a mapping response is missing required data."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


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


async def map_feature_async(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> PlateauFeature:
    """Asynchronously return ``feature`` augmented with mapping data."""

    return (await map_features_async(session, [feature], mapping_types))[0]


def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> PlateauFeature:
    """Return ``feature`` augmented with mapping information."""

    return asyncio.run(map_feature_async(session, feature, mapping_types))


def _build_mapping_prompt(
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig],
) -> str:
    """Return a prompt requesting mappings for ``features``."""

    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    items = load_mapping_items(tuple(cfg.dataset for cfg in mapping_types.values()))
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


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
) -> list[PlateauFeature]:
    """Return ``features`` merged with mapping ``payload``.

    Any mappings referencing unknown item identifiers are dropped rather than
    causing an error. This allows the calling code to rerun generation without
    manual intervention when the agent invents IDs.
    """
    # Build a lookup of valid item identifiers for each mapping type to
    # prevent the agent from inventing IDs that do not exist in reference
    # datasets.
    catalogues = load_mapping_items(
        tuple(cfg.dataset for cfg in mapping_types.values())
    )
    valid_ids: dict[str, set[str]] = {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            # Each feature must appear in the response; fail fast when absent.
            raise MappingError(f"Missing mappings for feature {feature.feature_id}")
        update_data: dict[str, list[Contribution]] = {}
        for key in mapping_types.keys():
            values = mapped.get(key, [])
            if not values:
                # Log missing or empty mappings rather than failing outright so
                # feature generation can continue when the agent omits a
                # category.  An empty list is stored for the mapping type.
                logfire.warning(
                    f"Missing mappings: feature={feature.feature_id} key={key}"
                )
            else:
                valid_values: list[Contribution] = []
                for item in values:
                    if item.item not in valid_ids[key]:
                        # Drop invalid mapping references so feature generation
                        # can proceed without manual intervention. These
                        # entries may be regenerated in a future run.
                        logfire.warning(
                            f"Dropping unknown {key} ID {item.item} for feature"
                            f" {feature.feature_id}"
                        )
                        continue
                    valid_values.append(item)
                values = valid_values
            update_data[key] = values
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    return results


async def map_features_async(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> list[PlateauFeature]:
    """Asynchronously return ``features`` with mapping information."""

    mapping_types = mapping_types or load_mapping_type_config()

    async def one(
        key: str, cfg: MappingTypeConfig
    ) -> tuple[str, MappingTypeConfig, MappingResponse]:
        sub_session = session.derive()
        prompt = _build_mapping_prompt(features, {key: cfg})
        logfire.debug(f"Requesting {key} mappings for {len(features)} features")
        payload = await sub_session.ask_async(prompt, output_type=MappingResponse)
        return key, cfg, payload

    tasks = [one(k, cfg) for k, cfg in mapping_types.items()]
    results_payloads = await asyncio.gather(*tasks)

    results: list[PlateauFeature] = list(features)
    for key, cfg, payload in results_payloads:
        results = _merge_mapping_results(results, payload, {key: cfg})
    return results


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> list[PlateauFeature]:
    """Return ``features`` augmented with mapping information."""

    return asyncio.run(map_features_async(session, features, mapping_types))


__all__ = [
    "map_feature",
    "map_feature_async",
    "map_features",
    "map_features_async",
    "MappingError",
]
