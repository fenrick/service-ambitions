# SPDX-License-Identifier: MIT
"""Helpers for the CLI mapping subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Sequence, cast

from core import mapping
from core.canonical import canonicalise_record
from core.conversation import ConversationSession
from io_utils.loader import configure_mapping_data_dir, load_mapping_items
from models import (
    FeatureMappingRef,
    MappingFeatureGroup,
    MappingItem,
    MappingSet,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
)
from utils import ErrorHandler


def load_catalogue(
    mapping_data_dir: Path | None,
    settings,
    error_handler: ErrorHandler | None = None,
) -> tuple[dict[str, list[MappingItem]], str]:
    """Return mapping catalogue items and hash.

    Args:
        mapping_data_dir: Optional directory override for catalogue files.
        settings: Runtime settings containing ``mapping_data_dir`` and ``mapping_sets``.

    Returns:
        A tuple of ``(items, catalogue_hash)`` where ``items`` is a mapping of
        field name to ``MappingItem`` list.

    Side Effects:
        Adjusts the global loader state by configuring the mapping data
        directory before reading items from disk.
    """

    configure_mapping_data_dir(mapping_data_dir or settings.mapping_data_dir)
    return load_mapping_items(settings.mapping_sets, error_handler=error_handler)


async def _apply_mapping_sets(
    features: Sequence[PlateauFeature],
    items: dict[str, list[MappingItem]],
    settings,
    cache_mode: Literal["off", "read", "refresh", "write"],
    catalogue_hash: str,
) -> list[PlateauFeature]:
    """Return features with all mapping sets applied.

    Args:
        features: Plateau features requiring mapping enrichment.
        items: Catalogue items keyed by mapping field.
        settings: Runtime settings providing mapping configuration.
        cache_mode: Strategy controlling mapping cache usage.
        catalogue_hash: Hash of the catalogue for cache invalidation.

    Returns:
        List of features augmented with mapping contributions.
    """

    mapped = list(features)
    for cfg in settings.mapping_sets:  # Apply each mapping set sequentially.
        mapped = await mapping.map_set(
            cast(ConversationSession, object()),
            cfg.field,
            items[cfg.field],
            mapped,
            service_name="svc",
            service_description="desc",
            plateau=1,
            strict=settings.strict_mapping,
            diagnostics=settings.diagnostics,
            cache_mode=cache_mode,
            catalogue_hash=catalogue_hash,
        )
    return mapped


def _group_plateau_mappings(
    plateau: PlateauResult,
    mapping_sets: Sequence[MappingSet],
    catalogue_lookup: dict[str, dict[str, str]],
    features_by_id: dict[str, PlateauFeature],
) -> None:
    """Populate mapping groups for a single plateau in place.

    Args:
        plateau: Plateau result to mutate.
        mapping_sets: Mapping set configurations.
        catalogue_lookup: Mapping field to item name lookup.
        features_by_id: Lookup of mapped features keyed by ``feature_id``.

    Side Effects:
        ``plateau.mappings`` is replaced with grouped mapping references.
    """

    mapped_feats = [features_by_id[f.feature_id] for f in plateau.features]
    plateau.mappings = {}
    for cfg in mapping_sets:  # Build groups for each mapping set.
        groups: dict[str, list[FeatureMappingRef]] = {}
        for feat in mapped_feats:  # Aggregate contributions per mapping item.
            for contrib in feat.mappings.get(cfg.field, []):
                groups.setdefault(contrib.item, []).append(
                    FeatureMappingRef(
                        feature_id=feat.feature_id,
                        description=feat.description,
                    )
                )
        catalogue = catalogue_lookup[cfg.field]
        plateau.mappings[cfg.field] = [
            MappingFeatureGroup(
                id=item_id,
                name=catalogue.get(item_id, item_id),
                mappings=sorted(refs, key=lambda r: r.feature_id),
            )
            for item_id, refs in sorted(groups.items())
        ]


def _assemble_mapping_groups(
    evolutions: Sequence[ServiceEvolution],
    mapped: Sequence[PlateauFeature],
    items: dict[str, list[MappingItem]],
    settings,
) -> None:
    """Populate plateau mapping groups on ``evolutions`` in place.

    Args:
        evolutions: Service evolutions to update.
        mapped: Features with mapping contributions applied.
        items: Catalogue items keyed by mapping field.
        settings: Runtime settings providing mapping configuration.

    Side Effects:
        Each plateau within ``evolutions`` gains grouped mapping references.
    """

    catalogue_lookup = {
        cfg.field: {item.id: item.name for item in items[cfg.field]}
        for cfg in settings.mapping_sets
    }
    features_by_id = {f.feature_id: f for f in mapped}
    for evo in evolutions:  # Traverse each evolution.
        for plateau in evo.plateaus:  # Populate mappings per plateau.
            _group_plateau_mappings(
                plateau, settings.mapping_sets, catalogue_lookup, features_by_id
            )


async def remap_features(
    evolutions: Sequence[ServiceEvolution],
    items: dict[str, list[MappingItem]],
    settings,
    cache_mode: Literal["off", "read", "refresh", "write"],
    catalogue_hash: str,
) -> None:
    """Populate feature mappings on the provided evolutions.

    Args:
        evolutions: Mutable collection of service evolutions to update.
        items: Catalogue items keyed by mapping field.
        settings: Runtime settings providing mapping configuration.
        cache_mode: Strategy controlling mapping cache usage.
        catalogue_hash: Hash of the catalogue for cache invalidation.

    Side Effects:
        Evolutions are mutated in place; each plateau gains populated
        ``mappings`` derived from the mapping catalogue and remote service.
    """

    features = [f for evo in evolutions for p in evo.plateaus for f in p.features]
    mapped = await _apply_mapping_sets(
        features, items, settings, cache_mode, catalogue_hash
    )
    _assemble_mapping_groups(evolutions, mapped, items, settings)


def write_output(evolutions: Iterable[ServiceEvolution], output_path: Path) -> None:
    """Write mapped evolutions to ``output_path`` as canonical JSON lines.

    Args:
        evolutions: Iterable of evolutions with mappings applied.
        output_path: Destination file for JSONL output.

    Side Effects:
        Creates or overwrites ``output_path`` with one line per evolution.
    """

    with output_path.open("w", encoding="utf-8") as fh:
        for evo in evolutions:
            record = canonicalise_record(evo.model_dump(mode="json"))
            fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
