# SPDX-License-Identifier: MIT
"""Helpers for the CLI mapping subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Sequence, cast

import loader
import mapping
from canonical import canonicalise_record
from conversation import ConversationSession
from loader import configure_mapping_data_dir, load_mapping_items
from models import (
    FeatureMappingRef,
    MappingFeatureGroup,
    MappingItem,
    ServiceEvolution,
)


def load_catalogue(
    mapping_data_dir: Path | None, settings
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
    return load_mapping_items(loader.MAPPING_DATA_DIR, settings.mapping_sets)


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
    mapped = features
    for cfg in settings.mapping_sets:
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

    catalogue_lookup = {
        cfg.field: {item.id: item.name for item in items[cfg.field]}
        for cfg in settings.mapping_sets
    }
    by_id = {f.feature_id: f for f in mapped}
    for evo in evolutions:
        for plateau in evo.plateaus:
            mapped_feats = [by_id[f.feature_id] for f in plateau.features]
            plateau.mappings = {}
            for cfg in settings.mapping_sets:
                groups: dict[str, list[FeatureMappingRef]] = {}
                for feat in mapped_feats:
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
