# SPDX-License-Identifier: MIT
"""Utilities for deterministic serialisation of service records."""

from __future__ import annotations

from typing import Any, Dict, List


def _sort_contributions(mappings: Dict[str, List[Dict[str, Any]]]) -> None:
    """Sort each mapping contribution list by ``item``."""

    for items in mappings.values():
        items.sort(key=lambda c: c.get("item", ""))


def _sort_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``feature`` with sorted mapping lists."""

    mappings = feature.get("mappings")
    if isinstance(mappings, dict):
        _sort_contributions(mappings)
    return feature


def _sort_feature_refs(refs: List[Dict[str, Any]]) -> None:
    """Order feature references by ``feature_id``."""

    refs.sort(key=lambda r: r.get("feature_id", ""))


def _sort_mapping_groups(groups: List[Dict[str, Any]]) -> None:
    """Order mapping groups by ``id`` and their feature refs by ``feature_id``."""

    for group in groups:
        refs = group.get("mappings")
        if isinstance(refs, list):
            _sort_feature_refs(refs)
    groups.sort(key=lambda g: g.get("id", ""))


def _sort_grouped_mappings(mappings: Dict[str, List[Dict[str, Any]]]) -> None:
    """Sort grouped mappings by mapping type and group identifier."""

    for groups in mappings.values():
        _sort_mapping_groups(groups)


def _sort_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ``features`` sorted by ``feature_id`` with mappings ordered."""

    return sorted(
        (_sort_feature(f) for f in features), key=lambda f: f.get("feature_id", "")
    )


def canonicalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``record`` with lists sorted for deterministic output."""

    service = record.get("service")
    if isinstance(service, dict):
        feats = service.get("features")
        if isinstance(feats, list):
            service["features"] = _sort_features(feats)

    for plateau in record.get("plateaus", []):
        if isinstance(plateau, dict):
            feats = plateau.get("features")
            if isinstance(feats, list):
                plateau["features"] = _sort_features(feats)
            mappings = plateau.get("mappings")
            if isinstance(mappings, dict):
                _sort_grouped_mappings(mappings)

    meta = record.get("meta")
    if isinstance(meta, dict):
        if meta.get("seed") is None:
            meta["seed"] = 0
        if meta.get("context_window") is None:
            meta["context_window"] = 0
        if meta.get("diagnostics") is None:
            meta["diagnostics"] = False
        if meta.get("catalogue_hash") is None:
            meta["catalogue_hash"] = ""

    return record


__all__ = ["canonicalise_record"]
