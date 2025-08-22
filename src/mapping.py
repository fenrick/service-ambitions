"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import logfire
from pydantic import ValidationError

from loader import MAPPING_DATA_DIR, load_mapping_items
from mapping_prompt import render_set_prompt
from models import (
    Contribution,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
)

if TYPE_CHECKING:
    from conversation import ConversationSession


_quarantine_logger: Callable[[Path], None] | None = None
"""Callback invoked with paths of quarantined mapping files."""


def set_quarantine_logger(callback: Callable[[Path], None] | None) -> None:
    """Register ``callback`` to receive paths of quarantined mapping files."""

    global _quarantine_logger
    _quarantine_logger = callback


def _quarantine_unknown_ids(data: Mapping[str, set[str]], service_id: str) -> Path:
    """Persist unknown mapping identifiers for ``service_id`` across all sets."""

    file_path = Path("quarantine/mapping") / service_id / "unknown_ids.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: sorted(v) for k, v in data.items() if v}
    file_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    logfire.warning(
        "Quarantined unknown mapping identifiers",
        path=str(file_path),
    )
    if _quarantine_logger is not None:
        _quarantine_logger(file_path)
    return file_path


def _clean_mapping_values(
    key: str,
    values: list[Contribution],
    valid_ids: dict[str, set[str]],
) -> tuple[list[Contribution], list[str]]:
    """Return filtered ``values`` and collect unknown IDs for ``key``."""

    valid: list[Contribution] = []
    unknown_ids: list[str] = []
    for item in values:
        if item.item not in valid_ids[key]:
            unknown_ids.append(item.item)
            continue
        valid.append(item)
    return valid, unknown_ids


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    catalogue_items: Mapping[str, list[MappingItem]] | None = None,
    service: str = "unknown",
) -> list[PlateauFeature]:
    """Return ``features`` merged with mapping ``payload``.

    Any mappings referencing unknown item identifiers are dropped rather than
    causing an error. Features with no valid mappings are tallied per set and
    logged once. This allows the calling code to rerun generation without
    manual intervention when the agent invents IDs.
    """

    catalogues = catalogue_items or load_mapping_items(MAPPING_DATA_DIR)
    valid_ids: dict[str, set[str]] = {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    dropped: dict[str, set[str]] = {}
    missing: dict[str, int] = {}
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            raise MappingError(f"Missing mappings for feature {feature.feature_id}")
        update_data = {}
        for key in mapping_types.keys():
            original = mapped.get(key, [])
            cleaned, unknown = _clean_mapping_values(key, original, valid_ids)
            if unknown:
                dropped.setdefault(key, set()).update(unknown)
            if not cleaned:
                missing[key] = missing.get(key, 0) + 1
            update_data[key] = cleaned
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    if dropped:
        for key, ids in dropped.items():
            logfire.warning(
                "Dropped unknown mapping IDs",
                key=key,
                count=len(ids),
            )
        _quarantine_unknown_ids(dropped, service)
    if missing:
        for key, count in missing.items():
            logfire.warning(f"{key}.missing={count}")
    return results


async def map_set(
    session: "ConversationSession",
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    *,
    service: str | None = None,
    strict: bool = False,
) -> list[PlateauFeature]:
    """Return ``features`` with ``set_name`` mappings populated.

    The agent is queried twice to obtain a valid :class:`MappingResponse`. The
    second attempt appends a hint instructing the model to return JSON only. If
    both attempts fail, the raw response is written to
    ``quarantine/mapping/<service>/<set>.txt`` and an empty mapping list is
    returned. When ``strict`` is ``True`` a :class:`MappingError` is raised
    instead of returning partial results.
    """

    cfg = MappingTypeConfig(dataset=set_name, label=set_name)
    prompt = render_set_prompt(set_name, list(items), features)

    try:
        raw = await session.ask_async(prompt)
        payload = MappingResponse.model_validate_json(raw)
    except (ValidationError, ValueError):
        hint_prompt = f"{prompt}\nReturn valid JSON only."
        raw = await session.ask_async(hint_prompt)
        try:
            payload = MappingResponse.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            svc = service or "unknown"
            qdir = Path("quarantine/mapping") / svc
            qdir.mkdir(parents=True, exist_ok=True)
            qfile = qdir / f"{set_name}.txt"
            qfile.write_text(raw, encoding="utf-8")
            if _quarantine_logger is not None:
                _quarantine_logger(qfile)
            if strict:
                raise MappingError(
                    f"Invalid mapping response for {svc}/{set_name}"
                ) from exc
            return [
                feat.model_copy(update={"mappings": feat.mappings | {set_name: []}})
                for feat in features
            ]

    return _merge_mapping_results(
        features,
        payload,
        {set_name: cfg},
        catalogue_items={set_name: list(items)},
        service=service or "unknown",
    )


class MappingError(RuntimeError):
    """Raised when a mapping response is missing required data."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


__all__ = [
    "map_set",
    "MappingError",
    "set_quarantine_logger",
]
