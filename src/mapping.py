# SPDX-License-Identifier: MIT
"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import logfire
from pydantic import ValidationError

from loader import MAPPING_DATA_DIR, load_mapping_items
from mapping_prompt import render_set_prompt
from models import (
    Contribution,
    MappingDiagnosticsResponse,
    MappingFeature,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
    StrictModel,
)
from quarantine import QuarantineWriter
from settings import load_settings
from telemetry import record_mapping_set

if TYPE_CHECKING:
    from conversation import ConversationSession


_writer = QuarantineWriter()

UNKNOWN_ID_LOG_LIMIT = 5
"""Maximum number of unknown IDs to include in warning logs."""


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    catalogue_items: Mapping[str, list[MappingItem]] | None = None,
    service: str = "unknown",
    strict: bool = False,
) -> tuple[list[PlateauFeature], int]:
    """Return ``features`` merged with mapping ``payload`` and unknown count.

    All mapping identifiers are validated against ``valid_ids``. Invented
    identifiers are removed from the result and logged with up to
    ``UNKNOWN_ID_LOG_LIMIT`` examples per set. When ``strict`` is ``True`` the
    presence of unknown or missing identifiers raises :class:`MappingError`.
    """

    catalogues = catalogue_items or load_mapping_items(
        MAPPING_DATA_DIR, load_settings().mapping_sets
    )
    valid_ids: dict[str, set[str]] = {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    dropped: dict[str, list[str]] = {}
    missing: dict[str, int] = {}
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            raise MappingError(f"Missing mappings for feature {feature.feature_id}")
        update_data = {}
        for key in mapping_types.keys():
            original = mapped.get(key, [])
            cleaned: list[Contribution] = []
            unknown: list[str] = []
            for item in original:
                # Guard against invented identifiers by dropping unknown IDs.
                if item.item not in valid_ids[key]:
                    unknown.append(item.item)
                    continue
                cleaned.append(item)
            if unknown:
                dropped.setdefault(key, []).extend(unknown)
            if not cleaned:
                # Track features missing any valid mappings for this set.
                missing[key] = missing.get(key, 0) + 1
            update_data[key] = cleaned
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    unknown_total = 0
    if dropped:
        for key, ids in dropped.items():
            logfire.warning(
                "Dropped unknown mapping IDs",
                set_name=key,
                count=len(ids),
                examples=ids[:UNKNOWN_ID_LOG_LIMIT],
            )
            unknown_total += len(ids)
            _writer.write(key, service or "unknown", "unknown_ids", sorted(ids))
        if strict:
            raise MappingError("Unknown mapping identifiers returned")
    if missing:
        for key, count in missing.items():
            logfire.warning(f"{key}.missing={count}")
        if strict:
            raise MappingError("Mappings missing for one or more features")
    return results, unknown_total


async def map_set(
    session: "ConversationSession",
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    *,
    service: str | None = None,
    strict: bool = False,
    diagnostics: bool | None = None,
    use_local_cache: bool = False,
) -> list[PlateauFeature]:
    """Return ``features`` with ``set_name`` mappings populated.

    The agent is queried twice to obtain a valid :class:`MappingResponse`. The
    second attempt appends a hint instructing the model to return JSON only. If
    both attempts fail, the raw response is written to
    ``quarantine/mapping/<service>/<set>.txt`` and an empty mapping list is
    returned. When ``strict`` is ``True`` a :class:`MappingError` is raised
    instead of returning partial results. When ``use_local_cache`` is ``True``
    responses are cached under ``.cache/mapping`` using ``hash(prompt)`` so
    repeated requests can bypass network calls.
    """

    cfg = MappingTypeConfig(dataset=set_name, label=set_name)
    use_diag = (
        diagnostics
        if diagnostics is not None
        else getattr(session, "diagnostics", False)
    )
    prompt = render_set_prompt(set_name, list(items), features, diagnostics=use_diag)
    model_type: type[StrictModel] = (
        MappingDiagnosticsResponse if use_diag else MappingResponse
    )
    cache_file: Path | None = None
    payload: StrictModel | None = None
    if use_local_cache:
        cache_dir = Path(".cache") / "mapping" / set_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{hash(prompt)}.json"
        if cache_file.exists():
            # Cache hit, parse payload from disk and skip the network.
            cached = cache_file.read_text(encoding="utf-8")
            payload = model_type.model_validate_json(cached)

    start = time.monotonic()
    retries = 0
    tokens = 0
    cost = 0.0
    if payload is None:
        # Cache miss triggers a network request.
        try:
            raw = await session.ask_async(prompt, output_type=model_type)
            tokens += getattr(session, "last_tokens", 0)
            cost += getattr(session, "last_cost", 0.0)
            payload = (
                raw
                if isinstance(raw, StrictModel)
                else model_type.model_validate_json(raw)
            )
        except (ValidationError, ValueError):
            retries = 1
            hint_prompt = f"{prompt}\nReturn valid JSON only."
            raw = await session.ask_async(hint_prompt, output_type=model_type)
            tokens += getattr(session, "last_tokens", 0)
            cost += getattr(session, "last_cost", 0.0)
            try:
                payload = (
                    raw
                    if isinstance(raw, StrictModel)
                    else model_type.model_validate_json(raw)
                )
            except (ValidationError, ValueError) as exc:
                svc = service or "unknown"
                text = raw if isinstance(raw, str) else raw.model_dump()
                _writer.write(set_name, svc, "json_parse_error", text)
                if strict:
                    raise MappingError(
                        f"Invalid mapping response for {svc}/{set_name}"
                    ) from exc
                record_mapping_set(
                    set_name,
                    features=len(features),
                    mapped_ids=0,
                    unknown_ids=0,
                    retries=retries,
                    latency=time.monotonic() - start,
                    tokens=tokens,
                    cost=cost,
                )
                return [
                    feat.model_copy(update={"mappings": feat.mappings | {set_name: []}})
                    for feat in features
                ]
        if cache_file:
            # Persist successful responses for future runs.
            cache_file.write_text(
                raw if isinstance(raw, str) else raw.model_dump_json(),
                encoding="utf-8",
            )

    if use_diag:
        diag_payload = cast(MappingDiagnosticsResponse, payload)
        plain = [
            MappingFeature(
                feature_id=feat.feature_id,
                mappings={
                    key: [Contribution(item=c.item) for c in vals]
                    for key, vals in feat.mappings.items()
                },
            )
            for feat in diag_payload.features
        ]
        payload_norm = MappingResponse(features=plain)
    else:
        payload_norm = cast(MappingResponse, payload)

    merged, unknown_count = _merge_mapping_results(
        features,
        payload_norm,
        {set_name: cfg},
        catalogue_items={set_name: list(items)},
        service=service or "unknown",
        strict=strict,
    )
    latency = time.monotonic() - start
    mapped_ids = sum(len(f.mappings.get(set_name, [])) for f in merged)
    record_mapping_set(
        set_name,
        features=len(features),
        mapped_ids=mapped_ids,
        unknown_ids=unknown_count,
        retries=retries,
        latency=latency,
        tokens=tokens,
        cost=cost,
    )
    return merged


class MappingError(RuntimeError):
    """Raised when a mapping response is missing required data."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


__all__ = [
    "map_set",
    "MappingError",
]
