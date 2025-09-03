# SPDX-License-Identifier: MIT
"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import logfire
from pydantic import ValidationError
from pydantic_core import from_json, to_json

from io_utils.loader import load_mapping_items, load_prompt_text
from io_utils.quarantine import QuarantineWriter
from models import (
    Contribution,
    FeatureMappingRef,
    MappingDiagnosticsResponse,
    MappingFeature,
    MappingFeatureGroup,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
    StrictModel,
)
from observability.telemetry import record_mapping_set
from runtime.environment import RuntimeEnv
from utils import CacheManager, ErrorHandler, JSONCacheManager, LoggingErrorHandler

from .mapping_prompt import render_set_prompt

if TYPE_CHECKING:
    from .conversation import ConversationSession


_writer = QuarantineWriter()

_cache_manager: CacheManager = JSONCacheManager()
_error_handler: ErrorHandler = LoggingErrorHandler()


def configure_cache_manager(manager: CacheManager) -> None:
    """Override the cache manager used for mapping results."""

    global _cache_manager
    _cache_manager = manager


def configure_error_handler(handler: ErrorHandler) -> None:
    """Override the error handler used for mapping operations."""

    global _error_handler
    _error_handler = handler


UNKNOWN_ID_LOG_LIMIT = 5
"""Maximum number of unknown IDs to include in warning logs."""


def _sanitize(value: str) -> str:
    """Return ``value`` with newlines and tabs replaced by spaces."""

    return value.replace("\n", " ").replace("\t", " ")


def _json_bytes(value: Any, *, sort_keys: bool = False, **kwargs: Any) -> bytes:
    """Return ``value`` serialised to JSON bytes."""

    if sort_keys and isinstance(value, dict):
        value = dict(sorted(value.items()))
    try:
        return cast(Any, to_json)(value, **kwargs)
    except TypeError:  # pragma: no cover - legacy pydantic-core
        if "sort_keys" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("sort_keys")
        return cast(Any, to_json)(value, **kwargs)


def _features_hash(features: Sequence[PlateauFeature]) -> str:
    """Return a SHA256 hash summarising ``features``."""

    digests = []
    for feat in features:
        canonical = _json_bytes(
            {
                "ref": _sanitize(feat.feature_id),
                "name": _sanitize(feat.name),
                "description": _sanitize(feat.description),
            },
            sort_keys=True,
        )
        digests.append(hashlib.sha256(canonical).hexdigest())
    combined = "".join(sorted(digests))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def build_cache_key(
    model_name: str,
    set_name: str,
    catalogue_hash: str,
    features: Sequence[PlateauFeature],
    diagnostics: bool,
) -> str:
    """Return a deterministic cache key for mapping responses.

    The key incorporates the model, catalogue hash, prompt template version,
    diagnostics flag and a hash of the feature definitions.
    """

    template_name = "mapping_prompt_diagnostics" if diagnostics else "mapping_prompt"
    try:
        template_text = load_prompt_text(template_name)
    except FileNotFoundError:
        template_text = ""
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
    parts = [
        model_name,
        set_name,
        catalogue_hash,
        template_hash,
        str(int(diagnostics)),
        _features_hash(features),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def cache_path(service: str, plateau: int, set_name: str, key: str) -> Path:
    """Return canonical cache path for ``service`` and ``set_name``.

    Cache files are grouped by context, service identifier and plateau level.
    """

    try:
        settings = RuntimeEnv.instance().settings
        cache_root = settings.cache_dir
        context = settings.context_id
    except Exception:  # pragma: no cover - settings unavailable
        cache_root = Path(".cache")
        context = "unknown"

    path = (
        cache_root
        / context
        / service
        / str(plateau)
        / "mappings"
        / set_name
        / f"{key}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _discover_cache_file(
    service: str, plateau: int, set_name: str, key: str
) -> tuple[Path, Path]:
    """Return existing cache path and canonical destination.

    Searches the service cache directory for ``set_name`` entries that may have
    been stored under legacy locations (e.g. without plateau information or
    within an ``unknown`` folder). The first match is returned alongside the
    canonical path.
    """

    canonical = cache_path(service, plateau, set_name, key)
    if canonical.exists():
        return canonical, canonical

    service_root = canonical.parents[3]
    for candidate in service_root.glob(f"**/mappings/**/{set_name}/{key}.json"):
        return candidate, canonical

    return canonical, canonical


def cache_write_json_atomic(path: Path, content: Any) -> None:
    """Atomically write ``content`` as pretty JSON to ``path``."""

    _cache_manager.write_json_atomic(path, content)


# Backwards compatibility for renamed helpers
_build_cache_key = build_cache_key
_cache_path = cache_path


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

    env = RuntimeEnv.instance()
    catalogues = (
        catalogue_items
        or load_mapping_items(env.settings.mapping_sets, error_handler=_error_handler)[
            0
        ]
    )
    valid_ids: dict[str, set[str]] = {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    dropped: dict[str, list[str]] = {}
    missing: dict[str, int] = {}
    missing_features: dict[str, list[str]] = {}
    for feature in features:
        update_data: dict[str, list[Contribution]]
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            # Record and warn when the response omits a feature entirely.
            logfire.warning("missing mapping", feature_id=feature.feature_id)
            for key in mapping_types.keys():
                missing_features.setdefault(key, []).append(feature.feature_id)
            update_data = {key: [] for key in mapping_types.keys()}
            merged = feature.model_copy(
                update={"mappings": feature.mappings | update_data}
            )
            results.append(merged)
            continue
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
    if missing_features:
        for key, ids in missing_features.items():
            logfire.warning(
                "Missing mapping features",
                set_name=key,
                count=len(ids),
                examples=ids[:UNKNOWN_ID_LOG_LIMIT],
            )
        if strict:
            raise MappingError("Mappings missing for one or more features")
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
    service_name: str,
    service_description: str,
    plateau: int,
    service: str | None = None,
    strict: bool = False,
    diagnostics: bool | None = None,
    cache_mode: Literal["off", "read", "refresh", "write"] = "off",
    catalogue_hash: str = "",
) -> list[PlateauFeature]:
    """Return ``features`` with ``set_name`` mappings populated.

    Args:
        session: Conversation with the language model.
        set_name: Mapping dataset to populate.
        items: Catalogue items available for mapping.
        features: Features requiring enrichment.
        service_name: Human readable name of the service.
        service_description: Description of the service at ``plateau``.
        plateau: Numeric plateau level being mapped.
        service: Optional service identifier used for caching.
        strict: Raise :class:`MappingError` instead of returning partial results.
        diagnostics: Enable diagnostics mode to request rationales.
        cache_mode: Local cache behaviour.
        catalogue_hash: SHA256 digest representing the loaded mapping catalogues.

    The agent is queried up to twice to obtain a valid :class:`MappingResponse`.
    The second attempt appends a hint directing the model to stick to the
    defined fields. If both attempts fail, the raw response is written to
    ``quarantine/mapping/<service>/<set>.txt`` and an empty mapping list is
    returned. When ``strict`` is ``True`` a :class:`MappingError`` is raised
    instead of returning partial results. ``cache_mode`` controls local caching
    behaviour. ``catalogue_hash`` should be the SHA256 digest returned by
    :func:`loader.load_mapping_items` so cache keys vary when catalogue data
    changes:

    - ``"off"``: bypass the cache entirely.
    - ``"read"``: use cached content when available, otherwise fetch and write.
    - ``"refresh"``: ignore any existing cache and always overwrite.
    - ``"write"``: avoid reading and only write responses when the file is
      absent.
    """

    cfg = MappingTypeConfig(dataset=set_name, label=set_name)
    use_diag = (
        diagnostics
        if diagnostics is not None
        else getattr(session, "diagnostics", False)
    )
    model_obj = getattr(session, "client", None)
    model_name = getattr(getattr(model_obj, "model", None), "model_name", "")
    key = build_cache_key(model_name, set_name, catalogue_hash, features, use_diag)
    model_type = cast(
        type[StrictModel],
        getattr(
            model_obj,
            "output_type",
            MappingDiagnosticsResponse if use_diag else MappingResponse,
        ),
    )
    cache_file: Path | None = None
    payload: StrictModel | None = None
    write_after_call = False
    cache_hit = False
    if cache_mode != "off":
        svc = service or "unknown"
        candidate, cache_file = _discover_cache_file(svc, plateau, set_name, key)
        exists_before = candidate.exists()
        if cache_mode == "read" and exists_before:
            try:
                with candidate.open("rb") as fh:
                    data = from_json(fh.read())
                payload = model_type.model_validate(data)
                cache_hit = True
                if candidate != cache_file:
                    cache_write_json_atomic(cache_file, payload.model_dump())
                    candidate.unlink()
            except (ValidationError, ValueError) as exc:
                _error_handler.handle(f"Invalid cache file: {candidate}", exc)
                raise RuntimeError(f"Invalid cache file: {candidate}") from exc
        if cache_mode == "refresh":
            write_after_call = True
        elif cache_mode == "write" and not exists_before:
            write_after_call = True
        elif cache_mode == "read" and not exists_before:
            write_after_call = True

    cache_state = (
        "refresh" if cache_mode == "refresh" else ("hit" if cache_hit else "miss")
    )
    logfire.info(
        "mapping_set",
        set_name=set_name,
        cache=cache_state,
        cache_key=key,
        features=len(features),
    )

    start = time.monotonic()
    retries = 0
    tokens = 0
    if payload is None:
        # Cache miss or refresh triggers a network request.
        prompt = render_set_prompt(
            set_name,
            list(items),
            features,
            service_name=service_name,
            service_description=service_description,
            plateau=plateau,
            diagnostics=use_diag,
        )
        should_log_prompt = use_diag and getattr(session, "log_prompts", False)
        if should_log_prompt:
            features_json = _json_bytes(
                [
                    {
                        "id": f.feature_id,
                        "name": f.name,
                        "description": f.description,
                    }
                    for f in sorted(features, key=lambda fe: fe.feature_id)
                ],
                indent=2,
            ).decode("utf-8")
            logfire.debug(
                "mapping_prompt",
                set_name=set_name,
                features=features_json,
            )
            session_log_prompts = getattr(session, "log_prompts", False)
            session.log_prompts = False
        try:
            payload = await session.ask_async(prompt)
            tokens += getattr(session, "last_tokens", 0)
        except Exception as exc:  # noqa: BLE001
            svc = service or "unknown"
            _writer.write(set_name, svc, "json_parse_error", str(exc))
            _error_handler.handle("Invalid mapping response", exc)
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
            )
            if should_log_prompt:
                session.log_prompts = session_log_prompts
            return [
                feat.model_copy(update={"mappings": feat.mappings | {set_name: []}})
                for feat in features
            ]
        if cache_file and write_after_call:
            # Persist successful responses for future runs.
            data = payload.model_dump() if hasattr(payload, "model_dump") else payload
            cache_write_json_atomic(cache_file, data)
        if should_log_prompt:
            session.log_prompts = session_log_prompts

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
    )
    return merged


def group_features_by_mapping(
    features: Sequence[PlateauFeature],
    mapping_type: str,
    catalogue: Sequence[MappingItem],
) -> list[MappingFeatureGroup]:
    """Return mapping items keyed to features referencing them.

    Args:
        features: Plateau features potentially containing mappings.
        mapping_type: Mapping category to group by, such as ``"applications"``.
        catalogue: Reference data providing mapping item names.

    Returns:
        List of :class:`MappingFeatureGroup` entries sorted by mapping ID.
        Features without mappings for ``mapping_type`` are ignored.
    """

    groups: dict[str, list[FeatureMappingRef]] = {}
    for feat in features:
        items = feat.mappings.get(mapping_type, [])
        if not items:
            continue  # Skip features lacking mappings for this type.
        for contrib in items:
            groups.setdefault(contrib.item, []).append(
                FeatureMappingRef(
                    feature_id=feat.feature_id,
                    description=feat.description,
                )
            )

    name_lookup = {item.id: item.name for item in catalogue}
    result = [
        MappingFeatureGroup(
            id=item_id,
            name=name_lookup.get(item_id, item_id),
            mappings=sorted(refs, key=lambda r: r.feature_id),
        )
        for item_id, refs in sorted(groups.items())
    ]
    return result


class MappingError(RuntimeError):
    """Raised when a mapping response is missing required data."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


__all__ = [
    "map_set",
    "group_features_by_mapping",
    "MappingError",
    "build_cache_key",
    "cache_path",
]
