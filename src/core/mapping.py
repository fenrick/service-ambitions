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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import logfire
from pydantic import ValidationError
from pydantic_core import from_json, to_json

from constants import DEFAULT_CACHE_DIR
from .conversation import _prompt_cache_key
from io_utils.loader import load_mapping_items, load_mapping_meta
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
from utils import ErrorHandler, LoggingErrorHandler

from .mapping_prompt import render_set_prompt

if TYPE_CHECKING:
    from .conversation import ConversationSession
from .dry_run import DryRunInvocation
from .cache_utils import (
    cache_write_json_atomic,
    configure_cache_manager as _configure_cache_manager,
)

_writer = QuarantineWriter()

_error_handler: ErrorHandler = LoggingErrorHandler()
configure_cache_manager = _configure_cache_manager


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
        return cast(bytes, cast(Any, to_json)(value, **kwargs))
    except TypeError:  # pragma: no cover - legacy pydantic-core
        if "sort_keys" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("sort_keys")
        return cast(bytes, cast(Any, to_json)(value, **kwargs))


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


def build_cache_key(*args: Any, **kwargs: Any) -> str:  # pragma: no cover - legacy shim
    """Deprecated: mapping cache is now keyed by full prompt+history.

    Retained for callers that still import this symbol; returns a short hash of
    arguments only to keep deterministic but non-conflicting values. Not used
    by the engine.
    """
    payload = "|".join([str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(service: str, plateau: int, set_name: str, key: str) -> Path:
    """Return canonical cache path for ``service`` and ``set_name``.

    Cache files are grouped by context, service identifier and plateau level.
    """
    try:
        settings = RuntimeEnv.instance().settings
        cache_root = settings.cache_dir
        context = settings.context_id
    except RuntimeError:  # pragma: no cover - settings unavailable
        cache_root = DEFAULT_CACHE_DIR
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
def _catalogues_for_merge(
    mapping_types: Mapping[str, MappingTypeConfig],
    catalogue_items: Mapping[str, list[MappingItem]] | None,
) -> Mapping[str, list[MappingItem]]:
    """Return catalogue mapping for ``mapping_types``."""
    env = RuntimeEnv.instance()
    return (
        catalogue_items
        or load_mapping_items(env.settings.mapping_sets, error_handler=_error_handler)[
            0
        ]
    )


def _build_valid_ids(
    mapping_types: Mapping[str, MappingTypeConfig],
    catalogues: Mapping[str, list[MappingItem]],
) -> dict[str, set[str]]:
    """Return valid identifier lookup for each mapping type."""
    return {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }


def _merge_feature(
    feature: PlateauFeature,
    keys: Sequence[str],
    mapped_lookup: Mapping[str, dict[str, list[Contribution]]],
    valid_ids: Mapping[str, set[str]],
    dropped: dict[str, list[str]],
    missing: dict[str, int],
    missing_features: dict[str, list[str]],
) -> PlateauFeature:
    """Merge mappings for a single ``feature``."""
    mapped = mapped_lookup.get(feature.feature_id)
    update_data: dict[str, list[Contribution]]
    if mapped is None:
        logfire.warning("missing mapping", feature_id=feature.feature_id)
        for key in keys:
            missing_features.setdefault(key, []).append(feature.feature_id)
        update_data = {key: [] for key in keys}
        return feature.model_copy(update={"mappings": feature.mappings | update_data})
    update_data = {}
    for key in keys:
        original = mapped.get(key, [])
        cleaned = [item for item in original if item.item in valid_ids[key]]
        unknown = [item.item for item in original if item.item not in valid_ids[key]]
        if unknown:
            dropped.setdefault(key, []).extend(unknown)
        if not cleaned:
            missing[key] = missing.get(key, 0) + 1
        update_data[key] = cleaned
    return feature.model_copy(update={"mappings": feature.mappings | update_data})


def _merge_all_features(
    features: Sequence[PlateauFeature],
    keys: Sequence[str],
    mapped_lookup: Mapping[str, dict[str, list[Contribution]]],
    valid_ids: Mapping[str, set[str]],
) -> tuple[
    list[PlateauFeature],
    dict[str, list[str]],
    dict[str, int],
    dict[str, list[str]],
]:
    """Return merged features and tracking dictionaries."""
    results: list[PlateauFeature] = []
    dropped: dict[str, list[str]] = {}
    missing: dict[str, int] = {}
    missing_features: dict[str, list[str]] = {}
    for feature in features:
        merged = _merge_feature(
            feature, keys, mapped_lookup, valid_ids, dropped, missing, missing_features
        )
        results.append(merged)
    return results, dropped, missing, missing_features


def _build_facets_model(
    set_name: str, schema: object
) -> tuple["TypeAdapter", bool] | tuple[None, bool]:
    """Return a ``TypeAdapter`` for the dataset facets and a required flag.

    When ``schema`` is falsy or imports are unavailable, returns ``(None, False)``.
    """
    try:
        from typing import Literal, Optional, Any as _Any
        from pydantic import TypeAdapter, create_model
    except Exception:  # pragma: no cover - defensive import
        return (None, False)

    if not schema:
        return (None, False)

    fields: dict[str, tuple[_Any, _Any]] = {}
    any_required = False
    for f in cast(list[dict[str, object]], schema):
        fid = f.get("id")
        ftype = f.get("type")
        required = bool(f.get("required"))
        if not isinstance(fid, str) or not isinstance(ftype, str):
            continue
        any_required = any_required or required
        if ftype == "boolean":
            fpy: _Any = bool
        elif ftype == "integer":
            fpy = int
        elif ftype == "number":
            fpy = float
        elif ftype == "enum":
            opts_raw = cast(list[dict[str, object]] | None, f.get("options"))
            opts = [
                o.get("id")
                for o in (opts_raw or [])
                if isinstance(o, dict) and isinstance(o.get("id"), str)
            ]
            fpy = cast(_Any, Literal[tuple(opts)]) if opts else str
        else:
            fpy = str
        if required:
            fields[fid] = (fpy, ...)
        else:
            fields[fid] = (Optional[fpy], None)

    FacetsModel = create_model(f"Facets_{set_name}", **cast(dict[str, Any], fields))
    return TypeAdapter(FacetsModel), any_required


def _collect_facet_violations(
    payload: MappingResponse,
    set_name: str,
    facets_adapter: "TypeAdapter" | None,
    any_required: bool,
) -> list[dict[str, object]]:
    """Return a list of facet validation violations for ``set_name``."""
    if facets_adapter is None:
        return []
    violations: list[dict[str, object]] = []
    for feat in payload.features:
        contribs = feat.mappings.get(set_name, [])
        for c in contribs:
            facets = getattr(c, "facets", None)
            if facets is None and any_required:
                violations.append(
                    {
                        "feature_id": feat.feature_id,
                        "item": c.item,
                        "reason": "facets object missing",
                    }
                )
                continue
            if facets is None:
                continue
            try:
                facets_adapter.validate_python(facets)
            except Exception as exc:  # pragma: no cover - defensive
                violations.append(
                    {
                        "feature_id": feat.feature_id,
                        "item": c.item,
                        "reason": "invalid facets",
                        "error": str(exc),
                    }
                )
    return violations


def _validate_facets_for_set(
    payload: MappingResponse,
    set_name: str,
    *,
    strict: bool,
    service: str | None,
) -> None:
    """Validate contribution facets using a dynamic Pydantic model.

    Builds a Facets model from the dataset's facet schema and validates each
    contribution's ``facets`` object for ``set_name``. When any facet is marked
    required in the dataset, the ``facets`` object itself is required.
    """
    try:
        settings = RuntimeEnv.instance().settings
        meta = load_mapping_meta(settings.mapping_sets)
        cfg = meta.get(set_name, {}) if isinstance(meta, dict) else {}
        schema = cfg.get("facets") if isinstance(cfg, dict) else None
    except (RuntimeError, OSError, ValueError, TypeError):  # pragma: no cover - defensive
        schema = None
    adapter, any_required = _build_facets_model(set_name, schema)
    violations = _collect_facet_violations(payload, set_name, adapter, any_required)
    if not violations:
        return
    svc = service or "unknown"
    _writer.write(set_name, svc, "facet_validation", violations)
    if strict:
        raise MappingError("Required facet validation failed")


def _log_unknown_ids(
    dropped: Mapping[str, list[str]],
    service: str,
    strict: bool,
) -> int:
    """Log and optionally raise for unknown identifiers."""
    if not dropped:
        return 0
    unknown_total = 0
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
    return unknown_total


def _log_missing_features(
    missing_features: Mapping[str, list[str]],
    strict: bool,
) -> None:
    """Log and optionally raise for features missing mappings."""
    if not missing_features:
        return
    for key, ids in missing_features.items():
        logfire.warning(
            "Missing mapping features",
            set_name=key,
            count=len(ids),
            examples=ids[:UNKNOWN_ID_LOG_LIMIT],
        )
    if strict:
        raise MappingError("Mappings missing for one or more features")


def _log_missing_counts(missing: Mapping[str, int], strict: bool) -> None:
    """Log and optionally raise for features lacking valid mappings."""
    if not missing:
        return
    for key, count in missing.items():
        logfire.warning(f"{key}.missing={count}")
    if strict:
        raise MappingError("Mappings missing for one or more features")


def _read_mapping_cache(
    candidate: Path, cache_file: Path, model_type: type[StrictModel]
) -> StrictModel:
    """Return cached mapping payload from ``candidate``."""
    try:
        with candidate.open("rb") as fh:
            data = from_json(fh.read())
        payload = model_type.model_validate(data)
        if candidate != cache_file:
            cache_write_json_atomic(cache_file, payload.model_dump())
            candidate.unlink()
        return payload
    except (ValidationError, ValueError) as exc:
        _error_handler.handle(f"Invalid cache file: {candidate}", exc)
        raise RuntimeError(f"Invalid cache file: {candidate}") from exc


@dataclass
class CacheInfo:
    """Details about cache interaction for a mapping request."""

    payload: StrictModel | None
    cache_file: Path | None
    write_after_call: bool
    cache_hit: bool


def _load_cache(
    cache_mode: Literal["off", "read", "refresh", "write"],
    service: str | None,
    plateau: int,
    set_name: str,
    key: str,
    model_type: type[StrictModel],
) -> CacheInfo:
    """Return cache information for the mapping request."""
    if cache_mode == "off":
        return CacheInfo(None, None, False, False)
    svc = service or "unknown"
    candidate, cache_file = _discover_cache_file(svc, plateau, set_name, key)
    exists_before = candidate.exists()
    payload: StrictModel | None = None
    cache_hit = False
    if cache_mode == "read" and exists_before:
        payload = _read_mapping_cache(candidate, cache_file, model_type)
        cache_hit = True
    write_after_call = cache_mode == "refresh" or (
        cache_mode in {"write", "read"} and not exists_before
    )
    return CacheInfo(payload, cache_file, write_after_call, cache_hit)


def _cache_state(cache_mode: str, cache_hit: bool) -> str:
    """Return human readable cache status."""
    if cache_mode == "refresh":
        return "refresh"
    return "hit" if cache_hit else "miss"


def _prepare_prompt(
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    service_name: str,
    service_description: str,
    plateau: int,
    diagnostics: bool,
    session: "ConversationSession",
) -> tuple[str, bool, bool]:
    """Return rendered prompt and prompt logging state."""
    # Resolve optional facet schema for this mapping set (by field name)
    try:
        settings = RuntimeEnv.instance().settings
        meta = load_mapping_meta(settings.mapping_sets)
        facets_meta = (
            meta.get(set_name, {}).get("facets") if isinstance(meta, dict) else None
        )
        facets_seq = list(facets_meta) if isinstance(facets_meta, list) else None
    except (RuntimeError, OSError, ValueError, TypeError):  # pragma: no cover - defensive
        facets_seq = None

    prompt = render_set_prompt(
        set_name,
        list(items),
        features,
        service_name=service_name,
        service_description=service_description,
        plateau=plateau,
        diagnostics=diagnostics,
        facets_meta=facets_seq,
    )
    should_log_prompt = diagnostics and getattr(session, "log_prompts", False)
    original_flag = getattr(session, "log_prompts", False)
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
        session.log_prompts = False
    return prompt, should_log_prompt, original_flag


async def _request_mapping_payload(
    session: "ConversationSession",
    prompt: str,
    cache_file: Path | None,
    write_after_call: bool,
    set_name: str,
    features: Sequence[PlateauFeature],
    strict: bool,
    service: str | None,
    start: float,
    retries: int,
    model_type: type[StrictModel],
) -> tuple[StrictModel | None, int, list[PlateauFeature] | None]:
    """Return mapping payload or fallback features on error."""
    try:
        response = await session.ask_async(prompt)
        tokens = getattr(session, "last_tokens", 0)
        payload = (
            cast(StrictModel, response)
            if hasattr(response, "model_dump")
            else model_type.model_validate(response)
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - narrow types are impractical here due to heterogeneous user data; logged and re-raised upstream (see issue #501)
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
            tokens=0,
        )
        empty = [
            feat.model_copy(update={"mappings": feat.mappings | {set_name: []}})
            for feat in features
        ]
        return None, 0, empty
    if cache_file and write_after_call:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        cache_write_json_atomic(cache_file, data)
    return payload, tokens, None


async def _fetch_and_cache(
    params: FetchParams,
    prompt: str,
) -> tuple[StrictModel | None, int, list[PlateauFeature] | None]:
    """Fetch payload from model and optionally write to cache."""
    payload, tokens, fallback = await _request_mapping_payload(
        params.session,
        prompt,
        params.cache_file,
        params.write_after_call,
        params.set_name,
        params.features,
        params.strict,
        params.service,
        params.start,
        params.retries,
        params.model_type,
    )
    return payload, tokens, fallback


def _normalise_payload(payload: StrictModel, use_diag: bool) -> MappingResponse:
    """Return mapping response normalised for diagnostics."""
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
        return MappingResponse(features=plain)
    return cast(MappingResponse, payload)


def _record_metrics(
    set_name: str,
    features: Sequence[PlateauFeature],
    merged: Sequence[PlateauFeature],
    unknown_count: int,
    start: float,
    retries: int,
    tokens: int,
) -> None:
    """Record telemetry for a mapping operation."""
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


# Backwards compatibility for renamed helpers
_build_cache_key = build_cache_key
_cache_path = cache_path


@dataclass
class MapSetParams:
    """Options controlling ``map_set`` behaviour."""

    service_name: str
    service_description: str
    plateau: int
    service: str | None = None
    strict: bool = False
    diagnostics: bool | None = None
    cache_mode: Literal["off", "read", "refresh", "write"] = "off"
    catalogue_hash: str = ""


@dataclass
class ModelContext:
    """Derived model details for mapping requests (no cache key)."""

    cfg: MappingTypeConfig
    use_diag: bool
    model_name: str
    model_type: type[StrictModel]


@dataclass
class FetchParams:
    """Parameters required to fetch mapping payloads."""

    session: "ConversationSession"
    set_name: str
    items: Sequence[MappingItem]
    features: Sequence[PlateauFeature]
    service_name: str
    service_description: str
    plateau: int
    use_diag: bool
    model_type: type[StrictModel]
    cache_file: Path | None
    write_after_call: bool
    strict: bool
    service: str | None
    start: float
    retries: int


def _build_context(
    session: "ConversationSession",
    set_name: str,
    features: Sequence[PlateauFeature],
    params: MapSetParams,
) -> ModelContext:
    """Derive model and cache details for mapping."""
    cfg = MappingTypeConfig(dataset=set_name, label=set_name)
    use_diag = (
        params.diagnostics
        if params.diagnostics is not None
        else getattr(session, "diagnostics", False)
    )
    model_obj = getattr(session, "client", None)
    model_name = getattr(getattr(model_obj, "model", None), "model_name", "")
    # Resolve the expected Pydantic model type for cache validation.
    # Some agents expose a `NativeOutput` wrapper rather than a model class.
    # In that case, fall back to the known response models based on diagnostics.
    _out = getattr(model_obj, "output_type", None)
    if hasattr(_out, "model_validate"):
        model_type = cast(type[StrictModel], _out)
    else:
        model_type = cast(
            type[StrictModel],
            MappingDiagnosticsResponse if use_diag else MappingResponse,
        )
    return ModelContext(cfg, use_diag, model_name, model_type)


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    catalogue_items: Mapping[str, list[MappingItem]] | None = None,
    service: str = "unknown",
    strict: bool = False,
) -> tuple[list[PlateauFeature], int]:
    """Return ``features`` merged with mapping ``payload`` and unknown count."""
    catalogues = _catalogues_for_merge(mapping_types, catalogue_items)
    valid_ids = _build_valid_ids(mapping_types, catalogues)
    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    keys = list(mapping_types.keys())
    results, dropped, missing, missing_features = _merge_all_features(
        features, keys, mapped_lookup, valid_ids
    )
    unknown_total = _log_unknown_ids(dropped, service or "unknown", strict)
    _log_missing_features(missing_features, strict)
    _log_missing_counts(missing, strict)
    return results, unknown_total


async def map_set(
    session: "ConversationSession",
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
    params: MapSetParams,
) -> list[PlateauFeature]:
    """Return ``features`` with ``set_name`` mappings populated.

    Args:
        session: Conversation with the language model.
        set_name: Mapping dataset to populate.
        items: Catalogue items available for mapping.
        features: Features requiring enrichment.
        params: Options controlling mapping behaviour.

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
    with logfire.span(
        "mapping.map_set", attributes={"set_name": set_name, "features": len(features)}
    ) as span:
        context = _build_context(session, set_name, features, params)
        # Render prompt to compute the new key based on full prompt+history
        prompt, should_log_prompt, original_flag = _prepare_prompt(
            set_name,
            items,
            features,
            params.service_name,
            params.service_description,
            params.plateau,
            context.use_diag,
            session,
        )
        stage = f"mapping_{set_name}"
        history = session.history_context_text() if hasattr(session, "history_context_text") else ""
        key = _prompt_cache_key(prompt, context.model_name, stage, history)
        cache = _load_cache(
            params.cache_mode,
            params.service,
            params.plateau,
            set_name,
            key,
            context.model_type,
        )
        cache_state = _cache_state(params.cache_mode, cache.cache_hit)
        span.set_attribute("cache", cache_state)
        span.set_attribute("cache_key", key)
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
        payload = cache.payload
    if payload is None:
        # Dry-run: halt before any agent invocation when cache is missing.
        try:
            _settings = RuntimeEnv.instance().settings
        except RuntimeError:  # pragma: no cover - defensive
            _settings = None
        if getattr(_settings, "dry_run", False):
            svc = params.service or "unknown"
            stage_name = f"mapping_{set_name}"
            model_name = getattr(
                getattr(getattr(session, "client", None), "model", None),
                "model_name",
                "",
            )
            raise DryRunInvocation(
                stage=stage_name,
                model=model_name,
                cache_file=cache.cache_file,
                service_id=svc,
            )
        fetch_params = FetchParams(
            session=session,
            set_name=set_name,
            items=items,
            features=features,
            service_name=params.service_name,
            service_description=params.service_description,
            plateau=params.plateau,
            use_diag=context.use_diag,
            model_type=context.model_type,
            cache_file=cache.cache_file,
            write_after_call=cache.write_after_call,
            strict=params.strict,
            service=params.service,
            start=start,
            retries=retries,
        )
        payload, tokens, fallback = await _fetch_and_cache(fetch_params, prompt)
        if fallback is not None:
            return fallback
    # Restore prompt logging flag when diagnostics temporarily logged the prompt
    if should_log_prompt:
        session.log_prompts = original_flag

    payload_norm = _normalise_payload(cast(StrictModel, payload), context.use_diag)
    _validate_facets_for_set(
        payload_norm,
        set_name,
        strict=params.strict,
        service=params.service,
    )
    merged, unknown_count = _merge_mapping_results(
        features,
        payload_norm,
        {set_name: context.cfg},
        catalogue_items={set_name: list(items)},
        service=params.service or "unknown",
        strict=params.strict,
    )
    _record_metrics(set_name, features, merged, unknown_count, start, retries, tokens)
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
    "MapSetParams",
    "map_set",
    "group_features_by_mapping",
    "MappingError",
    "build_cache_key",
    "cache_path",
]
