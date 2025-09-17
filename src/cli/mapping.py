# SPDX-License-Identifier: MIT
"""Helpers for the CLI mapping subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

from core import mapping
from core.canonical import canonicalise_record
from core.conversation import ConversationSession
from io_utils.loader import (
    configure_mapping_data_dir,
    configure_prompt_dir,
    load_evolution_prompt,
    load_mapping_items,
    load_roles,
)
from models import (
    MappingDiagnosticsResponse,
    MappingFeatureGroup,
    MappingItem,
    MappingResponse,
    PlateauFeature,
    PlateauRole,
    Role,
    ServiceEvolution,
)
from models.factory import ModelFactory
from runtime.settings import Settings
from utils import ErrorHandler

Agent: Any
NativeOutput: Any
try:  # pragma: no cover - optional dependency guard for lightweight environments
    from pydantic_ai import Agent as _AgentType
    from pydantic_ai import NativeOutput as _NativeOutputType

    Agent = _AgentType
    NativeOutput = _NativeOutputType
except Exception:  # pragma: no cover - fallback when pydantic-ai is unavailable
    Agent = None
    NativeOutput = None


def load_catalogue(
    mapping_data_dir: Path | None,
    settings: Settings,
    error_handler: ErrorHandler | None = None,
) -> tuple[dict[str, list[MappingItem]], str]:
    """Return mapping catalogue items and hash.

    Args:
        mapping_data_dir: Optional directory override for catalogue files.
        settings: Runtime settings containing ``mapping_data_dir`` and ``mapping_sets``.
        error_handler: Optional error handler for reporting invalid catalogue
            files. When ``None``, the default loader handler is used.

    Returns:
        A tuple of ``(items, catalogue_hash)`` where ``items`` is a mapping of
        field name to ``MappingItem`` list.

    Side Effects:
        Adjusts the global loader state by configuring the mapping data
        directory before reading items from disk.
    """
    configure_mapping_data_dir(mapping_data_dir or settings.mapping_data_dir)
    return load_mapping_items(settings.mapping_sets, error_handler=error_handler)


SessionFactory = Callable[[ServiceEvolution], ConversationSession]


def _build_mapping_session_factory(
    settings: Settings,
    *,
    allow_prompt_logging: bool,
    transcripts_dir: Path | None,
    cache_mode: Literal["off", "read", "refresh", "write"],
) -> SessionFactory:
    """Return a factory that produces mapping ``ConversationSession`` instances."""
    if Agent is None or NativeOutput is None:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "pydantic-ai is required for mapping runs. Install the optional "
            "dependencies or provide a custom session_factory."
        )

    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)
    factory = ModelFactory(
        settings.model,
        settings.openai_api_key,
        stage_models=getattr(settings, "models", None),
        reasoning=settings.reasoning,
        seed=None,
        web_search=settings.web_search,
    )
    map_model = factory.get("mapping")
    map_output = MappingDiagnosticsResponse if settings.diagnostics else MappingResponse
    agent = Agent(
        map_model,
        instructions=system_prompt,
        output_type=NativeOutput(
            [map_output],
            name=(
                "diagnostic_feature_mappings"
                if settings.diagnostics
                else "feature_mappings"
            ),
            description=(
                "Return mapping contributions for each feature. Provide a "
                "'features' list where each item includes feature_id and a "
                "'mappings' object keyed by mapping type, each containing a "
                "list of contributions. When diagnostics are enabled, include "
                "rationales for each contribution."
            ),
        ),
    )

    def _factory(evo: ServiceEvolution) -> ConversationSession:
        session = ConversationSession(
            agent,
            stage="mapping",
            diagnostics=settings.diagnostics,
            log_prompts=allow_prompt_logging,
            transcripts_dir=transcripts_dir,
            use_local_cache=settings.use_local_cache,
            cache_mode=cache_mode,
        )
        session.add_parent_materials(evo.service)
        return session

    return _factory


def _enrich_feature(
    feature: PlateauFeature,
    role_lookup: dict[str, Role],
) -> PlateauFeature:
    """Attach role metadata to ``feature`` without altering mapping shape.

    The mapping contribution objects are left unchanged to preserve the
    canonical JSON shape expected by golden tests and downstream tooling.
    """
    if getattr(feature, "role", None) is None and feature.customer_type in role_lookup:
        feature.role = role_lookup[feature.customer_type]
    return feature


async def remap_features(
    evolutions: Sequence[ServiceEvolution],
    items: dict[str, list[MappingItem]],
    settings: Settings,
    cache_mode: Literal["off", "read", "refresh", "write"],
    catalogue_hash: str,
    *,
    allow_prompt_logging: bool = False,
    transcripts_dir: Path | None = None,
    session_factory: SessionFactory | None = None,
) -> None:
    """Populate feature mappings on the provided evolutions.

    Args:
        evolutions: Mutable collection of service evolutions to update.
        items: Catalogue items keyed by mapping field.
        settings: Runtime settings providing mapping configuration.
        cache_mode: Strategy controlling mapping cache usage.
        catalogue_hash: Hash of the catalogue for cache invalidation.
        allow_prompt_logging: Enable prompt logging on derived sessions.
        transcripts_dir: Optional directory for transcript output when diagnostics
            or logging are enabled.
        session_factory: Optional factory used to construct conversation sessions
            (primarily for tests); defaults to the real model-backed factory.

    Side Effects:
        Evolutions are mutated in place; each plateau gains populated
        ``mappings`` derived from the mapping catalogue and remote service.
    """
    role_lookup = {r.role_id: r for r in load_roles(settings.roles_file)}
    mapping_sets = list(getattr(settings, "mapping_sets", []) or [])
    use_local_cache = getattr(settings, "use_local_cache", True)
    effective_cache_mode = cache_mode if use_local_cache else "off"

    session_factory = session_factory or _build_mapping_session_factory(
        settings,
        allow_prompt_logging=allow_prompt_logging,
        transcripts_dir=transcripts_dir,
        cache_mode=effective_cache_mode,
    )

    for evo in evolutions:
        service_session = session_factory(evo)
        for plateau in evo.plateaus:
            if not plateau.features:
                plateau.mappings = {cfg.field: [] for cfg in mapping_sets}
                plateau.roles = []
                continue

            plateau_session = service_session.derive()
            plateau_session.stage = f"mapping_{plateau.plateau}"
            features = list(plateau.features)
            mapping_groups: dict[str, list[MappingFeatureGroup]] = {}

            for cfg in mapping_sets:
                set_session = plateau_session.derive()
                set_session.stage = f"{plateau_session.stage}_{cfg.field}"
                set_session._pending_features = list(features)
                params = mapping.MapSetParams(
                    service_name=evo.service.name,
                    service_description=plateau.service_description,
                    plateau=plateau.plateau,
                    service=evo.service.service_id,
                    strict=settings.strict_mapping,
                    diagnostics=settings.diagnostics,
                    cache_mode=effective_cache_mode,
                    catalogue_hash=catalogue_hash,
                )
                features = await mapping.map_set(
                    set_session,
                    cfg.field,
                    items[cfg.field],
                    list(features),
                    params,
                )
                mapping_groups[cfg.field] = mapping.group_features_by_mapping(
                    features, cfg.field, items[cfg.field]
                )

            plateau.features = [_enrich_feature(feat, role_lookup) for feat in features]
            plateau.roles = _build_role_groups(plateau.features, role_lookup)
            plateau.mappings = mapping_groups


def _build_role_groups(
    features: Sequence[PlateauFeature],
    role_lookup: dict[str, Role],
) -> list[PlateauRole]:
    """Return plateau roles populated from ``features``."""
    role_groups: dict[str, list[PlateauFeature]] = {}
    for feat in features:
        role_groups.setdefault(feat.customer_type, []).append(feat)
    return [
        PlateauRole(
            role_id=role_id,
            name=role_lookup[role_id].name if role_id in role_lookup else role_id,
            description=(
                role_lookup[role_id].description if role_id in role_lookup else ""
            ),
            features=sorted(group, key=lambda x: x.feature_id),
        )
        for role_id, group in sorted(role_groups.items())
    ]


def iter_serialised_evolutions(
    evolutions: Iterable[ServiceEvolution],
) -> Iterator[str]:
    """Yield canonical JSON representations for ``evolutions``.

    Args:
        evolutions: Iterable of evolutions with mappings applied.

    Yields:
        JSON strings in canonical form ready for JSONL output.
    """
    schemas: dict[str, Any] = {}
    for evo in evolutions:
        record = canonicalise_record(evo.model_dump(mode="json"))
        meta = record.get("meta", {})
        if isinstance(meta, dict):
            meta["schemas"] = schemas
            record["meta"] = meta
        yield json.dumps(record, separators=(",", ":"), sort_keys=True)


def write_output(evolutions: Iterable[ServiceEvolution], output_path: Path) -> None:
    """Write mapped evolutions to ``output_path`` as canonical JSON lines.

    Args:
        evolutions: Iterable of evolutions with mappings applied.
        output_path: Destination file for JSONL output.

    Side Effects:
        Creates or overwrites ``output_path`` with one line per evolution.
    """
    with output_path.open("w", encoding="utf-8") as fh:
        for line in iter_serialised_evolutions(evolutions):
            fh.write(line + "\n")
