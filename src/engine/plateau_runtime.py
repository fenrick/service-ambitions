"""Runtime container for plateau generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import logfire
from pydantic import ValidationError
from pydantic_core import from_json
from tqdm import tqdm  # type: ignore[import-untyped]

from core.conversation import ConversationSession
from core.dry_run import DryRunInvocation
from core.mapping import (
    MapSetParams,
    cache_write_json_atomic,
    group_features_by_mapping,
    map_set,
)
from io_utils.loader import load_mapping_items, load_prompt_text
from models import (
    FeatureItem,
    MappingFeatureGroup,
    MappingItem,
    MappingSet,
    PlateauFeature,
    PlateauFeaturesResponse,
)
from runtime.environment import RuntimeEnv
from utils import ShortCodeRegistry
from core.conversation import _prompt_cache_key, _prompt_cache_path


@dataclass
class PlateauRuntime:
    """Hold generation artefacts for a single plateau."""

    plateau: int
    plateau_name: str
    description: str
    features: list[PlateauFeature] = field(default_factory=list)
    mappings: dict[str, list[MappingFeatureGroup]] = field(default_factory=dict)
    success: bool = False

    def _feature_cache_hashed_path(self, service: str, session: ConversationSession, *, service_name: str) -> Path:
        """Return hashed prompt-cache path for features.

        The key incorporates full prompt text and prior service context.
        """
        prompt = self._build_plateau_prompt(
            service_name=service_name, description=self.description, roles=[]
        )
        stage = f"features_{self.plateau}"
        model_name = getattr(getattr(session.client, "model", None), "model_name", "")
        key = _prompt_cache_key(prompt, model_name, stage, session.history_context_text())
        return _prompt_cache_path(service, stage, key)

    def _to_feature(
        self, item: FeatureItem, role: str, code_registry: ShortCodeRegistry
    ) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``."""
        canonical = f"{item.name}|{role}|{self.plateau_name}"
        feature_id = code_registry.generate(canonical)
        return PlateauFeature(
            feature_id=feature_id,
            name=item.name,
            description=item.description,
            score=item.score,
            customer_type=role,
        )

    def _build_plateau_prompt(
        self,
        *,
        service_name: str,
        description: str,
        roles: Sequence[str],
    ) -> str:
        """Return a prompt requesting features for this plateau."""
        template = load_prompt_text("plateau_prompt")
        roles_str = ", ".join(f'"{r}"' for r in roles)
        return template.format(
            service_name=service_name,
            service_description=description,
            plateau=str(self.plateau),
            roles=str(roles_str),
        )

    def _collect_features(
        self,
        payload: PlateauFeaturesResponse,
        *,
        roles: Sequence[str],
        code_registry: ShortCodeRegistry,
    ) -> list[PlateauFeature]:
        """Return PlateauFeature records extracted from ``payload``."""
        features: list[PlateauFeature] = []
        for role in roles:
            raw_features = payload.features.get(role, [])
            for item in raw_features:
                features.append(self._to_feature(item, role, code_registry))
        return features

    def _load_cached_payload(
        self,
        service_id: str,
        session: ConversationSession,
        service_name: str,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> tuple[PlateauFeaturesResponse | None, Path | None]:
        """Return cached feature payload when available."""
        payload: PlateauFeaturesResponse | None = None
        cache_file: Path | None = None
        if use_local_cache and cache_mode != "off":
            cache_file = self._feature_cache_hashed_path(service_id, session, service_name=service_name)
            candidate = cache_file
            if cache_mode == "read" and candidate.exists():
                try:
                    with candidate.open("rb") as fh:
                        data = from_json(fh.read())
                    payload = PlateauFeaturesResponse.model_validate(data)
                except (ValidationError, ValueError) as exc:
                    raise RuntimeError(f"Invalid feature cache: {candidate}") from exc
        return payload, cache_file

    async def _dispatch_features(
        self,
        session: ConversationSession,
        *,
        service_id: str,
        service_name: str,
        roles: Sequence[str],
        cache_file: Path | None,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> PlateauFeaturesResponse:
        """Return feature payload from a single prompt and optionally cache it."""
        prompt = self._build_plateau_prompt(
            service_name=service_name,
            description=self.description,
            roles=roles,
        )
        logfire.info("Requesting features", plateau=self.plateau, service=service_id)
        raw = await session.ask_async(prompt)
        logfire.info("Validating features", pleateau=self.plateau, service=service_id)
        payload = PlateauFeaturesResponse.model_validate(raw)
        # Respect cache mode semantics for writes:
        # - off: never write
        # - refresh: always write (overwrite)
        # - read/write: write only when the file is absent
        if use_local_cache and cache_mode != "off":
            target = cache_file or self._feature_cache_hashed_path(service_id, session, service_name=service_name)
            exists_before = target.exists()
            should_write = cache_mode == "refresh" or (
                cache_mode in {"read", "write"} and not exists_before
            )
            if should_write:
                cache_write_json_atomic(target, payload.model_dump())
        return payload

    async def generate_features(
        self,
        session: ConversationSession,
        *,
        service_id: str,
        service_name: str,
        roles: Sequence[str],
        code_registry: ShortCodeRegistry,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> None:
        """Populate ``self.features`` using ``session``."""
        payload, cache_file = self._load_cached_payload(
            service_id, session, service_name, use_local_cache, cache_mode
        )
        if payload is None:
            # In dry-run mode, halt before making the agent call.
            if getattr(RuntimeEnv.instance().settings, "dry_run", False):
                stage_name = f"features_{self.plateau}"
                model_name = getattr(
                    getattr(getattr(session, "client", None), "model", None),
                    "model_name",
                    "",
                )
                raise DryRunInvocation(
                    stage=stage_name,
                    model=model_name,
                    cache_file=cache_file,
                    service_id=service_id,
                )
            payload = await self._dispatch_features(
                session,
                service_id=service_id,
                service_name=service_name,
                roles=roles,
                cache_file=cache_file,
                use_local_cache=use_local_cache,
                cache_mode=cache_mode,
            )

        self.features = self._collect_features(
            payload, roles=roles, code_registry=code_registry
        )

    async def _run_mapping_set(
        self,
        session: ConversationSession,
        cfg: MappingSet,
        *,
        items: Mapping[str, Sequence[MappingItem]],
        service_name: str,
        service_id: str,
        service_description: str,
        strict: bool,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
        catalogue_hash: str,
    ) -> list[MappingFeatureGroup]:
        """Return grouped mapping results for a single mapping configuration.

        Args:
            session: Base conversation session for the mapping request.
            cfg: Mapping set configuration to evaluate.
            items: Catalogue items keyed by mapping field.
            service_name: Human readable service name used in prompts.
            service_id: Identifier for caching operations.
            service_description: Description of the service for context.
            strict: Raise :class:`MappingError` on invalid responses when ``True``.
            use_local_cache: Enable local caching of mapping responses.
            cache_mode: Cache behaviour controlling reads and writes.
            catalogue_hash: Digest representing loaded mapping catalogues.

        Returns:
            Grouped mapping results keyed by mapping item identifier.
        """
        set_session = session.derive()
        # Include plateau in stage name so any ancillary tooling can scope by level.
        set_session.stage = f"mapping_{self.plateau}_{cfg.field}"
        params = MapSetParams(
            service_name=service_name,
            service_description=service_description,
            plateau=self.plateau,
            service=service_id,
            strict=strict,
            cache_mode=(cache_mode if use_local_cache else "off"),
            catalogue_hash=catalogue_hash,
        )
        result = await map_set(
            set_session,
            cfg.field,
            items[cfg.field],
            list(self.features),
            params,
        )
        return group_features_by_mapping(result, cfg.field, items[cfg.field])

    async def generate_mappings(
        self,
        session: ConversationSession,
        *,
        service_name: str,
        service_id: str,
        service_description: str,
        strict: bool,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
        progress: tqdm[Any] | None = None,
    ) -> None:
        """Populate ``self.mappings`` for ``self.features``."""
        settings = RuntimeEnv.instance().settings
        items, catalogue_hash = load_mapping_items(settings.mapping_sets)

        groups: dict[str, list[MappingFeatureGroup]] = {}
        for cfg in settings.mapping_sets:
            groups[cfg.field] = await self._run_mapping_set(
                session,
                cfg,
                items=items,
                service_name=service_name,
                service_id=service_id,
                service_description=service_description,
                strict=strict,
                use_local_cache=use_local_cache,
                cache_mode=cache_mode,
                catalogue_hash=catalogue_hash,
            )
            # Tick once per mapping LLM call.
            if progress:
                progress.update(1)

        self.mappings = groups
        self.success = True

    def set_results(
        self,
        *,
        features: list[PlateauFeature],
        mappings: dict[str, list[MappingFeatureGroup]],
    ) -> None:
        """Store ``features`` and ``mappings`` for this plateau."""
        with logfire.span(
            "plateau_runtime.set_results",
            attributes={"plateau": self.plateau_name},
        ):
            self.features = list(features)
            self.mappings = mappings
            self.success = True
            logfire.debug(
                "Stored plateau results",
                plateau=self.plateau_name,
                feature_count=len(self.features),
                mapping_sets=len(self.mappings),
            )

    def status(self) -> bool:
        """Return ``True`` when generation succeeded."""
        logfire.debug(
            "Plateau status checked",
            plateau=self.plateau_name,
            success=self.success,
        )
        return self.success
