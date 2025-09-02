"""Runtime container for plateau generation outputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import logfire
from pydantic import ValidationError
from pydantic_core import from_json

from core.conversation import ConversationSession
from core.mapping import cache_write_json_atomic, group_features_by_mapping, map_set
from io_utils.loader import load_mapping_items, load_prompt_text
from models import (
    FeatureItem,
    MappingFeatureGroup,
    MappingItem,
    MappingSet,
    PlateauFeature,
    PlateauFeaturesResponse,
    RoleFeaturesResponse,
)
from runtime.environment import RuntimeEnv
from shortcode import ShortCodeRegistry
from utils.cache_paths import feature_cache


@dataclass
class PlateauRuntime:
    """Hold generation artefacts for a single plateau."""

    plateau: int
    plateau_name: str
    description: str
    features: list[PlateauFeature] = field(default_factory=list)
    mappings: dict[str, list[MappingFeatureGroup]] = field(default_factory=dict)
    success: bool = False

    def _feature_cache_path(self, service: str) -> Path:
        """Return canonical cache path for features."""

        return feature_cache(service, self.plateau)

    def _discover_feature_cache(self, service: str) -> tuple[Path, Path]:
        """Return existing feature cache and canonical destination."""

        canonical = self._feature_cache_path(service)
        if canonical.exists():
            return canonical, canonical

        service_root = canonical.parents[1]
        for candidate in service_root.glob("**/features.json"):
            return candidate, canonical

        return canonical, canonical

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
        required_count: int,
    ) -> str:
        """Return a prompt requesting features for this plateau."""

        template = load_prompt_text("plateau_prompt")
        roles_str = ", ".join(f'"{r}"' for r in roles)
        return template.format(
            required_count=required_count,
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

    def _validate_roles(
        self,
        role_data: dict[str, Any],
        *,
        roles: Sequence[str],
        required_count: int,
    ) -> tuple[dict[str, list[FeatureItem]], list[str], dict[str, int]]:
        """Return valid roles, invalid names and missing counts.

        Args:
            role_data: Raw features keyed by role from the model.
            roles: Allowed role names.
            required_count: Required features per role.

        Returns:
            A tuple of (valid features per role, invalid role names,
            missing feature counts).
        """

        valid: dict[str, list[FeatureItem]] = {}
        invalid: list[str] = []
        missing: dict[str, int] = {}
        for role in roles:
            items = role_data.get(role, [])
            try:
                role_block = RoleFeaturesResponse(features=items)
            except Exception:  # noqa: BLE001
                invalid.append(role)
                valid[role] = []
                continue
            valid[role] = list(role_block.features)
            if len(role_block.features) < required_count:
                missing[role] = required_count - len(role_block.features)
        return valid, invalid, missing

    async def _request_role_features_async(
        self,
        level: int,
        role: str,
        description: str,
        count: int,
        session: ConversationSession,
    ) -> list[FeatureItem]:
        """Return ``count`` features for ``role`` when initial parsing fails."""

        prompt = (
            f"Previous output returned invalid features for role '{role}'.\nProvide"
            f" exactly {count} unique features for this role at plateau"
            f" {level}.\n\nService description:\n{description}"
        )
        payload = await session.ask_async(prompt)
        return payload.features

    async def _recover_invalid_roles(
        self,
        invalid: list[str],
        *,
        level: int,
        description: str,
        session: ConversationSession,
        required_count: int,
    ) -> dict[str, list[FeatureItem]]:
        """Return features for roles that failed validation."""

        fixes: dict[str, list[FeatureItem]] = {}
        for role in invalid:
            fixes[role] = await self._request_role_features_async(
                level, role, description, required_count, session
            )
        return fixes

    def _enforce_min_features(
        self,
        valid: dict[str, list[FeatureItem]],
        *,
        roles: Sequence[str],
        required: int,
    ) -> None:
        """Ensure each role has at least ``required`` features."""

        for role in roles:
            items = valid.get(role, [])
            if len(items) < required:
                raise ValueError(
                    f"Expected at least {required} features for '{role}', got"
                    f" {len(items)} after retry"
                )

    async def _request_missing_features_async(
        self,
        level: int,
        role: str,
        description: str,
        missing: int,
        session: ConversationSession,
    ) -> list[FeatureItem]:
        """Return additional features for ``role`` to meet the required count."""

        prompt = (
            "Previous output returned insufficient features for role"
            f" '{role}'.\nProvide exactly {missing} additional unique features for this"
            f" role at plateau {level}.\n\nService description:\n{description}"
        )
        payload = await session.ask_async(prompt)
        return payload.features

    def _load_cached_payload(
        self,
        service_id: str,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> tuple[PlateauFeaturesResponse | None, Path | None]:
        """Return cached feature payload when available."""

        payload: PlateauFeaturesResponse | None = None
        cache_file: Path | None = None
        if use_local_cache and cache_mode != "off":
            candidate, cache_file = self._discover_feature_cache(service_id)
            if cache_mode == "read" and candidate.exists():
                try:
                    with candidate.open("rb") as fh:
                        data = from_json(fh.read())
                    payload = PlateauFeaturesResponse.model_validate(data)
                    if candidate != cache_file:
                        cache_write_json_atomic(cache_file, payload.model_dump())
                        candidate.unlink()
                except (ValidationError, ValueError) as exc:
                    raise RuntimeError(f"Invalid feature cache: {candidate}") from exc
        return payload, cache_file

    async def _dispatch_feature_prompt(
        self,
        session: ConversationSession,
        *,
        service_id: str,
        service_name: str,
        roles: Sequence[str],
        required_count: int,
    ) -> PlateauFeaturesResponse:
        """Return features generated via LLM prompt."""

        prompt = self._build_plateau_prompt(
            service_name=service_name,
            description=self.description,
            roles=roles,
            required_count=required_count,
        )
        logfire.info("Requesting features", plateau=self.plateau, service=service_id)
        return await session.ask_async(prompt)

    async def _recover_feature_shortfalls(
        self,
        valid: dict[str, list[FeatureItem]],
        invalid_roles: list[str],
        missing: dict[str, int],
        *,
        level: int,
        description: str,
        session: ConversationSession,
        required_count: int,
        roles: Sequence[str],
    ) -> dict[str, list[FeatureItem]]:
        """Return roles with recovered features.

        Args:
            valid: Initially valid features keyed by role.
            invalid_roles: Roles that failed validation.
            missing: Number of missing features per role.
            level: Plateau level for follow-up prompts.
            description: Service description for context.
            session: Conversation session used for prompts.
            required_count: Minimum features per role.
            roles: All role names.

        Returns:
            Updated mapping of roles to feature lists.
        """

        fixes = await self._recover_invalid_roles(
            invalid_roles,
            level=level,
            description=description,
            session=session,
            required_count=required_count,
        )
        valid.update(fixes)
        tasks = {
            role: asyncio.create_task(
                self._request_missing_features_async(
                    level, role, description, need, session
                )
            )
            for role, need in missing.items()
        }
        if tasks:
            results = await asyncio.gather(*tasks.values())
            for role, extras in zip(tasks.keys(), results, strict=False):
                valid[role].extend(extras)
        self._enforce_min_features(valid, roles=roles, required=required_count)
        return valid

    async def _dispatch_and_cache_features(
        self,
        session: ConversationSession,
        *,
        service_id: str,
        service_name: str,
        roles: Sequence[str],
        required_count: int,
        cache_file: Path | None,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> PlateauFeaturesResponse:
        """Return validated feature payload and optionally cache it.

        Args:
            session: Conversation session used to query the model.
            service_id: Identifier for cache storage.
            service_name: Human-readable service name for prompts.
            roles: Customer roles requiring features.
            required_count: Number of features to collect per role.
            cache_file: Destination for cached payload.
            use_local_cache: Whether caching is enabled.
            cache_mode: Caching behaviour mode.

        Returns:
            Validated feature payload.
        """

        payload = await self._dispatch_feature_prompt(
            session,
            service_id=service_id,
            service_name=service_name,
            roles=roles,
            required_count=required_count,
        )
        role_data = payload.features
        valid, invalid_roles, missing = self._validate_roles(
            role_data, roles=roles, required_count=required_count
        )
        valid = await self._recover_feature_shortfalls(
            valid,
            invalid_roles,
            missing,
            level=self.plateau,
            description=self.description,
            session=session,
            required_count=required_count,
            roles=roles,
        )
        payload = PlateauFeaturesResponse(
            features={role: list(valid.get(role, [])) for role in roles}
        )
        if use_local_cache and cache_mode != "off":
            cache_write_json_atomic(
                cache_file or self._feature_cache_path(service_id),
                payload.model_dump(),
            )
        return payload

    async def generate_features(
        self,
        session: ConversationSession,
        *,
        service_id: str,
        service_name: str,
        roles: Sequence[str],
        required_count: int,
        code_registry: ShortCodeRegistry,
        use_local_cache: bool,
        cache_mode: Literal["off", "read", "refresh", "write"],
    ) -> None:
        """Populate ``self.features`` using ``session``."""

        payload, cache_file = self._load_cached_payload(
            service_id, use_local_cache, cache_mode
        )
        if payload is None:
            payload = await self._dispatch_and_cache_features(
                session,
                service_id=service_id,
                service_name=service_name,
                roles=roles,
                required_count=required_count,
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
        set_session.stage = f"mapping_{cfg.field}"
        result = await map_set(
            set_session,
            cfg.field,
            items[cfg.field],
            list(self.features),
            service_name=service_name,
            service_description=service_description,
            plateau=self.plateau,
            service=service_id,
            strict=strict,
            cache_mode=(cache_mode if use_local_cache else "off"),
            catalogue_hash=catalogue_hash,
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
