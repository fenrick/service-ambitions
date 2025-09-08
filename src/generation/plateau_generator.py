# SPDX-License-Identifier: MIT
"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is handled in an isolated
``ConversationSession`` so generation and mapping for one level do not leak
history into another while still reusing the same underlying agent.
"""

from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

import logfire
from pydantic_core import to_json

from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from io_utils.loader import (
    load_plateau_definitions,
    load_prompt_text,
    load_role_ids,
)
from models import (
    FeatureItem,
    PlateauDescriptionsResponse,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceFeaturePlateau,
    ServiceInput,
    ServiceMeta,
)
from runtime.environment import RuntimeEnv
from utils import ShortCodeRegistry
from utils.cache_paths import feature_cache

# Settings and token scheduling are no longer required after simplification.

A_NON_EMPTY_STRING = "'description' must be a non-empty string"


# Snapshot of plateau and role metadata loaded on first use.


@lru_cache(maxsize=1)
def plateau_definitions() -> list[ServiceFeaturePlateau]:
    """Return service feature plateau definitions from configuration."""
    settings = RuntimeEnv.instance().settings
    return load_plateau_definitions(settings.mapping_data_dir)


@lru_cache(maxsize=1)
def default_plateau_map() -> dict[str, int]:
    """Return mapping of plateau name to numeric level."""
    return {p.name: idx + 1 for idx, p in enumerate(plateau_definitions())}


@lru_cache(maxsize=1)
def default_plateau_names() -> list[str]:
    """Return plateau names in ascending maturity order."""
    return [p.name for p in plateau_definitions()]


@lru_cache(maxsize=1)
def default_role_ids() -> list[str]:
    """Return core role identifiers."""
    settings = RuntimeEnv.instance().settings
    return load_role_ids(settings.roles_file)


def _feature_cache_path(service: str, plateau: int) -> Path:
    """Return canonical cache path for features at ``plateau``."""
    return feature_cache(service, plateau)


def _discover_feature_cache(service: str, plateau: int) -> tuple[Path, Path]:
    """Return existing feature cache and canonical destination."""
    canonical = _feature_cache_path(service, plateau)
    if canonical.exists():
        return canonical, canonical

    service_root = canonical.parents[1]
    for candidate in service_root.glob("**/features.json"):
        return candidate, canonical

    return canonical, canonical


class PlateauGenerator:
    """Generate plateau features and service evolution summaries.

    Local caching is enabled by default in read-only mode to reuse mapping
    results between runs.
    """

    def __init__(
        self,
        session: ConversationSession,
        roles: Sequence[str] | None = None,
        *,
        description_session: ConversationSession | None = None,
        mapping_session: ConversationSession | None = None,
        strict: bool = False,
        use_local_cache: bool = True,
        cache_mode: Literal["off", "read", "refresh", "write"] = "read",
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for feature generation.
            roles: Role identifiers to include during generation.
            description_session: Session used for plateau descriptions.
            mapping_session: Session used for feature mapping.
            strict: Enforce feature and mapping completeness when ``True``.
            use_local_cache: Read and write mapping results from the cache
                directory (default ``${XDG_CACHE_HOME}/service-ambitions`` and
                falling back to ``/tmp/service-ambitions`` when
                ``XDG_CACHE_HOME`` is unset) when ``True``. Caching is enabled
                by default.
            cache_mode: Caching strategy controlling read/write behaviour.
                Defaults to ``"read"`` for read-only access.
        """
        self.session = session
        self.description_session = description_session or session
        self.mapping_session = mapping_session or session
        self.roles = list(roles or default_role_ids())
        self.strict = strict
        self.use_local_cache = use_local_cache
        self.cache_mode: Literal["off", "read", "refresh", "write"] = cache_mode
        self._service: ServiceInput | None = None
        # Track quarantine file paths for invalid plateau descriptions.
        self.quarantined_descriptions: list[Path] = []
        # Registry used for deterministic feature codes.
        self.code_registry = ShortCodeRegistry()

    def _quarantine_description(self, plateau_name: str, raw: str) -> Path:
        """Persist ``raw`` text for ``plateau_name`` and record its path."""
        # Create the quarantine directory if it does not yet exist.
        qdir = Path("quarantine/descriptions")
        qdir.mkdir(parents=True, exist_ok=True)
        file_path = qdir / f"{plateau_name}.txt"
        file_path.write_text(raw, encoding="utf-8")
        logfire.warning(f"Quarantined plateau description at {file_path}")
        self.quarantined_descriptions.append(file_path)
        return file_path

    def _request_description(
        self, level: int, session: ConversationSession | None = None
    ) -> str:
        """Return the service description for ``level``.

        This helper delegates to :meth:`_request_descriptions` using the
        plateau's name to keep all description handling in one place. Any
        parsing or sanitisation errors are handled by the batch method and an
        empty string is returned when the model cannot provide a valid
        description.
        """
        session = session or self.description_session
        plateau_name = next(
            (n for n, lvl in default_plateau_map().items() if lvl == level),
            f"plateau_{level}",
        )
        return self._request_descriptions([plateau_name], session).get(plateau_name, "")

    @staticmethod
    def _sanitize_description(text: str) -> str:
        """Remove any model-added preamble from ``text``.

        Some models prepend strings like "Prepared plateau-1 description for X:"
        before the actual description. This helper strips that prefix if present.
        """
        pattern = r"^Prepared plateau-\d+ description for [^:]+:\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    def _build_descriptions_prompt(self, plateau_names: Sequence[str]) -> str:
        """Return a prompt requesting descriptions for ``plateau_names``."""
        lines: list[str] = []
        for name in plateau_names:
            try:
                level = default_plateau_map()[name]
            except KeyError as exc:
                raise ValueError(f"Unknown plateau name: {name}") from exc
            lines.append(f"{level}. {name}")
        plateaus_str = "\n".join(lines)
        template = load_prompt_text("plateau_descriptions_prompt")
        return template.format(plateaus=plateaus_str)

    def _normalize_descriptions_payload(
        self, payload: PlateauDescriptionsResponse | dict[str, object]
    ) -> tuple[PlateauDescriptionsResponse | None, str]:
        """Return validated model and raw JSON string for ``payload``.

        If validation fails, returns ``(None, raw)`` where ``raw`` captures the
        serialised payload for quarantine purposes.
        """
        if isinstance(payload, PlateauDescriptionsResponse):
            return payload, payload.model_dump_json()
        if isinstance(payload, dict):
            try:
                model = PlateauDescriptionsResponse.model_validate(payload)
                return model, model.model_dump_json()
            except Exception:  # noqa: BLE001 - defensive normalisation
                return None, to_json(payload).decode()
        return None, str(payload)

    def _request_descriptions_common(
        self,
        plateau_names: Sequence[str],
        payload: PlateauDescriptionsResponse | dict[str, object],
    ) -> dict[str, str]:
        """Parse ``payload`` description data for ``plateau_names``.

        Accepts either a validated :class:`PlateauDescriptionsResponse` or a
        ``dict`` (e.g. when loaded from cache without automatic validation).
        """
        results: dict[str, str] = {}
        model, raw = self._normalize_descriptions_payload(payload)
        if model is None:
            # Unable to normalise the payload; quarantine entries and return empty.
            for name in plateau_names:
                self._quarantine_description(name, raw)
                results[name] = ""
            return results
        item_map = {item.plateau_name: item for item in model.descriptions}
        for name in plateau_names:
            item = item_map.get(name)
            if item is None:
                # Missing entry for the plateau name.
                self._quarantine_description(name, raw)
                results[name] = ""
                continue
            try:
                cleaned = self._sanitize_description(item.description)
                if not cleaned:
                    raise ValueError(A_NON_EMPTY_STRING)
                results[name] = cleaned
            except Exception as exc:  # noqa: BLE001 - tolerate provider/client glitches; safe fallback path (see issue #501)
                self._quarantine_description(name, raw)
                logfire.error(f"Invalid plateau description for {name}: {exc}")
                results[name] = ""
        return results

    def _request_descriptions(
        self,
        plateau_names: Sequence[str],
        session: ConversationSession | None = None,
    ) -> dict[str, str]:
        session = session or self.description_session
        prompt = self._build_descriptions_prompt(plateau_names)
        logfire.info("Requesting plateau descriptions", plateaus=list(plateau_names))
        payload = session.ask(prompt)
        return self._request_descriptions_common(plateau_names, payload)

    async def _request_descriptions_async(
        self,
        plateau_names: Sequence[str],
        session: ConversationSession | None = None,
    ) -> dict[str, str]:
        """Asynchronously return descriptions for ``plateau_names``."""
        session = session or self.description_session
        prompt = self._build_descriptions_prompt(plateau_names)
        logfire.info("Requesting plateau descriptions", plateaus=list(plateau_names))
        payload = await session.ask_async(prompt)
        return self._request_descriptions_common(plateau_names, payload)

    def _to_feature(
        self, item: FeatureItem, role: str, plateau_name: str
    ) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``.

        Args:
            item: Raw feature details returned by the agent.
            role: Role the feature applies to.
            plateau_name: Plateau the feature belongs to.

        Returns:
            Plateau feature populated with the provided metadata.
        """
        canonical = f"{item.name}|{role}|{plateau_name}"
        feature_id = self.code_registry.generate(canonical)
        return PlateauFeature(
            feature_id=feature_id,
            name=item.name,
            description=item.description,
            score=item.score,
            customer_type=role,
        )

    def _build_plateau_prompt(self, level: int, description: str) -> str:
        """Return a prompt requesting features for ``level``."""
        template = load_prompt_text("plateau_prompt")
        roles_str = ", ".join(f'"{r}"' for r in self.roles)
        return template.format(
            service_name=self._service.name if self._service else "",
            service_description=description,
            plateau=str(level),
            roles=str(roles_str),
        )

    def _prepare_sessions(self, service_input: ServiceInput) -> None:
        """Attach ``service_input`` to all conversation sessions."""
        self._service = service_input
        self.session.add_parent_materials(service_input)
        self.description_session.add_parent_materials(service_input)
        self.mapping_session.add_parent_materials(service_input)

    async def _schedule_plateaus(
        self,
        runtimes: Sequence[PlateauRuntime],
        service_input: ServiceInput,
    ) -> list[PlateauRuntime]:
        """Populate ``runtimes`` with plateau results."""
        results: list[PlateauRuntime] = []
        for runtime in runtimes:
            plateau_session = ConversationSession(
                self.session.client,
                stage=f"features_{runtime.plateau}",
                use_local_cache=self.use_local_cache,
                cache_mode=self.cache_mode,
            )
            plateau_session.add_parent_materials(service_input)
            await self.generate_plateau_async(runtime, session=plateau_session)
            results.append(runtime)
        return results

    def _validate_plateau_results(
        self,
        results: Sequence[PlateauResult],
        plateau_names: Sequence[str],
        role_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> tuple[list[PlateauResult], dict[str, bool]]:
        """Validate ``results`` and normalise feature fields.

        Args:
            results: Plateau results with features for each maturity level.
            plateau_names: Ordered plateau names included in the evolution.
            role_ids: Roles expected to appear across all features.
            strict: Enforce presence of features for all roles and non-empty
                mapping lists when ``True``.

        Returns:
            Tuple of validated plateau results and a mapping indicating which
            roles were encountered.
        """
        plateaus: list[PlateauResult] = []
        roles_seen: dict[str, bool] = {r: False for r in role_ids}
        for result in results:
            if result.plateau_name not in plateau_names:
                raise ValueError(f"Unknown plateau name: {result.plateau_name}")
            valid: list[PlateauFeature] = []
            for feat in result.features:
                if feat.customer_type not in role_ids:
                    raise ValueError(f"Unknown customer_type: {feat.customer_type}")
                roles_seen[feat.customer_type] = True
                feature_id = feat.feature_id.strip()
                if not feature_id:
                    raise ValueError("feature_id must be non-empty")
                valid.append(
                    PlateauFeature(
                        feature_id=feature_id,
                        name=feat.name.strip(),
                        description=feat.description.strip(),
                        score=feat.score,
                        customer_type=feat.customer_type,
                        mappings=feat.mappings,
                    )
                )
            if strict and (
                not result.mappings
                or any(len(v) == 0 for v in result.mappings.values())
            ):
                raise ValueError(
                    f"Plateau {result.plateau_name} has incomplete mappings"
                )
            plateaus.append(
                PlateauResult(
                    plateau=result.plateau,
                    plateau_name=result.plateau_name,
                    service_description=result.service_description,
                    features=valid,
                    mappings=result.mappings,
                )
            )
        return plateaus, roles_seen

    async def _write_transcript(
        self,
        transcripts_dir: Path,
        service_input: ServiceInput,
        evolution: ServiceEvolution,
    ) -> None:
        """Persist a transcript for ``service_input`` and ``evolution``."""
        payload = {
            "request": service_input.model_dump(mode="json"),
            "response": evolution.model_dump(mode="json"),
        }
        data = to_json(payload).decode()
        path = transcripts_dir / f"{service_input.service_id}.json"
        await asyncio.to_thread(
            path.write_text,
            data,
            encoding="utf-8",
        )

    async def _assemble_evolution(
        self,
        service_input: ServiceInput,
        results: Sequence[PlateauResult],
        plateau_names: Sequence[str],
        role_ids: Sequence[str],
        meta: ServiceMeta,
        transcripts_dir: Path | None,
        *,
        strict: bool = False,
    ) -> ServiceEvolution:
        """Return ``ServiceEvolution`` from plateau ``results``."""
        plateaus, roles_seen = self._validate_plateau_results(
            results, plateau_names, role_ids, strict=strict
        )
        evolution = ServiceEvolution(
            meta=meta, service=service_input, plateaus=plateaus
        )
        if transcripts_dir is not None:
            await self._write_transcript(transcripts_dir, service_input, evolution)
        if strict:
            missing = [r for r, seen in roles_seen.items() if not seen]
            if missing:
                raise ValueError(
                    f"No features generated for roles: {', '.join(sorted(missing))}"
                )
        return evolution

    @logfire.instrument()
    async def generate_plateau_async(
        self,
        runtime: PlateauRuntime,
        *,
        session: ConversationSession | None = None,
    ) -> PlateauRuntime:
        """Populate ``runtime`` with plateau artefacts."""
        if self._service is None:
            raise ValueError(
                "ServiceInput not set. Call generate_service_evolution first."
            )

        session = session or self.session
        await runtime.generate_features(
            session,
            service_id=self._service.service_id,
            service_name=self._service.name,
            roles=self.roles,
            code_registry=self.code_registry,
            use_local_cache=self.use_local_cache,
            cache_mode=self.cache_mode,
        )

        map_session = ConversationSession(
            self.mapping_session.client,
            stage=self.mapping_session.stage,
            use_local_cache=self.use_local_cache,
            cache_mode=self.cache_mode,
        )
        map_session.add_parent_materials(self._service)
        await runtime.generate_mappings(
            map_session,
            service_name=self._service.name,
            service_id=self._service.service_id,
            service_description=runtime.description,
            strict=self.strict,
            use_local_cache=self.use_local_cache,
            cache_mode=self.cache_mode,
        )
        return runtime

    def generate_plateau(
        self,
        runtime: PlateauRuntime,
        *,
        session: ConversationSession | None = None,
    ) -> PlateauRuntime:
        """Synchronously populate ``runtime`` with mapped features."""
        return asyncio.run(self.generate_plateau_async(runtime, session=session))

    async def _init_runtimes(
        self, runtimes: Sequence[PlateauRuntime] | None
    ) -> list[PlateauRuntime]:
        """Create default plateau runtimes when none are provided."""
        if runtimes is not None:
            return list(runtimes)
        names = default_plateau_names()
        desc_map = await self._request_descriptions_async(
            names, session=self.description_session
        )
        pmap = default_plateau_map()
        return [
            PlateauRuntime(
                plateau=pmap[name],
                plateau_name=name,
                description=desc_map[name],
            )
            for name in names
        ]

    def _resolve_role_ids(self, role_ids: Sequence[str] | None) -> list[str]:
        """Return explicit role identifiers or fall back to defaults."""
        return list(role_ids or self.roles)

    def _collect_results(
        self, runtimes: Sequence[PlateauRuntime], results: Sequence[PlateauRuntime]
    ) -> tuple[list[PlateauResult], list[str]]:
        """Transform plateau runtimes into evolution results."""
        plateau_names = [r.plateau_name for r in runtimes]
        plateau_results = [
            PlateauResult(
                plateau=r.plateau,
                plateau_name=r.plateau_name,
                service_description=r.description,
                features=r.features,
                mappings=r.mappings,
            )
            for r in results
            if r.status()
        ]
        return plateau_results, plateau_names

    def _log_quarantines(self) -> None:
        """Emit a warning when any plateau descriptions were quarantined."""
        if self.quarantined_descriptions:
            logfire.warning(
                f"Quarantined {len(self.quarantined_descriptions)} plateau"
                " description(s)",
                paths=[str(p) for p in self.quarantined_descriptions],
            )

    async def generate_service_evolution_async(
        self,
        service_input: ServiceInput,
        runtimes: Sequence[PlateauRuntime] | None = None,
        role_ids: Sequence[str] | None = None,
        *,
        transcripts_dir: Path | None = None,
        meta: ServiceMeta,
    ) -> ServiceEvolution:
        """Asynchronously return service evolution for provided ``runtimes``.

        Args:
            service_input: Source service details to evolve.
            runtimes: Plateau runtimes to process. When ``None`` the defaults
                from :func:`default_plateau_names` are used.
            role_ids: Optional subset of role identifiers to include.
            transcripts_dir: Directory to persist per-service transcripts. ``None``
                disables transcript persistence.
            meta: Metadata object applied to the resulting evolution (e.g. model
                configuration, mapping types, web search flag).
        """
        self.quarantined_descriptions.clear()
        self._prepare_sessions(service_input)

        with logfire.span("generate_service_evolution") as span:
            span.set_attribute("service.id", service_input.service_id)
            if service_input.customer_type:
                span.set_attribute("customer_type", service_input.customer_type)
            runtimes = await self._init_runtimes(runtimes)
            role_ids = self._resolve_role_ids(role_ids)
            results = await self._schedule_plateaus(runtimes, service_input)
            plateau_results, plateau_names = self._collect_results(runtimes, results)
            evolution = await self._assemble_evolution(
                service_input,
                plateau_results,
                plateau_names,
                role_ids,
                meta,
                transcripts_dir,
                strict=self.strict,
            )
            self._log_quarantines()
            return evolution

    def generate_service_evolution(
        self,
        service_input: ServiceInput,
        runtimes: Sequence[PlateauRuntime] | None = None,
        role_ids: Sequence[str] | None = None,
        *,
        transcripts_dir: Path | None = None,
        meta: ServiceMeta,
    ) -> ServiceEvolution:
        """Return service evolution for provided plateau runtimes."""
        return asyncio.run(
            self.generate_service_evolution_async(
                service_input,
                runtimes,
                role_ids,
                transcripts_dir=transcripts_dir,
                meta=meta,
            )
        )
