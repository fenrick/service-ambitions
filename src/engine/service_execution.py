"""Service evolution execution helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import logfire
from pydantic_ai import Agent
from pydantic_core import to_json

from core.canonical import canonicalise_record
from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from engine.service_runtime import ServiceRuntime
from io_utils.loader import load_mapping_items
from model_factory import ModelFactory
from models import (
    MappingDiagnosticsResponse,
    MappingResponse,
    PlateauDescriptionsResponse,
    PlateauFeaturesResponse,
    ServiceInput,
    ServiceMeta,
)
from persistence import atomic_write
from plateau_generator import (
    PlateauGenerator,
    default_plateau_map,
    default_plateau_names,
)
from quarantine import QuarantineWriter
from runtime.environment import RuntimeEnv
from settings import Settings
from utils import ErrorHandler

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
EVOLUTIONS_GENERATED = logfire.metric_counter("evolutions_generated")
LINES_WRITTEN = logfire.metric_counter("lines_written")
_writer = QuarantineWriter()


class ServiceExecution:
    """Execute a single service evolution run."""

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        factory: ModelFactory,
        system_prompt: str,
        transcripts_dir: Path | None,
        role_ids: Sequence[str],
        temp_output_dir: Path | None,
        allow_prompt_logging: bool,
        error_handler: ErrorHandler,
    ) -> None:
        self.runtime = runtime
        self.factory = factory
        self.system_prompt = system_prompt
        self.transcripts_dir = transcripts_dir
        self.role_ids = role_ids
        self.temp_output_dir = temp_output_dir
        self.allow_prompt_logging = allow_prompt_logging
        self.error_handler = error_handler

    def _build_generator(
        self, settings: Settings
    ) -> tuple[PlateauGenerator, str, str, str]:
        """Construct plateau generator and return model names.

        Args:
            settings: Runtime configuration.

        Returns:
            Tuple of the generator and stage model names.
        """

        desc_model = self.factory.get("descriptions")
        feat_model = self.factory.get("features")
        map_model = self.factory.get("mapping")

        desc_name = self.factory.model_name("descriptions")
        feat_name = self.factory.model_name("features")
        map_name = self.factory.model_name("mapping")

        desc_agent = Agent(
            desc_model,
            instructions=self.system_prompt,
            output_type=PlateauDescriptionsResponse,
        )
        feat_agent = Agent(
            feat_model,
            instructions=self.system_prompt,
            output_type=PlateauFeaturesResponse,
        )
        map_agent = Agent(
            map_model,
            instructions=self.system_prompt,
            output_type=(
                MappingDiagnosticsResponse if settings.diagnostics else MappingResponse
            ),
        )

        desc_session = ConversationSession(
            desc_agent,
            stage="descriptions",
            diagnostics=settings.diagnostics,
            log_prompts=self.allow_prompt_logging,
            transcripts_dir=self.transcripts_dir,
            use_local_cache=settings.use_local_cache,
            cache_mode=settings.cache_mode,
        )
        feat_session = ConversationSession(
            feat_agent,
            stage="features",
            diagnostics=settings.diagnostics,
            log_prompts=self.allow_prompt_logging,
            transcripts_dir=self.transcripts_dir,
            use_local_cache=settings.use_local_cache,
            cache_mode=settings.cache_mode,
        )
        map_session = ConversationSession(
            map_agent,
            stage="mapping",
            diagnostics=settings.diagnostics,
            log_prompts=self.allow_prompt_logging,
            transcripts_dir=self.transcripts_dir,
            use_local_cache=settings.use_local_cache,
            cache_mode=settings.cache_mode,
        )

        generator = PlateauGenerator(
            feat_session,
            required_count=settings.features_per_role,
            roles=self.role_ids,
            description_session=desc_session,
            mapping_session=map_session,
            strict=settings.strict,
            use_local_cache=settings.use_local_cache,
            cache_mode=settings.cache_mode,
        )
        return generator, desc_name, feat_name, map_name

    def _ensure_run_meta(
        self,
        settings: Settings,
        desc_name: str,
        feat_name: str,
        map_name: str,
        feat_model,
    ) -> None:
        """Initialise and store run metadata in ``RuntimeEnv``."""

        env = RuntimeEnv.instance()
        if env.run_meta is not None:  # run metadata already exists
            return
        models_map = {
            "descriptions": desc_name,
            "features": feat_name,
            "mapping": map_name,
            "search": self.factory.model_name("search"),
        }
        _, catalogue_hash = load_mapping_items(settings.mapping_sets)
        context_window = getattr(feat_model, "max_input_tokens", 0)
        env.run_meta = ServiceMeta(
            run_id=str(uuid4()),
            seed=self.factory.seed,
            models=models_map,
            web_search=getattr(self.factory, "_web_search", False),
            mapping_types=sorted(getattr(settings, "mapping_types", {}).keys()),
            context_window=context_window,
            diagnostics=settings.diagnostics,
            catalogue_hash=catalogue_hash,
            created=datetime.now(timezone.utc),
        )

    async def _prepare_runtimes(
        self, generator: PlateauGenerator
    ) -> list[PlateauRuntime]:
        """Return plateau runtimes with descriptions."""

        names = list(default_plateau_names())
        desc_map = await generator._request_descriptions_async(
            names, session=generator.description_session
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

    def _write_temp_output(
        self, service: ServiceInput, record: dict[str, object]
    ) -> None:
        """Persist intermediate JSON record when enabled."""

        if self.temp_output_dir is None:
            return
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(
            self.temp_output_dir / f"{service.service_id}.json",
            [to_json(record).decode()],
        )

    async def run(self) -> bool:
        """Populate ``runtime`` and report success.

        Returns:
            ``True`` when evolution generation succeeds. Generated artefacts
            are stored on :attr:`runtime` and are not returned to callers.
        """

        service = self.runtime.service
        desc_name = self.factory.model_name("descriptions")
        feat_name = self.factory.model_name("features")
        map_name = self.factory.model_name("mapping")
        attrs = {
            "service_id": service.service_id,
            "service_name": service.name,
            "descriptions_model": desc_name,
            "features_model": feat_name,
            "mapping_model": map_name,
        }
        with logfire.span("generate_evolution_for_service", attributes=attrs):
            try:
                SERVICES_PROCESSED.add(1)
                settings = RuntimeEnv.instance().settings
                generator, desc_name, feat_name, map_name = self._build_generator(
                    settings
                )
                feat_model = self.factory.get("features")
                self._ensure_run_meta(
                    settings, desc_name, feat_name, map_name, feat_model
                )
                runtimes = await self._prepare_runtimes(generator)
                env = RuntimeEnv.instance()
                meta = env.run_meta
                assert meta is not None  # mypy safeguard
                evolution = await generator.generate_service_evolution_async(
                    service,
                    runtimes,
                    transcripts_dir=self.transcripts_dir,
                    meta=meta,
                )
                record = canonicalise_record(evolution.model_dump(mode="json"))
                self._write_temp_output(service, record)
                self.runtime.plateaus = runtimes
                self.runtime.line = to_json(record).decode()
                self.runtime.success = True
                return True
            except Exception as exc:  # noqa: BLE001
                quarantine_file = await asyncio.to_thread(
                    _writer.write,
                    "evolution",
                    service.service_id,
                    "schema_mismatch",
                    service.model_dump(),
                )
                self.error_handler.handle(
                    "Failed to generate evolution for "
                    f"{service.service_id}; quarantined {quarantine_file}",
                    exc,
                )
                return False
