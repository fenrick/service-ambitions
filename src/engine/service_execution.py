"""Service evolution execution helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import logfire
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_core import to_json

from core.canonical import canonicalise_record
from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from engine.service_runtime import ServiceRuntime
from generation.plateau_generator import (
    PlateauGenerator,
    default_plateau_map,
    default_plateau_names,
)
from io_utils.loader import load_mapping_items
from io_utils.persistence import atomic_write
from io_utils.quarantine import QuarantineWriter
from models import (
    MappingDiagnosticsResponse,
    MappingResponse,
    PlateauDescriptionsResponse,
    PlateauFeaturesResponse,
    ServiceInput,
    ServiceMeta,
)
from models.factory import ModelFactory
from runtime.environment import RuntimeEnv
from runtime.settings import Settings
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
        # Cache runtime settings for use across helper methods
        self.settings: Settings = RuntimeEnv.instance().settings
        # Initialised in ``_build_generator``
        self.generator: PlateauGenerator | None = None
        self.desc_name = ""
        self.feat_name = ""
        self.map_name = ""
        self.feat_model: object | None = None

    def refresh_settings(self) -> None:
        """Refresh cached settings from :class:`RuntimeEnv`.

        Side effects:
            Updates ``self.settings`` to reflect any changes in the runtime
            configuration.
        """

        self.settings = RuntimeEnv.instance().settings

    def _build_generator(self) -> None:
        """Construct plateau generator and cache model names.

        Side effects:
            Sets ``generator``, ``desc_name``, ``feat_name``, ``map_name`` and
            ``feat_model`` attributes using the current ``settings``.
        """

        settings = self.settings

        desc_model = self.factory.get("descriptions")
        feat_model = self.factory.get("features")
        map_model = self.factory.get("mapping")

        self.desc_name = self.factory.model_name("descriptions")
        self.feat_name = self.factory.model_name("features")
        self.map_name = self.factory.model_name("mapping")
        self.feat_model = feat_model

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

        self.generator = PlateauGenerator(
            feat_session,
            required_count=settings.features_per_role,
            roles=self.role_ids,
            description_session=desc_session,
            mapping_session=map_session,
            strict=settings.strict,
            use_local_cache=settings.use_local_cache,
            cache_mode=settings.cache_mode,
        )

    def _ensure_run_meta(self) -> None:
        """Initialise and store run metadata in ``RuntimeEnv``."""

        env = RuntimeEnv.instance()
        if env.run_meta is not None:  # run metadata already exists
            return
        models_map = {
            "descriptions": self.desc_name,
            "features": self.feat_name,
            "mapping": self.map_name,
            "search": self.factory.model_name("search"),
        }
        _, catalogue_hash = load_mapping_items(
            self.settings.mapping_sets, error_handler=self.error_handler
        )
        context_window = getattr(self.feat_model, "max_input_tokens", 0)
        env.run_meta = ServiceMeta(
            run_id=str(uuid4()),
            seed=self.factory.seed,
            models=models_map,
            web_search=getattr(self.factory, "_web_search", False),
            mapping_types=sorted(getattr(self.settings, "mapping_types", {}).keys()),
            context_window=context_window,
            diagnostics=self.settings.diagnostics,
            catalogue_hash=catalogue_hash,
            created=datetime.now(timezone.utc),
        )

    async def _prepare_runtimes(self) -> list[PlateauRuntime]:
        """Return plateau runtimes with descriptions."""

        generator = self.generator
        if generator is None:  # pragma: no cover - defensive
            raise RuntimeError("Generator not initialised")

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

        self.refresh_settings()
        service = self.runtime.service
        self._build_generator()
        if self.generator is None:
            raise RuntimeError("Plateau generator is not initialised")
        attrs = {
            "service_id": service.service_id,
            "service_name": service.name,
            "descriptions_model": self.desc_name,
            "features_model": self.feat_name,
            "mapping_model": self.map_name,
        }
        with logfire.span("generate_evolution_for_service", attributes=attrs):
            try:
                SERVICES_PROCESSED.add(1)
                self._ensure_run_meta()
                runtimes = await self._prepare_runtimes()
                env = RuntimeEnv.instance()
                meta = env.run_meta
                if meta is None:
                    raise RuntimeError("Run metadata is not initialised")
                evolution = await self.generator.generate_service_evolution_async(
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
            except RuntimeError:
                raise
            except (ValidationError, ValueError, OSError) as exc:
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
