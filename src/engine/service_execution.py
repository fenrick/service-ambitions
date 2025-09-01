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

import loader
from canonical import canonicalise_record
from conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from loader import load_mapping_items
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
from utils import ErrorHandler

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
EVOLUTIONS_GENERATED = logfire.metric_counter("evolutions_generated")
LINES_WRITTEN = logfire.metric_counter("lines_written")

_RUN_META: ServiceMeta | None = None
_writer = QuarantineWriter()


class ServiceExecution:
    """Execute a single service evolution run.

    The :class:`ServiceExecution` class encapsulates the logic required to
    generate a service evolution and persist the result.  The ``run`` method
    performs the generation and stores intermediate artefacts on the instance
    while ``finalise`` writes the persisted output.
    """

    def __init__(
        self,
        service: ServiceInput,
        *,
        factory: ModelFactory,
        system_prompt: str,
        transcripts_dir: Path | None,
        role_ids: Sequence[str],
        lock: asyncio.Lock,
        output,
        new_ids: set[str],
        temp_output_dir: Path | None,
        allow_prompt_logging: bool,
        error_handler: ErrorHandler,
    ) -> None:
        self.service = service
        self.factory = factory
        self.system_prompt = system_prompt
        self.transcripts_dir = transcripts_dir
        self.role_ids = role_ids
        self.lock = lock
        self.output = output
        self.new_ids = new_ids
        self.temp_output_dir = temp_output_dir
        self.allow_prompt_logging = allow_prompt_logging
        self.error_handler = error_handler
        self.line: str | None = None

    async def run(self) -> bool:
        """Generate the evolution for ``service`` and store the result.

        Returns:
            ``True`` on success, ``False`` otherwise.

        Side effects:
            Updates metrics, writes quarantine files on failure and stores the
            generated line on the instance for later persistence.
        """

        desc_name = self.factory.model_name("descriptions")
        feat_name = self.factory.model_name("features")
        map_name = self.factory.model_name("mapping")
        attrs = {
            "service_id": self.service.service_id,
            "service_name": self.service.name,
            "descriptions_model": desc_name,
            "features_model": feat_name,
            "mapping_model": map_name,
            "output_path": getattr(self.output, "name", ""),
        }
        with logfire.span("generate_evolution_for_service", attributes=attrs):
            try:
                SERVICES_PROCESSED.add(1)
                desc_model = self.factory.get("descriptions")
                feat_model = self.factory.get("features")
                map_model = self.factory.get("mapping")

                settings = RuntimeEnv.instance().settings

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
                        MappingDiagnosticsResponse
                        if settings.diagnostics
                        else MappingResponse
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
                global _RUN_META
                if _RUN_META is None:
                    models_map = {
                        "descriptions": desc_name,
                        "features": feat_name,
                        "mapping": map_name,
                        "search": self.factory.model_name("search"),
                    }
                    _, catalogue_hash = load_mapping_items(
                        loader.MAPPING_DATA_DIR, settings.mapping_sets
                    )
                    context_window = getattr(feat_model, "max_input_tokens", 0)
                    _RUN_META = ServiceMeta(
                        run_id=str(uuid4()),
                        seed=self.factory.seed,
                        models=models_map,
                        web_search=getattr(self.factory, "_web_search", False),
                        mapping_types=sorted(
                            getattr(settings, "mapping_types", {}).keys()
                        ),
                        context_window=context_window,
                        diagnostics=settings.diagnostics,
                        catalogue_hash=catalogue_hash,
                        created=datetime.now(timezone.utc),
                    )
                plateau_names = list(default_plateau_names())
                desc_map = await generator._request_descriptions_async(
                    plateau_names, session=desc_session
                )
                runtimes = [
                    PlateauRuntime(
                        plateau=default_plateau_map()[name],
                        plateau_name=name,
                        description=desc_map[name],
                    )
                    for name in plateau_names
                ]
                evolution = await generator.generate_service_evolution_async(
                    self.service,
                    runtimes,
                    transcripts_dir=self.transcripts_dir,
                    meta=_RUN_META,
                )
                record = canonicalise_record(evolution.model_dump(mode="json"))
                if self.temp_output_dir is not None:
                    self.temp_output_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write(
                        self.temp_output_dir / f"{self.service.service_id}.json",
                        [to_json(record).decode()],
                    )
                self.line = to_json(record).decode()
                return True
            except Exception as exc:  # noqa: BLE001
                quarantine_file = await asyncio.to_thread(
                    _writer.write,
                    "evolution",
                    self.service.service_id,
                    "schema_mismatch",
                    self.service.model_dump(),
                )
                self.error_handler.handle(
                    "Failed to generate evolution for "
                    f"{self.service.service_id}; quarantined {quarantine_file}",
                    exc,
                )
                return False

    async def finalise(self) -> None:
        """Persist the previously generated line to disk."""

        if self.line is None:
            return
        async with self.lock:
            await asyncio.to_thread(self.output.write, f"{self.line}\n")
            self.new_ids.add(self.service.service_id)
            EVOLUTIONS_GENERATED.add(1)
            LINES_WRITTEN.add(1)
        logfire.info(
            "Generated evolution",
            service_id=self.service.service_id,
            output_path=getattr(self.output, "name", ""),
        )
