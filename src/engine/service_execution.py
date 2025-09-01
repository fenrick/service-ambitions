"""Service evolution execution helpers."""

from __future__ import annotations

import argparse
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
from loader import load_mapping_items
from model_factory import ModelFactory
from models import ServiceInput, ServiceMeta
from persistence import atomic_write
from plateau_generator import PlateauGenerator
from quarantine import QuarantineWriter

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
        settings,
        args: argparse.Namespace,
        system_prompt: str,
        transcripts_dir: Path | None,
        role_ids: Sequence[str],
        lock: asyncio.Lock,
        output,
        new_ids: set[str],
        temp_output_dir: Path | None,
    ) -> None:
        self.service = service
        self.factory = factory
        self.settings = settings
        self.args = args
        self.system_prompt = system_prompt
        self.transcripts_dir = transcripts_dir
        self.role_ids = role_ids
        self.lock = lock
        self.output = output
        self.new_ids = new_ids
        self.temp_output_dir = temp_output_dir
        self.line: str | None = None

    async def run(self) -> bool:
        """Generate the evolution for ``service`` and store the result.

        Returns ``True`` on success, ``False`` otherwise.  Generated artefacts
        are kept on the instance for later persistence.
        """

        desc_name = self.factory.model_name(
            "descriptions", self.args.descriptions_model or self.args.model
        )
        feat_name = self.factory.model_name(
            "features", self.args.features_model or self.args.model
        )
        map_name = self.factory.model_name(
            "mapping", self.args.mapping_model or self.args.model
        )
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
                desc_model = self.factory.get(
                    "descriptions", self.args.descriptions_model or self.args.model
                )
                feat_model = self.factory.get(
                    "features", self.args.features_model or self.args.model
                )
                map_model = self.factory.get(
                    "mapping", self.args.mapping_model or self.args.model
                )

                desc_agent = Agent(desc_model, instructions=self.system_prompt)
                feat_agent = Agent(feat_model, instructions=self.system_prompt)
                map_agent = Agent(map_model, instructions=self.system_prompt)

                desc_session = ConversationSession(
                    desc_agent,
                    stage="descriptions",
                    diagnostics=self.settings.diagnostics,
                    log_prompts=self.args.allow_prompt_logging,
                    transcripts_dir=self.transcripts_dir,
                    use_local_cache=self.args.use_local_cache,
                    cache_mode=self.args.cache_mode,
                )
                feat_session = ConversationSession(
                    feat_agent,
                    stage="features",
                    diagnostics=self.settings.diagnostics,
                    log_prompts=self.args.allow_prompt_logging,
                    transcripts_dir=self.transcripts_dir,
                    use_local_cache=self.args.use_local_cache,
                    cache_mode=self.args.cache_mode,
                )
                map_session = ConversationSession(
                    map_agent,
                    stage="mapping",
                    diagnostics=self.settings.diagnostics,
                    log_prompts=self.args.allow_prompt_logging,
                    transcripts_dir=self.transcripts_dir,
                    use_local_cache=self.args.use_local_cache,
                    cache_mode=self.args.cache_mode,
                )
                generator = PlateauGenerator(
                    feat_session,
                    required_count=self.settings.features_per_role,
                    roles=self.role_ids,
                    description_session=desc_session,
                    mapping_session=map_session,
                    strict=self.args.strict,
                    use_local_cache=self.args.use_local_cache,
                    cache_mode=self.args.cache_mode,
                )
                global _RUN_META
                if _RUN_META is None:
                    models_map = {
                        "descriptions": desc_name,
                        "features": feat_name,
                        "mapping": map_name,
                        "search": self.factory.model_name(
                            "search", self.args.search_model or self.args.model
                        ),
                    }
                    _, catalogue_hash = load_mapping_items(
                        loader.MAPPING_DATA_DIR, self.settings.mapping_sets
                    )
                    context_window = getattr(feat_model, "max_input_tokens", 0)
                    _RUN_META = ServiceMeta(
                        run_id=str(uuid4()),
                        seed=self.args.seed,
                        models=models_map,
                        web_search=getattr(self.factory, "_web_search", False),
                        mapping_types=sorted(
                            getattr(self.settings, "mapping_types", {}).keys()
                        ),
                        context_window=context_window,
                        diagnostics=self.settings.diagnostics,
                        catalogue_hash=catalogue_hash,
                        created=datetime.now(timezone.utc),
                    )
                evolution = await generator.generate_service_evolution_async(
                    self.service,
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
                logfire.exception(
                    "Failed to generate evolution",
                    service_id=self.service.service_id,
                    error=str(exc),
                    quarantine_file=str(quarantine_file),
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
