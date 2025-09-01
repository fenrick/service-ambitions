"""High-level service processing engine."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from itertools import islice
from pathlib import Path

import logfire
from tqdm import tqdm  # type: ignore[import-untyped]

from engine.service_execution import ServiceExecution
from loader import (
    configure_mapping_data_dir,
    configure_prompt_dir,
    load_evolution_prompt,
    load_role_ids,
)
from model_factory import ModelFactory
from models import ServiceInput
from persistence import atomic_write, read_lines
from service_loader import load_services

# Helper functions migrated from cli for reuse.


def _prepare_paths(output: Path, resume: bool) -> tuple[Path, Path]:
    """Return paths used for output and resume tracking."""
    part_path = output.with_suffix(
        output.suffix + ".tmp" if not resume else output.suffix + ".tmp.part"
    )
    processed_path = output.with_name("processed_ids.txt")
    return part_path, processed_path


def _load_resume_state(
    processed_path: Path, output_path: Path, resume: bool
) -> tuple[set[str], list[str]]:
    """Return previously processed IDs and existing output lines."""
    processed_ids = set(read_lines(processed_path)) if resume else set()
    existing_lines = read_lines(output_path) if resume else []
    return processed_ids, existing_lines


def _ensure_transcripts_dir(path: str | None, output: Path) -> Path:
    """Create and return the directory used to store transcripts."""
    transcripts_dir = Path(path) if path is not None else output.parent / "_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return transcripts_dir


def _load_services_list(
    input_file: str, max_services: int | None, processed_ids: set[str]
) -> list[ServiceInput]:
    """Return services filtered for ``processed_ids`` and ``max_services``."""
    with load_services(Path(input_file)) as svc_iter:
        if max_services is not None:
            svc_iter = islice(svc_iter, max_services)
        return [s for s in svc_iter if s.service_id not in processed_ids]


def _save_results(
    *,
    resume: bool,
    part_path: Path,
    output_path: Path,
    existing_lines: list[str],
    processed_ids: set[str],
    new_ids: set[str],
    processed_path: Path,
) -> set[str]:
    """Persist generated lines and update processed IDs."""
    if resume:
        new_lines = read_lines(part_path)
        atomic_write(output_path, [*existing_lines, *new_lines])
        part_path.unlink(missing_ok=True)
        processed_ids.update(new_ids)
    else:
        os.replace(part_path, output_path)
        processed_ids = new_ids
    atomic_write(processed_path, sorted(processed_ids))
    return processed_ids


class ProcessingEngine:
    """Coordinate service evolution generation across multiple services."""

    def __init__(
        self, args: argparse.Namespace, settings, transcripts_dir: Path | None
    ) -> None:
        self.args = args
        self.settings = settings
        self.transcripts_dir = transcripts_dir
        self.output_path = Path(args.output_file)
        self.part_path, self.processed_path = _prepare_paths(
            self.output_path, args.resume
        )
        self.processed_ids, self.existing_lines = _load_resume_state(
            self.processed_path, self.output_path, args.resume
        )
        if self.transcripts_dir is None and not args.no_logs:
            self.transcripts_dir = _ensure_transcripts_dir(
                args.transcripts_dir, self.output_path
            )
        self.new_ids: set[str] = set()
        self.executions: list[ServiceExecution] = []
        self.success = False

    async def run(self) -> bool:
        """Run evolutions for all loaded services."""

        with logfire.span("processing_engine.run"):
            logfire.info(
                "Starting processing engine",
                input_file=self.args.input_file,
            )
            use_web_search = (
                self.args.web_search
                if self.args.web_search is not None
                else self.settings.web_search
            )
            factory = ModelFactory(
                self.settings.model,
                self.settings.openai_api_key,
                stage_models=getattr(self.settings, "models", None),
                reasoning=self.settings.reasoning,
                seed=self.args.seed,
                web_search=use_web_search,
            )
            configure_prompt_dir(self.settings.prompt_dir)
            if self.args.mapping_data_dir is None and not self.settings.diagnostics:
                raise RuntimeError("--mapping-data-dir is required in production mode")
            configure_mapping_data_dir(
                self.args.mapping_data_dir or self.settings.mapping_data_dir
            )
            system_prompt = load_evolution_prompt(
                self.settings.context_id, self.settings.inspiration
            )
            role_ids = load_role_ids(Path(self.args.roles_file))
            services = _load_services_list(
                self.args.input_file, self.args.max_services, self.processed_ids
            )
            if self.args.dry_run:
                logfire.info("Validated services", count=len(services))
                return True
            concurrency = (
                self.args.concurrency
                if self.args.concurrency is not None
                else self.settings.concurrency
            )
            if concurrency < 1:
                raise ValueError("concurrency must be a positive integer")
            sem = asyncio.Semaphore(concurrency)
            lock = asyncio.Lock()
            progress = (
                tqdm(total=len(services))
                if self.args.progress and sys.stdout.isatty()
                else None
            )
            temp_output_dir = (
                Path(self.args.temp_output_dir)
                if self.args.temp_output_dir is not None
                else None
            )
            success = True

            async def run_one(service: ServiceInput) -> None:
                nonlocal success
                async with sem:
                    execution = ServiceExecution(
                        service,
                        factory=factory,
                        settings=self.settings,
                        args=self.args,
                        system_prompt=system_prompt,
                        transcripts_dir=self.transcripts_dir,
                        role_ids=role_ids,
                        lock=lock,
                        output=None,
                        new_ids=self.new_ids,
                        temp_output_dir=temp_output_dir,
                    )
                    self.executions.append(execution)
                    if not await execution.run():
                        success = False
                if progress:
                    progress.update(1)

            async with asyncio.TaskGroup() as tg:
                for svc in services:
                    tg.create_task(run_one(svc))
            if progress:
                progress.close()
            self.success = success
            logfire.info("Processing engine completed", success=success)
            return success

    async def finalise(self) -> None:
        """Write successful results to disk."""
        with logfire.span("processing_engine.finalise"):
            if not self.executions:
                logfire.debug("No executions to finalise")
                return
            output = await asyncio.to_thread(self.part_path.open, "w", encoding="utf-8")
            try:
                for exec in self.executions:
                    exec.output = output
                    await exec.finalise()
            finally:
                await asyncio.to_thread(output.close)
            _save_results(
                resume=self.args.resume,
                part_path=self.part_path,
                output_path=self.output_path,
                existing_lines=self.existing_lines,
                processed_ids=self.processed_ids,
                new_ids=self.new_ids,
                processed_path=self.processed_path,
            )
            logfire.info("Finalised processing engine", output=str(self.output_path))


__all__ = ["ProcessingEngine"]
