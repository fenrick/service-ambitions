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
from runtime.environment import RuntimeEnv
from service_loader import load_services
from settings import Settings
from utils import ErrorHandler, LoggingErrorHandler

# Helper functions migrated from cli for reuse.


def _prepare_paths(output: Path, resume: bool) -> tuple[Path, Path]:
    """Return paths used for output and resume tracking.

    Args:
        output: Final output file.
        resume: Whether resuming a previous run.

    Returns:
        Tuple of temporary output path and processed IDs path.

    Side effects:
        None.
    """
    part_path = output.with_suffix(
        output.suffix + ".tmp" if not resume else output.suffix + ".tmp.part"
    )
    processed_path = output.with_name("processed_ids.txt")
    return part_path, processed_path


def _load_resume_state(
    processed_path: Path, output_path: Path, resume: bool
) -> tuple[set[str], list[str]]:
    """Return previously processed IDs and existing output lines.

    Args:
        processed_path: Location of the processed IDs file.
        output_path: Existing output file.
        resume: Whether resuming a previous run.

    Returns:
        Set of processed IDs and list of existing output lines.

    Side effects:
        Reads from the file system when ``resume`` is ``True``.
    """
    processed_ids = set(read_lines(processed_path)) if resume else set()
    existing_lines = read_lines(output_path) if resume else []
    return processed_ids, existing_lines


def _ensure_transcripts_dir(path: str | None, output: Path) -> Path:
    """Create and return the directory used to store transcripts.

    Args:
        path: Optional directory override.
        output: Path to the main output file.

    Returns:
        Directory where transcripts are stored.

    Side effects:
        Creates the directory if it does not exist.
    """
    transcripts_dir = Path(path) if path is not None else output.parent / "_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return transcripts_dir


def _load_services_list(
    input_file: str, max_services: int | None, processed_ids: set[str]
) -> list[ServiceInput]:
    """Return services filtered for ``processed_ids`` and ``max_services``.

    Args:
        input_file: Path to the services definition file.
        max_services: Optional cap on number of services.
        processed_ids: IDs already processed in previous runs.

    Returns:
        List of services pending processing.

    Side effects:
        Reads services from the file system.
    """
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
    """Persist generated lines and update processed IDs.

    Args:
        resume: Whether the engine resumed a previous run.
        part_path: Temporary output file path.
        output_path: Final output file path.
        existing_lines: Lines read from a previous run.
        processed_ids: IDs processed before this invocation.
        new_ids: IDs generated during this run.
        processed_path: File storing processed IDs.

    Returns:
        Updated set of all processed IDs.

    Side effects:
        Moves or writes files on disk.
    """
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

    def __init__(self, args: argparse.Namespace, transcripts_dir: Path | None) -> None:
        self.args = args
        self.transcripts_dir = transcripts_dir
        self.output_path = Path(args.output_file)
        self.part_path, self.processed_path = _prepare_paths(
            self.output_path, args.resume
        )
        self.processed_ids, self.existing_lines = _load_resume_state(
            self.processed_path, self.output_path, args.resume
        )
        if self.transcripts_dir is None:
            self.transcripts_dir = _ensure_transcripts_dir(
                args.transcripts_dir, self.output_path
            )
        self.new_ids: set[str] = set()
        self.executions: list[ServiceExecution] = []
        self.success = False

    def _create_model_factory(self, settings: Settings) -> ModelFactory:
        """Create a model factory from settings.

        Args:
            settings: Global configuration.

        Returns:
            Configured :class:`ModelFactory` instance.

        Side effects:
            None.
        """

        return ModelFactory(
            settings.model,
            settings.openai_api_key,
            stage_models=getattr(settings, "models", None),
            reasoning=settings.reasoning,
            seed=self.args.seed,
            web_search=settings.web_search,
        )

    def _load_services(
        self, settings: Settings
    ) -> tuple[str, list[str], list[ServiceInput]]:
        """Load system prompt, role identifiers and service definitions.

        Args:
            settings: Global configuration.

        Returns:
            Tuple of system prompt, role IDs and filtered services.

        Side effects:
            Configures prompt and mapping directories based on ``settings``.
        """

        configure_prompt_dir(settings.prompt_dir)
        configure_mapping_data_dir(settings.mapping_data_dir)
        system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)
        role_ids = load_role_ids(Path(self.args.roles_file))
        services = _load_services_list(
            self.args.input_file, self.args.max_services, self.processed_ids
        )
        return system_prompt, role_ids, services

    def _setup_concurrency(
        self, settings: Settings
    ) -> tuple[asyncio.Semaphore, asyncio.Lock]:
        """Create concurrency primitives.

        Args:
            settings: Global configuration.

        Returns:
            Semaphore and lock controlling concurrent access.

        Side effects:
            None.
        """
        concurrency = settings.concurrency
        if concurrency < 1:  # Guard against invalid configuration
            raise ValueError("concurrency must be a positive integer")
        return asyncio.Semaphore(concurrency), asyncio.Lock()

    def _create_progress(self, total: int) -> tqdm | None:
        """Create a progress bar if enabled.

        Args:
            total: Total number of services.

        Returns:
            :class:`tqdm` progress bar or ``None`` when disabled.

        Side effects:
            None.
        """
        if self.args.progress and sys.stdout.isatty():
            return tqdm(total=total)
        # Progress bars are disabled in non-interactive environments.
        return None

    def _prepare_models(
        self,
    ) -> tuple[ModelFactory, str, list[str], list[ServiceInput]]:
        """Initialise shared models and service definitions.

        Returns:
            Model factory, system prompt, role IDs and services.

        Side effects:
            Configures prompt and mapping directories.
        """

        settings = RuntimeEnv.instance().settings
        factory = self._create_model_factory(settings)
        system_prompt, role_ids, services = self._load_services(settings)
        return factory, system_prompt, role_ids, services

    def _init_sessions(
        self, total: int
    ) -> tuple[asyncio.Semaphore, asyncio.Lock, tqdm | None, Path | None, ErrorHandler]:
        """Create concurrency, progress and error-handling helpers.

        Args:
            total: Number of services to process.

        Returns:
            Semaphore, lock, optional progress bar, temporary output directory and
            error handler.

        Side effects:
            May create the temporary output directory.
        """

        settings = RuntimeEnv.instance().settings
        sem, lock = self._setup_concurrency(settings)
        progress = self._create_progress(total)
        temp_output_dir = (
            Path(self.args.temp_output_dir)
            if self.args.temp_output_dir is not None
            else None
        )
        error_handler: ErrorHandler = LoggingErrorHandler()
        return sem, lock, progress, temp_output_dir, error_handler

    async def _generate_evolution(
        self,
        services: list[ServiceInput],
        factory: ModelFactory,
        system_prompt: str,
        role_ids: list[str],
        sem: asyncio.Semaphore,
        lock: asyncio.Lock,
        progress: tqdm | None,
        temp_output_dir: Path | None,
        error_handler: ErrorHandler,
    ) -> bool:
        """Run service executions concurrently.

        Args:
            services: Services to process.
            factory: Shared model factory.
            system_prompt: System prompt for all stages.
            role_ids: Valid role identifiers.
            sem: Semaphore limiting concurrency.
            lock: Lock guarding output writes.
            progress: Progress bar to update or ``None``.
            temp_output_dir: Directory for temporary output artefacts.
            error_handler: Handler for execution errors.

        Returns:
            ``True`` when all executions succeed, ``False`` otherwise.

        Side effects:
            Updates ``self.executions`` and ``self.new_ids``.
        """

        success = True

        async def run_one(service: ServiceInput) -> None:
            nonlocal success
            async with sem:
                execution = ServiceExecution(
                    service,
                    factory=factory,
                    system_prompt=system_prompt,
                    transcripts_dir=self.transcripts_dir,
                    role_ids=role_ids,
                    lock=lock,
                    output=None,
                    new_ids=self.new_ids,
                    temp_output_dir=temp_output_dir,
                    allow_prompt_logging=self.args.allow_prompt_logging,
                    error_handler=error_handler,
                )
                self.executions.append(execution)
                if not await execution.run():
                    success = False
            if progress:
                progress.update(1)

        async with asyncio.TaskGroup() as tg:
            for svc in services:
                tg.create_task(run_one(svc))

        return success

    async def run(self) -> bool:
        """Orchestrate the evolution workflow."""

        with logfire.span("processing_engine.run"):
            logfire.info(
                "Starting processing engine",
                input_file=self.args.input_file,
            )
            factory, system_prompt, role_ids, services = self._prepare_models()
            if self.args.dry_run:
                logfire.info("Validated services", count=len(services))
                return True
            sem, lock, progress, temp_output_dir, error_handler = self._init_sessions(
                len(services)
            )
            success = await self._generate_evolution(
                services,
                factory,
                system_prompt,
                role_ids,
                sem,
                lock,
                progress,
                temp_output_dir,
                error_handler,
            )
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
