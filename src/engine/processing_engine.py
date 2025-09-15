"""High-level service processing engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any, cast

import logfire
from tqdm import tqdm  # type: ignore[import-untyped]

from engine.service_execution import (
    EVOLUTIONS_GENERATED,
    LINES_WRITTEN,
    ServiceExecution,
)
from engine.service_runtime import ServiceRuntime
from io_utils.loader import (
    configure_mapping_data_dir,
    configure_prompt_dir,
    load_evolution_prompt,
    load_plateau_definitions,
    load_role_ids,
)
from io_utils.persistence import atomic_write, read_lines
from io_utils.service_loader import load_services
from models import ServiceInput
from models.factory import ModelFactory
from runtime.environment import RuntimeEnv
from runtime.settings import Settings
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
    with logfire.span(
        "processing_engine.prepare_paths",
        attributes={"output": str(output), "resume": resume},
    ):
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
    with logfire.span(
        "processing_engine.load_resume_state",
        attributes={
            "processed_path": str(processed_path),
            "output_path": str(output_path),
            "resume": resume,
        },
    ):
        processed_ids = set(read_lines(processed_path)) if resume else set()
        existing_lines = read_lines(output_path) if resume else []
        return processed_ids, existing_lines


def _load_resume_meta(state_path: Path, resume: bool) -> dict[str, Any]:
    """Return metadata from a previous run when resuming."""
    with logfire.span(
        "processing_engine.load_resume_meta",
        attributes={"state_path": str(state_path), "resume": resume},
    ):
        if not resume:
            return {}
        lines = read_lines(state_path)
        if not lines:
            raise FileNotFoundError(state_path)
        try:
            return cast(dict[str, Any], json.loads(lines[0]))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid resume metadata: {state_path}") from exc


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
    with logfire.span(
        "processing_engine.ensure_transcripts_dir",
        attributes={"path": path, "output": str(output)},
    ):
        transcripts_dir = (
            Path(path) if path is not None else output.parent / "_transcripts"
        )
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
    with logfire.span(
        "processing_engine.load_services_list",
        attributes={
            "input_file": input_file,
            "max_services": max_services if max_services is not None else 0,
            "processed_ids": len(processed_ids),
        },
    ):
        with load_services(Path(input_file)) as svc_iter:
            if max_services is not None:
                svc_iter = islice(svc_iter, max_services)
            result = [s for s in svc_iter if s.service_id not in processed_ids]
            logfire.debug("Loaded service list", count=len(result))
            return result


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
        self.state_path = self.output_path.with_name("resume_state.json")
        input_path = Path(args.input_file)
        if args.resume and not input_path.exists():
            raise FileNotFoundError(input_path)
        self.input_hash = (
            sha256(input_path.read_bytes()).hexdigest() if input_path.exists() else ""
        )
        meta = _load_resume_meta(self.state_path, args.resume)
        if args.resume:
            if meta.get("input_hash") != self.input_hash:
                raise ValueError("Resume refused: input file has changed")
            if meta.get("output_path") != str(self.output_path):
                raise ValueError("Resume refused: output path has changed")
        if self.transcripts_dir is None:
            self.transcripts_dir = _ensure_transcripts_dir(
                args.transcripts_dir, self.output_path
            )
        self.new_ids: set[str] = set()
        self.runtimes: list[ServiceRuntime] = []
        self.success = False
        self.factory: ModelFactory | None = None
        self.system_prompt: str | None = None
        self.role_ids: list[str] | None = None
        self.services: list[ServiceInput] | None = None
        self.sem: asyncio.Semaphore | None = None
        self.progress: tqdm | None = None
        self.temp_output_dir: Path | None = None
        self.error_handler: ErrorHandler | None = None
        # Cache runtime settings to avoid repeated environment lookups
        self.settings: Settings = RuntimeEnv.instance().settings

    def refresh_settings(self) -> None:
        """Refresh cached settings from :class:`RuntimeEnv`.

        Side effects:
            Updates ``self.settings`` to reflect the current runtime
            configuration.
        """
        self.settings = RuntimeEnv.instance().settings

    def _create_model_factory(self) -> ModelFactory:
        """Create a model factory from ``self.settings``."""
        settings = self.settings
        return ModelFactory(
            settings.model,
            settings.openai_api_key,
            stage_models=getattr(settings, "models", None),
            reasoning=settings.reasoning,
            seed=self.args.seed,
            web_search=settings.web_search,
        )

    def _load_services(self) -> tuple[str, list[str], list[ServiceInput]]:
        """Load system prompt, role identifiers and service definitions."""
        settings = self.settings
        configure_prompt_dir(settings.prompt_dir)
        configure_mapping_data_dir(settings.mapping_data_dir)
        system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)
        role_ids = load_role_ids(self.settings.roles_file)
        services = _load_services_list(
            self.args.input_file, self.args.max_services, self.processed_ids
        )
        return system_prompt, role_ids, services

    def _setup_concurrency(self) -> asyncio.Semaphore:
        """Create a semaphore limiting concurrent executions."""
        concurrency = self.settings.concurrency
        if concurrency < 1:  # Guard against invalid configuration
            raise ValueError("concurrency must be a positive integer")
        return asyncio.Semaphore(concurrency)

    def _create_progress(self, total: int) -> tqdm | None:
        """Create a progress bar if enabled.

        Args:
            total: Total number of progress ticks.

        Returns:
            :class:`tqdm` progress bar or ``None`` when disabled.

        Side effects:
            None.
        """
        if self.args.progress and sys.stdout.isatty() and not self.args.json_logs:
            return tqdm(total=total)
        # Progress bars are disabled in non-interactive environments or when
        # structured logs are requested.
        return None

    def _prepare_models(self) -> None:
        """Initialise shared models and service definitions."""
        self.factory = self._create_model_factory()
        system_prompt, role_ids, services = self._load_services()
        self.system_prompt = system_prompt
        self.role_ids = role_ids
        self.services = services

    def _init_sessions(self) -> None:
        """Create concurrency, progress and error-handling helpers.

        The total number of progress ticks is derived from ``self.services``
        multiplied by the number of LLM calls per service.
        """
        # Progress advances per LLM call. Compute dynamically based on the
        # number of plateaus (P) and mapping sets (M):
        #   1 (descriptions) + P (features) + P*M (mappings) + 1 (assembly)
        # = P*(M + 1) + 2 per service.
        services_count = len(self.services or [])
        try:
            plateaus_count = len(
                load_plateau_definitions(self.settings.mapping_data_dir)
            )
        except Exception:
            plateaus_count = 0
        mapping_sets_count = len(getattr(self.settings, "mapping_sets", []) or [])
        per_service_ticks = plateaus_count * (mapping_sets_count + 1) + 2
        total = services_count * max(per_service_ticks, 1)
        self.sem = self._setup_concurrency()
        self.progress = self._create_progress(total)
        self.temp_output_dir = (
            Path(self.args.temp_output_dir)
            if self.args.temp_output_dir is not None
            else None
        )
        self.error_handler = LoggingErrorHandler()

    async def _run_service(self, service: ServiceInput) -> bool:
        """Execute ``service`` and update runtime state.

        Args:
            service: Service to evolve.

        Returns:
            ``True`` when the execution succeeds, ``False`` otherwise.
        """
        # Validate prerequisites before running the service.
        if self.factory is None:
            raise RuntimeError("Model factory is not initialised")
        if self.system_prompt is None:
            raise RuntimeError("System prompt is not loaded")
        if self.role_ids is None:
            raise RuntimeError("Role identifiers are not loaded")
        if self.sem is None:
            raise RuntimeError("Concurrency semaphore is not configured")
        if self.error_handler is None:
            raise RuntimeError("Error handler is not configured")
        async with self.sem:
            with logfire.span(
                "processing_engine.run_service",
                attributes={"service_id": service.service_id},
            ):
                runtime = ServiceRuntime(service)
                execution = ServiceExecution(
                    runtime,
                    factory=self.factory,
                    system_prompt=self.system_prompt,
                    transcripts_dir=self.transcripts_dir,
                    role_ids=self.role_ids,
                    temp_output_dir=self.temp_output_dir,
                    allow_prompt_logging=self.args.allow_prompt_logging,
                    error_handler=self.error_handler,
                    progress=self.progress,
                )
                success = await execution.run()
                self.runtimes.append(runtime)
        # Progress updates are handled per LLM call within ServiceExecution and
        # underlying generators; no per-service tick here.
        return success

    async def _generate_evolution(self) -> bool:
        """Run service executions concurrently.

        Returns:
            ``True`` when all executions succeed, ``False`` otherwise.

        Side effects:
            Updates ``self.executions`` and ``self.new_ids``.

        Notes:
            Operates on services stored in ``self.services``.
        """
        services = self.services or []
        with logfire.span(
            "processing_engine.generate_evolution", attributes={"count": len(services)}
        ):
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._run_service(svc)) for svc in services]
            return all(task.result() for task in tasks)

    async def run(self) -> bool:
        """Execute the full processing pipeline.

        Returns:
            ``True`` when all service executions succeed. Results are stored on
            per-service runtimes and written out by :meth:`finalise`.
        """
        self.refresh_settings()
        with logfire.span("processing_engine.run"):
            logfire.info(
                "Starting processing engine",
                input_file=self.args.input_file,
            )
            self._prepare_models()
            self._init_sessions()
            try:
                success = await self._generate_evolution()
            except Exception as exc:
                # When running in dry-run mode, allow graceful halt when an agent
                # invocation would be required (cache miss). Otherwise re-raise.
                if getattr(self.settings, "dry_run", False):
                    try:
                        # Try to introspect DryRunInvocation details if available
                        from core.dry_run import DryRunInvocation

                        if isinstance(exc, DryRunInvocation):
                            cache_path = str(exc.cache_file) if exc.cache_file else ""
                            logfire.info(
                                "Dry-run halted before agent invocation",
                                stage=exc.stage,
                                model=exc.model,
                                service_id=exc.service_id,
                                cache_file=cache_path,
                            )
                        else:
                            logfire.info("Dry-run halted", error=str(exc))
                    except Exception:  # pragma: no cover - defensive
                        logfire.info("Dry-run halted", error=str(exc))
                    self.success = False
                    return False
                raise
            if self.progress:
                self.progress.close()
            self.success = success
            logfire.info("Processing engine completed", success=success)
            return success

    def _save_results(self) -> None:
        """Persist generated lines and update processed IDs.

        Side effects:
            Moves or writes files on disk.
        """
        if self.args.resume:
            # Merge new output with existing lines when resuming.
            new_lines = read_lines(self.part_path)
            atomic_write(self.output_path, [*self.existing_lines, *new_lines])
            self.part_path.unlink(missing_ok=True)
            self.processed_ids.update(self.new_ids)
        else:
            # Move temp file into final destination on fresh runs.
            os.replace(self.part_path, self.output_path)
            self.processed_ids = set(self.new_ids)
        atomic_write(self.processed_path, sorted(self.processed_ids))
        settings_dump = (
            self.settings.model_dump()
            if hasattr(self.settings, "model_dump")
            else vars(self.settings)
        )
        state = {
            "input_hash": self.input_hash,
            "output_path": str(self.output_path),
            "settings": settings_dump,
        }
        atomic_write(self.state_path, [json.dumps(state, default=str)])

    async def finalise(self) -> None:
        """Write successful results to disk."""
        with logfire.span("processing_engine.finalise"):
            if not self.runtimes:
                logfire.debug("No executions to finalise")
                return
            output = await asyncio.to_thread(self.part_path.open, "w", encoding="utf-8")
            try:
                for runtime in self.runtimes:
                    if not runtime.success:
                        continue
                    # Successful runtimes must produce an output line.
                    if runtime.line is None:
                        raise RuntimeError(
                            "Runtime completed successfully without producing a line"
                        )
                    await asyncio.to_thread(output.write, f"{runtime.line}\n")
                    self.new_ids.add(runtime.service.service_id)
                    EVOLUTIONS_GENERATED.add(1)
                    LINES_WRITTEN.add(1)
                    logfire.info(
                        "Generated evolution",
                        service_id=runtime.service.service_id,
                        output_path=getattr(output, "name", ""),
                    )
            finally:
                await asyncio.to_thread(output.close)
            self._save_results()
            logfire.info("Finalised processing engine", output=str(self.output_path))


__all__ = ["ProcessingEngine"]
