"""Command-line interface for generating service ambitions and evolutions."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Coroutine, Sequence, cast

from pydantic_ai import Agent
from tqdm import tqdm

from conversation import ConversationSession
from diagnostics import validate_jsonl
from generator import AmbitionModel, ServiceAmbitionGenerator
from loader import (
    configure_prompt_dir,
    load_ambition_prompt,
    load_evolution_prompt,
    load_role_ids,
)
from mapping import init_embeddings
from model_factory import ModelFactory
from models import ServiceInput
from monitoring import LOG_FILE_NAME, init_logfire, logfire
from persistence import atomic_write, read_lines
from plateau_generator import PlateauGenerator
from service_loader import load_services
from settings import load_settings

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
EVOLUTIONS_GENERATED = logfire.metric_counter("evolutions_generated")
LINES_WRITTEN = logfire.metric_counter("lines_written")


def _configure_logging(args: argparse.Namespace, settings) -> None:
    """Configure the logging subsystem."""

    # CLI-specified level takes precedence over configured default
    level_name = settings.log_level

    if args.verbose == 1:
        # Single -v flag bumps log level to INFO for clearer output
        level_name = "INFO"
    elif args.verbose >= 2:
        # Two or more -v flags enable DEBUG for deep troubleshooting
        level_name = "DEBUG"

    logging.basicConfig(
        filename=LOG_FILE_NAME,
        level=getattr(logging, level_name.upper(), logging.INFO),
        force=True,
    )
    if settings.logfire_token:
        # Initialize logfire only when a token is configured
        init_logfire(settings.logfire_token)


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


async def _generate_evolution_for_service(
    service: ServiceInput,
    *,
    factory: ModelFactory,
    settings,
    args: argparse.Namespace,
    system_prompt: str,
    transcripts_dir: Path,
    role_ids: Sequence[str],
    mapping_batch_size: int,
    mapping_parallel_types: bool,
    mapping_strict: bool,
    lock: asyncio.Lock,
    output,
    new_ids: set[str],
) -> None:
    """Generate evolution for ``service`` and record results."""

    desc_name = factory.model_name(
        "descriptions", args.descriptions_model or args.model
    )
    feat_name = factory.model_name("features", args.features_model or args.model)
    map_name = factory.model_name("mapping", args.mapping_model or args.model)
    attrs = {
        "service_id": service.service_id,
        "service_name": service.name,
        "descriptions_model": desc_name,
        "features_model": feat_name,
        "mapping_model": map_name,
        "output_path": getattr(output, "name", ""),
    }
    with logfire.span("generate_evolution_for_service", attributes=attrs):
        try:
            SERVICES_PROCESSED.add(1)
            desc_model = factory.get(
                "descriptions", args.descriptions_model or args.model
            )
            feat_model = factory.get("features", args.features_model or args.model)
            map_model = factory.get("mapping", args.mapping_model or args.model)

            desc_agent = Agent(desc_model, instructions=system_prompt)
            feat_agent = Agent(feat_model, instructions=system_prompt)
            map_agent = Agent(map_model, instructions=system_prompt)

            desc_session = ConversationSession(desc_agent, stage="descriptions")
            feat_session = ConversationSession(feat_agent, stage="features")
            map_session = ConversationSession(map_agent, stage="mapping")
            generator = PlateauGenerator(
                feat_session,
                required_count=settings.features_per_role,
                roles=role_ids,
                description_session=desc_session,
                mapping_session=map_session,
                mapping_batch_size=mapping_batch_size,
                mapping_parallel_types=mapping_parallel_types,
                mapping_strict=mapping_strict,
            )
            evolution = await generator.generate_service_evolution_async(
                service, transcripts_dir=transcripts_dir
            )
            line = f"{evolution.model_dump_json()}\n"
            async with lock:
                output.write(line)
                new_ids.add(service.service_id)
                EVOLUTIONS_GENERATED.add(1)
                LINES_WRITTEN.add(1)
            logfire.info(
                "Generated evolution",
                service_id=service.service_id,
                output_path=getattr(output, "name", ""),
            )
        except Exception as exc:  # noqa: BLE001
            quarantine_dir = Path("quarantine")
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            quarantine_file = quarantine_dir / f"{service.service_id}.json"
            quarantine_file.write_text(service.model_dump_json(indent=2))
            logfire.exception(
                "Failed to generate evolution",
                service_id=service.service_id,
                error=str(exc),
                quarantine_file=str(quarantine_file),
            )


async def _cmd_generate_ambitions(args: argparse.Namespace, settings) -> None:
    """Generate service ambitions and write them to disk."""
    output_path = Path(args.output_file)
    attrs = {"output_path": str(output_path), "resume": args.resume}
    with logfire.span("cmd_generate_ambitions", attributes=attrs) as span:
        try:
            if args.validate_only:
                count = validate_jsonl(output_path, AmbitionModel)
                logfire.info(
                    "Validated output",
                    lines=count,
                    output_path=str(output_path),
                )
                return

            # Load prompt components from the configured directory
            configure_prompt_dir(settings.prompt_dir)
            system_prompt = load_ambition_prompt(
                settings.context_id, settings.inspiration
            )

            use_web_search = (
                args.web_search if args.web_search is not None else settings.web_search
            )
            factory = ModelFactory(
                settings.model,
                settings.openai_api_key,
                stage_models=getattr(settings, "models", None),
                reasoning=settings.reasoning,
                seed=args.seed,
                web_search=use_web_search,
            )

            model_name = factory.model_name(
                "features", args.features_model or args.model
            )
            span.set_attribute("model_name", model_name)
            logfire.info("Generating ambitions", model=model_name)
            model = factory.get("features", args.features_model or args.model)
            concurrency = args.concurrency or settings.concurrency
            token_weighting = (
                args.token_weighting
                if args.token_weighting is not None
                else settings.token_weighting
            )
            generator = ServiceAmbitionGenerator(
                model,
                concurrency=concurrency,
                batch_size=settings.batch_size,
                request_timeout=settings.request_timeout,
                retries=settings.retries,
                retry_base_delay=settings.retry_base_delay,
                expected_output_tokens=args.expected_output_tokens,
                token_weighting=token_weighting,
            )

            part_path, processed_path = _prepare_paths(output_path, args.resume)
            processed_ids, existing_lines = _load_resume_state(
                processed_path, output_path, args.resume
            )
            transcripts_dir = _ensure_transcripts_dir(args.transcripts_dir, output_path)
            services = _load_services_list(
                args.input_file,
                args.max_services,
                processed_ids if args.resume else set(),
            )

            if args.dry_run:
                logfire.info(
                    "Validated services",
                    count=len(services),
                    resume=args.resume,
                )
                return

            show_progress = args.progress and sys.stdout.isatty()
            progress = tqdm(total=len(services)) if show_progress else None
            new_ids = await generator.generate_async(
                services,
                system_prompt,
                str(part_path),
                progress=progress,
                transcripts_dir=transcripts_dir,
            )
            if progress:
                progress.close()

            SERVICES_PROCESSED.add(len(new_ids))
            LINES_WRITTEN.add(len(new_ids))

            processed_ids = _save_results(
                resume=args.resume,
                part_path=part_path,
                output_path=output_path,
                existing_lines=existing_lines,
                processed_ids=processed_ids,
                new_ids=new_ids,
                processed_path=processed_path,
            )
            logfire.info(
                "Results written",
                output_path=str(output_path),
                lines_written=len(new_ids),
                resume=args.resume,
            )
        except Exception as exc:  # noqa: BLE001
            logfire.exception(
                "Ambition generation failed",
                output_path=str(output_path),
                resume=args.resume,
                error=str(exc),
            )
            raise


async def _cmd_generate_evolution(args: argparse.Namespace, settings) -> None:
    """Generate service evolution summaries."""

    use_web_search = (
        args.web_search if args.web_search is not None else settings.web_search
    )
    factory = ModelFactory(
        settings.model,
        settings.openai_api_key,
        stage_models=getattr(settings, "models", None),
        reasoning=settings.reasoning,
        seed=args.seed,
        web_search=use_web_search,
    )

    # Warm mapping embeddings upfront so subsequent requests reuse cached vectors.
    # Failures are logged by ``init_embeddings`` and do not interrupt startup.
    await init_embeddings()

    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)

    role_ids = load_role_ids(Path(args.roles_file))
    mapping_batch_size = args.mapping_batch_size or settings.mapping_batch_size
    mapping_parallel_types = (
        args.mapping_parallel_types
        if args.mapping_parallel_types is not None
        else settings.mapping_parallel_types
    )
    mapping_strict = (
        args.mapping_strict
        if args.mapping_strict is not None
        else settings.mapping_strict
    )

    output_path = Path(args.output_file)
    part_path, processed_path = _prepare_paths(output_path, args.resume)
    processed_ids, existing_lines = _load_resume_state(
        processed_path, output_path, args.resume
    )
    transcripts_dir = _ensure_transcripts_dir(args.transcripts_dir, output_path)
    services = _load_services_list(args.input_file, args.max_services, processed_ids)

    if args.dry_run:
        logfire.info(f"Validated {len(services)} services")
        return

    concurrency = args.concurrency or settings.concurrency
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    new_ids: set[str] = set()
    show_progress = args.progress and sys.stdout.isatty()
    progress = tqdm(total=len(services)) if show_progress else None

    async def run_one(service: ServiceInput) -> None:
        async with sem:
            await _generate_evolution_for_service(
                service,
                factory=factory,
                settings=settings,
                args=args,
                system_prompt=system_prompt,
                transcripts_dir=transcripts_dir,
                role_ids=role_ids,
                mapping_batch_size=mapping_batch_size,
                mapping_parallel_types=mapping_parallel_types,
                mapping_strict=mapping_strict,
                lock=lock,
                output=output,
                new_ids=new_ids,
            )
            if progress:
                progress.update(1)

    with open(part_path, "w", encoding="utf-8") as output:
        async with asyncio.TaskGroup() as tg:
            for service in services:
                tg.create_task(run_one(service))
    if progress:
        progress.close()

    processed_ids = _save_results(
        resume=args.resume,
        part_path=part_path,
        output_path=output_path,
        existing_lines=existing_lines,
        processed_ids=processed_ids,
        new_ids=new_ids,
        processed_path=processed_path,
    )


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""

    settings = load_settings()
    if settings.logfire_token:
        init_logfire(settings.logfire_token)

    parser = argparse.ArgumentParser(
        description="Service ambitions utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        help=(
            "Global chat model name (default openai:gpt-5). "
            "Can also be set via the MODEL env variable."
        ),
    )
    common.add_argument(
        "--descriptions-model",
        help="Model for plateau descriptions (default openai:o4-mini)",
    )
    common.add_argument(
        "--features-model",
        help=(
            "Model for feature generation (default openai:gpt-5; "
            "use openai:o4-mini for lower cost)"
        ),
    )
    common.add_argument(
        "--mapping-model",
        help="Model for feature mapping (default openai:o4-mini)",
    )
    common.add_argument(
        "--search-model",
        help="Model for web search (default openai:gpt-4o-search-preview)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for info, -vv for debug)",
    )
    common.add_argument(
        "--concurrency",
        type=int,
        help="Number of services to process concurrently",
    )
    common.add_argument(
        "--max-services",
        type=int,
        help="Process at most this many services",
    )
    common.add_argument(
        "--mapping-batch-size",
        type=int,
        help="Number of features per mapping request batch",
    )
    common.add_argument(
        "--mapping-parallel-types",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable parallel mapping type requests",
    )
    common.add_argument(
        "--mapping-strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Raise an error when mappings remain empty after retries",
    )
    common.add_argument(
        "--token-weighting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable token-based concurrency weighting",
    )
    common.add_argument(
        "--seed",
        type=int,
        help="Seed random number generation for reproducible output",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without calling the API",
    )
    common.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar during execution",
    )
    common.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume processing using processed_ids.txt",
    )
    common.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable web search when prompts need external lookups. "
            "Adds latency and cost"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    amb = subparsers.add_parser(
        "generate-ambitions",
        parents=[common],
        help="Generate service ambitions",
        description="Generate service ambitions",
    )
    amb.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help="Path to the services JSONL file",
    )
    amb.add_argument(
        "--output-file",
        default="ambitions.jsonl",
        help="File to write the results",
    )
    amb.add_argument(
        "--transcripts-dir",
        help=(
            "Directory to store per-service request/response transcripts. "
            "Defaults to a '_transcripts' folder beside the output file."
        ),
    )
    amb.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output file and exit",
    )
    amb.add_argument(
        "--expected-output-tokens",
        type=int,
        default=256,
        help="Anticipated tokens per response for concurrency tuning",
    )
    amb.set_defaults(func=_cmd_generate_ambitions)

    evo = subparsers.add_parser(
        "generate-evolution",
        parents=[common],
        help="Generate service evolution",
    )
    evo.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help="Path to the services JSONL file",
    )
    evo.add_argument(
        "--output-file",
        default="evolution.jsonl",
        help="File to write the results",
    )
    evo.add_argument(
        "--transcripts-dir",
        help=(
            "Directory to store per-service request/response transcripts. "
            "Defaults to a '_transcripts' folder beside the output file."
        ),
    )
    evo.add_argument(
        "--roles-file",
        default="data/roles.json",
        help="Path to the roles definition JSON file",
    )
    evo.set_defaults(func=_cmd_generate_evolution)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    _configure_logging(args, settings)

    result = args.func(args, settings)
    if inspect.isawaitable(result):
        # Cast ensures that asyncio.run receives a proper Coroutine
        asyncio.run(cast(Coroutine[Any, Any, Any], result))

    logfire.force_flush()


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
