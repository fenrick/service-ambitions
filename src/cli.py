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
from typing import Any, Coroutine, Iterable, cast

import logfire  # type: ignore[import-not-found]
from pydantic_ai import Agent
from tqdm import tqdm

from conversation import ConversationSession
from diagnostics import validate_jsonl
from generator import AmbitionModel, ServiceAmbitionGenerator
from loader import (
    configure_prompt_dir,
    load_ambition_prompt,
    load_evolution_prompt,
    load_plateau_definitions,
    load_roles,
)
from model_factory import ModelFactory
from models import ServiceInput
from monitoring import LOG_FILE_NAME, init_logfire
from persistence import atomic_write, read_lines
from plateau_generator import PlateauGenerator
from service_loader import load_services
from settings import load_settings


def _default_plateaus() -> list[str]:
    """Return all plateau names from configuration."""

    return [p.name for p in load_plateau_definitions()]


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


async def _cmd_generate_ambitions(args: argparse.Namespace, settings) -> None:
    """Generate service ambitions and write them to disk."""

    output_path = Path(args.output_file)
    if args.validate_only:
        count = validate_jsonl(output_path, AmbitionModel)
        logfire.info(f"Validated {count} lines in {output_path}")
        return

    # Load prompt components from the configured directory
    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_ambition_prompt(settings.context_id, settings.inspiration)

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

    feat_name = factory.model_name("features", args.features_model or args.model)
    logfire.info(f"Generating evolution using features model {feat_name}")
    model_name = factory.model_name("features", args.features_model or args.model)
    logfire.info(f"Generating ambitions using model {model_name}")
    model = factory.get("features", args.features_model or args.model)
    concurrency = args.concurrency or settings.concurrency
    generator = ServiceAmbitionGenerator(
        model,
        concurrency=concurrency,
        batch_size=settings.batch_size,
        request_timeout=settings.request_timeout,
        retries=settings.retries,
        retry_base_delay=settings.retry_base_delay,
        expected_output_tokens=args.expected_output_tokens,
        token_weighting=settings.token_weighting,
    )

    part_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
        if not args.resume
        else output_path.suffix + ".tmp.part"
    )
    processed_path = output_path.with_name("processed_ids.txt")

    processed_ids: set[str] = set(read_lines(processed_path)) if args.resume else set()
    existing_lines: list[str] = read_lines(output_path) if args.resume else []
    transcripts_dir = (
        Path(args.transcripts_dir)
        if args.transcripts_dir is not None
        else output_path.parent / "_transcripts"
    )
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    with load_services(Path(args.input_file)) as svc_iter:
        if args.max_services is not None:
            # Limit processing to the requested number of services.
            svc_iter = islice(svc_iter, args.max_services)

        if args.resume:
            svc_iter = (s for s in svc_iter if s.service_id not in processed_ids)

        show_progress = args.progress and sys.stdout.isatty()
        services_list: list[ServiceInput] | None = None
        services: Iterable[ServiceInput]
        if args.dry_run or show_progress:
            services_list = list(svc_iter)
            services = services_list
        else:
            services = svc_iter

        if args.dry_run:
            logfire.info(f"Validated {len(services_list or [])} services")
            return

        if show_progress:
            progress = tqdm(total=len(services_list or []))
        else:
            progress = None
        new_ids = await generator.generate_async(
            services,
            system_prompt,
            str(part_path),
            progress=progress,
            transcripts_dir=transcripts_dir,
        )
        if progress:
            progress.close()

    if args.resume:
        new_lines = read_lines(part_path)
        atomic_write(output_path, [*existing_lines, *new_lines])
        part_path.unlink(missing_ok=True)
        processed_ids.update(new_ids)
    else:
        os.replace(part_path, output_path)
        processed_ids = new_ids

    atomic_write(processed_path, sorted(processed_ids))
    logfire.info(f"Results written to {output_path}")


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

    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)

    roles = load_roles(Path(args.roles_file))
    role_ids = [r.role_id for r in roles]
    mapping_batch_size = args.mapping_batch_size or settings.mapping_batch_size
    mapping_parallel_types = (
        args.mapping_parallel_types
        if args.mapping_parallel_types is not None
        else settings.mapping_parallel_types
    )

    output_path = Path(args.output_file)
    part_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
        if not args.resume
        else output_path.suffix + ".tmp.part"
    )
    processed_path = output_path.with_name("processed_ids.txt")

    processed_ids: set[str] = set(read_lines(processed_path)) if args.resume else set()
    existing_lines: list[str] = read_lines(output_path) if args.resume else []

    with load_services(Path(args.input_file)) as svc_iter:
        if args.max_services is not None:
            svc_iter = islice(svc_iter, args.max_services)
        services = [s for s in svc_iter if s.service_id not in processed_ids]

    if args.dry_run:
        logfire.info(f"Validated {len(services)} services")
        return

    concurrency = args.concurrency or settings.concurrency
    if concurrency < 1:
        # A zero semaphore would deadlock all tasks, so fail fast on invalid input.
        raise ValueError("concurrency must be a positive integer")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    new_ids: set[str] = set()
    show_progress = args.progress and sys.stdout.isatty()
    progress = tqdm(total=len(services)) if show_progress else None

    async def run_one(service: ServiceInput) -> None:
        async with sem:
            try:
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
                )
                evolution = await generator.generate_service_evolution_async(service)
                line = f"{evolution.model_dump_json()}\n"
                async with lock:
                    output.write(line)
                    new_ids.add(service.service_id)
                logfire.info(f"Generated evolution for {service.name}")
            except Exception as exc:  # noqa: BLE001
                quarantine_dir = Path("quarantine")
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                quarantine_file = quarantine_dir / f"{service.service_id}.json"
                quarantine_file.write_text(service.model_dump_json(indent=2))
                logfire.error(f"Failed to generate evolution for {service.name}: {exc}")
            finally:
                if progress:
                    progress.update(1)

    with open(part_path, "w", encoding="utf-8") as output:
        async with asyncio.TaskGroup() as tg:
            for service in services:
                tg.create_task(run_one(service))
    if progress:
        progress.close()

    if args.resume:
        new_lines = read_lines(part_path)
        atomic_write(output_path, [*existing_lines, *new_lines])
        part_path.unlink(missing_ok=True)
        processed_ids.update(new_ids)
    else:
        os.replace(part_path, output_path)
        processed_ids = new_ids

    atomic_write(processed_path, sorted(processed_ids))


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
        help="Chat model name. Can also be set via the MODEL env variable.",
    )
    common.add_argument(
        "--descriptions-model",
        help="Model for plateau descriptions",
    )
    common.add_argument(
        "--features-model",
        help="Model for feature generation",
    )
    common.add_argument(
        "--mapping-model",
        help="Model for feature mapping",
    )
    common.add_argument(
        "--search-model",
        help="Model for web search",
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
        help="Enable or disable web search for model browsing",
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
        "--mapping-batch-size",
        type=int,
        help="Number of features per mapping request batch",
    )
    evo.add_argument(
        "--mapping-parallel-types",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable parallel mapping type requests",
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
