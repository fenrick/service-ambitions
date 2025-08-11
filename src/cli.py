"""Command-line interface for generating service ambitions and evolutions."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable

import logfire
from pydantic_ai import Agent
from tqdm import tqdm

from conversation import ConversationSession
from generator import ServiceAmbitionGenerator, build_model
from loader import (
    configure_prompt_dir,
    load_ambition_prompt,
    load_evolution_prompt,
    load_plateau_definitions,
    load_services,
)
from models import ServiceInput
from monitoring import LOG_FILE_NAME, init_logfire
from persistence import atomic_write, read_lines
from plateau_generator import PlateauGenerator
from settings import load_settings

logger = logging.getLogger(__name__)


def _default_plateaus() -> list[str]:
    """Return plateau names from configuration."""

    # Use only the first four plateaus to keep default scope manageable
    return [p.name for p in load_plateau_definitions()[:4]]


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

    # Load prompt components from the configured directory
    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_ambition_prompt(settings.context_id, settings.inspiration)

    # Prefer model specified on the CLI, falling back to settings
    model_name = args.model or settings.model
    logger.info("Generating ambitions using model %s", model_name)

    model = build_model(
        model_name,
        settings.openai_api_key,
        seed=args.seed,
        reasoning=settings.reasoning,
    )
    concurrency = args.concurrency or settings.concurrency
    generator = ServiceAmbitionGenerator(
        model,
        concurrency=concurrency,
        request_timeout=settings.request_timeout,
        retries=settings.retries,
        retry_base_delay=settings.retry_base_delay,
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
            logger.info("Validated %d services", len(services_list or []))
            return

        if show_progress:
            progress = tqdm(total=len(services_list or []))
        else:
            progress = None
        new_ids = await generator.generate_async(
            services, system_prompt, str(part_path), progress=progress
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
    logger.info("Results written to %s", output_path)


async def _cmd_generate_evolution(args: argparse.Namespace, settings) -> None:
    """Generate service evolution summaries."""

    # Allow CLI model override, defaulting to configured model
    model_name = args.model or settings.model
    model = build_model(
        model_name,
        settings.openai_api_key,
        seed=args.seed,
        reasoning=settings.reasoning,
    )

    # Load and assemble the system prompt so each conversation begins with
    # the situational context, definitions and inspirations.
    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)

    concurrency = args.concurrency or settings.concurrency
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")

    semaphore = asyncio.Semaphore(concurrency)

    async def process_service(service: ServiceInput) -> tuple[str, str, str]:
        """Return the evolution JSON for ``service``.

        Args:
            service: Service specification to evolve across plateaus.

        Returns:
            A tuple of ``(service_id, service_name, json_payload)``.

        The function instantiates a fresh :class:`PlateauGenerator` and
        conversation session per service so each evolution runs in isolation and
        does not leak chat history between services.
        """

        async with semaphore:
            agent = Agent(model, instructions=system_prompt)
            session = ConversationSession(agent)
            generator = PlateauGenerator(session)
            evolution = await generator.generate_service_evolution(service)
            return (
                service.service_id,
                service.name,
                evolution.model_dump_json(),
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
        logger.info("Validated %d services", len(services))
        return

    new_ids: set[str] = set()
    total_services = len(services)
    show_progress = args.progress and sys.stdout.isatty()
    progress = tqdm(total=total_services) if show_progress else None

    def _batched(seq: list[ServiceInput], size: int) -> Iterable[list[ServiceInput]]:
        """Yield ``seq`` in lists of at most ``size`` items."""

        it = iter(seq)
        while batch := list(islice(it, size)):
            yield batch

    # Create tasks in manageable chunks to avoid scheduling every service at once.
    batch_size = settings.batch_size or max(1, concurrency * 5)
    with open(part_path, "w", encoding="utf-8") as output:
        for chunk in _batched(services, batch_size):
            tasks = [asyncio.create_task(process_service(svc)) for svc in chunk]
            for task in asyncio.as_completed(tasks):
                svc_id, name, payload = await task
                output.write(f"{payload}\n")
                new_ids.add(svc_id)
                logger.info("Generated evolution for %s", name)
                if progress:
                    progress.update(1)
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


async def main_async() -> None:
    """Parse arguments and dispatch to the requested subcommand."""
    settings = load_settings()
    if settings.logfire_token:
        # Enable logfire integration when a token is provided
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

    # Define subcommands for the supported operations
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
    evo.set_defaults(func=_cmd_generate_evolution)

    # Parse the user's command-line selections
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Configure logging prior to executing the command
    _configure_logging(args, settings)

    # Execute the requested subcommand function
    await args.func(args, settings)


def main() -> None:
    """Entry point for command-line execution."""
    try:
        asyncio.run(main_async())
    finally:
        # Ensure telemetry is flushed even when errors occur
        logfire.force_flush()


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
