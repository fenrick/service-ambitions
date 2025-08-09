"""Command-line interface for generating service ambitions and evolutions."""

from __future__ import annotations

import argparse
import asyncio
import logging

import logfire
from pydantic_ai import Agent

from conversation import ConversationSession
from generator import ServiceAmbitionGenerator, build_model
from loader import (
    configure_prompt_dir,
    load_plateau_definitions,
    load_prompt,
    load_services,
)
from models import ServiceInput
from monitoring import init_logfire
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
        level=getattr(logging, level_name.upper(), logging.INFO), force=True
    )
    if settings.logfire_token:
        # Initialize logfire only when a token is configured
        init_logfire(settings.logfire_token)


async def _cmd_generate_ambitions(args: argparse.Namespace, settings) -> None:
    """Generate service ambitions and write them to disk."""

    # Load prompt components from the configured directory
    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_prompt(settings.context_id, settings.inspiration)
    with load_services(args.input_file) as service_iter:
        services = list(service_iter)
    logger.debug("Loaded %d services from %s", len(services), args.input_file)

    # Prefer model specified on the CLI, falling back to settings
    model_name = args.model or settings.model
    logger.info("Generating ambitions using model %s", model_name)

    model = build_model(model_name, settings.openai_api_key)
    generator = ServiceAmbitionGenerator(model, concurrency=settings.concurrency)
    await generator.generate(services, system_prompt, args.output_file)
    logger.info("Results written to %s", args.output_file)


async def _cmd_generate_evolution(args: argparse.Namespace, settings) -> None:
    """Generate service evolution summaries."""

    # Allow CLI model override, defaulting to configured model
    model_name = args.model or settings.model
    model = build_model(model_name, settings.openai_api_key)

    # Load and assemble the system prompt so each conversation begins with
    # the situational context, definitions and inspirations.
    configure_prompt_dir(settings.prompt_dir)
    system_prompt = load_prompt(settings.context_id, settings.inspiration)

    if settings.concurrency < 1:
        raise ValueError("concurrency must be a positive integer")

    semaphore = asyncio.Semaphore(settings.concurrency)

    async def process_service(service: ServiceInput) -> tuple[str, str]:
        """Return the evolution JSON for ``service``.

        Args:
            service: Service specification to evolve across plateaus.

        Returns:
            A tuple containing the service name and its generated evolution in
            JSON format.

        The function instantiates a fresh :class:`PlateauGenerator` and
        conversation session per service so each evolution runs in isolation and
        does not leak chat history between services.
        """

        async with semaphore:
            agent = Agent(model, instructions=system_prompt)
            session = ConversationSession(agent)
            generator = PlateauGenerator(session)
            evolution = await generator.generate_service_evolution(service)
            return service.name, evolution.model_dump_json()

    with (
        load_services(args.input_file) as services,
        open(args.output_file, "w", encoding="utf-8") as output,
    ):
        tasks = [asyncio.create_task(process_service(svc)) for svc in services]
        for task in asyncio.as_completed(tasks):
            name, payload = await task
            output.write(f"{payload}\n")
            logger.info("Generated evolution for %s", name)


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

    # Configure logging prior to executing the command
    _configure_logging(args, settings)

    # Execute the requested subcommand function
    await args.func(args, settings)

    # Flush telemetry once the command completes
    logfire.force_flush()


def main() -> None:
    """Entry point for command-line execution."""
    asyncio.run(main_async())


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
