"""Command-line interface for service ambitions and evolutions."""

from __future__ import annotations

import argparse
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
from monitoring import init_logfire
from plateau_generator import PlateauGenerator
from settings import load_settings

logger = logging.getLogger(__name__)


def _default_plateaus() -> list[str]:
    """Return plateau names from configuration."""

    return [p.name for p in load_plateau_definitions()[:4]]


def _configure_logging(args: argparse.Namespace, settings) -> None:
    """Configure the logging subsystem."""

    level_name = args.log_level or settings.log_level
    if args.verbose == 1:
        level_name = "INFO"
    elif args.verbose >= 2:
        level_name = "DEBUG"
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO), force=True
    )
    if settings.logfire_token or args.logfire_service:
        init_logfire(args.logfire_service, settings.logfire_token)


def _cmd_generate_ambitions(args: argparse.Namespace, settings) -> None:
    """Generate service ambitions and write them to disk."""

    configure_prompt_dir(args.prompt_dir)
    system_prompt = load_prompt(args.context_id, args.inspirations_id)
    services = list(load_services(args.input_file))
    logger.debug("Loaded %d services from %s", len(services), args.input_file)

    model_name = args.model or settings.model
    logger.info("Generating ambitions using model %s", model_name)

    model = build_model(model_name, settings.openai_api_key)
    generator = ServiceAmbitionGenerator(model, concurrency=args.concurrency)
    generator.generate(services, system_prompt, args.output_file)
    logger.info("Results written to %s", args.output_file)
    logfire.force_flush()


def _cmd_generate_evolution(args: argparse.Namespace, settings) -> None:
    """Generate service evolution summaries."""

    model_name = args.model or settings.model
    model = build_model(model_name, settings.openai_api_key)
    agent = Agent(model)

    with open(args.output_file, "w", encoding="utf-8") as output:
        for service in load_services(args.input_file):
            session = ConversationSession(agent)
            generator = PlateauGenerator(session)
            evolution = generator.generate_service_evolution(
                service, args.plateaus, args.customers
            )
            output.write(f"{evolution.model_dump_json()}\n")
            logger.info("Generated evolution for %s", service.name)
    logfire.force_flush()


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""

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
        "--log-level",
        help="Logging level. Can also be set via the LOG_LEVEL env variable.",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for info, -vv for debug)",
    )
    common.add_argument(
        "--logfire-service",
        help="Enable Logfire telemetry for the given service name",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    amb = subparsers.add_parser(
        "generate-ambitions",
        parents=[common],
        help="Generate service ambitions",
        description="Generate service ambitions",
    )
    amb.add_argument(
        "--prompt-dir", default="prompts", help="Directory containing prompt components"
    )
    amb.add_argument(
        "--context-id", default="university", help="Situational context identifier"
    )
    amb.add_argument(
        "--inspirations-id", default="general", help="Inspirations identifier"
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
        "--concurrency",
        type=int,
        default=5,
        help="Number of services to process concurrently (must be > 0)",
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
        "--plateaus",
        nargs="+",
        default=_default_plateaus(),
        help="Plateau names to evaluate",
    )
    evo.add_argument(
        "--customers",
        nargs="+",
        default=["learners", "staff", "community"],
        help="Customer types to evaluate",
    )
    evo.set_defaults(func=_cmd_generate_evolution)

    args = parser.parse_args()
    settings = load_settings()
    _configure_logging(args, settings)
    args.func(args, settings)


if __name__ == "__main__":
    main()
