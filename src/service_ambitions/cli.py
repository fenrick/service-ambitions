"""Command-line interface for service ambitions generation."""

from __future__ import annotations

import argparse
import logging

from .generator import ServiceAmbitionGenerator, build_model
from .loader import load_prompt, load_services
from .monitoring import init_logfire
from .settings import load_settings

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and generate ambitions for each service."""

    parser = argparse.ArgumentParser(description="Generate service ambitions")
    parser.add_argument(
        "--prompt-dir",
        default="prompts",
        help="Directory containing prompt components",
    )
    parser.add_argument(
        "--context-id",
        default="university",
        help="Situational context identifier",
    )
    parser.add_argument(
        "--inspirations-id",
        default="general",
        help="Inspirations identifier",
    )
    parser.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help="Path to the services JSONL file",
    )
    parser.add_argument(
        "--output-file", default="ambitions.jsonl", help="File to write the results"
    )
    parser.add_argument(
        "--model",
        help="Chat model name. Can also be set via the MODEL env variable.",
    )
    parser.add_argument(
        "--log-level",
        help="Logging level. Can also be set via the LOG_LEVEL env variable.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of services to process concurrently",
    )
    parser.add_argument(
        "--logfire-service",
        help="Enable Logfire telemetry for the given service name",
    )
    args = parser.parse_args()

    settings = load_settings()

    log_level = args.log_level or settings.log_level
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    api_key = settings.openai_api_key

    if settings.logfire_token or args.logfire_service:
        init_logfire(args.logfire_service, settings.logfire_token)

    system_prompt = load_prompt(args.prompt_dir, args.context_id, args.inspirations_id)
    services = list(load_services(args.input_file))

    model_name = args.model or settings.model

    try:
        model = build_model(model_name, api_key)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to initialize model %s: %s", model_name, exc)
        raise

    generator = ServiceAmbitionGenerator(model, concurrency=args.concurrency)
    generator.generate(services, system_prompt, args.output_file)
