"""Command-line interface for service ambitions generation."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from .generator import ServiceAmbitionGenerator, build_model
from .loader import load_prompt, load_services

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and generate ambitions for each service."""

    parser = argparse.ArgumentParser(description="Generate service ambitions")
    parser.add_argument(
        "--prompt-file", default="prompt.md", help="Path to the system prompt file"
    )
    parser.add_argument(
        "--prompt-id",
        help="Prompt template identifier. Overrides --prompt-file when provided.",
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
        default=os.getenv("MODEL", "o4-mini"),
        help="Chat model name. Can also be set via the MODEL env variable.",
    )
    parser.add_argument(
        "--response-format",
        default=os.getenv("RESPONSE_FORMAT"),
        help=(
            "Optional response format passed to ChatOpenAI. "
            "Can also be set via the RESPONSE_FORMAT env variable."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level. Can also be set via the LOG_LEVEL env variable.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of services to process concurrently",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it via a .env file or a secret manager."
        )

    prompt_file = args.prompt_file
    if args.prompt_id:
        prompt_file = f"prompt-{args.prompt_id}.md"
    system_prompt = load_prompt(prompt_file)
    services = list(load_services(args.input_file))

    try:
        model = build_model(args.model, api_key, args.response_format)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to initialize model %s: %s", args.model, exc)
        raise

    generator = ServiceAmbitionGenerator(model, concurrency=args.concurrency)
    generator.generate(services, system_prompt, args.output_file)
