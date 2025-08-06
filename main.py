"""Command-line tool for generating service ambitions using a chat model."""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, Iterator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown

logger = logging.getLogger(__name__)


def load_prompt(path: str) -> str:
    """Return the system prompt from ``path``.

    Args:
        path: Location of the prompt markdown file.

    Returns:
        The contents of the prompt file.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """

    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error("Prompt file not found: %s", path)
        raise FileNotFoundError(
            f"Prompt file not found. Please create a {path} file in the current "
            "directory."
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading prompt file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the prompt file: {exc}"
        ) from exc


def load_services(path: str) -> Iterator[Dict[str, Any]]:
    """Yield services from ``path`` in JSON Lines format.

    Args:
        path: Location of the services JSONL file.

    Yields:
        Parsed service definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If any line cannot be parsed as JSON.
    """

    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    except FileNotFoundError:
        logger.error("Services file not found: %s", path)
        raise FileNotFoundError(
            f"Services file not found. Please create a {path} file in the current "
            "directory."
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading services file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the services file: {exc}"
        ) from exc


async def process_service(
    service: Dict[str, Any], model, prompt: str
) -> Dict[str, Any]:
    """Generate ambitions for ``service`` asynchronously.

    Args:
        service: Service definition.
        model: Chat model used to generate ambitions.
        prompt: System prompt string.

    Returns:
        Parsed JSON response from the model.
    """

    prompt_template = ChatPromptTemplate(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

    service_details = json.dumps(service)
    prompt_message = prompt_template.invoke(
        {"system_prompt": prompt, "user_prompt": service_details}
    )
    try:
        response = await model.ainvoke(prompt_message)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Model invocation failed for service %s: %s",
            service.get("name", "unknown"),
            exc,
        )
        raise
    return parse_json_markdown(response.content)


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
        "--model-provider",
        default=os.getenv("MODEL_PROVIDER", "openai"),
        help=(
            "Chat model provider. Can also be set via the "
            "MODEL_PROVIDER env variable."
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
        model = init_chat_model(model=args.model, model_provider=args.model_provider)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Failed to initialize model %s from provider %s: %s",
            args.model,
            args.model_provider,
            exc,
        )
        raise

    try:
        output_file = open(args.output_file, "w", encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to open %s: %s", args.output_file, exc)
        raise

    async def process_all() -> None:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def worker(service: Dict[str, Any]) -> None:
            async with semaphore:
                logger.info("Processing service %s", service.get("name", "unknown"))
                try:
                    result = await process_service(service, model, system_prompt)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(
                        "Failed to process service %s: %s",
                        service.get("name", "unknown"),
                        exc,
                    )
                    return
                output_file.write(f"{json.dumps(result)}\n")

        await asyncio.gather(*(worker(service) for service in services))

    try:
        asyncio.run(process_all())
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to write results to %s: %s", args.output_file, exc)
        raise
    finally:
        output_file.close()


if __name__ == "__main__":
    main()
