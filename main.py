"""Command-line tool for generating service ambitions using a chat model."""

import argparse
import json
import os
from typing import Dict, Any, Iterator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown


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
        raise FileNotFoundError(
            f"Prompt file not found. Please create a {path} file in the current directory."
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
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
        raise FileNotFoundError(
            f"Services file not found. Please create a {path} file in the current directory."
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"An error occurred while reading the services file: {exc}"
        ) from exc


def process_service(service: Dict[str, Any], model, prompt: str) -> Dict[str, Any]:
    """Generate ambitions for ``service``.

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
    response = model.invoke(prompt_message)
    return parse_json_markdown(response.content)


def main() -> None:
    """Parse arguments and generate ambitions for each service."""

    parser = argparse.ArgumentParser(description="Generate service ambitions")
    parser.add_argument(
        "--prompt-file", default="prompt.md", help="Path to the system prompt file"
    )
    parser.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help="Path to the services JSONL file",
    )
    parser.add_argument(
        "--output-file", default="ambitions.jsonl", help="File to write the results"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it via a .env file or a secret manager."
        )

    system_prompt = load_prompt(args.prompt_file)
    services = load_services(args.input_file)

    model = init_chat_model(model="o4-mini", model_provider="openai")

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for i, service in enumerate(services, start=1):
            print(f"Processing service {i}: {service.get('name', 'unknown')}")
            try:
                result = process_service(service, model, system_prompt)
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"Failed to process service {service.get('name', 'unknown')}: {exc}"
                )
                continue
            output_file.write(f"{json.dumps(result)}\n")


if __name__ == "__main__":
    main()
