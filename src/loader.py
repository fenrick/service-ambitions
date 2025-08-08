"""Utilities for loading prompts and services."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterator, List, TypeVar

import logfire
from pydantic import TypeAdapter

from models import ServiceFeaturePlateau

logger = logging.getLogger(__name__)

# Directory containing prompt templates. Updated via ``configure_prompt_dir``.
PROMPT_DIR = "prompts"


def configure_prompt_dir(path: str) -> None:
    """Set the base directory for prompt templates."""

    global PROMPT_DIR
    PROMPT_DIR = path


@logfire.instrument()
def _read_file(path: str) -> str:
    """Return the contents of ``path``.

    Args:
        path: File location.

    Returns:
        The file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """

    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:  # pragma: no cover - logging
        logger.error("Prompt file not found: %s", path)
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading prompt file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the prompt file: {exc}"
        ) from exc


T = TypeVar("T")


@logfire.instrument()
def _read_json_file(path: str, schema: type[T]) -> T:
    """Return JSON data loaded from ``path`` validated against ``schema``."""

    try:
        adapter = TypeAdapter(schema)
        return adapter.validate_json(_read_file(path))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading JSON file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the JSON file: {exc}"
        ) from exc


@logfire.instrument()
def load_prompt_text(prompt_name: str, base_dir: str | None = None) -> str:
    """Return the contents of a prompt template.

    The function locates ``prompt_name`` within ``base_dir`` (defaulting to the
    globally configured :data:`PROMPT_DIR`) and returns the stripped file
    contents. The ``.md`` suffix is added automatically if missing.

    Args:
        prompt_name: Name of the prompt file without directory components.
        base_dir: Optional override for the base directory containing prompts.

    Returns:
        Prompt template text.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """

    directory = base_dir or PROMPT_DIR
    filename = prompt_name if prompt_name.endswith(".md") else f"{prompt_name}.md"
    return _read_file(os.path.join(directory, filename))


@logfire.instrument()
@lru_cache(maxsize=None)
def load_mapping_items(base_dir: str = "data") -> Dict[str, list[Dict[str, str]]]:
    """Return mapping reference data from ``base_dir``.

    Args:
        base_dir: Directory containing mapping data files.

    Returns:
        A dictionary with keys ``information``, ``applications``, and
        ``technologies`` mapping to lists of item dictionaries.

    Raises:
        FileNotFoundError: If a required file is missing.
        RuntimeError: If a file cannot be read or parsed.
    """

    files = {
        "information": "information.json",
        "applications": "applications.json",
        "technologies": "technologies.json",
    }
    data: Dict[str, list[Dict[str, str]]] = {}
    for key, filename in files.items():
        path = os.path.join(base_dir, filename)
        data[key] = _read_json_file(path, list[Dict[str, str]])
    return data


@logfire.instrument()
@lru_cache(maxsize=None)
def load_plateau_definitions(
    base_dir: str = "data", filename: str = "service_feature_plateaus.json"
) -> List[ServiceFeaturePlateau]:
    """Return service feature plateau definitions from ``base_dir``.

    Args:
        base_dir: Directory containing data files.
        filename: Plateau definitions file name.

    Returns:
        List of :class:`ServiceFeaturePlateau` records.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """

    path = os.path.join(base_dir, filename)
    try:
        return _read_json_file(path, list[ServiceFeaturePlateau])
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Invalid plateau definition data in %s: %s", path, exc)
        raise RuntimeError(f"Invalid plateau definitions: {exc}") from exc


@logfire.instrument()
def load_prompt(
    context_id: str,
    inspirations_id: str,
    base_dir: str | None = None,
    plateaus_file: str = "service_feature_plateaus.md",
    definitions_file: str = "definitions.md",
    task_file: str = "task_definition.md",
    response_file: str = "response_structure.md",
) -> str:
    """Assemble the system prompt from modular components.

    Args:
        context_id: Identifier for the situational context file within the
            prompts directory.
        inspirations_id: Identifier for the inspirations file within the
            prompts directory.
        base_dir: Optional override for the base prompts directory.
        plateaus_file: Filename of the service feature plateaus component.
        definitions_file: Filename of the definitions component.
        task_file: Filename of the task definition component.
        response_file: Filename of the response structure component.

    Returns:
        Combined prompt text.

    Raises:
        FileNotFoundError: If a component file does not exist.
        RuntimeError: If a component file cannot be read.
    """

    directory = base_dir or PROMPT_DIR
    components = [
        os.path.join(directory, "situational_context", f"{context_id}.md"),
        os.path.join(directory, plateaus_file),
        os.path.join(directory, definitions_file),
        os.path.join(directory, "inspirations", f"{inspirations_id}.md"),
        os.path.join(directory, task_file),
        os.path.join(directory, response_file),
    ]
    parts = [_read_file(path) for path in components]
    return "\n\n".join(parts)


@logfire.instrument()
def load_services(path: str) -> Iterator[Dict[str, Any]]:
    """Yield services from ``path`` in JSON Lines format."""

    adapter = TypeAdapter(Dict[str, Any])
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    service = adapter.validate_json(line)
                except Exception as exc:  # pragma: no cover - logging
                    logger.error("Invalid service entry in %s: %s", path, exc)
                    raise RuntimeError("Invalid service definition") from exc
                service_id = service.get("service_id")
                if service_id is not None and not isinstance(service_id, str):
                    logger.error("service_id must be a string: %s", service_id)
                    raise RuntimeError("service_id must be a string")
                jobs = service.get("jobs_to_be_done")
                if jobs is not None and not isinstance(jobs, list):
                    logger.error("jobs_to_be_done must be a list: %s", jobs)
                    raise RuntimeError("jobs_to_be_done must be a list")
                yield service
    except FileNotFoundError:  # pragma: no cover - logging
        logger.error("Services file not found: %s", path)
        raise FileNotFoundError(
            "Services file not found. Please create a %s file in the current directory."
            % path
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading services file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the services file: {exc}"
        ) from exc
