"""Utilities for loading prompts and services."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator

import logfire

logger = logging.getLogger(__name__)


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


@logfire.instrument()
def load_plateau_prompt(base_dir: str, filename: str = "plateau_prompt.md") -> str:
    """Return the plateau feature generation prompt template.

    Args:
        base_dir: Directory containing prompt templates.
        filename: Plateau prompt file name.

    Returns:
        Prompt template text.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """

    return _read_file(os.path.join(base_dir, filename))


@logfire.instrument()
def load_mapping_prompt(base_dir: str, filename: str = "mapping_prompt.md") -> str:
    """Return the feature mapping prompt template.

    Args:
        base_dir: Directory containing prompt templates.
        filename: Mapping prompt file name.

    Returns:
        Prompt template text.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """

    return _read_file(os.path.join(base_dir, filename))


@logfire.instrument()
def load_prompt(
    base_dir: str,
    context_id: str,
    inspirations_id: str,
    plateaus_file: str = "service_feature_plateaus.md",
    definitions_file: str = "definitions.md",
    task_file: str = "task_definition.md",
    response_file: str = "response_structure.md",
) -> str:
    """Assemble the system prompt from modular components.

    Args:
        base_dir: Directory containing prompt components.
        context_id: Identifier for the situational context file within
            ``base_dir/situational_context``.
        inspirations_id: Identifier for the inspirations file within
            ``base_dir/inspirations``.
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

    components = [
        os.path.join(base_dir, "situational_context", f"{context_id}.md"),
        os.path.join(base_dir, plateaus_file),
        os.path.join(base_dir, definitions_file),
        os.path.join(base_dir, "inspirations", f"{inspirations_id}.md"),
        os.path.join(base_dir, task_file),
        os.path.join(base_dir, response_file),
    ]
    parts = [_read_file(path) for path in components]
    return "\n\n".join(parts)


@logfire.instrument()
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
    except FileNotFoundError:  # pragma: no cover - logging
        logger.error("Services file not found: %s", path)
        raise FileNotFoundError(
            (
                "Services file not found. Please create a %s file in the current "
                "directory."
            )
            % path
        ) from None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading services file %s: %s", path, exc)
        raise RuntimeError(
            f"An error occurred while reading the services file: {exc}"
        ) from exc
