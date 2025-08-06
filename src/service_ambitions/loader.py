"""Utilities for loading prompts and services."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator

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
    except FileNotFoundError:  # pragma: no cover - logging
        logger.error("Prompt file not found: %s", path)
        raise FileNotFoundError(
            (
                "Prompt file not found. Please create a %s file in the current "
                "directory."
            )
            % path
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
