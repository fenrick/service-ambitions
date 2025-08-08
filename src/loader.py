"""Utilities for loading prompts, configuration and data files.

The helpers in this module centralise file-system access for prompts, mapping
metadata and service definitions. Several functions are cached to avoid
re-reading static content, and many include lightweight error handling so
callers receive concise exceptions.
"""

from __future__ import annotations

import logging
import os
from contextlib import closing, contextmanager
from functools import lru_cache
from typing import Dict, Generator, Iterator, List, Sequence, TypeVar

import logfire
from pydantic import TypeAdapter

from models import (
    AppConfig,
    MappingItem,
    MappingTypeConfig,
    ServiceFeaturePlateau,
    ServiceInput,
)

logger = logging.getLogger(__name__)

# Directory containing prompt templates.  Mutable so tests or callers may point
# to alternative directories via ``configure_prompt_dir``.
PROMPT_DIR = "prompts"


def configure_prompt_dir(path: str) -> None:
    """Set the base directory for prompt templates.

    Side Effects:
        Updates the module-level :data:`PROMPT_DIR` used by other loading
        helpers so tests or CLI options may override where templates are sourced
        from.
    """

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
    """Return JSON data loaded from ``path`` validated against ``schema``.

    ``schema`` may be any type understood by :class:`pydantic.TypeAdapter`, such
    as a Pydantic model or standard container type.
    """

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
    contents. The ``.md`` suffix is added automatically if missing. Results are
    not cached so callers should apply their own caching where appropriate.

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
def load_mapping_items(
    mapping_types: Sequence[str] | None = None,
    base_dir: str = "data",
) -> Dict[str, list[MappingItem]]:
    """Return mapping reference data for ``mapping_types`` from ``base_dir``.

    Args:
        mapping_types: Mapping dataset names to load. Defaults to the standard
            information, applications and technologies datasets.
        base_dir: Directory containing mapping data files.

    Returns:
        A dictionary mapping each ``mapping_type`` to a list of
        :class:`MappingItem`.

    Raises:
        FileNotFoundError: If a required file is missing.
        RuntimeError: If a file cannot be read or parsed.
    """

    datasets = mapping_types or ("information", "applications", "technologies")
    files = {name: f"{name}.json" for name in datasets}
    data: Dict[str, list[MappingItem]] = {}
    for key, filename in files.items():
        path = os.path.join(base_dir, filename)
        data[key] = _read_json_file(path, list[MappingItem])
    return data


@logfire.instrument()
@lru_cache(maxsize=None)
def load_app_config(base_dir: str = "config", filename: str = "app.json") -> AppConfig:
    """Return application configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    path = os.path.join(base_dir, filename)
    return _read_json_file(path, AppConfig)


@logfire.instrument()
@lru_cache(maxsize=None)
def load_mapping_type_config(
    base_dir: str = "config",
    filename: str = "app.json",
) -> Dict[str, MappingTypeConfig]:
    """Return mapping type configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    return load_app_config(base_dir, filename).mapping_types


@logfire.instrument()
@lru_cache(maxsize=None)
def load_plateau_definitions(
    base_dir: str = "data",
    filename: str = "service_feature_plateaus.json",
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


@contextmanager
def load_services(path: str) -> Iterator[Iterator[ServiceInput]]:
    """Yield services from ``path`` in JSON Lines format.

    Each line is parsed as JSON and validated against :class:`ServiceInput`.

    Args:
        path: Location of the services JSONL file.

    Yields:
        Parsed service definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If any line cannot be parsed as JSON or contains invalid
            fields.
    """

    def load_services_int(path: str) -> Generator[ServiceInput, None, None]:
        with logfire.span("Calling loader.load_services"):
            adapter = TypeAdapter(ServiceInput)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield adapter.validate_json(line)
                        except Exception as exc:  # pragma: no cover - logging
                            logger.error("Invalid service entry in %s: %s", path, exc)
                            raise RuntimeError("Invalid service definition") from exc
            except FileNotFoundError:  # pragma: no cover - logging
                logger.error("Services file not found: %s", path)
                raise FileNotFoundError(
                    "Services file not found. Please create a %s file in the current"
                    " directory." % path
                ) from None
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error reading services file %s: %s", path, exc)
                raise RuntimeError(
                    f"An error occurred while reading the services file: {exc}"
                ) from exc

    with closing(load_services_int(path)) as items:
        yield items
