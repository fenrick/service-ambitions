"""Utilities for loading prompts, configuration and data files.

The helpers in this module centralise file-system access for prompts, mapping
metadata and service definitions. Several functions are cached to avoid
re-reading static content, and many include lightweight error handling so
callers receive concise exceptions.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Generator, Iterator, Sequence, TypeVar, cast

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
PROMPT_DIR = Path("prompts")

# Core role statement for all system prompts. This line anchors the model's
# objective before any contextual material is provided.
NORTH_STAR = (
    "You are the world's leading service designer and enterprise architect; your"
    " job is to produce strictly-valid JSON structured outputs aligned to the"
    " schema."
)


def configure_prompt_dir(path: Path | str) -> None:
    """Set the base directory for prompt templates.

    Side Effects:
        Updates the module-level :data:`PROMPT_DIR` used by other loading
        helpers so tests or CLI options may override where templates are sourced
        from.
    """

    global PROMPT_DIR
    PROMPT_DIR = Path(path)


@logfire.instrument()
def _read_file(path: Path) -> str:
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
        with path.open("r", encoding="utf-8") as file:
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
def _read_json_file(path: Path, schema: type[T]) -> T:
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
def load_prompt_text(prompt_name: str, base_dir: Path | None = None) -> str:
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
    return _read_file(directory / filename)


@logfire.instrument()
@lru_cache(maxsize=None)
def load_mapping_items(
    mapping_types: Sequence[str] | None = None,
    base_dir: Path | str = Path("data"),
) -> dict[str, list[MappingItem]]:
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
    base_path = Path(base_dir)
    files = {name: f"{name}.json" for name in datasets}
    data: dict[str, list[MappingItem]] = {}
    for key, filename in files.items():
        path = base_path / filename
        data[key] = _read_json_file(path, list[MappingItem])
    return data


@lru_cache(maxsize=None)
@logfire.instrument()
def load_app_config(
    base_dir: Path | str = Path("config"),
    filename: Path | str = Path("app.json"),
) -> AppConfig:
    """Return application configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    path = Path(base_dir) / Path(filename)
    return _read_json_file(path, AppConfig)


@lru_cache(maxsize=None)
@logfire.instrument()
def load_mapping_type_config(
    base_dir: Path | str = Path("config"),
    filename: Path | str = Path("app.json"),
) -> dict[str, MappingTypeConfig]:
    """Return mapping type configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    return load_app_config(base_dir, filename).mapping_types


@lru_cache(maxsize=None)
@logfire.instrument()
def load_plateau_definitions(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("service_feature_plateaus.json"),
) -> list[ServiceFeaturePlateau]:
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

    path = Path(base_dir) / Path(filename)
    try:
        return _read_json_file(path, list[ServiceFeaturePlateau])
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Invalid plateau definition data in %s: %s", path, exc)
        raise RuntimeError(f"Invalid plateau definitions: {exc}") from exc


@lru_cache(maxsize=None)
@logfire.instrument()
def load_definitions(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("definitions.json"),
    keys: Sequence[str] | None = None,
) -> str:
    """Return formatted definition text as a numbered Markdown list.

    Args:
        base_dir: Directory containing the definitions file.
        filename: Name of the definitions JSON file.
        keys: Optional identifiers selecting definitions to include. When ``None``,
            all entries are returned in file order.

    Returns:
        Markdown formatted definition text.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """

    path = Path(base_dir) / Path(filename)
    data: dict[str, object] = _read_json_file(path, dict[str, object])
    bullets = list(cast(list[dict[str, str]], data.get("bullets", [])))
    if keys:
        bullets = [b for b in bullets if b.get("name") in keys]
    lines = [f"## {data.get('title', 'Definitions')}", ""]
    for idx, item in enumerate(bullets, start=1):
        lines.append(f"{idx}. **{item['name']}**: {item['description']}")
    return "\n".join(lines)


@logfire.instrument()
def load_plateau_text(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("service_feature_plateaus.json"),
) -> str:
    """Return plateau descriptions as a numbered Markdown list.

    Args:
        base_dir: Directory containing plateau definition data.
        filename: JSON file containing :class:`ServiceFeaturePlateau` records.

    Returns:
        Markdown text describing each plateau.

    Raises:
        FileNotFoundError: If the data file does not exist.
        RuntimeError: If the data cannot be read or parsed.
    """

    plateaus = load_plateau_definitions(base_dir, filename)
    lines = ["## Service feature plateaus", ""]
    for idx, plateau in enumerate(plateaus, start=1):
        lines.append(f"{idx}. **{plateau.name}**: {plateau.description}")
    return "\n".join(lines)


@logfire.instrument()
def load_evolution_prompt(
    context_id: str,
    inspirations_id: str,
    base_dir: Path | None = None,
    definitions_dir: Path | str = Path("data"),
    definitions_file: Path | str = Path("definitions.json"),
    definition_keys: Sequence[str] | None = None,
    plateaus_dir: Path | str = Path("data"),
    plateaus_file: Path | str = Path("service_feature_plateaus.json"),
) -> str:
    """Assemble the system prompt from modular components.

    Args:
        context_id: Identifier for the situational context file within the
            prompts directory.
        inspirations_id: Identifier for the inspirations file within the
            prompts directory.
        base_dir: Optional override for the base prompts directory.
        definitions_dir: Directory containing definition data.
        definitions_file: Definitions file name.
        definition_keys: Optional identifiers selecting which definitions to
            include.
        plateaus_dir: Directory containing plateau definition data.
        plateaus_file: Plateau definitions file name.

    Returns:
        Combined prompt text.

    Raises:
        FileNotFoundError: If a component file does not exist.
        RuntimeError: If a component file cannot be read.
    """

    directory = base_dir or PROMPT_DIR
    components = [
        NORTH_STAR,
        load_prompt_text(f"situational_context/{context_id}", directory),
        load_plateau_text(plateaus_dir, plateaus_file),
        load_definitions(definitions_dir, definitions_file, definition_keys),
        load_prompt_text(f"inspirations/{inspirations_id}", directory),
    ]
    return "\n\n".join(components)


@logfire.instrument()
def load_ambition_prompt(
    context_id: str,
    inspirations_id: str,
    base_dir: Path | None = None,
    definitions_dir: Path | str = Path("data"),
    definitions_file: Path | str = Path("definitions.json"),
    definition_keys: Sequence[str] | None = None,
    task_file: Path | str = Path("task_definition.md"),
    response_file: Path | str = Path("response_structure.md"),
    plateaus_dir: Path | str = Path("data"),
    plateaus_file: Path | str = Path("service_feature_plateaus.json"),
) -> str:
    """Assemble the system prompt from modular components.

    Args:
        context_id: Identifier for the situational context file within the
            prompts directory.
        inspirations_id: Identifier for the inspirations file within the
            prompts directory.
        base_dir: Optional override for the base prompts directory.
        definitions_dir: Directory containing definition data.
        definitions_file: Definitions file name.
        definition_keys: Optional identifiers selecting which definitions to
            include.
        task_file: Filename of the task definition component.
        response_file: Filename of the response structure component.
        plateaus_dir: Directory containing plateau definition data.
        plateaus_file: Plateau definitions file name.

    Returns:
        Combined prompt text.

    Raises:
        FileNotFoundError: If a component file does not exist.
        RuntimeError: If a component file cannot be read.
    """

    directory = base_dir or PROMPT_DIR
    components = [
        NORTH_STAR,
        load_prompt_text(f"situational_context/{context_id}", directory),
        load_plateau_text(plateaus_dir, plateaus_file),
        load_definitions(definitions_dir, definitions_file, definition_keys),
        load_prompt_text(f"inspirations/{inspirations_id}", directory),
        load_prompt_text(str(task_file), directory),
        load_prompt_text(str(response_file), directory),
    ]
    return "\n\n".join(components)


# Backward compatibility alias
load_prompt = load_evolution_prompt


@logfire.instrument()
def _load_service_entries(path: Path | str) -> Generator[ServiceInput, None, None]:
    """Yield services from ``path`` while validating each JSON line."""

    path_obj = Path(path)
    with logfire.span("Calling loader.load_services"):
        adapter = TypeAdapter(ServiceInput)
        try:
            with path_obj.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue  # Skip blank lines.
                    try:
                        # Validate each line against the schema before yielding.
                        yield adapter.validate_json(line)
                    except Exception as exc:  # pragma: no cover - logging
                        logger.error("Invalid service entry in %s: %s", path_obj, exc)
                        raise RuntimeError("Invalid service definition") from exc
        except FileNotFoundError:  # pragma: no cover - logging
            logger.error("Services file not found: %s", path_obj)
            raise FileNotFoundError(
                "Services file not found. Please create a %s file in the current"
                " directory." % path_obj
            ) from None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error reading services file %s: %s", path_obj, exc)
            raise RuntimeError(
                f"An error occurred while reading the services file: {exc}"
            ) from exc


class ServiceLoader:
    """Iterator and context manager for service definitions."""

    def __init__(self, path: Path | str) -> None:
        self._path = path

    def __iter__(self) -> Iterator[ServiceInput]:
        return _load_service_entries(self._path)

    def __enter__(self) -> Iterator[ServiceInput]:
        return self.__iter__()

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no cleanup
        return None


def load_services(path: Path | str) -> ServiceLoader:
    """Return an iterable over services defined in ``path``.

    The returned object can be used either as an iterator or as a context
    manager, mirroring the previous behaviour of this function while allowing
    direct iteration as used in tests.
    """

    return ServiceLoader(path)
