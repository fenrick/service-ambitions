"""Utilities for loading prompts, configuration and reference data.

The helpers in this module centralise file-system access for prompt templates,
mapping metadata and other shared data files. Several functions are cached to
avoid re-reading static content, and many include lightweight error handling so
callers receive concise exceptions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import logfire
from pydantic import TypeAdapter

from models import (
    AppConfig,
    DefinitionBlock,
    MappingItem,
    MappingSet,
    MappingTypeConfig,
    Role,
    ServiceFeaturePlateau,
)

FEATURE_PLATEAUS_JSON = "service_feature_plateaus.json"

DEFINITIONS_JSON = "definitions.json"

# Directory containing prompt templates.  Mutable so tests or callers may point
# to alternative directories via ``configure_prompt_dir``.
PROMPT_DIR = Path("prompts")

# Directory containing mapping reference data. Updated via
# :func:`configure_mapping_data_dir` so callers may override the default ``data``
# location.
MAPPING_DATA_DIR = Path("data")

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


def configure_mapping_data_dir(path: Path | str) -> None:
    """Set the base directory for mapping reference data.

    Side Effects:
        Updates :data:`MAPPING_DATA_DIR` used by catalogue loaders and clears
        cached data so subsequent calls honour the new location.
    """

    global MAPPING_DATA_DIR
    MAPPING_DATA_DIR = Path(path)
    _load_mapping_items.cache_clear()


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
    except FileNotFoundError:
        logfire.error(f"Prompt file not found: {path}")
        raise
    except Exception as exc:
        logfire.error(f"Error reading prompt file {path}: {exc}")
        raise RuntimeError(
            f"An error occurred while reading the prompt file: {exc}"
        ) from exc


T = TypeVar("T")


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
    except Exception as exc:
        logfire.error(f"Error reading JSON file {path}: {exc}")
        raise RuntimeError(
            f"An error occurred while reading the JSON file: {exc}"
        ) from exc


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


def load_mapping_items(
    data_dir: Path, sets: Sequence[MappingSet]
) -> dict[str, list[MappingItem]]:
    """Return mapping reference data for ``sets`` sourced from ``data_dir``."""

    key = tuple((s.file, s.field) for s in sets)
    return _load_mapping_items(data_dir, key)


@lru_cache(maxsize=None)
def _load_mapping_items(
    data_dir: Path, key: Tuple[Tuple[str, str], ...]
) -> dict[str, list[MappingItem]]:
    """Load mapping items using a hashable key for caching."""

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Mapping data directory not found: {data_dir}")

    data: dict[str, list[MappingItem]] = {}
    for file, field in key:
        path = data_dir / file
        try:
            items = _read_json_file(path, list[MappingItem])
        except FileNotFoundError:
            raise
        except Exception:
            continue
        data[field] = sorted(items, key=lambda item: item.id)
    return data


@lru_cache(maxsize=None)
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
def load_mapping_type_config(
    base_dir: Path | str = Path("config"),
    filename: Path | str = Path("app.json"),
) -> dict[str, MappingTypeConfig]:
    """Return mapping type configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    return load_app_config(base_dir, filename).mapping_types


@lru_cache(maxsize=None)
def load_plateau_definitions(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path(FEATURE_PLATEAUS_JSON),
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
    except Exception as exc:
        logfire.error(f"Invalid plateau definition data in {path}: {exc}")
        raise RuntimeError(f"Invalid plateau definitions: {exc}") from exc


@lru_cache(maxsize=None)
def load_roles(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("roles.json"),
) -> list[Role]:
    """Return role definitions from ``base_dir`` or a direct file path.

    Args:
        base_dir: Directory containing data files or the roles file itself.
        filename: Roles definitions file name when ``base_dir`` is a directory.

    Returns:
        List of :class:`Role` records.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """

    base_path = Path(base_dir)
    # If ``base_dir`` points to a directory append ``filename``; otherwise treat
    # it as the full path to the roles file.
    path = base_path / Path(filename) if base_path.is_dir() else base_path
    try:
        return _read_json_file(path, list[Role])
    except Exception as exc:
        logfire.error(f"Invalid role data in {path}: {exc}")
        raise RuntimeError(f"Invalid roles: {exc}") from exc


@lru_cache(maxsize=None)
def load_role_ids(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("roles.json"),
) -> list[str]:
    """Return role identifiers from ``base_dir`` or a direct file path.

    Args:
        base_dir: Directory containing data files or the roles file itself.
        filename: Roles definitions file name when ``base_dir`` is a directory.

    Returns:
        List of role identifier strings.

    Raises:
        FileNotFoundError: If the roles file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """

    return [role.role_id for role in load_roles(base_dir, filename)]


@lru_cache(maxsize=None)
def load_definitions(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path(DEFINITIONS_JSON),
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
    try:
        data = _read_json_file(path, DefinitionBlock)
    except Exception as exc:
        logfire.error(f"Invalid definition data in {path}: {exc}")
        raise RuntimeError(f"Invalid definitions: {exc}") from exc

    bullets = data.bullets
    if keys:
        bullets = [item for item in bullets if item.name in keys]
    lines = [f"## {data.title}", ""]
    for idx, item in enumerate(bullets, start=1):
        lines.append(f"{idx}. **{item.name}**: {item.description}")
    return "\n".join(lines)


def load_plateau_text(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path(FEATURE_PLATEAUS_JSON),
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


def load_evolution_prompt(
    context_id: str,
    inspirations_id: str,
    base_dir: Path | None = None,
    definitions_dir: Path | str = Path("data"),
    definitions_file: Path | str = Path(DEFINITIONS_JSON),
    definition_keys: Sequence[str] | None = None,
    plateaus_dir: Path | str = Path("data"),
    plateaus_file: Path | str = Path(FEATURE_PLATEAUS_JSON),
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


def load_ambition_prompt(
    context_id: str,
    inspirations_id: str,
    base_dir: Path | None = None,
    definitions_dir: Path | str = Path("data"),
    definitions_file: Path | str = Path(DEFINITIONS_JSON),
    definition_keys: Sequence[str] | None = None,
    task_file: Path | str = Path("task_definition.md"),
    response_file: Path | str = Path("response_structure.md"),
    plateaus_dir: Path | str = Path("data"),
    plateaus_file: Path | str = Path(FEATURE_PLATEAUS_JSON),
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
