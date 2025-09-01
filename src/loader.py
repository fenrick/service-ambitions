# SPDX-License-Identifier: MIT
"""Utilities for loading prompts, configuration and reference data.

The helpers in this module centralise file-system access for prompt templates,
mapping metadata and other shared data files. Several functions are cached to
avoid re-reading static content, and many include lightweight error handling so
callers receive concise exceptions.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Sequence, TypeVar

import logfire
import yaml
from pydantic import TypeAdapter
from pydantic_core import to_json

from models import (
    AppConfig,
    DefinitionBlock,
    MappingItem,
    MappingSet,
    MappingTypeConfig,
    Role,
    ServiceFeaturePlateau,
)
from utils import (
    ErrorHandler,
    FileMappingLoader,
    FilePromptLoader,
    LoggingErrorHandler,
    MappingLoader,
    PromptLoader,
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

_prompt_loader: PromptLoader = FilePromptLoader(PROMPT_DIR)
_mapping_loader: MappingLoader = FileMappingLoader(MAPPING_DATA_DIR)
_error_handler: ErrorHandler = LoggingErrorHandler()

# Core role statement for all system prompts. This line anchors the model's
# objective before any contextual material is provided.
NORTH_STAR = (
    "You are the world's leading service designer and enterprise architect; your"
    " job is to produce strictly-valid JSON structured outputs aligned to the"
    " schema."
)


def configure_prompt_dir(path: Path | str) -> None:
    """Set the base directory for prompt templates."""

    global PROMPT_DIR, _prompt_loader
    PROMPT_DIR = Path(path)
    _prompt_loader = FilePromptLoader(PROMPT_DIR)


def configure_mapping_data_dir(path: Path | str) -> None:
    """Set the base directory for mapping reference data."""

    global MAPPING_DATA_DIR, _mapping_loader
    MAPPING_DATA_DIR = Path(path)
    _mapping_loader = FileMappingLoader(MAPPING_DATA_DIR)


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


def _sanitize(value: str) -> str:
    """Return ``value`` with newlines and tabs replaced by spaces."""

    return value.replace("\n", " ").replace("\t", " ")


def _append_list(lines: list[str], label: str, values: Sequence[str]) -> None:
    """Append ``values`` under ``label`` using Markdown bullet rules.

    If ``values`` contains a single entry, it is added inline after the label.
    Otherwise the values are rendered as an indented sub-list.
    """

    if not values:  # No values to record
        return
    if len(values) == 1:  # Single value rendered inline
        lines.append(f"   - {label}: {values[0]}")
        return
    lines.append(f"   - {label}:")
    for value in values:
        lines.append(f"     - {value}")


def compile_catalogue_for_set(
    items: Sequence[MappingItem],
) -> tuple[list[MappingItem], str]:
    """Return ``items`` sorted by ID and their SHA256 digest.

    The digest is derived from a deterministic JSON serialisation of the
    catalogue items, ensuring consistent hashes across runs regardless of input
    order or whitespace variations.
    """

    ordered = sorted(items, key=lambda item: item.id)
    canonical = [
        {
            "id": _sanitize(item.id),
            "name": _sanitize(item.name),
            "description": _sanitize(item.description),
        }
        for item in ordered
    ]
    try:
        serialised = to_json(canonical, sort_keys=True).decode("utf-8")  # type: ignore[call-arg]
    except TypeError:  # Fallback for older pydantic-core versions
        serialised = to_json(canonical).decode("utf-8")
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ordered, digest


def _read_yaml_file(path: Path, schema: type[T]) -> T:
    """Return YAML data loaded from ``path`` validated against ``schema``."""

    try:
        adapter = TypeAdapter(schema)
        return adapter.validate_python(yaml.safe_load(_read_file(path)))
    except FileNotFoundError:
        raise
    except Exception as exc:
        logfire.error(f"Error reading YAML file {path}: {exc}")
        raise RuntimeError(
            f"An error occurred while reading the YAML file: {exc}"
        ) from exc


def load_prompt_text(prompt_name: str, base_dir: Path | None = None) -> str:
    """Return the contents of a prompt template."""

    loader = _prompt_loader if base_dir is None else FilePromptLoader(Path(base_dir))
    try:
        return loader.load(prompt_name)
    except Exception as exc:
        _error_handler.handle(f"Error loading prompt {prompt_name}", exc)
        raise


def load_mapping_items(
    data_dir: Path, sets: Sequence[MappingSet]
) -> tuple[dict[str, list[MappingItem]], str]:
    """Return mapping reference data and a combined catalogue hash."""

    loader = (
        _mapping_loader if data_dir == MAPPING_DATA_DIR else FileMappingLoader(data_dir)
    )
    try:
        return loader.load(sets)
    except Exception as exc:
        _error_handler.handle("Error loading mapping items", exc)
        raise


@lru_cache(maxsize=None)
def load_app_config(
    base_dir: Path | str = Path("config"),
    filename: Path | str = Path("app.yaml"),
) -> AppConfig:
    """Return application configuration from ``base_dir``.

    Results are cached for the lifetime of the process.
    """

    path = Path(base_dir) / Path(filename)
    return _read_yaml_file(path, AppConfig)


@lru_cache(maxsize=None)
def load_mapping_type_config(
    base_dir: Path | str = Path("config"),
    filename: Path | str = Path("app.yaml"),
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
        bullets = [item for item in bullets if item.id in keys]
    lines = [f"## {data.title}", ""]
    for idx, item in enumerate(bullets, start=1):
        header = f"{idx}. **{item.name}**"
        if item.aliases:  # Show alternative terms in-line
            header += f" ({', '.join(item.aliases)})"
        lines.append(header)
        if item.short_definition:  # Include concise explanation when present
            lines.append(f"   - Short definition: {item.short_definition}")
        lines.append(f"   - Definition: {item.definition}")
        _append_list(lines, "Decision rules", item.decision_rules)
        _append_list(lines, "Use when", item.use_when)
        _append_list(lines, "Avoid confusion with", item.avoid_confusion_with)
        _append_list(lines, "Examples", item.examples)
        _append_list(lines, "Non-examples", item.non_examples)
        if item.related_terms:  # Join identifiers inline
            lines.append(f"   - Related terms: {', '.join(item.related_terms)}")
        if item.tags:  # Tag list summarises categorisation
            lines.append(f"   - Tags: {', '.join(item.tags)}")
        if item.owner:  # Identify definition steward
            lines.append(f"   - Owner: {item.owner}")
        if item.last_updated:  # Track provenance
            lines.append(f"   - Last updated: {item.last_updated}")
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
        lines.append(f"{idx}. **{plateau.name}**")
        lines.append(f"   - Core idea: {plateau.description.core_idea}")
        lines.append("   - Key characteristics:")
        for characteristic in plateau.description.key_characteristics:
            lines.append(f"     - {characteristic}")
        lines.append(
            f"   - What it feels like: {plateau.description.what_it_feels_like}"
        )
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
load_prompt = load_ambition_prompt
