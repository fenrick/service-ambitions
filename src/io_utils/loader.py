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
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_json

from models import (
    AppConfig,
    DefinitionBlock,
    DefinitionItem,
    MappingItem,
    MappingSet,
    Role,
    ServiceFeaturePlateau,
)
from runtime.environment import RuntimeEnv
from utils import (
    ErrorHandler,
    FileMappingLoader,
    FilePromptLoader,
    LoggingErrorHandler,
)

FEATURE_PLATEAUS_JSON = "service_feature_plateaus.json"

DEFINITIONS_JSON = "definitions.json"

# Core role statement for all system prompts. This line anchors the model's
# objective before any contextual material is provided.
NORTH_STAR = (
    "You are the world's leading service designer and enterprise architect; your"
    " job is to produce strictly-valid JSON structured outputs aligned to the"
    " schema."
)


def configure_prompt_dir(path: Path | str) -> None:
    """Set the base directory for prompt templates."""
    env = RuntimeEnv.instance()
    env.prompt_loader = FilePromptLoader(Path(path))
    clear_prompt_cache()


def configure_mapping_data_dir(path: Path | str) -> None:
    """Set the base directory for mapping reference data."""
    env = RuntimeEnv.instance()
    env.mapping_loader = FileMappingLoader(Path(path))
    clear_mapping_cache()


def _read_file(path: Path, error_handler: ErrorHandler | None = None) -> str:
    """Return the contents of ``path``.

    Args:
        path: File location.
        error_handler: Processor for any errors encountered.

    Returns:
        The file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    handler = error_handler or LoggingErrorHandler()
    with logfire.span("fs.read_text", attributes={"path": str(path)}):
        try:
            with path.open("r", encoding="utf-8") as file:
                text = file.read().strip()
                logfire.debug("Read text file", path=str(path), bytes=len(text))
                return text
        except FileNotFoundError as exc:
            handler.handle(f"Prompt file not found: {path}", exc)
            raise
        except OSError as exc:
            handler.handle(f"Error reading prompt file {path}", exc)
            raise RuntimeError(
                f"An error occurred while reading the prompt file: {exc}"
            ) from exc


T = TypeVar("T")


def _read_json_file(
    path: Path,
    schema: type[T],
    error_handler: ErrorHandler | None = None,
) -> T:
    """Return JSON data loaded from ``path`` validated against ``schema``.

    ``schema`` may be any type understood by :class:`pydantic.TypeAdapter`, such
    as a Pydantic model or standard container type.
    """
    handler = error_handler or LoggingErrorHandler()
    with logfire.span("fs.read_json", attributes={"path": str(path)}):
        try:
            adapter = TypeAdapter(schema)
            return adapter.validate_json(_read_file(path, handler))
        except FileNotFoundError:
            raise
        except (RuntimeError, ValidationError, ValueError) as exc:
            handler.handle(f"Error reading JSON file {path}", exc)
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


def _read_yaml_file(
    path: Path,
    schema: type[T],
    error_handler: ErrorHandler | None = None,
) -> T:
    """Return YAML data loaded from ``path`` validated against ``schema``.

    Args:
        path: File location.
        schema: Pydantic-compatible schema to validate against.
        error_handler: Processor for any errors encountered.
    """
    handler = error_handler or LoggingErrorHandler()
    with logfire.span("fs.read_yaml", attributes={"path": str(path)}):
        try:
            adapter = TypeAdapter(schema)
            return adapter.validate_python(yaml.safe_load(_read_file(path, handler)))
        except FileNotFoundError:
            raise
        except (RuntimeError, ValidationError, yaml.YAMLError, ValueError) as exc:
            handler.handle(f"Error reading YAML file {path}", exc)
            raise RuntimeError(
                f"An error occurred while reading the YAML file: {exc}"
            ) from exc


@lru_cache(maxsize=None)
def load_prompt_text(prompt_name: str, base_dir: Path | str | None = None) -> str:
    """Return the contents of a prompt template.

    Results are memoised for the lifetime of the process. Use
    :func:`clear_prompt_cache` to invalidate the cache when template files
    change.
    """
    env = RuntimeEnv.instance()
    loader = env.prompt_loader if base_dir is None else FilePromptLoader(Path(base_dir))
    with logfire.span(
        "loader.load_prompt_text",
        attributes={
            "name": prompt_name,
            "base_dir": str(base_dir or getattr(env, "prompt_loader", "")),
        },
    ):
        try:
            return loader.load(prompt_name)
        except (OSError, ValueError, RuntimeError) as exc:
            LoggingErrorHandler().handle(f"Error loading prompt {prompt_name}", exc)
            raise


def clear_prompt_cache() -> None:
    """Invalidate memoised prompt text."""
    load_prompt_text.cache_clear()
    RuntimeEnv.instance().prompt_loader.clear_cache()


def clear_mapping_cache() -> None:
    """Invalidate memoised mapping catalogues."""
    RuntimeEnv.instance().mapping_loader.clear_cache()


def load_mapping_items(
    sets: Sequence[MappingSet],
    data_dir: Path | str | None = None,
    error_handler: ErrorHandler | None = None,
) -> tuple[dict[str, list[MappingItem]], str]:
    """Return mapping reference data and a combined catalogue hash.

    Args:
        sets: Catalogue definitions to load. When empty, object‑form datasets are
            auto‑discovered from ``data_dir`` (or the runtime mapping directory).
        data_dir: Optional directory override for catalogue files.
        error_handler: Processor for any errors encountered.
    """
    loader = (
        FileMappingLoader(Path(data_dir))
        if data_dir is not None
        else RuntimeEnv.instance().mapping_loader
    )
    handler = error_handler or LoggingErrorHandler()
    with logfire.span(
        "loader.load_mapping_items",
        attributes={
            "sets": len(sets),
            "data_dir": str(data_dir) if data_dir is not None else None,
        },
    ):
        try:
            if not sets:
                # Auto‑discover object‑form datasets: files that embed a "field".
                if not isinstance(
                    loader, FileMappingLoader
                ):  # pragma: no cover - defensive
                    raise RuntimeError("Auto-discovery requires FileMappingLoader")
                return loader.discover()
            return loader.load(sets)
        except (OSError, ValueError, RuntimeError) as exc:
            handler.handle("Error loading mapping items", exc)
            raise


def load_mapping_meta(
    sets: Sequence[MappingSet], data_dir: Path | str | None = None
) -> dict[str, dict[str, object]]:
    """Return mapping dataset metadata keyed by field.

    The metadata includes optional attributes carried in object‑form datasets
    such as ``label`` and ``facets``. When called with a plain list‑form
    dataset, the result may be empty for that field.
    """
    loader = (
        FileMappingLoader(Path(data_dir))
        if data_dir is not None
        else RuntimeEnv.instance().mapping_loader
    )
    if not isinstance(loader, FileMappingLoader):  # pragma: no cover - defensive
        return {}
    # Ensure caches are warm and return the meta snapshot.
    return loader.meta(sets)


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
def load_plateau_definitions(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path(FEATURE_PLATEAUS_JSON),
    error_handler: ErrorHandler | None = None,
) -> list[ServiceFeaturePlateau]:
    """Return service feature plateau definitions from ``base_dir``.

    Args:
        base_dir: Directory containing data files.
        filename: Plateau definitions file name.
        error_handler: Processor for any errors encountered.

    Returns:
        List of :class:`ServiceFeaturePlateau` records.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """
    handler = error_handler or LoggingErrorHandler()
    path = Path(base_dir) / Path(filename)
    with logfire.span(
        "loader.load_plateau_definitions", attributes={"path": str(path)}
    ):
        try:
            return _read_json_file(path, list[ServiceFeaturePlateau], handler)
        except (RuntimeError, ValidationError, ValueError) as exc:
            handler.handle(f"Invalid plateau definition data in {path}", exc)
            raise RuntimeError(f"Invalid plateau definitions: {exc}") from exc


@lru_cache(maxsize=None)
def load_roles(
    base_dir: Path | str = Path("data"),
    filename: Path | str = Path("roles.json"),
    error_handler: ErrorHandler | None = None,
) -> list[Role]:
    """Return role definitions from ``base_dir`` or a direct file path.

    Args:
        base_dir: Directory containing data files or the roles file itself.
        filename: Roles definitions file name when ``base_dir`` is a directory.
        error_handler: Processor for any errors encountered.

    Returns:
        List of :class:`Role` records.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read or parsed.
    """
    handler = error_handler or LoggingErrorHandler()
    base_path = Path(base_dir)
    # If ``base_dir`` points to a directory append ``filename``; otherwise treat
    # it as the full path to the roles file.
    path = base_path / Path(filename) if base_path.is_dir() else base_path
    with logfire.span("loader.load_roles", attributes={"path": str(path)}):
        try:
            return _read_json_file(path, list[Role], handler)
        except (RuntimeError, ValidationError, ValueError) as exc:
            handler.handle(f"Invalid role data in {path}", exc)
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
    with logfire.span(
        "loader.load_role_ids",
        attributes={"path": str(Path(base_dir) / Path(filename))},
    ):
        roles = load_roles(base_dir, filename)
        ids = [role.role_id for role in roles]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate role_id values found in roles.json")
        return ids


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
    with logfire.span(
        "loader.load_definitions",
        attributes={"path": str(path), "keys": None if keys is None else len(keys)},
    ):
        try:
            data = _read_json_file(path, DefinitionBlock)
        except (RuntimeError, ValidationError, ValueError) as exc:
            logfire.error(f"Invalid definition data in {path}: {exc}")
            raise RuntimeError(f"Invalid definitions: {exc}") from exc

    bullets = data.bullets
    if keys:
        bullets = [item for item in bullets if item.id in keys]
    lines = [f"## {data.title}", ""]
    for idx, item in enumerate(bullets, start=1):
        lines.extend(_format_definition_item(idx, item))
    return "\n".join(lines)


def _format_definition_item(idx: int, item: DefinitionItem) -> list[str]:
    """Return formatted lines for a single definition item."""
    header = f"{idx}. **{item.name}**"
    if item.aliases:
        header += f" ({', '.join(item.aliases)})"
    lines = [header]
    if item.short_definition:
        lines.append(f"   - Short definition: {item.short_definition}")
    lines.append(f"   - Definition: {item.definition}")
    for label, values in [
        ("Decision rules", item.decision_rules),
        ("Use when", item.use_when),
        ("Avoid confusion with", item.avoid_confusion_with),
        ("Examples", item.examples),
        ("Non-examples", item.non_examples),
    ]:
        _append_list(lines, label, values)
    for label, value in [
        ("Related terms", item.related_terms and ", ".join(item.related_terms)),
        ("Tags", item.tags and ", ".join(item.tags)),
        ("Owner", item.owner),
        ("Last updated", item.last_updated),
    ]:
        if value:
            lines.append(f"   - {label}: {value}")
    return lines


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
    with logfire.span(
        "loader.load_plateau_text",
        attributes={"path": str(Path(base_dir) / Path(filename))},
    ):
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
    with logfire.span(
        "loader.load_evolution_prompt",
        attributes={
            "context_id": context_id,
            "inspirations_id": inspirations_id,
        },
    ):
        components = [
            NORTH_STAR,
            load_prompt_text(f"situational_context/{context_id}", base_dir),
            load_plateau_text(plateaus_dir, plateaus_file),
            load_definitions(definitions_dir, definitions_file, definition_keys),
            load_prompt_text(f"inspirations/{inspirations_id}", base_dir),
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
        plateaus_dir: Directory containing plateau definition data.
        plateaus_file: Plateau definitions file name.

    Returns:
        Combined prompt text.

    Raises:
        FileNotFoundError: If a component file does not exist.
        RuntimeError: If a component file cannot be read.
    """
    with logfire.span(
        "loader.load_ambition_prompt",
        attributes={
            "context_id": context_id,
            "inspirations_id": inspirations_id,
            "task_file": str(task_file),
        },
    ):
        components = [
            NORTH_STAR,
            load_prompt_text(f"situational_context/{context_id}", base_dir),
            load_plateau_text(plateaus_dir, plateaus_file),
            load_definitions(definitions_dir, definitions_file, definition_keys),
            load_prompt_text(f"inspirations/{inspirations_id}", base_dir),
            load_prompt_text(str(task_file), base_dir),
        ]
        return "\n\n".join(components)


# Backward compatibility alias
load_prompt = load_ambition_prompt
