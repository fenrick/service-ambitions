"""Input and output helpers for prompts, mappings and files.

Exports:
    configure_prompt_dir: Set the base directory for prompt templates.
    configure_mapping_data_dir: Set the base directory for mapping catalogues.
    load_prompt_text: Read a prompt template from disk.
    load_mapping_items: Load mapping catalogue entries.
    load_services: Iterate over service definitions in a JSONL file.
    read_lines: Return existing lines from a file.
    atomic_write: Write files atomically.
    validate_jsonl: Validate JSONL files against a Pydantic model.
    QuarantineWriter: Persist invalid payloads and maintain a manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .diagnostics import validate_jsonl
from .persistence import atomic_write, read_lines
from .quarantine import QuarantineWriter
from .service_loader import load_services

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from models import MappingItem, MappingSet
    from utils import ErrorHandler


def configure_prompt_dir(path: Path | str) -> None:
    """Set the base directory for prompt templates."""
    from .loader import configure_prompt_dir as _configure_prompt_dir

    _configure_prompt_dir(path)


def configure_mapping_data_dir(path: Path | str) -> None:
    """Set the base directory for mapping catalogues."""
    from .loader import configure_mapping_data_dir as _configure_mapping_data_dir

    _configure_mapping_data_dir(path)


def load_prompt_text(prompt_name: str, base_dir: Path | str | None = None) -> str:
    """Return the contents of a prompt template."""
    from .loader import load_prompt_text as _load_prompt_text

    return _load_prompt_text(prompt_name, base_dir)


def load_mapping_items(
    sets: Sequence["MappingSet"],
    data_dir: Path | str | None = None,
    error_handler: "ErrorHandler" | None = None,
) -> tuple[dict[str, list["MappingItem"]], str]:
    """Return mapping reference data and a catalogue hash."""
    from .loader import load_mapping_items as _load_mapping_items

    return _load_mapping_items(sets, data_dir, error_handler)


__all__ = [
    "configure_prompt_dir",
    "configure_mapping_data_dir",
    "load_prompt_text",
    "load_mapping_items",
    "load_services",
    "read_lines",
    "atomic_write",
    "validate_jsonl",
    "QuarantineWriter",
]
