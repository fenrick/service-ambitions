# SPDX-License-Identifier: MIT
"""Utilities for validating dataset inputs."""

from __future__ import annotations

import json
from pathlib import Path

from io_utils import validate_jsonl
from models import MappingItem, ServiceInput


def validate_data_dir(data_dir: Path) -> None:
    """Validate JSON inputs within ``data_dir``.

    Parameters
    ----------
    data_dir:
        Directory containing ``services.jsonl`` and an optional ``catalogue``
        subdirectory with JSON files.

    Raises:
    ------
    FileNotFoundError
        If ``services.jsonl`` is missing.
    ValueError
        If any record fails validation.
    """
    services_file = data_dir / "services.jsonl"
    if not services_file.is_file():
        raise FileNotFoundError(services_file)

    count = validate_jsonl(services_file, ServiceInput)
    print(f"{services_file}: {count} valid records")

    catalogue_dir = data_dir / "catalogue"
    if not catalogue_dir.is_dir():
        return

    for path in sorted(catalogue_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        try:
            items = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(items, list):
            raise ValueError(f"{path} does not contain a list of items")
        for idx, item in enumerate(items, start=1):
            try:
                MappingItem.model_validate(item)
            except Exception as exc:  # pragma: no cover - unexpected errors
                raise ValueError(f"{path} item {idx} invalid: {exc}") from exc
        print(f"{path}: {len(items)} valid items")
