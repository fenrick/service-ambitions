"""Tests for :func:`loader.load_definitions`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader import load_definitions


def test_load_definitions_roundtrip(tmp_path: Path) -> None:
    """Definitions file is rendered as Markdown with numbered bullets."""

    content = {
        "title": "Key definitions",
        "bullets": [
            {"name": "Alpha", "description": "First"},
            {"name": "Beta", "description": "Second"},
        ],
    }
    file = tmp_path / "definitions.json"
    file.write_text(json.dumps(content), encoding="utf-8")

    text = load_definitions(tmp_path, file.name)

    assert "## Key definitions" in text
    assert "1. **Alpha**: First" in text
    assert "2. **Beta**: Second" in text


def test_load_definitions_validation_error(tmp_path: Path) -> None:
    """Invalid definition entries raise ``RuntimeError``."""

    content = {
        "title": "Key definitions",
        "bullets": [{"name": "Alpha"}],
    }
    file = tmp_path / "definitions.json"
    file.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_definitions(tmp_path, file.name)
