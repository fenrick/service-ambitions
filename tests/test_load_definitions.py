# SPDX-License-Identifier: MIT
"""Tests for :func:`loader.load_definitions`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from io_utils.loader import load_definitions


def test_load_definitions_roundtrip(tmp_path: Path) -> None:
    """All definition fields are rendered in the output."""

    content = {
        "title": "Key definitions",
        "bullets": [
            {
                "id": "a",
                "name": "Alpha",
                "aliases": ["A"],
                "short_definition": "Brief",
                "definition": "First letter",
                "decision_rules": ["Rule one"],
                "use_when": ["Example use"],
                "avoid_confusion_with": ["Beta"],
                "examples": ["An example"],
                "non_examples": ["A counterexample"],
                "related_terms": ["omega"],
                "tags": ["tag1"],
                "owner": "Team",
                "last_updated": "2024-01-01",
            }
        ],
    }
    file = tmp_path / "definitions.json"
    file.write_text(json.dumps(content), encoding="utf-8")

    text = load_definitions(tmp_path, file.name)

    assert "## Key definitions" in text
    assert "1. **Alpha** (A)" in text
    assert "Short definition: Brief" in text
    assert "Definition: First letter" in text
    assert "Decision rules:" in text and "Rule one" in text
    assert "Use when:" in text and "Example use" in text
    assert "Avoid confusion with:" in text and "Beta" in text
    assert "Examples:" in text and "An example" in text
    assert "Non-examples:" in text and "A counterexample" in text
    assert "Related terms: omega" in text
    assert "Tags: tag1" in text
    assert "Owner: Team" in text
    assert "Last updated: 2024-01-01" in text


def test_load_definitions_validation_error(tmp_path: Path) -> None:
    """Invalid definition entries raise ``RuntimeError``."""

    content = {
        "title": "Key definitions",
        "bullets": [{"id": "a", "name": "Alpha"}],
    }
    file = tmp_path / "definitions.json"
    file.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_definitions(tmp_path, file.name)
