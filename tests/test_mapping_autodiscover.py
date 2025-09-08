# SPDX-License-Identifier: MIT
"""Tests for auto-discovering mapping datasets from object-form files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from io_utils.loader import load_mapping_items


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def test_autodiscover_uses_object_form_only(tmp_path: Path) -> None:
    # Object-form datasets with embedded fields
    apps = {
        "field": "applications",
        "label": "Applications",
        "items": [{"id": "A", "name": "App", "description": "d"}],
    }
    tech = {
        "field": "technologies",
        "label": "Technologies",
        "items": [{"id": "T", "name": "Tech", "description": "d"}],
    }
    # List-only dataset (ignored in discovery)
    info = [{"id": "I", "name": "Info", "description": "d"}]

    _write_json(tmp_path / "applications.json", apps)
    _write_json(tmp_path / "technologies.json", tech)
    _write_json(tmp_path / "information.json", info)

    data, digest = load_mapping_items([], data_dir=tmp_path)

    assert set(data.keys()) == {"applications", "technologies"}
    assert [i.id for i in data["applications"]] == ["A"]
    assert [i.id for i in data["technologies"]] == ["T"]
    assert isinstance(digest, str) and len(digest) == 64
