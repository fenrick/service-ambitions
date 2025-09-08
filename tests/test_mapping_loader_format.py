# SPDX-License-Identifier: MIT
"""Tests for flexible mapping dataset file formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models import MappingSet
from utils.mapping_loader import FileMappingLoader


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def test_loader_supports_object_and_list_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange datasets in both formats
    obj = {
        "field": "applications",
        "label": "Applications",
        "items": [
            {"id": "A1", "name": "CRM", "description": "Customer mgmt"},
        ],
    }
    lst = [
        {"id": "T1", "name": "Cloud", "description": "Public cloud"},
    ]
    _write_json(tmp_path / "apps.json", obj)
    _write_json(tmp_path / "tech.json", lst)

    loader = FileMappingLoader(tmp_path)
    sets = [
        MappingSet(name="Apps", file="apps.json", field="applications"),
        MappingSet(name="Tech", file="tech.json", field="technologies"),
    ]

    # Act
    data, digest = loader.load(sets)

    # Assert
    assert set(data.keys()) == {"applications", "technologies"}
    assert [i.id for i in data["applications"]] == ["A1"]
    assert [i.id for i in data["technologies"]] == ["T1"]
    assert isinstance(digest, str) and len(digest) == 64


def test_loader_warns_on_field_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dataset declares field "applications" but config says "apps"
    obj = {
        "field": "applications",
        "label": "Applications",
        "items": [
            {"id": "A1", "name": "CRM", "description": "Customer mgmt"},
        ],
    }
    _write_json(tmp_path / "apps.json", obj)

    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "utils.mapping_loader.logfire",
        "warning",
        lambda msg, **kw: warnings.append((msg, kw)),
    )

    loader = FileMappingLoader(tmp_path)
    sets = [MappingSet(name="Apps", file="apps.json", field="apps")]
    data, _ = loader.load(sets)

    assert "applications" in warnings[0][1].get("file_field", "")
    assert [i.id for i in data["apps"]] == ["A1"]


def test_loader_digest_includes_embedded_meta(tmp_path: Path) -> None:
    base = {
        "field": "applications",
        "label": "Applications",
        "items": [
            {"id": "A1", "name": "CRM", "description": "Customer mgmt"},
        ],
    }
    path = tmp_path / "apps.json"
    _write_json(path, base)
    sets = [MappingSet(name="Apps", file="apps.json", field="applications")]

    # First load
    loader1 = FileMappingLoader(tmp_path)
    _, digest1 = loader1.load(sets)

    # Change embedded label and reload via a fresh loader to bypass cache
    base["label"] = "Business Applications"
    _write_json(path, base)
    loader2 = FileMappingLoader(tmp_path)
    _, digest2 = loader2.load(sets)

    assert digest1 != digest2
