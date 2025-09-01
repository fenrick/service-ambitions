# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path

import pytest

import utils.mapping_loader as mapping_loader
from loader import compile_catalogue_for_set, load_mapping_items
from models import MappingSet
from utils.mapping_loader import FileMappingLoader


def test_load_mapping_items_missing_dir(tmp_path: Path) -> None:
    """load_mapping_items should error when directory is absent."""

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_mapping_items(missing, [])


def test_load_mapping_items_sorted(tmp_path: Path) -> None:
    """Items are returned sorted by identifier while preserving duplicates."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    items = [
        {"id": "2", "name": "B", "description": "b"},
        {"id": "1", "name": "A", "description": "a"},
        {"id": "2", "name": "C", "description": "c"},
    ]
    (data_dir / "applications.json").write_text(json.dumps(items), encoding="utf-8")

    sets = [MappingSet(name="Apps", file="applications.json", field="applications")]
    result, catalogue_hash = load_mapping_items(data_dir, sets)

    ids = [item.id for item in result["applications"]]
    names = [item.name for item in result["applications"]]
    assert ids == ["1", "2", "2"]
    assert names == ["A", "B", "C"]
    _, per_digest = compile_catalogue_for_set(result["applications"])
    expected = hashlib.sha256(f"applications:{per_digest}".encode("utf-8")).hexdigest()
    assert catalogue_hash == expected


def test_file_mapping_loader_caches(monkeypatch, tmp_path: Path) -> None:
    """FileMappingLoader caches data and can be reset."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    items = [{"id": "1", "name": "A", "description": "a"}]
    (data_dir / "apps.json").write_text(json.dumps(items), encoding="utf-8")
    sets = [MappingSet(name="Apps", file="apps.json", field="apps")]
    loader = FileMappingLoader(data_dir)
    loader.load(sets)

    def boom(*_a, **_k):
        raise RuntimeError("disk access")

    monkeypatch.setattr(mapping_loader, "_read_json_file", boom)
    loader.load(sets)  # uses cache

    loader.clear_cache()
    with pytest.raises(RuntimeError):
        loader.load(sets)
