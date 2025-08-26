# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path

import pytest

from loader import compile_catalogue_for_set, load_mapping_items
from models import MappingSet


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
