import json
from pathlib import Path

import pytest

from loader import load_mapping_items


def test_load_mapping_items_missing_dir(tmp_path: Path) -> None:
    """load_mapping_items should error when directory is absent."""

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_mapping_items(missing)


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

    result = load_mapping_items(data_dir)

    ids = [item.id for item in result["applications"]]
    names = [item.name for item in result["applications"]]
    assert ids == ["1", "2", "2"]
    assert names == ["A", "B", "C"]
