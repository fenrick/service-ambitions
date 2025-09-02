# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path

import pytest

import utils.mapping_loader as mapping_loader
from io_utils.loader import (
    _read_file,
    _read_json_file,
    compile_catalogue_for_set,
    load_mapping_items,
)
from models import MappingSet
from utils.error_handler import ErrorHandler
from utils.mapping_loader import FileMappingLoader


def test_load_mapping_items_missing_dir(tmp_path: Path) -> None:
    """load_mapping_items should error when directory is absent."""

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_mapping_items([], data_dir=missing)


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
    result, catalogue_hash = load_mapping_items(sets, data_dir=data_dir)

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


class DummyHandler(ErrorHandler):
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.exceptions: list[Exception | None] = []

    def handle(self, message: str, exc: Exception | None = None) -> None:  # noqa: D401
        self.messages.append(message)
        self.exceptions.append(exc)


def test_read_file_invokes_handler(tmp_path: Path) -> None:
    """_read_file should delegate errors to the handler."""

    handler = DummyHandler()
    missing = tmp_path / "absent.txt"
    with pytest.raises(FileNotFoundError):
        _read_file(missing, error_handler=handler)

    assert handler.messages == [f"Prompt file not found: {missing}"]
    assert isinstance(handler.exceptions[0], FileNotFoundError)


def test_read_json_file_invokes_handler(tmp_path: Path) -> None:
    """_read_json_file should delegate errors to the handler."""

    handler = DummyHandler()
    bad = tmp_path / "data.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _read_json_file(bad, list[str], error_handler=handler)

    assert handler.messages == [f"Error reading JSON file {bad}"]
    assert handler.exceptions[0] is not None
