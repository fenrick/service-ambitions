"""Tests for persistence utilities."""

import json
from pathlib import Path
from unittest.mock import patch

from io_utils.persistence import atomic_write


def test_atomic_write_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    atomic_write(path, [json.dumps({"a": 1})])
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_atomic_write_flushes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    flush_called = False

    real_open = open

    def open_wrapper(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        original_flush = handle.flush

        def tracked_flush() -> None:
            nonlocal flush_called
            flush_called = True
            original_flush()

        handle.flush = tracked_flush
        return handle

    with (
        patch("io_utils.persistence.open", open_wrapper),
        patch("io_utils.persistence.os.fsync") as fsync,
    ):
        atomic_write(path, ["data"])

    assert flush_called
    fsync.assert_called()
