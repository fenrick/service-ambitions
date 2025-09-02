# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest
from pydantic import BaseModel

from io_utils.diagnostics import validate_jsonl


class Item(BaseModel):
    x: int


def test_validate_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    assert validate_jsonl(path, Item) == 2


def test_validate_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"x": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_jsonl(path, Item)
