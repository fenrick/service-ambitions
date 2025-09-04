import importlib
import shutil
import sys
from pathlib import Path

import pytest

cli = importlib.import_module("cli.main")


def _prepare_dataset(tmp_path: Path, valid: bool) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    src = Path(
        "tests/fixtures/services-valid.jsonl"
        if valid
        else "tests/fixtures/services-invalid.jsonl"
    )
    (data_dir / "services.jsonl").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copytree("tests/fixtures/catalogue", data_dir / "catalogue")
    return data_dir


def test_validate_data_dir_passes(monkeypatch, tmp_path: Path) -> None:
    data_dir = _prepare_dataset(tmp_path, True)
    monkeypatch.setattr(sys, "argv", ["main", "validate", "--data", str(data_dir)])
    cli.main()


def test_validate_data_dir_fails(monkeypatch, tmp_path: Path) -> None:
    data_dir = _prepare_dataset(tmp_path, False)
    monkeypatch.setattr(sys, "argv", ["main", "validate", "--data", str(data_dir)])
    with pytest.raises(ValueError, match="Line 1 invalid"):
        cli.main()
