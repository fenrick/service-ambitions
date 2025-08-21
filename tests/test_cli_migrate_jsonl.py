"""Tests for the migrate-jsonl CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cli import _cmd_migrate_jsonl


def test_migrate_jsonl_migrates_records(tmp_path: Path) -> None:
    """Records should be rewritten with the target schema version."""

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        '{"schema_version": "1.0", "service": {}, "plateaus": []}\n',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        from_version="1.0",
        to_version="1.1",
    )

    _cmd_migrate_jsonl(args, None)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["meta"]["schema_version"] == "1.1"


def test_migrate_jsonl_rejects_unknown_versions(tmp_path: Path) -> None:
    """Unsupported version pairs should raise ``ValueError``."""

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        from_version="0.9",
        to_version="1.0",
    )

    with pytest.raises(ValueError):
        _cmd_migrate_jsonl(args, None)
