"""Tests for the migrate-jsonl CLI command and helper."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cli
from migrate_jsonl import migrate_jsonl, migrate_record

LEGACY = {
    "id": "svc-1",
    "name": "Legacy",
    "description": "desc",
    "jobs": ["job"],
    "features": [{"id": "feat-1", "name": "Feature", "description": "fdesc"}],
}

EXPECTED = {
    "service_id": "svc-1",
    "name": "Legacy",
    "description": "desc",
    "jobs_to_be_done": [{"name": "job"}],
    "features": [{"feature_id": "feat-1", "name": "Feature", "description": "fdesc"}],
}


def test_migrate_record():
    """migrate_record should convert 1.0 objects to 1.x schema."""

    assert migrate_record(LEGACY) == EXPECTED


def test_helper_migrate_jsonl(tmp_path: Path):
    """migrate_jsonl should read and write migrated records."""

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(json.dumps(LEGACY) + "\n", encoding="utf-8")

    written = migrate_jsonl(input_path, output_path)
    assert written == 1

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload == EXPECTED


def test_cli_migrate_jsonl(tmp_path: Path, monkeypatch):
    """CLI command should migrate files on disk."""

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(json.dumps(LEGACY) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(logfire_token=None, log_level="INFO"),
    )
    argv = [
        "main",
        "migrate-jsonl",
        "--input-file",
        str(input_path),
        "--output-file",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload == EXPECTED
