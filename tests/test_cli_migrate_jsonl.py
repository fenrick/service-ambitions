import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli
import migration


def test_cli_migrates_jsonl(tmp_path, monkeypatch):
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        '{"schema_version": "1.0", "val": 1}\n{"schema_version": "1.0", "val": 2}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"

    settings = SimpleNamespace(log_level="INFO", logfire_token=None)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    calls = []

    def fake_migrate(record, source, target):
        calls.append((source, target))
        return migration.migrate_record(record, source, target)

    monkeypatch.setattr(migration, "migrate_record", fake_migrate)

    argv = [
        "main",
        "migrate-jsonl",
        "--from",
        "1.0",
        "--to",
        "2.0",
        "--input-file",
        str(input_path),
        "--output-file",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    contents = output_path.read_text(encoding="utf-8").splitlines()
    out_lines = [json.loads(line) for line in contents]
    assert all(item["schema_version"] == "2.0" for item in out_lines)
    assert calls == [("1.0", "2.0"), ("1.0", "2.0")]
