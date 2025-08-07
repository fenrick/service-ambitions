import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import service_ambitions.cli as cli
import service_ambitions.generator as generator
from service_ambitions.generator import ServiceAmbitionGenerator


def test_cli_generates_output(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text(
        "You are a helpful assistant.", encoding="utf-8"
    )
    (base / "service_feature_plateaus.md").write_text("p", encoding="utf-8")
    (base / "definitions.md").write_text("d", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n{"name": "beta"}\n')

    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    async def fake_process_service(self, service, prompt):
        return {"service": service["name"], "prompt": prompt[:3]}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "--prompt-dir",
        str(base),
        "--context-id",
        "ctx",
        "--inspirations-id",
        "insp",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--concurrency",
        "2",
        "--model",
        "test",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    lines = output_file.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"service": "alpha", "prompt": "You"},
        {"service": "beta", "prompt": "You"},
    ]


def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["main"])
    with pytest.raises(RuntimeError):
        cli.main()


def test_cli_switches_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "alpha.md").write_text("Alpha", encoding="utf-8")
    (base / "situational_context" / "beta.md").write_text("Beta", encoding="utf-8")
    (base / "service_feature_plateaus.md").write_text("p", encoding="utf-8")
    (base / "definitions.md").write_text("d", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n')
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    async def fake_process_service(self, service, prompt):
        return {"prompt": prompt}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "--context-id",
        "beta",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--model",
        "test",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    line = json.loads(output_file.read_text().strip())
    assert line["prompt"].startswith("Beta")


def test_cli_model_instantiation_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "university.md").write_text("ctx", encoding="utf-8")
    (base / "service_feature_plateaus.md").write_text("p", encoding="utf-8")
    (base / "definitions.md").write_text("d", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n')
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("MODEL", "test-model")

    captured: dict[str, str] = {}

    def fake_build_model(model_name: str, api_key: str):
        captured["model"] = model_name
        captured["api_key"] = api_key
        return "test"

    monkeypatch.setattr(generator, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "build_model", fake_build_model)

    async def fake_process_service(self, service, prompt):
        return {"ok": True}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert captured["model"] == "test-model"
    assert captured["api_key"] == "dummy"


def test_cli_enables_logfire(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "university.md").write_text("ctx", encoding="utf-8")
    (base / "service_feature_plateaus.md").write_text("p", encoding="utf-8")
    (base / "definitions.md").write_text("d", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n')
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("LOGFIRE_TOKEN", "lf-key")

    async def fake_process_service(self, service, prompt):
        return {"ok": True}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    captured: dict[str, str | None] = {}

    def fake_configure(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

    dummy_module = SimpleNamespace(configure=fake_configure)
    monkeypatch.setitem(sys.modules, "logfire", dummy_module)

    argv = [
        "main",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--logfire-service",
        "demo",
        "--model",
        "test",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert captured["token"] == "lf-key"
    assert captured["service_name"] == "demo"
