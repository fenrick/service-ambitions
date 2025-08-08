import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import generator
from generator import ServiceAmbitionGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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

    async def fake_process_service(self, service, prompt=None):
        assert prompt is not None
        return {"service": service["name"], "prompt": prompt[:3]}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "generate-ambitions",
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
    monkeypatch.setattr(sys, "argv", ["main", "generate-ambitions"])
    with pytest.raises(RuntimeError) as excinfo:
        cli.main()
    assert "openai_api_key" in str(excinfo.value)


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

    async def fake_process_service(self, service, prompt=None):
        assert prompt is not None
        return {"prompt": prompt}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "generate-ambitions",
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

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "generate-ambitions",
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

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    captured: dict[str, object] = {}
    called: dict[str, bool] = {"installed": False}

    def fake_configure(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

    def fake_install(
        modules, *, min_duration, check_imported_modules="error"
    ):  # type: ignore[no-untyped-def]
        called["installed"] = True
        captured["modules"] = modules
        captured["min_duration"] = min_duration
        captured["check"] = check_imported_modules

    dummy_module = SimpleNamespace(
        configure=fake_configure,
        instrument_system_metrics=lambda **kwargs: None,
        install_auto_tracing=fake_install,
    )
    monkeypatch.setitem(sys.modules, "logfire", dummy_module)

    argv = [
        "main",
        "generate-ambitions",
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
    assert captured["modules"] == []
    assert captured["min_duration"] == 0
    assert called["installed"]


def test_cli_rejects_invalid_concurrency(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(cli, "load_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(cli, "load_services", lambda *a, **k: [])
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: object())

    argv = ["main", "generate-ambitions", "--concurrency", "0", "--model", "test"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="concurrency must be a positive integer"):
        cli.main()


def test_cli_help_shows_parameters(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main", "generate-ambitions", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "Generate service ambitions" in out
    assert "--concurrency" in out


def test_cli_verbose_logging(tmp_path, monkeypatch, capsys):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "service_feature_plateaus.md").write_text("p", encoding="utf-8")
    (base / "definitions.md").write_text("d", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    argv = [
        "main",
        "generate-ambitions",
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
        "--model",
        "test",
        "-v",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert "Processing service alpha" in capsys.readouterr().err
