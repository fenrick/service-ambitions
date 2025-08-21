import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import generator
from generator import ServiceAmbitionGenerator
from monitoring import LOG_FILE_NAME

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyFactory:
    def __init__(self, *a, **k):
        pass

    def model_name(self, stage, override=None):
        return "dummy"

    def get(self, stage, override=None):
        return object()


cli.ModelFactory = DummyFactory  # type: ignore[assignment,misc]


def test_cli_generates_output(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text(
        "You are a helpful assistant.", encoding="utf-8"
    )
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n{"name": "beta"}\n', encoding="utf-8")

    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=2,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        models=None,
        web_search=False,
        token_weighting=True,
    )

    async def fake_process_service(self, service, prompt=None):
        assert prompt is not None
        return {"service": service["name"], "prompt": prompt[:3]}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--model",
        "test",
        "--max-services",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"service": "alpha", "prompt": "You"}
    ]


def test_cli_dry_run_skips_processing(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        models=None,
        web_search=False,
        token_weighting=True,
    )

    called = {"ran": False}

    async def fake_generate_async(
        self, services, prompt, output_path, progress=None, transcripts_dir=None
    ):
        called["ran"] = True

    monkeypatch.setattr(ServiceAmbitionGenerator, "generate_async", fake_generate_async)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert not output_file.exists()
    assert not called["ran"]


def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["main", "generate-ambitions"])
    with pytest.raises(RuntimeError) as excinfo:
        cli.main()
    assert "openai_api_key" in str(excinfo.value)


def test_cli_switches_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "alpha.md").write_text("Alpha", encoding="utf-8")
    (base / "situational_context" / "beta.md").write_text("Beta", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"name": "d", "description": "d"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": "p"}]',
        encoding="utf-8",
    )
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="test",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="beta",
        inspiration="general",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )

    async def fake_process_service(self, service, prompt=None):
        assert prompt is not None
        return {"prompt": prompt}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

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

    line = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert line["prompt"].startswith("Beta")


def test_cli_model_instantiation_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "university.md").write_text("ctx", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"name": "d", "description": "d"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": "p"}]',
        encoding="utf-8",
    )
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="test-model",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="university",
        inspiration="general",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )

    captured: dict[str, str] = {}

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed=None,
        reasoning=None,
        web_search=False,
    ):
        captured["model"] = model_name
        captured["api_key"] = api_key
        captured["seed"] = seed
        captured["web_search"] = web_search
        return "test"

    monkeypatch.setattr(generator, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "build_model", fake_build_model)

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

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
    assert captured["seed"] is None
    assert captured["web_search"] is False


def test_cli_seed_sets_random(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )

    captured: dict[str, int | None] = {}

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed=None,
        reasoning=None,
        web_search=False,
    ):
        captured["seed"] = seed
        captured["web_search"] = web_search
        return "test"

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}, 0

    monkeypatch.setattr(generator, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "build_model", fake_build_model)
    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--seed",
        "123",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert captured["seed"] == 123
    assert captured["web_search"] is False
    assert random.random() == pytest.approx(random.Random(123).random())


def test_cli_enables_logfire(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "university.md").write_text("ctx", encoding="utf-8")
    (base / "inspirations" / "general.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"name": "d", "description": "d"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": "p"}]',
        encoding="utf-8",
    )
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="test",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="university",
        inspiration="general",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token="lf-key",
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    captured: dict[str, object] = {}
    called: dict[str, bool] = {"installed": False}

    def fake_configure(**kwargs):
        captured.update(kwargs)

    def fake_install(modules, *, min_duration, check_imported_modules="error"):
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
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert captured["token"] == "lf-key"
    assert captured["service_name"] == "demo"
    assert captured["modules"] == []
    assert captured["min_duration"] == 0
    assert called["installed"]


def test_cli_no_logs_disables_logging(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token="token",
        reasoning=None,
        models=None,
        web_search=False,
        token_weighting=True,
        request_timeout=60,
        retries=5,
        retry_base_delay=0.5,
    )

    called = {"init": False}

    def fake_init(token):
        called["init"] = True

    async def fake_generate_async(
        self, services, prompt, output_path, progress=None, transcripts_dir=None
    ):
        assert transcripts_dir is None
        Path(output_path).write_text("{}\n", encoding="utf-8")
        return {"svc-1"}

    monkeypatch.setattr(ServiceAmbitionGenerator, "generate_async", fake_generate_async)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "init_logfire", fake_init)

    seen: list[bool] = []

    class DummySession:
        def __init__(self, *a, log_prompts=True, **k):
            seen.append(log_prompts)

    monkeypatch.setattr(cli, "ConversationSession", DummySession)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--no-logs",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)

    cli.main()

    assert output_file.exists()
    assert not (tmp_path / LOG_FILE_NAME).exists()
    assert not (tmp_path / "_transcripts").exists()
    assert not called["init"]
    assert all(not flag for flag in seen)


def test_cli_rejects_invalid_concurrency(monkeypatch):
    monkeypatch.setattr(cli, "load_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(cli, "load_services", lambda *a, **k: [])
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: object())

    settings = SimpleNamespace(
        model="test",
        log_level="INFO",
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        concurrency=0,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = ["main", "generate-ambitions", "--model", "test"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="concurrency must be a positive integer"):
        cli.main()


def test_cli_help_shows_parameters(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main", "generate-ambitions", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "Generate service ambitions" in out
    assert "--input-file" in out
    assert "--expected-output-tokens" in out


def test_cli_passes_expected_output_tokens(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        models=None,
        web_search=False,
        token_weighting=True,
    )

    captured: dict[str, int] = {}

    def fake_init(
        self,
        model,
        concurrency=5,
        batch_size=None,
        request_timeout=60,
        retries=5,
        retry_base_delay=0.5,
        expected_output_tokens=256,
        token_weighting=True,
    ) -> None:
        captured["expected"] = expected_output_tokens

    monkeypatch.setattr(ServiceAmbitionGenerator, "__init__", fake_init)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--dry-run",
        "--expected-output-tokens",
        "512",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert captured["expected"] == 512


def test_cli_verbose_logging(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n', encoding="utf-8")
    output_file = tmp_path / "output.jsonl"

    settings = SimpleNamespace(
        model="test",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
        token_weighting=True,
    )

    async def fake_process_service(self, service, prompt=None):
        return {"ok": True}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "-v",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    monkeypatch.chdir(tmp_path)

    cli.main()

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = Path(LOG_FILE_NAME)
    assert "Processing service alpha" in log_file.read_text(encoding="utf-8")


def test_cli_resume_skips_processed(tmp_path, monkeypatch):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("c", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("i", encoding="utf-8")
    (base / "task_definition.md").write_text("t", encoding="utf-8")
    (base / "response_structure.md").write_text("r", encoding="utf-8")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text(
        '{"service_id": "1", "name": "alpha", "description": "d", "jobs_to_be_done":'
        ' [{"name": "j"}]}\n{"service_id": "2", "name": "beta", "description": "d",'
        ' "jobs_to_be_done": [{"name": "j"}]}\n',
        encoding="utf-8",
    )

    output_file = tmp_path / "output.jsonl"
    output_file.write_text('{"service_id": "1"}\n', encoding="utf-8")
    (tmp_path / "processed_ids.txt").write_text("1\n", encoding="utf-8")

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(base),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        web_search=False,
    )

    processed: list[str] = []

    async def fake_process_service(self, service, prompt=None):
        processed.append(service.service_id)
        return {"service_id": service.service_id}, 0

    monkeypatch.setattr(
        ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    argv = [
        "main",
        "generate-ambitions",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--continue",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert processed == ["2"]
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert (tmp_path / "processed_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines() == ["1", "2"]


def test_cli_flushes_logfire_on_error(monkeypatch):
    """``logfire.force_flush`` runs even when the CLI raises."""

    async def boom():
        raise RuntimeError("fail")

    called: dict[str, bool] = {"flushed": False}

    def fake_flush() -> None:
        called["flushed"] = True

    monkeypatch.setattr(cli, "main_async", boom)
    monkeypatch.setattr(cli.logfire, "force_flush", fake_flush)

    with pytest.raises(RuntimeError):
        cli.main()

    assert called["flushed"]


def test_cli_validate_only(tmp_path, monkeypatch):
    output_file = tmp_path / "out.jsonl"
    output_file.write_text('{"a": 1}\n', encoding="utf-8")

    settings = SimpleNamespace(
        model="cfg",
        log_level="INFO",
        prompt_dir=str(tmp_path),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        batch_size=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        openai_api_key="dummy",
        logfire_token=None,
        reasoning=None,
        models=None,
        web_search=False,
        token_weighting=True,
    )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    called = {"ran": False}

    async def fake_generate_async(self, *a, **k):
        called["ran"] = True

    monkeypatch.setattr(ServiceAmbitionGenerator, "generate_async", fake_generate_async)

    argv = [
        "main",
        "generate-ambitions",
        "--output-file",
        str(output_file),
        "--validate-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert not called["ran"]
