import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.processing_engine import ProcessingEngine
from engine.service_runtime import ServiceRuntime
from models import ServiceInput
from runtime.environment import RuntimeEnv


def _make_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_file=str(tmp_path / "out.jsonl"),
        resume=False,
        transcripts_dir=None,
        seed=0,
        roles_file=str(tmp_path / "roles.json"),
        input_file="services.json",
        max_services=None,
        progress=False,
        temp_output_dir=None,
        dry_run=False,
        allow_prompt_logging=False,
    )


def _make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt",
        openai_api_key="key",
        models=None,
        reasoning=None,
        prompt_dir=Path("prompts"),
        mapping_data_dir=Path("mapping"),
        context_id="ctx",
        inspiration="insp",
        concurrency=2,
        web_search=False,
    )


def test_prepare_models_uses_runtimeenv(monkeypatch, tmp_path):
    settings = _make_settings()
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    captured = {}

    def fake_create_model_factory():
        captured["factory"] = engine.settings
        return "factory"

    def fake_load_services():
        captured["services"] = engine.settings
        svc = ServiceInput(
            service_id="svc",
            name="alpha",
            description="desc",
            jobs_to_be_done=[],
        )
        return "prompt", ["role"], [svc]

    monkeypatch.setattr(engine, "_create_model_factory", fake_create_model_factory)
    monkeypatch.setattr(engine, "_load_services", fake_load_services)

    engine._prepare_models()

    assert captured["factory"] is settings
    assert captured["services"] is settings
    assert engine.factory == "factory"
    assert engine.system_prompt == "prompt"
    assert engine.role_ids == ["role"]
    assert engine.services and engine.services[0].service_id == "svc"


def test_init_sessions_uses_runtimeenv(monkeypatch, tmp_path):
    settings = _make_settings()
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    captured = {}

    def fake_setup_concurrency():
        captured["settings"] = engine.settings
        return asyncio.Semaphore(1)

    monkeypatch.setattr(engine, "_setup_concurrency", fake_setup_concurrency)
    monkeypatch.setattr(engine, "_create_progress", lambda total: "progress")

    engine._init_sessions(5)

    assert captured["settings"] is settings
    assert isinstance(engine.sem, asyncio.Semaphore)
    assert engine.progress == "progress"
    assert engine.temp_output_dir is None
    assert engine.error_handler is not None


@pytest.mark.asyncio
async def test_finalise_writes_runtime_lines(tmp_path):
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    svc = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=[],
    )
    runtime = ServiceRuntime(svc)
    runtime.line = '{"ok": true}'
    runtime.success = True
    engine.runtimes.append(runtime)

    await engine.finalise()

    out_path = Path(args.output_file)
    assert out_path.exists()
    assert out_path.read_text().strip() == runtime.line
    assert engine.new_ids == {"svc"}


@pytest.mark.asyncio
async def test_generate_evolution_aggregates_success(monkeypatch, tmp_path):
    settings = _make_settings()
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    outcomes = {"a": True, "b": False}

    class DummyExecution:
        def __init__(self, runtime, **kwargs):
            self.runtime = runtime

        async def run(self) -> bool:  # pragma: no cover - trivial
            self.runtime.success = outcomes[self.runtime.service.service_id]
            return self.runtime.success

    monkeypatch.setattr("engine.service_execution.ServiceExecution", DummyExecution)
    monkeypatch.setattr("engine.processing_engine.ServiceExecution", DummyExecution)

    services = [
        ServiceInput(
            service_id="a",
            name="alpha",
            description="d",
            jobs_to_be_done=[],
        ),
        ServiceInput(
            service_id="b",
            name="beta",
            description="d",
            jobs_to_be_done=[],
        ),
    ]
    sem = asyncio.Semaphore(2)
    handler = SimpleNamespace(handle=lambda *a, **k: None)
    engine.factory = SimpleNamespace()
    engine.system_prompt = ""
    engine.role_ids = []
    engine.sem = sem
    engine.progress = None
    engine.temp_output_dir = None
    engine.error_handler = handler
    ok = await engine._generate_evolution(services)
    assert ok is False
    assert [r.service.service_id for r in engine.runtimes] == ["a", "b"]
