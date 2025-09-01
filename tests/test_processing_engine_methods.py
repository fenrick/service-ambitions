import asyncio
from pathlib import Path
from types import SimpleNamespace

from engine.processing_engine import ProcessingEngine
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

    def fake_create_model_factory(s):
        captured["factory"] = s
        return "factory"

    def fake_load_services(s):
        captured["services"] = s
        svc = ServiceInput(
            service_id="svc",
            name="alpha",
            description="desc",
            jobs_to_be_done=[],
        )
        return "prompt", ["role"], [svc]

    monkeypatch.setattr(engine, "_create_model_factory", fake_create_model_factory)
    monkeypatch.setattr(engine, "_load_services", fake_load_services)

    factory, system_prompt, role_ids, services = engine._prepare_models()

    assert captured["factory"] is settings
    assert captured["services"] is settings
    assert factory == "factory"
    assert system_prompt == "prompt"
    assert role_ids == ["role"]
    assert services[0].service_id == "svc"


def test_init_sessions_uses_runtimeenv(monkeypatch, tmp_path):
    settings = _make_settings()
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    captured = {}

    def fake_setup_concurrency(s):
        captured["settings"] = s
        return asyncio.Semaphore(1), asyncio.Lock()

    monkeypatch.setattr(engine, "_setup_concurrency", fake_setup_concurrency)
    monkeypatch.setattr(engine, "_create_progress", lambda total: "progress")

    sem, lock, progress, temp_dir, handler = engine._init_sessions(5)

    assert captured["settings"] is settings
    assert isinstance(sem, asyncio.Semaphore)
    assert isinstance(lock, asyncio.Lock)
    assert progress == "progress"
    assert temp_dir is None
    assert handler is not None
