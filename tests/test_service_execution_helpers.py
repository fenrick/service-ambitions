from pathlib import Path
from types import SimpleNamespace

import engine.service_execution as se
from engine.service_execution import ServiceExecution
from engine.service_runtime import ServiceRuntime
from models import ServiceInput
from runtime.environment import RuntimeEnv
from settings import Settings
from utils import ErrorHandler


def _settings() -> Settings:
    return Settings(
        model="dummy:model",
        log_level="INFO",
        prompt_dir=Path("."),
        context_id="ctx",
        inspiration="insp",
        concurrency=1,
        openai_api_key="key",
    )


def _factory():
    class Dummy:
        seed = 1

        def get(self, stage: str) -> object:  # pragma: no cover - simple stub
            return object()

        def model_name(self, stage: str) -> str:  # pragma: no cover - simple stub
            return f"{stage}-model"

    return Dummy()


def _runtime() -> ServiceRuntime:
    svc = ServiceInput(
        service_id="svc",
        name="alpha",
        description="d",
        jobs_to_be_done=[],
    )
    return ServiceRuntime(svc)


def _execution() -> ServiceExecution:
    RuntimeEnv.reset()
    RuntimeEnv.initialize(_settings())

    class DummyHandler(ErrorHandler):
        def handle(
            self, message: str, exc: Exception | None = None
        ) -> None:  # pragma: no cover
            pass

    return ServiceExecution(
        _runtime(),
        factory=_factory(),
        system_prompt="",
        transcripts_dir=None,
        role_ids=[],
        temp_output_dir=None,
        allow_prompt_logging=False,
        error_handler=DummyHandler(),
    )


def test_ensure_run_meta_initialises_once(monkeypatch):
    exec_obj = _execution()
    monkeypatch.setattr(se, "load_mapping_items", lambda *a, **k: ({}, "0" * 64))
    env = RuntimeEnv.instance()
    env.run_meta = None
    settings = env.settings
    exec_obj._ensure_run_meta(
        settings,
        "desc-model",
        "feat-model",
        "map-model",
        SimpleNamespace(max_input_tokens=0),
    )
    meta = RuntimeEnv.instance().run_meta
    assert meta is not None
    assert meta.models["descriptions"] == "desc-model"
    exec_obj._ensure_run_meta(
        settings,
        "other-desc",
        "other-feat",
        "other-map",
        SimpleNamespace(max_input_tokens=0),
    )
    assert RuntimeEnv.instance().run_meta is meta
    env.run_meta = None
