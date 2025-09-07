import asyncio
from pathlib import Path
from types import SimpleNamespace

import engine.service_execution as se
from engine.service_execution import ServiceExecution
from engine.service_runtime import ServiceRuntime
from models import ServiceInput
from runtime.environment import RuntimeEnv
from runtime.settings import Settings
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


def test_build_generator_sets_attributes(monkeypatch):
    exec_obj = _execution()

    monkeypatch.setattr(se, "Agent", lambda *a, **k: object())
    monkeypatch.setattr(se, "ConversationSession", lambda *a, **k: object())

    class DummyGenerator:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(se, "PlateauGenerator", DummyGenerator)

    exec_obj._build_generator()

    assert isinstance(exec_obj.generator, DummyGenerator)
    assert exec_obj.desc_name == "descriptions-model"
    assert exec_obj.feat_name == "features-model"
    assert exec_obj.map_name == "mapping-model"
    assert exec_obj.feat_model is not None


def test_ensure_run_meta_initialises_once(monkeypatch):
    exec_obj = _execution()
    monkeypatch.setattr(se, "load_mapping_items", lambda *a, **k: ({}, "0" * 64))
    env = RuntimeEnv.instance()
    env.run_meta = None
    exec_obj.desc_name = "desc-model"
    exec_obj.feat_name = "feat-model"
    exec_obj.map_name = "map-model"
    exec_obj.feat_model = SimpleNamespace(max_input_tokens=0)
    exec_obj._ensure_run_meta()
    meta = RuntimeEnv.instance().run_meta
    assert meta is not None
    assert meta.models["descriptions"] == "desc-model"
    exec_obj.desc_name = "other-desc"
    exec_obj.feat_name = "other-feat"
    exec_obj.map_name = "other-map"
    exec_obj._ensure_run_meta()
    assert RuntimeEnv.instance().run_meta is meta
    env.run_meta = None


def test_prepare_runtimes_uses_internal_generator(monkeypatch):
    exec_obj = _execution()

    class DummySession:
        def add_parent_materials(self, _service):
            return None

    class DummyGenerator:
        description_session = DummySession()

        async def _request_descriptions_async(self, names, session):
            return {n: f"{n}-desc" for n in names}

    exec_obj.generator = DummyGenerator()
    monkeypatch.setattr(se, "default_plateau_names", lambda: ["p1", "p2"])
    monkeypatch.setattr(se, "default_plateau_map", lambda: {"p1": 1, "p2": 2})

    runtimes = asyncio.run(exec_obj._prepare_runtimes())

    assert len(runtimes) == 2
    assert runtimes[0].plateau == 1
    assert runtimes[0].description == "p1-desc"
