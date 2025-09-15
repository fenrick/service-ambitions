"""Tests for graceful cancellation and resume behaviour."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.processing_engine import ProcessingEngine
from models import ServiceInput
from runtime.environment import RuntimeEnv
from test_processing_engine_methods import _make_args, _make_settings
from utils import LoggingErrorHandler


def _make_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProcessingEngine, SimpleNamespace, ServiceInput, ServiceInput]:
    settings = _make_settings()
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path)
    engine = ProcessingEngine(args, None)

    svc_a = ServiceInput(service_id="a", name="a", description="", jobs_to_be_done=[])
    svc_b = ServiceInput(service_id="b", name="b", description="", jobs_to_be_done=[])

    def fake_prepare_models() -> None:
        engine.factory = SimpleNamespace()
        engine.system_prompt = ""
        engine.role_ids = []
        engine.services = [svc_a, svc_b]

    def fake_init_sessions() -> None:
        engine.sem = asyncio.Semaphore(2)
        engine.progress = None
        engine.temp_output_dir = None
        engine.error_handler = LoggingErrorHandler()

    monkeypatch.setattr(engine, "_prepare_models", fake_prepare_models)
    monkeypatch.setattr(engine, "_init_sessions", fake_init_sessions)
    return engine, args, svc_a, svc_b


async def _cancel_after_first(
    engine: ProcessingEngine,
    svc_a: ServiceInput,
    svc_b: ServiceInput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_done = asyncio.Event()

    class DummyExecution:
        def __init__(self, runtime, **_: object) -> None:
            self.runtime = runtime

        async def run(self) -> bool:
            if self.runtime.service.service_id == "a":
                self.runtime.line = '{"id": "a"}'
                self.runtime.success = True
                first_done.set()
                return True
            await first_done.wait()
            await asyncio.sleep(0.05)
            self.runtime.line = '{"id": "b"}'
            self.runtime.success = True
            return True

    monkeypatch.setattr("engine.service_execution.ServiceExecution", DummyExecution)
    monkeypatch.setattr("engine.processing_engine.ServiceExecution", DummyExecution)

    async def run_engine() -> None:
        try:
            await engine.run()
        finally:
            await engine.finalise()

    task = asyncio.create_task(run_engine())
    await first_done.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_ctrl_c_flushes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, args, svc_a, svc_b = _make_engine(tmp_path, monkeypatch)
    await _cancel_after_first(engine, svc_a, svc_b, monkeypatch)

    out_path = Path(args.output_file)
    proc_path = out_path.with_name("processed_ids.txt")
    assert out_path.exists()
    assert proc_path.read_text(encoding="utf-8").split() == ["a"]
    assert out_path.read_text(encoding="utf-8").splitlines() == ['{"id": "a"}']


@pytest.mark.asyncio
async def test_resume_after_cancel_continues_from_processed_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, args, svc_a, svc_b = _make_engine(tmp_path, monkeypatch)
    await _cancel_after_first(engine, svc_a, svc_b, monkeypatch)

    resume_args = _make_args(tmp_path)
    resume_args.resume = True
    engine2 = ProcessingEngine(resume_args, None)

    def resume_prepare_models() -> None:
        engine2.factory = SimpleNamespace()
        engine2.system_prompt = ""
        engine2.role_ids = []
        services = [svc_a, svc_b]
        engine2.services = [
            s for s in services if s.service_id not in engine2.processed_ids
        ]

    def resume_init_sessions() -> None:
        engine2.sem = asyncio.Semaphore(2)
        engine2.progress = None
        engine2.temp_output_dir = None
        engine2.error_handler = LoggingErrorHandler()

    monkeypatch.setattr(engine2, "_prepare_models", resume_prepare_models)
    monkeypatch.setattr(engine2, "_init_sessions", resume_init_sessions)

    class ResumeExecution:
        def __init__(self, runtime, **_: object) -> None:
            self.runtime = runtime

        async def run(self) -> bool:
            self.runtime.line = f'{{"id": "{self.runtime.service.service_id}"}}'
            self.runtime.success = True
            return True

    monkeypatch.setattr("engine.service_execution.ServiceExecution", ResumeExecution)
    monkeypatch.setattr("engine.processing_engine.ServiceExecution", ResumeExecution)

    await engine2.run()
    await engine2.finalise()

    out_path = Path(resume_args.output_file)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    proc_path = out_path.with_name("processed_ids.txt")
    processed = proc_path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"id": "a"}', '{"id": "b"}']
    assert processed == ["a", "b"]
