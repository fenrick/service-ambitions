"""Tests for service execution behaviour."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.service_execution import ServiceExecution
from test_service_execution_helpers import _execution


@pytest.mark.asyncio
async def test_run_raises_when_generator_missing(monkeypatch):
    exec_obj = _execution()

    monkeypatch.setattr(ServiceExecution, "_build_generator", lambda self: None)

    with pytest.raises(RuntimeError, match="generator is not initialised"):
        await exec_obj.run()


@pytest.mark.asyncio
async def test_run_raises_when_run_meta_missing(monkeypatch):
    exec_obj = _execution()

    class DummyGenerator:  # pragma: no cover - simple stub
        async def generate_service_evolution_async(self, *a, **k):  # pragma: no cover
            return None

    monkeypatch.setattr(
        ServiceExecution,
        "_build_generator",
        lambda self: setattr(self, "generator", DummyGenerator()),
    )

    async def _fake_prepare_runtimes(self):
        return []

    monkeypatch.setattr(ServiceExecution, "_prepare_runtimes", _fake_prepare_runtimes)
    monkeypatch.setattr(ServiceExecution, "_ensure_run_meta", lambda self: None)
    dummy_env = SimpleNamespace(settings=exec_obj.settings, run_meta=None)
    monkeypatch.setattr("runtime.environment.RuntimeEnv.instance", lambda: dummy_env)

    with pytest.raises(RuntimeError, match="Run metadata is not initialised"):
        await exec_obj.run()
