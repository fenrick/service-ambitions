import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import generator
from models import ServiceInput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions

    async def run(self, user_prompt: str, output_type):  # pragma: no cover - stub
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt})
        )


def test_process_service_async(monkeypatch):
    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=["job"],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    result = asyncio.run(gen.process_service(service, "prompt"))
    assert json.loads(result["service"]) == service.model_dump()


def test_process_service_retries(monkeypatch):
    """Transient failures trigger retries with backoff."""

    attempts = {"count": 0}

    class FlakyAgent(DummyAgent):
        async def run(self, user_prompt: str, output_type):  # pragma: no cover - stub
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary")
            return await super().run(user_prompt, output_type)

    async def fast_sleep(_: float) -> None:
        """Skip real waiting during backoff."""

        return None

    monkeypatch.setattr(generator, "Agent", FlakyAgent)
    monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)

    service = ServiceInput(
        service_id="svc2",
        name="beta",
        description="desc",
        jobs_to_be_done=["job"],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    result = asyncio.run(gen.process_service(service, "prompt"))

    assert attempts["count"] == 3
    assert json.loads(result["service"]) == service.model_dump()


def test_generator_rejects_invalid_concurrency():
    with pytest.raises(ValueError):
        generator.ServiceAmbitionGenerator(SimpleNamespace(), concurrency=0)
