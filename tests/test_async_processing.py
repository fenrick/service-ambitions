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
                # Connection errors are considered transient and should retry.
                raise ConnectionError("temporary")
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


def test_with_retry_fails_fast_on_non_transient(monkeypatch):
    """Non-transient errors are not retried."""

    calls = {"count": 0}

    async def fail():
        calls["count"] += 1
        raise ValueError("boom")

    # Ensure a failure does not trigger a backoff sleep.
    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)

    with pytest.raises(ValueError):
        asyncio.run(
            generator._with_retry(
                lambda: fail(), request_timeout=0.1, attempts=5, base=0.01
            )
        )

    assert calls["count"] == 1


def test_with_retry_honours_retry_after(monkeypatch):
    """Retry-After hints extend the backoff delay."""

    class DummyRateLimitError(Exception):
        """Minimal RateLimitError stub with headers."""

        def __init__(self) -> None:
            self.response = SimpleNamespace(headers={"Retry-After": "10"})

    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise DummyRateLimitError()
        return "ok"

    slept = {"delay": 0.0}

    async def fake_sleep(seconds: float) -> None:
        slept["delay"] = seconds

    monkeypatch.setattr(generator, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(
        generator,
        "TRANSIENT_EXCEPTIONS",
        generator.TRANSIENT_EXCEPTIONS + (DummyRateLimitError,),
    )
    monkeypatch.setattr(generator.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(generator.random, "random", lambda: 0.0)

    result = asyncio.run(
        generator._with_retry(
            lambda: flaky(), request_timeout=0.1, attempts=2, base=0.1
        )
    )

    assert result == "ok"
    assert slept["delay"] == 10.0
