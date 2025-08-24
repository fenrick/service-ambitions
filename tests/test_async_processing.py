import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import generator
from models import ServiceInput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions

    async def run(self, user_prompt: str, output_type):
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt}),
            usage=lambda: SimpleNamespace(total_tokens=1),
        )


def test_process_service_async(monkeypatch):
    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    result, tokens, retries = asyncio.run(gen.process_service(service, "prompt"))
    assert json.loads(result["service"]) == service.model_dump()
    assert tokens == 1
    assert retries == 0


def test_process_service_retries(monkeypatch):
    """Transient failures trigger retries with backoff."""

    attempts = {"count": 0}

    class FlakyAgent(DummyAgent):
        async def run(self, user_prompt: str, output_type):
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
        jobs_to_be_done=[{"name": "job"}],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    result, tokens, retries = asyncio.run(gen.process_service(service, "prompt"))

    assert attempts["count"] == 3
    assert json.loads(result["service"]) == service.model_dump()
    assert tokens == 1
    assert retries == 2


def test_with_retry_logs_attempt(monkeypatch):
    """Retry events emit structured log entries."""

    events: list[tuple[str, dict[str, Any]]] = []

    async def flaky() -> str:
        if not events:
            raise ConnectionError("boom")
        return "ok"

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(
        generator.logfire,
        "warning",
        lambda msg, **attrs: events.append((msg, attrs)),
    )

    asyncio.run(
        generator._with_retry(
            lambda: flaky(), request_timeout=0.1, attempts=2, base=0.01
        )
    )

    assert events[0][0] == "Retrying request"
    assert events[0][1]["attempt"] == 1
    assert "backoff_delay" in events[0][1]


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


def test_with_retry_fails_fast_on_provider_api_error(monkeypatch):
    """Generic provider API errors are not retried."""

    from openai import OpenAIError

    calls = {"count": 0}

    async def fail() -> None:
        calls["count"] += 1
        raise OpenAIError("boom")

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)

    with pytest.raises(OpenAIError):
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
    throttled = {"called": 0}

    async def fake_sleep(seconds: float) -> None:
        slept["delay"] = seconds

    def fake_throttle(_: float) -> None:
        throttled["called"] += 1

    monkeypatch.setattr(generator, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(
        generator,
        "TRANSIENT_EXCEPTIONS",
        generator.TRANSIENT_EXCEPTIONS + (DummyRateLimitError,),
    )
    monkeypatch.setattr(generator.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(generator.random, "random", lambda: 0.0)

    result, retries = asyncio.run(
        generator._with_retry(
            lambda: flaky(),
            request_timeout=0.1,
            attempts=2,
            base=0.1,
            on_retry_after=fake_throttle,
        )
    )

    assert result == "ok"
    assert retries == 1
    assert slept["delay"] == 10.0
    assert throttled["called"] == 1


def test_generate_async_saves_transcripts(tmp_path, monkeypatch):
    """Setting ``transcripts_dir`` writes per-service transcripts."""

    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = ServiceInput(
        service_id="svc-123",
        name="alpha",
        description="Contact j.doe@example.com",
        jobs_to_be_done=[],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    out_file = tmp_path / "out.jsonl"
    transcripts = tmp_path / "t"

    async def run() -> None:
        await gen.generate_async(
            [service], "prompt", str(out_file), transcripts_dir=transcripts
        )

    asyncio.run(run())
    transcript_path = transcripts / "svc-123.json"
    assert transcript_path.exists()
    data = transcript_path.read_text(encoding="utf-8")
    assert "j.doe@example.com" not in data
    assert "svc-123" not in data
    assert "<redacted>" in data


@pytest.mark.asyncio()
async def test_process_all_fsyncs(tmp_path, monkeypatch):
    """Writer flushes and fsyncs periodically to ensure durability."""

    fsync_calls: list[int] = []

    def fake_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr(generator.os, "fsync", fake_fsync)

    async def fake_process_service(self, service, prompt=None):
        return {"id": service.service_id}, 1, 0

    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._prompt = "p"

    services = [
        ServiceInput(service_id="svc1", name="s1", description="d", jobs_to_be_done=[]),
        ServiceInput(service_id="svc2", name="s2", description="d", jobs_to_be_done=[]),
    ]

    await gen._process_all(services, str(tmp_path / "out.jsonl"))
    # Two lines plus a final sync at close.
    assert len(fsync_calls) == 3


@pytest.mark.asyncio()
async def test_run_one_counters_success(tmp_path, monkeypatch):
    """Successful runs update processed and token counters."""

    async def ok(self, service):
        return ({"line": service.service_id}, service.service_id, 1, 0, "success")

    class DummyCounter:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def add(self, value: int) -> None:
            self.calls.append(value)

        @property
        def value(self) -> int:
            return sum(self.calls)

    class DummyMetrics:
        def __init__(self) -> None:
            self.tokens: list[int] = []

        def record_tokens(self, count: int) -> None:
            self.tokens.append(count)

    processed = DummyCounter()
    failed = DummyCounter()
    monkeypatch.setattr(generator, "SERVICES_PROCESSED", processed)
    monkeypatch.setattr(generator, "SERVICES_FAILED", failed)
    monkeypatch.setattr(generator.ServiceAmbitionGenerator, "_process_service_line", ok)

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._limiter = asyncio.Semaphore(1)
    gen._metrics = DummyMetrics()
    outfile = tmp_path / "out.jsonl"
    handle = outfile.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    processed_set: set[str] = set()
    service = ServiceInput(
        service_id="s1", name="n", description="d", jobs_to_be_done=[]
    )

    await gen._run_one(service, handle, lock, processed_set, None, None)
    await asyncio.to_thread(handle.close)

    assert processed.value == 1
    assert failed.value == 0
    assert processed_set == {"s1"}
    assert outfile.read_text() == '{"line":"s1"}\n'
    assert gen._metrics.tokens == [1]


@pytest.mark.asyncio()
async def test_run_one_counters_failure(tmp_path, monkeypatch):
    """Failures increment the failed counter and release tokens."""

    async def bad(self, service):
        return (None, service.service_id, 0, 0, "error")

    class DummyCounter:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def add(self, value: int) -> None:
            self.calls.append(value)

        @property
        def value(self) -> int:
            return sum(self.calls)

    class DummyMetrics:
        def __init__(self) -> None:
            self.tokens: list[int] = []

        def record_tokens(self, count: int) -> None:
            self.tokens.append(count)

    processed = DummyCounter()
    failed = DummyCounter()
    monkeypatch.setattr(generator, "SERVICES_PROCESSED", processed)
    monkeypatch.setattr(generator, "SERVICES_FAILED", failed)
    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "_process_service_line", bad
    )

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._limiter = asyncio.Semaphore(1)
    gen._metrics = DummyMetrics()
    outfile = tmp_path / "out.jsonl"
    handle = outfile.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    processed_set: set[str] = set()
    service = ServiceInput(
        service_id="s2", name="n", description="d", jobs_to_be_done=[]
    )

    await gen._run_one(service, handle, lock, processed_set, None, None)
    await asyncio.to_thread(handle.close)

    assert processed.value == 0
    assert failed.value == 1
    assert processed_set == set()
    assert outfile.read_text() == ""
    assert gen._metrics.tokens == [0]
