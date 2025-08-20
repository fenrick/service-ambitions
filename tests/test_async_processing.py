import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

import backpressure
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
    result, tokens = asyncio.run(gen.process_service(service, "prompt"))
    assert json.loads(result["service"]) == service.model_dump()
    assert tokens == 1


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
    result, tokens = asyncio.run(gen.process_service(service, "prompt"))

    assert attempts["count"] == 3
    assert json.loads(result["service"]) == service.model_dump()
    assert tokens == 1


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

    result = asyncio.run(
        generator._with_retry(
            lambda: flaky(),
            request_timeout=0.1,
            attempts=2,
            base=0.1,
            on_retry_after=fake_throttle,
            metrics=generator.RollingMetrics(window=1),
        )
    )

    assert result == "ok"
    assert slept["delay"] == 10.0
    assert throttled["called"] == 1


def test_generate_async_saves_transcripts(tmp_path, monkeypatch):
    """Setting ``transcripts_dir`` writes per-service transcripts."""

    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = ServiceInput(
        service_id="svc", name="alpha", description="d", jobs_to_be_done=[]
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    out_file = tmp_path / "out.jsonl"
    transcripts = tmp_path / "t"

    async def run() -> None:
        await gen.generate_async(
            [service], "prompt", str(out_file), transcripts_dir=transcripts
        )

    asyncio.run(run())
    transcript_path = transcripts / "svc.json"
    assert transcript_path.exists()


@pytest.mark.asyncio()
async def test_weighted_acquisition(monkeypatch, tmp_path):
    """The generator acquires limiter permits proportional to token estimates."""

    captured: list[int] = []
    metrics_calls: list[SimpleNamespace] = []
    start_calls: list[int] = []
    end_calls: list[int] = []

    class DummyLimiter:
        def __call__(self, weight: int = 1):
            captured.append(weight)

            @asynccontextmanager
            async def manager():
                yield

            return manager()

        def throttle(self, _delay: float) -> None:
            pass

    async def fake_process_service(self, service, prompt=None):
        tokens = 2 if "svc1" in service.service_id else 1
        return {"id": service.service_id}, tokens

    def fake_estimate(prompt: str, expected: int) -> int:
        return 3 if "svc1" in prompt else 1

    monkeypatch.setattr(generator, "estimate_tokens", fake_estimate)
    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(
        backpressure.logfire,
        "metric",
        lambda n, v: metrics_calls.append(SimpleNamespace(name=n, value=v)),
    )

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._prompt = "p"
    gen._limiter = DummyLimiter()

    class DummyMetrics(generator.RollingMetrics):
        def record_start_tokens(self, count: int) -> None:
            start_calls.append(count)
            super().record_start_tokens(count)

        def record_end_tokens(self, count: int) -> None:
            end_calls.append(count)
            super().record_end_tokens(count)

    gen._metrics = DummyMetrics(window=1)
    services = [
        ServiceInput(service_id="svc1", name="s1", description="d", jobs_to_be_done=[]),
        ServiceInput(service_id="svc2", name="s2", description="d", jobs_to_be_done=[]),
    ]
    await gen._process_all(services, str(tmp_path / "out.jsonl"))
    assert sorted(captured) == [1, 3]
    names = {c.name for c in metrics_calls}
    assert {"tokens_per_second", "tokens_in_flight"} <= names
    assert sorted(start_calls) == [1, 3]
    assert sorted(end_calls) == [1, 2]


@pytest.mark.asyncio()
async def test_process_all_without_token_weighting(tmp_path, monkeypatch):
    """Token weighting may be disabled to use uniform permits."""

    captured: list[int] = []
    metrics_calls: list[SimpleNamespace] = []

    class DummyLimiter:
        def __call__(self, weight: int = 1):
            captured.append(weight)

            @asynccontextmanager
            async def manager():
                yield

            return manager()

        def throttle(self, _delay: float) -> None:
            pass

    async def fake_process_service(self, service, prompt=None):
        return {"id": service.service_id}, 1

    def fail_estimate(prompt: str, expected: int) -> int:
        raise AssertionError("estimate_tokens should be bypassed")

    monkeypatch.setattr(generator, "estimate_tokens", fail_estimate)
    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(
        backpressure.logfire,
        "metric",
        lambda n, v: metrics_calls.append(SimpleNamespace(name=n, value=v)),
    )

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace(), token_weighting=False)
    gen._prompt = "p"
    gen._limiter = DummyLimiter()
    gen._metrics = generator.RollingMetrics(window=1)
    services = [
        ServiceInput(service_id="svc1", name="s1", description="d", jobs_to_be_done=[]),
        ServiceInput(service_id="svc2", name="s2", description="d", jobs_to_be_done=[]),
    ]
    await gen._process_all(services, str(tmp_path / "out.jsonl"))
    assert captured == [1, 1]
    names = {c.name for c in metrics_calls}
    assert "tokens_per_second" not in names
    assert "tokens_in_flight" not in names


@pytest.mark.asyncio()
async def test_process_all_fsyncs(tmp_path, monkeypatch):
    """Writer flushes and fsyncs periodically to ensure durability."""

    fsync_calls: list[int] = []

    def fake_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr(generator.os, "fsync", fake_fsync)
    monkeypatch.setattr(generator, "estimate_tokens", lambda _p, _e: 1)

    class DummyLimiter:
        def __call__(self, weight: int = 1):
            @asynccontextmanager
            async def manager():
                yield

            return manager()

        def throttle(self, _delay: float) -> None:
            pass

    async def fake_process_service(self, service, prompt=None):
        return {"id": service.service_id}, 1

    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "process_service", fake_process_service
    )

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace(), flush_interval=1)
    gen._prompt = "p"
    gen._limiter = DummyLimiter()

    services = [
        ServiceInput(service_id="svc1", name="s1", description="d", jobs_to_be_done=[]),
        ServiceInput(service_id="svc2", name="s2", description="d", jobs_to_be_done=[]),
    ]

    await gen._process_all(services, str(tmp_path / "out.jsonl"))
    # Two lines plus a final sync at close.
    assert len(fsync_calls) == 3


@pytest.mark.asyncio()
async def test_run_one_counters_success(monkeypatch):
    """Successful runs update processed and token counters."""

    class DummyLimiter:
        def __call__(self, weight: int = 1):
            @asynccontextmanager
            async def manager():
                yield

            return manager()

        def throttle(self, _delay: float) -> None:
            pass

    async def ok(self, service, transcripts_dir):
        return ("line", service.service_id, 1)

    class DummyCounter:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def add(self, value: int) -> None:
            self.calls.append(value)

        @property
        def value(self) -> int:
            return sum(self.calls)

    processed = DummyCounter()
    failed = DummyCounter()
    tokens = DummyCounter()

    monkeypatch.setattr(generator, "SERVICES_PROCESSED", processed)
    monkeypatch.setattr(generator, "SERVICES_FAILED", failed)
    monkeypatch.setattr(generator, "TOKENS_IN_FLIGHT", tokens)
    monkeypatch.setattr(generator.ServiceAmbitionGenerator, "_process_service_line", ok)
    monkeypatch.setattr(generator, "estimate_tokens", lambda *_: 5)

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._limiter = DummyLimiter()
    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    service = ServiceInput(
        service_id="s1", name="n", description="d", jobs_to_be_done=[]
    )

    await gen._run_one(service, queue, None)

    assert processed.value == 1
    assert failed.value == 0
    assert tokens.calls == [5, -5]
    assert queue.qsize() == 1


@pytest.mark.asyncio()
async def test_run_one_counters_failure(monkeypatch):
    """Failures increment the failed counter and release tokens."""

    class DummyLimiter:
        def __call__(self, weight: int = 1):
            @asynccontextmanager
            async def manager():
                yield

            return manager()

        def throttle(self, _delay: float) -> None:
            pass

    async def bad(self, service, transcripts_dir):
        return (None, service.service_id, 0)

    class DummyCounter:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def add(self, value: int) -> None:
            self.calls.append(value)

        @property
        def value(self) -> int:
            return sum(self.calls)

    processed = DummyCounter()
    failed = DummyCounter()
    tokens = DummyCounter()

    monkeypatch.setattr(generator, "SERVICES_PROCESSED", processed)
    monkeypatch.setattr(generator, "SERVICES_FAILED", failed)
    monkeypatch.setattr(generator, "TOKENS_IN_FLIGHT", tokens)
    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "_process_service_line", bad
    )
    monkeypatch.setattr(generator, "estimate_tokens", lambda *_: 3)

    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    gen._limiter = DummyLimiter()
    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    service = ServiceInput(
        service_id="s2", name="n", description="d", jobs_to_be_done=[]
    )

    await gen._run_one(service, queue, None)

    assert processed.value == 0
    assert failed.value == 1
    assert tokens.calls == [3, -3]
    assert queue.qsize() == 0


def test_generate_async_consumes_in_batches(tmp_path, monkeypatch):
    """The generator only consumes inputs in ``batch_size`` chunks."""

    counter = {"count": 0}

    def services() -> Iterable[ServiceInput]:
        for i in range(10):  # Yield a fixed number of services lazily
            counter["count"] += 1
            yield ServiceInput(
                service_id=f"svc{i}",
                name=f"s{i}",
                description="d",
                jobs_to_be_done=[],
            )

    gen = generator.ServiceAmbitionGenerator(
        SimpleNamespace(), concurrency=2, batch_size=3
    )

    async def run() -> None:
        event = asyncio.Event()

        async def fake_process_service(self, service, prompt=None):
            await event.wait()
            return {"id": service.service_id}, 1

        monkeypatch.setattr(
            generator.ServiceAmbitionGenerator, "process_service", fake_process_service
        )

        task = asyncio.create_task(
            gen.generate_async(services(), "p", str(tmp_path / "out.jsonl"))
        )
        await asyncio.sleep(0.1)  # Allow the first batch to be scheduled
        assert counter["count"] == 3  # Only one batch should be consumed
        event.set()
        await task

    asyncio.run(run())
