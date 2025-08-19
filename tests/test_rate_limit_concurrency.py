"""Concurrency tests validating rate-limit recovery behaviour."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backpressure
import generator
from backpressure import AdaptiveSemaphore, RollingMetrics


class DummyRateLimitError(Exception):
    """Rate limit error stub exposing ``Retry-After`` headers."""

    def __init__(self) -> None:
        self.response = SimpleNamespace(headers={"Retry-After": "0"})


def test_with_retry_restores_permits(monkeypatch) -> None:
    """Rate-limit waves should throttle concurrency and then recover."""

    async def run() -> None:
        limiter = AdaptiveSemaphore(permits=2)
        throttled = {"count": 0}

        def recording_throttle(delay: float) -> None:
            throttled["count"] += 1
            limiter.throttle(delay)

        attempts = {"count": 0}

        async def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise DummyRateLimitError()
            return "ok"

        async def fast_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(backpressure.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(generator, "RateLimitError", DummyRateLimitError)
        monkeypatch.setattr(
            generator,
            "TRANSIENT_EXCEPTIONS",
            generator.TRANSIENT_EXCEPTIONS + (DummyRateLimitError,),
        )

        result = await generator._with_retry(
            flaky,
            request_timeout=0.1,
            attempts=2,
            base=0.1,
            on_retry_after=recording_throttle,
        )
        await asyncio.sleep(0)

        assert result == "ok"
        assert throttled["count"] == 1
        assert limiter.limit == 2

    asyncio.run(run())


def test_with_retry_records_metrics(monkeypatch) -> None:
    """Latency and rate limits are recorded during retries."""

    async def run() -> None:
        durations: list[float] = []
        rate_limits: list[int] = []

        metrics = RollingMetrics(window=1)

        def rec_latency(duration: float) -> None:
            durations.append(duration)

        def rec_rate_limit() -> None:
            rate_limits.append(1)

        metrics.record_latency = rec_latency  # type: ignore[method-assign]
        metrics.record_rate_limit = rec_rate_limit  # type: ignore[method-assign]

        attempt = {"count": 0}

        async def flaky() -> str:
            attempt["count"] += 1
            if attempt["count"] == 1:
                raise DummyRateLimitError()
            await asyncio.sleep(0.01)
            return "ok"

        async def fast_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(generator.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(generator, "RateLimitError", DummyRateLimitError)
        monkeypatch.setattr(
            generator,
            "TRANSIENT_EXCEPTIONS",
            generator.TRANSIENT_EXCEPTIONS + (DummyRateLimitError,),
        )

        result = await generator._with_retry(
            flaky,
            request_timeout=0.1,
            attempts=2,
            base=0.01,
            metrics=metrics,
        )

        assert result == "ok"
        assert rate_limits == [1]
        assert len(durations) == 1 and durations[0] >= 0.0

    asyncio.run(run())
