"""Concurrency tests validating rate-limit recovery behaviour."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backpressure
import generator
from backpressure import AdaptiveSemaphore


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
