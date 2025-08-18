import asyncio
from types import SimpleNamespace

import pytest

import backpressure
from backpressure import AdaptiveSemaphore, RollingMetrics

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_adaptive_semaphore_throttles():
    """Semaphore halves concurrency and restores it."""

    sem = AdaptiveSemaphore(4, ramp_interval=0.01)
    # Trigger a throttle event with a short delay for the test.
    sem.throttle(0.01)
    await asyncio.sleep(0.02)
    assert sem.limit == 2
    # Wait for ramp up to complete.
    await asyncio.sleep(0.05)
    assert sem.limit == 4


async def test_slow_start_on_consecutive_throttles():
    """Recovery slows with consecutive throttle events."""

    sem = AdaptiveSemaphore(4, ramp_interval=0.01, grace_period=0.05)
    sem.throttle(0.01)
    await asyncio.sleep(0.04)
    sem.throttle(0.01)
    await asyncio.sleep(0.02)
    assert sem.limit == 2
    # Not enough time has passed for slow-start to restore all permits.
    await asyncio.sleep(0.02)
    assert sem.limit < 4
    await asyncio.sleep(0.05)
    assert sem.limit == 4


async def test_slow_start_resets_after_grace():
    """Grace period clears slow-start state for future throttles."""

    sem = AdaptiveSemaphore(4, ramp_interval=0.01, grace_period=0.05)
    sem.throttle(0.01)
    await asyncio.sleep(0.04)
    sem.throttle(0.01)
    # Allow slow-start recovery to finish.
    await asyncio.sleep(0.1)
    assert sem.limit == 4
    # Wait past the grace period so counters reset.
    await asyncio.sleep(0.06)
    sem.throttle(0.01)
    await asyncio.sleep(0.02)
    assert sem.limit == 2
    await asyncio.sleep(0.05)
    assert sem.limit == 4


async def test_weighted_permits_respected():
    """Weighted acquisition blocks until enough permits are free."""

    sem = AdaptiveSemaphore(3)
    order: list[int] = []

    async def worker(weight: int, mark: int) -> None:
        async with sem(weight):
            order.append(mark)
            await asyncio.sleep(0.01)

    first = asyncio.create_task(worker(2, 1))
    await asyncio.sleep(0.001)
    second = asyncio.create_task(worker(2, 2))
    await asyncio.gather(first, second)

    assert order == [1, 2]


def test_rolling_metrics_reports(monkeypatch):
    """Metrics emit request rate and error rate."""

    calls: list[SimpleNamespace] = []

    def fake_metric(name: str, value: float) -> None:
        calls.append(SimpleNamespace(name=name, value=value))

    monkeypatch.setattr(backpressure.logfire, "metric", fake_metric)
    metrics = RollingMetrics(window=1)
    metrics.record_request()
    metrics.record_error()
    metrics.record_request()

    names = {c.name for c in calls}
    assert {"requests_per_second", "error_rate"} <= names
