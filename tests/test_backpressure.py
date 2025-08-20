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


async def test_consecutive_throttles_ramp_conservatively():
    """Repeated throttles double ramp interval and delay recovery."""

    sem = AdaptiveSemaphore(4, ramp_interval=0.05, grace_period=0.2)
    sem.throttle(0.05)
    # Allow first throttle to reduce permits.
    await asyncio.sleep(0.01)
    sem.throttle(0.05)
    await asyncio.sleep(0.02)

    # Limit is halved twice and ramp interval doubled.
    assert sem.limit == 1
    assert sem._ramp_interval == pytest.approx(0.1, abs=0.01)

    # Before delay + ramp interval no permits are restored.
    await asyncio.sleep(0.09)
    assert sem.limit == 1

    # After the recovery window begins, capacity ramps up slowly.
    await asyncio.sleep(0.06)
    assert sem.limit == 2


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


async def test_multi_weight_acquisition_release_counts():
    """Acquire and release multiple permits while tracking counts."""

    sem = AdaptiveSemaphore(5)
    await sem.acquire(3)
    assert sem.limit == 5
    assert sem._sem._value == 2
    await sem.acquire(2)
    assert sem._sem._value == 0
    sem.release(3)
    assert sem._sem._value == 3
    sem.release(2)
    assert sem._sem._value == 5
    assert sem.limit == 5


def test_rolling_metrics_reports(monkeypatch):
    """Metrics emit request, error and token rates."""

    calls: list[SimpleNamespace] = []
    info_calls: list[dict[str, float]] = []

    def fake_metric(name: str, value: float) -> None:
        calls.append(SimpleNamespace(kind="metric", name=name, value=value))

    def fake_info(msg: str, **kwargs: float) -> None:  # pragma: no cover - simple
        info_calls.append(kwargs)

    class FakeCounter:
        def __init__(self, name: str) -> None:
            self.name = name

        def add(self, value: float) -> None:  # pragma: no cover - simple proxy
            calls.append(SimpleNamespace(kind="counter", name=self.name, value=value))

    class FakeGauge:
        def __init__(self, name: str) -> None:
            self.name = name

        def set(self, value: float) -> None:  # pragma: no cover - simple proxy
            calls.append(SimpleNamespace(kind="gauge", name=self.name, value=value))

    monkeypatch.setattr(backpressure.logfire, "metric", fake_metric)
    monkeypatch.setattr(backpressure.logfire, "info", fake_info)
    monkeypatch.setattr(backpressure, "REQUESTS_TOTAL", FakeCounter("requests_total"))
    monkeypatch.setattr(backpressure, "ERRORS_TOTAL", FakeCounter("errors_total"))
    monkeypatch.setattr(backpressure, "TOKENS_IN_FLIGHT", FakeGauge("tokens_in_flight"))

    metrics = RollingMetrics(window=1)
    metrics.record_request()
    metrics.record_error(is_429=True)
    metrics.record_latency(0.1)
    metrics.record_start_tokens(5)
    metrics.record_end_tokens(5)
    metrics.record_request()

    names = {(c.kind, c.name) for c in calls}
    assert {
        ("metric", "requests_per_second"),
        ("metric", "error_rate"),
        ("metric", "tokens_per_second"),
        ("counter", "requests_total"),
        ("counter", "errors_total"),
        ("gauge", "tokens_in_flight"),
    } <= names
    assert any(
        {"tokens_per_sec", "rps", "error_rate", "rate_429", "avg_latency"}
        <= info.keys()
        for info in info_calls
    )
