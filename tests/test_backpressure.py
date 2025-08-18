import asyncio
from types import SimpleNamespace

import pytest

from backpressure import AdaptiveSemaphore, RollingMetrics


@pytest.mark.asyncio()
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


def test_rolling_metrics_reports(monkeypatch):
    """Metrics emit request rate and error rate."""

    calls: list[SimpleNamespace] = []

    def fake_metric(name: str, value: float) -> None:
        calls.append(SimpleNamespace(name=name, value=value))

    monkeypatch.setattr("backpressure.logfire", "metric", fake_metric)
    metrics = RollingMetrics(window=1)
    metrics.record_request()
    metrics.record_error()
    metrics.record_request()

    names = {c.name for c in calls}
    assert {"requests_per_second", "error_rate"} <= names
