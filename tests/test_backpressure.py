from types import SimpleNamespace

import backpressure
from backpressure import RollingMetrics


def test_rolling_metrics_reports(monkeypatch):
    """Metrics emit request, error and token rates."""

    calls: list[SimpleNamespace] = []
    info_calls: list[dict[str, float]] = []

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

    monkeypatch.setattr(backpressure.logfire, "info", fake_info)
    monkeypatch.setattr(backpressure, "REQUESTS_TOTAL", FakeCounter("requests_total"))
    monkeypatch.setattr(backpressure, "ERRORS_TOTAL", FakeCounter("errors_total"))
    monkeypatch.setattr(backpressure, "TOKENS_IN_FLIGHT", FakeGauge("tokens_in_flight"))
    monkeypatch.setattr(
        backpressure, "REQUESTS_PER_SECOND", FakeGauge("requests_per_second")
    )
    monkeypatch.setattr(backpressure, "ERROR_RATE", FakeGauge("error_rate"))
    monkeypatch.setattr(
        backpressure, "TOKENS_PER_SECOND", FakeGauge("tokens_per_second")
    )
    monkeypatch.setattr(backpressure, "RATE_429", FakeGauge("rate_429"))
    monkeypatch.setattr(backpressure, "AVG_LATENCY", FakeGauge("avg_latency"))

    metrics = RollingMetrics(window=1)
    metrics.record_request()
    metrics.record_error(is_429=True)
    metrics.record_latency(0.1)
    metrics.record_start_tokens(5)
    metrics.record_end_tokens(5)
    metrics.record_request()

    names = {(c.kind, c.name) for c in calls}
    assert {
        ("gauge", "requests_per_second"),
        ("gauge", "error_rate"),
        ("gauge", "tokens_per_second"),
        ("gauge", "rate_429"),
        ("gauge", "avg_latency"),
        ("counter", "requests_total"),
        ("counter", "errors_total"),
        ("gauge", "tokens_in_flight"),
    } <= names
    assert any(
        {"tokens_per_sec", "rps", "error_rate", "rate_429", "avg_latency"}
        <= info.keys()
        for info in info_calls
    )
