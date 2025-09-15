"""Tests for metric naming and span attributes."""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio()
async def test_span_attributes_present(monkeypatch):
    import logfire

    metric_names: list[str] = []

    def metric_counter(name: str):
        metric_names.append(name)
        return SimpleNamespace(add=lambda *a, **k: None)

    def metric_gauge(name: str):
        metric_names.append(name)
        return SimpleNamespace(set=lambda *a, **k: None)

    spans: list[tuple[str, dict]] = []

    class _Span:
        def __enter__(self):
            return SimpleNamespace(set_attribute=lambda *a, **k: None)

        def __exit__(self, exc_type, exc, tb):
            return False

    def span(name: str, attributes=None):
        spans.append((name, attributes))
        return _Span()

    monkeypatch.setattr(logfire, "metric_counter", metric_counter)
    monkeypatch.setattr(logfire, "metric_gauge", metric_gauge)
    monkeypatch.setattr(logfire, "span", span)

    from llm.queue import LLMQueue, LLMTaskMeta

    q = LLMQueue(max_concurrency=1)
    meta = LLMTaskMeta(
        stage="s",
        model_name="m",
        service_id="svc",
        request_id="req",
    )

    async def factory():
        return 42

    assert await q.submit(factory, meta=meta) == 42

    assert (
        "sa_llm_queue.submit",
        {
            "stage": "s",
            "model_name": "m",
            "service_id": "svc",
            "request_id": "req",
        },
    ) == spans[0]
    assert set(metric_names) == {
        "sa_llm_queue_inflight",
        "sa_llm_queue_submitted",
        "sa_llm_queue_completed",
    }
