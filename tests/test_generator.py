# SPDX-License-Identifier: MIT
import asyncio
from types import SimpleNamespace

import generator
from models import ServiceInput


class DummySpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}

    def __enter__(self):
        spans.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


async def fake_process_service(self, service, prompt=None):
    return {"id": service.service_id}, 3, 0.5, 1


spans: list[DummySpan] = []


def test_service_span_records_metrics(monkeypatch, tmp_path):
    spans.clear()
    monkeypatch.setattr(generator.logfire, "span", lambda name: DummySpan(name))
    monkeypatch.setattr(
        generator.ServiceAmbitionGenerator, "process_service", fake_process_service
    )
    monkeypatch.setattr(generator.os, "fsync", lambda fd: None)

    service = ServiceInput(
        service_id="svc1", name="svc", description="d", jobs_to_be_done=[]
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    asyncio.run(gen.generate_async([service], "prompt", str(tmp_path / "out.jsonl")))

    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["tokens.total"] == 3
    assert span.attributes["cost.estimate"] == 0.5
    assert span.attributes["retries"] == 1
    assert span.attributes["status"] == "success"
