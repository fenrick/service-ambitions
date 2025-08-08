import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import generator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions

    async def run(self, user_prompt: str, output_type):  # pragma: no cover - stub
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt})
        )


def test_process_service_async(monkeypatch):
    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = {"name": "alpha"}
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    result = asyncio.run(gen.process_service(service, "prompt"))
    assert json.loads(result["service"]) == service


def test_generator_rejects_invalid_concurrency():
    with pytest.raises(ValueError):
        generator.ServiceAmbitionGenerator(SimpleNamespace(), concurrency=0)
