"""Verify generator output against a locked golden file."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import generator
from models import ServiceInput


class DummyAgent:
    """Agent echoing the service prompt for deterministic output."""

    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions

    async def run(self, user_prompt: str, output_type):
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt})
        )


def test_sample_run_matches_golden(monkeypatch, tmp_path):
    """A small end-to-end run should match the stored golden JSONL file."""

    monkeypatch.setattr(generator, "Agent", DummyAgent)
    service = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    gen = generator.ServiceAmbitionGenerator(SimpleNamespace())
    out_path = tmp_path / "out.jsonl"
    asyncio.run(gen.generate_async([service], "prompt", str(out_path)))
    expected = (Path(__file__).parent / "golden" / "sample_run.jsonl").read_text()
    assert out_path.read_text() == expected
