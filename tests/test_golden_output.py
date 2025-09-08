# SPDX-License-Identifier: MIT
"""Verify simplified generator output against a locked golden file."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServiceInput:
    """Minimal service model for testing purposes."""

    service_id: str
    name: str
    description: str
    jobs_to_be_done: list[dict[str, Any]]
    features: list[dict[str, Any]] = field(default_factory=list)
    parent_id: str | None = None
    customer_type: str | None = None

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "service_id": self.service_id,
                "name": self.name,
                "parent_id": self.parent_id,
                "customer_type": self.customer_type,
                "description": self.description,
                "jobs_to_be_done": self.jobs_to_be_done,
                "features": self.features,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def model_dump(
        self, mode: str | None = None
    ) -> dict[str, Any]:  # pragma: no cover - passthrough
        return {
            "service_id": self.service_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "customer_type": self.customer_type,
            "description": self.description,
            "jobs_to_be_done": self.jobs_to_be_done,
            "features": self.features,
        }


class MiniGenerator:
    """Simplified generator that uses a provided agent class."""

    def __init__(self, agent_cls) -> None:
        self.agent = agent_cls()

    async def generate_async(
        self, services: list[ServiceInput], prompt: str, output_path: str
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for svc in services:
                resp = await self.agent.run(svc.model_dump_json())
                f.write(
                    json.dumps(resp.output.model_dump(), separators=(",", ":")) + "\n"
                )


def test_sample_run_matches_golden(tmp_path, dummy_agent):
    """A small end-to-end run should match the stored golden JSONL file."""
    service = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    gen = MiniGenerator(dummy_agent)
    out_path = tmp_path / "out.jsonl"
    asyncio.run(gen.generate_async([service], "prompt", str(out_path)))
    expected = (Path(__file__).parent / "golden" / "sample_run.jsonl").read_text()
    assert out_path.read_text() == expected
