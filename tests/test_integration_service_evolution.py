"""Integration test for four-plateau service evolution."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import Agent  # noqa: E402  pylint: disable=wrong-import-position

from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    Contribution,
    PlateauFeature,
    ServiceEvolution,
    ServiceInput,
)
from plateau_generator import (
    PlateauGenerator,
)  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Agent returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    async def run(self, prompt: str, message_history):  # pragma: no cover - stub
        self.prompts.append(prompt)
        return SimpleNamespace(output=self._responses.pop(0), new_messages=lambda: [])


def _feature_payload(count: int) -> str:
    items = [
        {
            "feature_id": f"f{i}",
            "name": f"Feat {i}",
            "description": f"Desc {i}",
            "score": {
                "level": 3,
                "label": "Defined",
                "justification": "test",
            },
        }
        for i in range(count)
    ]
    payload = {"learners": items, "academics": items, "professional_staff": items}
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_service_evolution_across_four_plateaus(monkeypatch) -> None:
    """``generate_service_evolution`` should aggregate all plateaus."""

    responses: list[str] = []
    for _ in range(4):
        responses.append(json.dumps({"description": "desc"}))
        responses.append(_feature_payload(5))
    agent = DummyAgent(responses)
    session = ConversationSession(cast(Agent[None, str], agent))
    generator = PlateauGenerator(session, required_count=5)

    map_calls = {"n": 0}

    def _fake_map_features(session, features):
        map_calls["n"] += 1
        results = []
        for feature in features:
            payload = feature.model_dump()
            payload["mappings"] = {
                "data": [Contribution(item="d", contribution="c")],
                "applications": [Contribution(item="a", contribution="c")],
                "technology": [Contribution(item="t", contribution="c")],
            }
            results.append(PlateauFeature(**payload))
        return results

    monkeypatch.setattr("plateau_generator.map_features", _fake_map_features)
    template = "{required_count} {service_name} {service_description} {plateau}"

    def fake_loader(name, *_, **__):
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)

    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    evolution = await generator.generate_service_evolution(
        service,
        ["Foundational", "Enhanced", "Experimental", "Disruptive"],
        ["learners", "academics", "professional_staff"],
    )
    assert isinstance(evolution, ServiceEvolution)
    assert len(evolution.plateaus) == 4
    assert sum(len(p.features) for p in evolution.plateaus) == 60
    assert all(len(p.features) >= 15 for p in evolution.plateaus)
    assert len(agent.prompts) == 8
    assert map_calls["n"] == 4
    assert len(agent.prompts) + map_calls["n"] == 12
    for plateau in evolution.plateaus:
        for feature in plateau.features:
            assert feature.mappings["data"]
            assert feature.mappings["applications"]
            assert feature.mappings["technology"]
