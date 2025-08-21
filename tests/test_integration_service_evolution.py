"""Integration test for four-plateau service evolution."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pydantic_ai import Agent

from conversation import ConversationSession
from models import (
    SCHEMA_VERSION,
    Contribution,
    PlateauFeature,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from plateau_generator import PlateauGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Agent returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def run_sync(self, prompt: str, message_history):
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
    payload = {
        "features": {
            "learners": items,
            "academics": items,
            "professional_staff": items,
        }
    }
    return json.dumps(payload)


def test_service_evolution_across_four_plateaus(monkeypatch) -> None:
    """``generate_service_evolution`` should aggregate all plateaus."""

    desc_payload = json.dumps(
        {
            "descriptions": [
                {"plateau": 1, "plateau_name": "Foundational", "description": "desc"},
                {"plateau": 2, "plateau_name": "Enhanced", "description": "desc"},
                {
                    "plateau": 3,
                    "plateau_name": "Experimental",
                    "description": "desc",
                },
                {"plateau": 4, "plateau_name": "Disruptive", "description": "desc"},
            ]
        }
    )
    responses: list[str] = [desc_payload]
    responses += [_feature_payload(5) for _ in range(4)]
    agent = DummyAgent(responses)
    session = ConversationSession(cast(Agent[None, str], agent))
    generator = PlateauGenerator(session, required_count=5)

    map_calls = {"n": 0}

    def _fake_map_features(session, features, *, strict=False):
        map_calls["n"] += 1
        results = []
        for feature in features:
            payload = feature.model_dump()
            payload["mappings"] = {
                "data": [Contribution(item="d", contribution=1.0)],
                "applications": [Contribution(item="a", contribution=1.0)],
                "technology": [Contribution(item="t", contribution=1.0)],
            }
            results.append(PlateauFeature(**payload))
        return results

    monkeypatch.setattr("plateau_generator.map_features", _fake_map_features)
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)

    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    meta = ServiceMeta(
        run_id="run",
        seed=None,
        models={},
        web_search=False,
        mapping_types=[],
        created=datetime.now(timezone.utc),
    )
    evolution = generator.generate_service_evolution(
        service,
        ["Foundational", "Enhanced", "Experimental", "Disruptive"],
        ["learners", "academics", "professional_staff"],
        meta=meta,
    )
    assert isinstance(evolution, ServiceEvolution)
    assert evolution.meta.schema_version == SCHEMA_VERSION
    assert len(evolution.plateaus) == 4
    assert sum(len(p.features) for p in evolution.plateaus) == 60
    assert all(len(p.features) >= 15 for p in evolution.plateaus)
    assert len(agent.prompts) == 5
    assert map_calls["n"] == 4
    assert len(agent.prompts) + map_calls["n"] == 9
    for plateau in evolution.plateaus:
        for feature in plateau.features:
            assert feature.mappings["data"]
            assert feature.mappings["applications"]
            assert feature.mappings["technology"]
