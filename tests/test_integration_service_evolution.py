# SPDX-License-Identifier: MIT
"""Integration test for four-plateau service evolution."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai import Agent

from conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from models import (
    SCHEMA_VERSION,
    FeatureMappingRef,
    MappingFeatureGroup,
    PlateauFeaturesResponse,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from plateau_generator import PlateauGenerator
from runtime.environment import RuntimeEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Agent returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def run_sync(self, prompt: str, message_history):
        self.prompts.append(prompt)
        payload_json = self._responses.pop(0)
        payload = PlateauFeaturesResponse.model_validate_json(payload_json)
        return SimpleNamespace(
            output=payload,
            new_messages=lambda: [],
            usage=lambda: SimpleNamespace(total_tokens=0),
        )

    async def run(self, prompt: str, message_history):
        return self.run_sync(prompt, message_history)


def _feature_payload(count: int) -> str:
    items = [
        {
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

    responses: list[str] = [_feature_payload(5) for _ in range(4)]
    agent = DummyAgent(responses)
    session = ConversationSession(
        cast(Agent[None, str], agent),
        use_local_cache=False,
        cache_mode="off",
    )
    RuntimeEnv.initialize(cast(Any, SimpleNamespace(mapping_data_dir=Path("data"))))
    generator = PlateauGenerator(
        session,
        required_count=5,
        use_local_cache=False,
        cache_mode="off",
    )

    map_calls = {"n": 0}

    async def _fake_map_features(self, session, features, **kwargs):
        map_calls["n"] += 1
        refs = [
            FeatureMappingRef(feature_id=f.feature_id, description=f.description)
            for f in features
        ]
        return {
            "data": [MappingFeatureGroup(id="d", name="d", mappings=refs.copy())],
            "applications": [
                MappingFeatureGroup(id="a", name="a", mappings=refs.copy())
            ],
            "technologies": [
                MappingFeatureGroup(id="t", name="t", mappings=refs.copy())
            ],
        }

    monkeypatch.setattr(PlateauGenerator, "_map_features", _fake_map_features)
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus}"
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
    runtimes = [
        PlateauRuntime(plateau=i + 1, plateau_name=n, description="desc")
        for i, n in enumerate(
            ["Foundational", "Enhanced", "Autonomous", "Outcome-Driven"]
        )
    ]
    evolution = generator.generate_service_evolution(
        service,
        runtimes,
        ["learners", "academics", "professional_staff"],
        meta=meta,
    )
    assert isinstance(evolution, ServiceEvolution)
    assert evolution.meta.schema_version == SCHEMA_VERSION
    assert len(evolution.plateaus) == 4
    assert sum(len(p.features) for p in evolution.plateaus) == 60
    assert all(len(p.features) >= 15 for p in evolution.plateaus)
    assert len(agent.prompts) == 4
    assert map_calls["n"] == 4
    assert len(agent.prompts) + map_calls["n"] == 8
    for plateau in evolution.plateaus:
        assert plateau.mappings["data"]
        assert plateau.mappings["applications"]
        assert plateau.mappings["technologies"]
