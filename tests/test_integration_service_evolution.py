"""Integration test for four-plateau service evolution."""

import json
import sys
from pathlib import Path
from typing import cast

from conversation import (  # noqa: E402  pylint: disable=wrong-import-position
    ConversationSession,
)
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    Contribution,
    PlateauFeature,
    ServiceEvolution,
    ServiceInput,
)
from plateau_generator import (  # noqa: E402  pylint: disable=wrong-import-position
    PlateauGenerator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Session returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover - simple stub
        pass


def _feature_payload(count: int) -> str:
    items = [
        {
            "feature_id": f"f{i}",
            "name": f"Feat {i}",
            "description": f"Desc {i}",
            "score": 0.5,
        }
        for i in range(count)
    ]
    payload = {"learners": items, "staff": items, "community": items}
    return json.dumps(payload)


def test_service_evolution_across_four_plateaus(monkeypatch) -> None:
    """``generate_service_evolution`` should aggregate all plateaus."""

    responses: list[str] = []
    for _ in range(4):
        responses.append(json.dumps({"description": "desc"}))
        responses.append(_feature_payload(5))
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=5)

    map_calls = {"n": 0}

    def _fake_map_features(session, features, prompt_dir="prompts"):
        map_calls["n"] += 1
        results = []
        for feature in features:
            payload = feature.model_dump()
            payload.update(
                data=[Contribution(item="d", contribution="c")],
                applications=[Contribution(item="a", contribution="c")],
                technology=[Contribution(item="t", contribution="c")],
            )
            results.append(PlateauFeature(**payload))
        return results

    monkeypatch.setattr("plateau_generator.map_features", _fake_map_features)
    template = "{required_count} {service_name} {service_description} {plateau}"
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )

    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=["job"],
    )
    evolution = generator.generate_service_evolution(
        service,
        ["Foundational", "Emerging", "Strategic", "Visionary"],
        ["learners", "staff", "community"],
    )
    assert isinstance(evolution, ServiceEvolution)
    assert len(evolution.plateaus) == 4
    assert sum(len(p.features) for p in evolution.plateaus) == 60
    assert all(len(p.features) >= 15 for p in evolution.plateaus)
    assert len(session.prompts) == 8
    assert map_calls["n"] == 4
    assert len(session.prompts) + map_calls["n"] == 12
    for plateau in evolution.plateaus:
        for feature in plateau.features:
            assert feature.data
            assert feature.applications
            assert feature.technology
