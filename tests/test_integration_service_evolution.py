"""Integration test for four-plateau service evolution."""

import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import (
    ServiceEvolution,
    ServiceInput,
)  # noqa: E402  pylint: disable=wrong-import-position
from plateau_generator import (
    PlateauGenerator,
)  # noqa: E402  pylint: disable=wrong-import-position


class DummySession:
    """Session returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        return self._responses.pop(0)

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover - simple stub
        pass


def _fake_map_feature(session, feature, prompt_dir):  # pragma: no cover - stub
    from mapping import MappedPlateauFeature, TypeContribution

    return MappedPlateauFeature(
        **feature.model_dump(),
        data=[TypeContribution(type="d", contribution="c")],
        applications=[TypeContribution(type="a", contribution="c")],
        technology=[TypeContribution(type="t", contribution="c")],
    )


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
    return json.dumps({"features": items})


def test_service_evolution_across_four_plateaus(monkeypatch) -> None:
    """``generate_service_evolution`` should aggregate all plateaus."""

    responses: list[str] = []
    for _ in range(4):
        responses.append(json.dumps({"description": "desc"}))
        responses.extend([_feature_payload(5) for _ in range(3)])
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=5)

    monkeypatch.setattr("plateau_generator.map_feature", _fake_map_feature)
    template = (
        "{required_count} {service_name} {service_description} "
        "{plateau} {customer_type}"
    )
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )

    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    evolution = generator.generate_service_evolution(service)
    assert isinstance(evolution, ServiceEvolution)
    assert len(evolution.results) == 60
    assert len(evolution.results) // 4 >= 15
