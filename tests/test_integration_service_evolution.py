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


def _feature_payload() -> str:
    return json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "name": "Feat",
                    "description": "Desc",
                    "score": 0.5,
                }
            ]
        }
    )


def test_service_evolution_across_four_plateaus(monkeypatch) -> None:
    """``generate_service_evolution`` should aggregate all plateaus."""

    responses: list[str] = []
    for _ in range(4):
        responses.append(json.dumps({"description": "desc"}))
        responses.extend([_feature_payload() for _ in range(3)])
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)

    monkeypatch.setattr("plateau_generator.map_feature", _fake_map_feature)

    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    evolution = generator.generate_service_evolution(service)

    assert isinstance(evolution, ServiceEvolution)
    assert len(evolution.results) == 12
