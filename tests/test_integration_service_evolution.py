"""Integration test for four-plateau service evolution."""

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conversation import (  # noqa: E402  pylint: disable=wrong-import-position
    ConversationSession,
)
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    ServiceEvolution,
    ServiceInput,
)
from plateau_generator import (  # noqa: E402  pylint: disable=wrong-import-position
    PlateauGenerator,
)


class DummySession:
    """Session returning queued JSON payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    async def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        return self._responses.pop(0)


async def _fake_map_feature(session, feature, prompt_dir):  # pragma: no cover - stub
    from mapping import MappedPlateauFeature

    return MappedPlateauFeature(**feature.model_dump(), mappings=[])


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

    responses = [_feature_payload() for _ in range(4)]
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)

    monkeypatch.setattr("plateau_generator.map_feature", _fake_map_feature)

    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    evolution = asyncio.run(
        generator.generate_service_evolution(service, ["a", "b", "c", "d"], ["retail"])
    )

    assert isinstance(evolution, ServiceEvolution)
    assert len(evolution.results) == 4
