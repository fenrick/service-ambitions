"""Tests for plateau feature generation."""

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conversation import ConversationSession
from models import ServiceInput
from plateau_generator import PlateauGenerator


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    async def ask(self, prompt: str) -> str:  # pragma: no cover - simple proxy
        return self._responses.pop(0)


def _feature_payload(count: int) -> str:
    items = [
        {
            "feature_id": f"f{i}",
            "name": f"Feature {i}",
            "description": f"Desc {i}",
            "score": 0.5,
        }
        for i in range(count)
    ]
    return json.dumps({"features": items})


def test_generate_plateau_returns_results() -> None:
    responses = [_feature_payload(5)] + ['{"mappings": []}'] * 15
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session))
    service = ServiceInput(name="svc", description="desc")

    results = asyncio.run(
        generator.generate_plateau(service, "alpha", "retail")
    )  # type: ignore[arg-type]

    assert len(results) == 5


def test_generate_plateau_raises_on_insufficient_features() -> None:
    session = DummySession([_feature_payload(3)])
    generator = PlateauGenerator(cast(ConversationSession, session))
    service = ServiceInput(name="svc", description="desc")

    with pytest.raises(ValueError):
        asyncio.run(
            generator.generate_plateau(service, "alpha", "retail")
        )  # type: ignore[arg-type]
