"""Tests for feature mapping."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapping import MappedPlateauFeature, map_feature
from models import PlateauFeature


class DummySession:
    """Simple stand-in for a conversation session."""

    def __init__(self, responses: list[str]) -> None:  # pragma: no cover - simple init
        self._responses = iter(responses)
        self.prompts: list[str] = []

    async def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.prompts.append(prompt)
        return next(self._responses)


def test_map_feature_returns_mappings() -> None:
    """``map_feature`` should populate mapping items with contributions."""

    session = DummySession(
        [
            '{"mappings": [{"item": "API", "contribution": "Enables integration."}]}',
            '{"mappings": []}',
            '{"mappings": []}',
        ]
    )
    feature = PlateauFeature(
        feature_id="f1", name="Integration", description="Allows external access"
    )

    result = asyncio.run(map_feature(session, feature))  # type: ignore[arg-type]

    assert isinstance(result, MappedPlateauFeature)
    assert result.mappings[0].item == "API"
    assert result.mappings[0].contribution == "Enables integration."


def test_map_feature_injects_reference_data() -> None:
    """The mapping prompts should include reference data lists."""

    session = DummySession(['{"mappings": []}'] * 3)
    feature = PlateauFeature(
        feature_id="f1", name="Integration", description="Allows external access"
    )
    asyncio.run(map_feature(session, feature))  # type: ignore[arg-type]

    assert len(session.prompts) == 3
    assert "User Data" in session.prompts[0]
    assert "Learning Platform" in session.prompts[1]
    assert "AI Engine" in session.prompts[2]
