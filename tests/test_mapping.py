"""Tests for feature mapping."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapping import MappedPlateauFeature, map_feature
from models import PlateauFeature


class DummySession:
    """Simple stand-in for a conversation session."""

    def __init__(self, response: str) -> None:  # pragma: no cover - simple init
        self._response = response

    async def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        return self._response


def test_map_feature_returns_mappings() -> None:
    """``map_feature`` should populate mapping items with contributions."""

    session = DummySession(
        '{"mappings": [{"item": "API", "contribution": "Enables integration."}]}'
    )
    feature = PlateauFeature(
        feature_id="f1", name="Integration", description="Allows external access"
    )

    result = asyncio.run(map_feature(session, feature))  # type: ignore[arg-type]

    assert isinstance(result, MappedPlateauFeature)
    assert result.mappings[0].item == "API"
    assert result.mappings[0].contribution == "Enables integration."
