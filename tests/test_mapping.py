"""Tests for feature mapping."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapping import MappedPlateauFeature, map_feature
from models import PlateauFeature


class DummySession:
    """Simple stand-in for a conversation session."""

    def __init__(self, responses: list[str]) -> None:  # pragma: no cover - simple init
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.prompts.append(prompt)
        return next(self._responses)


def test_map_feature_returns_mappings(monkeypatch) -> None:
    template = (
        "{feature_name} {feature_description} {category_label} "
        "{category_items} {category_key}"
    )
    monkeypatch.setattr("mapping.load_mapping_prompt", lambda *a, **k: template)
    """``map_feature`` should populate mapping items with contributions."""

    session = DummySession(
        [
            json.dumps(
                {"data": [{"type": "User Data", "contribution": "Personalises."}]}
            ),
            json.dumps(
                {
                    "applications": [
                        {"type": "Learning Platform", "contribution": "Delivers."}
                    ]
                }
            ),
            json.dumps(
                {"technology": [{"type": "AI Engine", "contribution": "Enhances."}]}
            ),
        ]
    )
    feature = PlateauFeature(
        feature_id="f1", name="Integration", description="Allows external access"
    )

    result = map_feature(session, feature)  # type: ignore[arg-type]

    assert isinstance(result, MappedPlateauFeature)
    assert result.data[0].type == "User Data"
    assert result.applications[0].type == "Learning Platform"
    assert result.technology[0].type == "AI Engine"


def test_map_feature_injects_reference_data(monkeypatch) -> None:
    template = (
        "{feature_name} {feature_description} {category_label} "
        "{category_items} {category_key}"
    )
    monkeypatch.setattr("mapping.load_mapping_prompt", lambda *a, **k: template)
    """The mapping prompts should include reference data lists."""

    session = DummySession(
        [
            json.dumps({"data": [{"type": "User", "contribution": "c"}]}),
            json.dumps({"applications": [{"type": "App", "contribution": "c"}]}),
            json.dumps({"technology": [{"type": "Tech", "contribution": "c"}]}),
        ]
    )
    feature = PlateauFeature(
        feature_id="f1", name="Integration", description="Allows external access"
    )
    map_feature(session, feature)  # type: ignore[arg-type]

    assert len(session.prompts) == 3
    assert "User Data" in session.prompts[0]
    assert "Learning Platform" in session.prompts[1]
    assert "AI Engine" in session.prompts[2]


def test_map_feature_rejects_invalid_json(monkeypatch) -> None:
    template = (
        "{feature_name} {feature_description} {category_label} "
        "{category_items} {category_key}"
    )
    monkeypatch.setattr("mapping.load_mapping_prompt", lambda *a, **k: template)
    """Invalid JSON responses should raise a ``ValueError``."""
    session = DummySession(["not-json"])
    feature = PlateauFeature(feature_id="f1", name="Integration", description="desc")
    with pytest.raises(ValueError):
        map_feature(session, feature)  # type: ignore[arg-type]
