"""Tests for feature mapping."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapping import map_feature, map_features
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
                {"data": [{"item": "User Data", "contribution": "Personalises."}]}
            ),
            json.dumps(
                {
                    "applications": [
                        {"item": "Learning Platform", "contribution": "Delivers."}
                    ]
                }
            ),
            json.dumps(
                {"technology": [{"item": "AI Engine", "contribution": "Enhances."}]}
            ),
        ]
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="Allows external access",
        score=0.5,
        customer_type="learners",
    )

    result = map_feature(session, feature)  # type: ignore[arg-type]

    assert isinstance(result, PlateauFeature)
    assert result.data[0].item == "User Data"
    assert result.applications[0].item == "Learning Platform"
    assert result.technology[0].item == "AI Engine"


def test_map_feature_injects_reference_data(monkeypatch) -> None:
    template = (
        "{feature_name} {feature_description} {category_label} "
        "{category_items} {category_key}"
    )
    monkeypatch.setattr("mapping.load_mapping_prompt", lambda *a, **k: template)
    """The mapping prompts should include reference data lists."""

    session = DummySession(
        [
            json.dumps({"data": [{"item": "User", "contribution": "c"}]}),
            json.dumps({"applications": [{"item": "App", "contribution": "c"}]}),
            json.dumps({"technology": [{"item": "Tech", "contribution": "c"}]}),
        ]
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="Allows external access",
        score=0.5,
        customer_type="learners",
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
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="desc",
        score=0.5,
        customer_type="learners",
    )
    with pytest.raises(ValueError):
        map_feature(session, feature)  # type: ignore[arg-type]


def test_map_features_returns_mappings(monkeypatch) -> None:
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda *a, **k: {
            "information": [{"id": "INF-1", "name": "User Data", "description": "d"}],
            "applications": [{"id": "APP-1", "name": "App", "description": "d"}],
            "technologies": [{"id": "TEC-1", "name": "Tech", "description": "d"}],
        },
    )

    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "data": [{"item": "INF-1", "contribution": "c"}],
                            "applications": [{"item": "APP-1", "contribution": "c"}],
                            "technology": [{"item": "TEC-1", "contribution": "c"}],
                        }
                    ]
                }
            )
        ]
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="Allows external access",
        score=0.5,
        customer_type="learners",
    )

    result = map_features(session, [feature])  # type: ignore[arg-type]

    assert result[0].data[0].item == "INF-1"
    assert "User Data" in session.prompts[0]


def test_map_features_validates_lists(monkeypatch) -> None:
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda *a, **k: {
            "information": [{"id": "INF-1", "name": "User Data", "description": "d"}],
            "applications": [{"id": "APP-1", "name": "App", "description": "d"}],
            "technologies": [{"id": "TEC-1", "name": "Tech", "description": "d"}],
        },
    )

    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "data": [],
                            "applications": [],
                            "technology": [],
                        }
                    ]
                }
            )
        ]
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="Allows external access",
        score=0.5,
        customer_type="learners",
    )

    with pytest.raises(ValueError):
        map_features(session, [feature])  # type: ignore[arg-type]
