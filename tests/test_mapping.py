"""Tests for feature mapping."""

import json
import sys
from pathlib import Path

import pytest

from mapping import map_feature, map_features
from models import MappingItem, PlateauFeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Simple stand-in for a conversation session."""

    def __init__(self, responses: list[str]) -> None:  # pragma: no cover - simple init
        self._responses = iter(responses)
        self.prompts: list[str] = []

    async def ask(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.prompts.append(prompt)
        return next(self._responses)


@pytest.mark.asyncio
async def test_map_feature_returns_mappings(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [
                MappingItem(id="APP-1", name="Learning Platform", description="d")
            ],
            "technologies": [
                MappingItem(id="TEC-1", name="AI Engine", description="d")
            ],
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
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": "c"}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": "c"}],
                        }
                    ]
                }
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

    result = await map_feature(session, feature)  # type: ignore[arg-type]

    assert isinstance(result, PlateauFeature)
    assert result.mappings["data"][0].item == "INF-1"
    assert result.mappings["applications"][0].item == "APP-1"
    assert result.mappings["technology"][0].item == "TEC-1"


@pytest.mark.asyncio
async def test_map_feature_injects_reference_data(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [
                MappingItem(id="APP-1", name="Learning Platform", description="d")
            ],
            "technologies": [
                MappingItem(id="TEC-1", name="AI Engine", description="d")
            ],
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
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": "c"}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": "c"}],
                        }
                    ]
                }
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
    await map_feature(session, feature)  # type: ignore[arg-type]

    assert len(session.prompts) == 3
    assert "User Data" in session.prompts[0]
    assert "Learning Platform" in session.prompts[1]
    assert "AI Engine" in session.prompts[2]


@pytest.mark.asyncio
async def test_map_feature_rejects_invalid_json(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [],
            "applications": [],
            "technologies": [],
        },
    )
    session = DummySession(["not-json"])
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="desc",
        score=0.5,
        customer_type="learners",
    )
    with pytest.raises(ValueError):
        await map_feature(session, feature)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_map_feature_flattens_nested_mappings(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [MappingItem(id="APP-1", name="App", description="d")],
            "technologies": [MappingItem(id="TEC-1", name="Tech", description="d")],
        },
    )
    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "mappings": {
                                "mappings": {
                                    "data": [{"item": "INF-1", "contribution": "c"}],
                                    "applications": [
                                        {"item": "APP-1", "contribution": "c"}
                                    ],
                                    "technology": [
                                        {"item": "TEC-1", "contribution": "c"}
                                    ],
                                }
                            },
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

    result = await map_feature(session, feature)  # type: ignore[arg-type]

    assert result.mappings["data"][0].item == "INF-1"
    assert result.mappings["applications"][0].item == "APP-1"
    assert result.mappings["technology"][0].item == "TEC-1"


@pytest.mark.asyncio
async def test_map_feature_flattens_repeated_mapping_keys(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [MappingItem(id="APP-1", name="App", description="d")],
            "technologies": [MappingItem(id="TEC-1", name="Tech", description="d")],
        },
    )
    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "mappings": {
                                "data": {
                                    "data": [{"item": "INF-1", "contribution": "c"}]
                                },
                                "applications": {
                                    "applications": [
                                        {"item": "APP-1", "contribution": "c"}
                                    ]
                                },
                                "technology": {
                                    "mappings": {
                                        "technology": [
                                            {"item": "TEC-1", "contribution": "c"}
                                        ]
                                    }
                                },
                            },
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

    result = await map_feature(session, feature)  # type: ignore[arg-type]

    assert result.mappings["data"][0].item == "INF-1"
    assert result.mappings["applications"][0].item == "APP-1"
    assert result.mappings["technology"][0].item == "TEC-1"


@pytest.mark.asyncio
async def test_map_features_returns_mappings(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [MappingItem(id="APP-1", name="App", description="d")],
            "technologies": [MappingItem(id="TEC-1", name="Tech", description="d")],
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
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": "c"}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": "c"}],
                        }
                    ]
                }
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

    result = await map_features(session, [feature])  # type: ignore[arg-type]

    assert result[0].mappings["data"][0].item == "INF-1"
    assert "User Data" in session.prompts[0]
    assert "App" in session.prompts[1]
    assert "Tech" in session.prompts[2]


@pytest.mark.asyncio
async def test_map_features_allows_empty_lists(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [MappingItem(id="APP-1", name="App", description="d")],
            "technologies": [MappingItem(id="TEC-1", name="Tech", description="d")],
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

    result = await map_features(session, [feature])  # type: ignore[arg-type]

    assert result[0].mappings["data"] == []
    assert result[0].mappings["applications"] == []
    assert result[0].mappings["technology"] == []
