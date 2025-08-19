"""Tests for feature mapping."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mapping
from mapping import map_feature, map_feature_async, map_features_async
from models import MappingItem, MappingTypeConfig, MaturityScore, PlateauFeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Simple stand-in for a conversation session."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    async def ask(self, prompt: str, output_type=None):
        self.prompts.append(prompt)
        response = next(self._responses)
        if output_type is None:
            return response
        return output_type.model_validate_json(response)

    def derive(self):
        return self

    async def ask_async(self, prompt: str, output_type=None):
        return await self.ask(prompt, output_type)


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
                            "data": [{"item": "INF-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": 0.5}],
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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_feature_async(session, feature)

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
                            "data": [{"item": "INF-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": 0.5}],
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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )
    await map_feature_async(session, feature)

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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )
    with pytest.raises(ValueError):
        await map_feature_async(session, feature)


def test_map_feature_ignores_unknown_ids(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr(
        "mapping.load_mapping_items",
        lambda types, *a, **k: {
            "information": [MappingItem(id="INF-1", name="User Data", description="d")],
            "applications": [],
            "technologies": [],
        },
    )

    class SyncSession:
        def __init__(self, responses: list[str]) -> None:
            self._responses = iter(responses)

        def ask(self, prompt: str, output_type=None):
            response = next(self._responses)
            if output_type is None:
                return response
            return output_type.model_validate_json(response)

    session = SyncSession(
        [
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "data": [{"item": "BAD", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps({"features": [{"feature_id": "f1", "applications": []}]}),
            json.dumps({"features": [{"feature_id": "f1", "technology": []}]}),
        ]
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="desc",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )
    result = map_feature(session, feature)

    assert result.mappings["data"] == []


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
                                    "data": [{"item": "INF-1", "contribution": 0.5}],
                                    "applications": [
                                        {"item": "APP-1", "contribution": 0.5}
                                    ],
                                    "technology": [
                                        {"item": "TEC-1", "contribution": 0.5}
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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_feature_async(session, feature)

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
                                    "data": [{"item": "INF-1", "contribution": 0.5}]
                                },
                                "applications": {
                                    "applications": [
                                        {"item": "APP-1", "contribution": 0.5}
                                    ]
                                },
                                "technology": {
                                    "mappings": {
                                        "technology": [
                                            {"item": "TEC-1", "contribution": 0.5}
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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_feature_async(session, feature)

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
                            "data": [{"item": "INF-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "applications": [{"item": "APP-1", "contribution": 0.5}],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "features": [
                        {
                            "feature_id": "f1",
                            "technology": [{"item": "TEC-1", "contribution": 0.5}],
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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_features_async(session, [feature])

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
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_features_async(session, [feature])

    assert result[0].mappings["data"] == []
    assert result[0].mappings["applications"] == []
    assert result[0].mappings["technology"] == []


@pytest.mark.asyncio
async def test_map_features_retries_on_empty(monkeypatch) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    items = {
        "information": [
            MappingItem(id="INF-1", name="User Data", description="user info"),
            MappingItem(id="INF-2", name="Traffic", description="traffic stats"),
        ]
    }
    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr("mapping.load_mapping_items", lambda types, *a, **k: items)

    initial = json.dumps({"features": [{"feature_id": "f1", "data": []}]})
    repaired = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "data": [{"item": "INF-1", "contribution": 0.9}],
                }
            ]
        }
    )
    session = DummySession([initial, repaired])
    feature = PlateauFeature(
        feature_id="f1",
        name="Integration",
        description="Uses user data",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await map_features_async(
        session,
        [feature],
        {"data": MappingTypeConfig(dataset="information", label="Info")},
    )

    assert len(session.prompts) == 2
    assert result[0].mappings["data"][0].item == "INF-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [True, False])
async def test_map_features_reprompts_per_feature(monkeypatch, parallel) -> None:
    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    items = {
        "information": [
            MappingItem(id="INF-1", name="User Data", description="user info"),
            MappingItem(id="INF-2", name="Traffic", description="traffic stats"),
        ]
    }

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr("mapping.load_mapping_items", lambda types, *a, **k: items)
    monkeypatch.setattr(
        "mapping._top_k_items", lambda features, dataset: items[dataset]
    )

    initial = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "data": [
                        {"item": "INF-1", "contribution": 0.8},
                        {"item": "INF-2", "contribution": 0.2},
                    ],
                },
                {"feature_id": "f2", "data": []},
            ]
        }
    )
    repaired = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f2",
                    "data": [
                        {"item": "INF-1", "contribution": 0.9},
                        {"item": "INF-2", "contribution": 0.1},
                    ],
                }
            ]
        }
    )

    session = DummySession([initial, repaired])
    features = [
        PlateauFeature(
            feature_id="f1",
            name="Integration",
            description="Allows external access",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        ),
        PlateauFeature(
            feature_id="f2",
            name="Export",
            description="Lets users export data",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        ),
    ]

    result = await map_features_async(
        session,
        features,
        {"data": MappingTypeConfig(dataset="information", label="Info")},
        batch_size=2,
        parallel_types=parallel,
    )

    assert len(session.prompts) == 2
    assert len(result[0].mappings["data"]) >= 2
    assert len(result[1].mappings["data"]) >= 2


@pytest.mark.asyncio
async def test_map_features_reprompts_missing_app_and_tech(monkeypatch) -> None:
    """Feature without apps/tech should be re-mapped with at least two items."""

    template = "{mapping_labels} {mapping_sections} {mapping_fields} {features}"

    def fake_loader(name, *_, **__):
        return template

    items = {
        "applications": [
            MappingItem(id="APP-1", name="App One", description=""),
            MappingItem(id="APP-2", name="App Two", description=""),
            MappingItem(id="APP-3", name="App Three", description=""),
        ],
        "technologies": [
            MappingItem(id="TEC-1", name="Tech One", description=""),
            MappingItem(id="TEC-2", name="Tech Two", description=""),
            MappingItem(id="TEC-3", name="Tech Three", description=""),
        ],
    }

    monkeypatch.setattr("mapping.load_prompt_text", fake_loader)
    monkeypatch.setattr("mapping.load_mapping_items", lambda types, *a, **k: items)
    monkeypatch.setattr(
        "mapping._top_k_items", lambda features, dataset: items[dataset]
    )

    initial_apps = json.dumps(
        {
            "features": [
                {"feature_id": "f1", "applications": []},
                {
                    "feature_id": "f2",
                    "applications": [
                        {"item": "APP-1", "contribution": 0.7},
                        {"item": "APP-2", "contribution": 0.3},
                    ],
                },
            ]
        }
    )
    initial_tech = json.dumps(
        {
            "features": [
                {"feature_id": "f1", "technology": []},
                {
                    "feature_id": "f2",
                    "technology": [
                        {"item": "TEC-1", "contribution": 0.6},
                        {"item": "TEC-2", "contribution": 0.4},
                    ],
                },
            ]
        }
    )
    repaired_apps = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "applications": [
                        {"item": "APP-1", "contribution": 0.6},
                        {"item": "APP-2", "contribution": 0.4},
                    ],
                }
            ]
        }
    )
    repaired_tech = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "technology": [
                        {"item": "TEC-1", "contribution": 0.6},
                        {"item": "TEC-2", "contribution": 0.4},
                    ],
                }
            ]
        }
    )

    session = DummySession([initial_apps, initial_tech, repaired_apps, repaired_tech])

    features = [
        PlateauFeature(
            feature_id="f1",
            name="Integration",
            description="Allows external access",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        ),
        PlateauFeature(
            feature_id="f2",
            name="Export",
            description="Lets users export data",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        ),
    ]

    result = await map_features_async(
        session,
        features,
        {
            "applications": MappingTypeConfig(dataset="applications", label="Apps"),
            "technology": MappingTypeConfig(dataset="technologies", label="Tech"),
        },
    )

    assert len(result[0].mappings["applications"]) >= 2
    assert len(result[0].mappings["technology"]) >= 2


def test_top_k_items_breaks_ties_lexicographically(monkeypatch) -> None:
    """Ensure TF-IDF ranking resolves equal scores by item identifier."""

    items = {
        "catalogue": [
            MappingItem(id="B", name="Same", description="entry"),
            MappingItem(id="A", name="Same", description="entry"),
        ]
    }
    monkeypatch.setattr("mapping.load_mapping_items", lambda types, *a, **k: items)
    mapping._catalogue_vectors.cache_clear()

    feature = PlateauFeature(
        feature_id="f1",
        name="Same",
        description="entry",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = mapping._top_k_items([feature], "catalogue", k=2)
    assert [item.id for item in result] == ["A", "B"]


def test_catalogue_vectorizer_cached(monkeypatch) -> None:
    """TfidfVectorizer fitting should occur only once per dataset."""

    calls: dict[str, int] = {"load": 0, "fit": 0}

    def fake_loader(types, *_, **__):
        calls["load"] += 1
        return {"catalogue": [MappingItem(id="A", name="One", description="")]}

    monkeypatch.setattr("mapping.load_mapping_items", fake_loader)

    orig_fit = mapping.TfidfVectorizer.fit

    def wrapped_fit(self, raw_documents, y=None):
        calls["fit"] += 1
        return orig_fit(self, raw_documents, y)

    monkeypatch.setattr(mapping.TfidfVectorizer, "fit", wrapped_fit)
    mapping._catalogue_vectors.cache_clear()

    feature = PlateauFeature(
        feature_id="f1",
        name="One",
        description="",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    mapping._top_k_items([feature], "catalogue", k=1)
    mapping._top_k_items([feature], "catalogue", k=1)

    assert calls["load"] == 1
    assert calls["fit"] == 1


@pytest.mark.asyncio
async def test_embedding_top_k_items_breaks_ties(monkeypatch) -> None:
    """Embedding selection also orders items lexicographically when scores tie."""

    items = [
        MappingItem(id="B", name="Same", description="entry"),
        MappingItem(id="A", name="Same", description="entry"),
    ]
    item_vecs = np.array([[1.0, 0.0], [1.0, 0.0]])

    async def fake_catalogue_embeddings(dataset: str):
        return item_vecs, items

    monkeypatch.setattr("mapping._catalogue_embeddings", fake_catalogue_embeddings)

    class DummyEmbeddings:
        async def create(self, model: str, input: list[str]):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
            )

    class DummyClient:
        embeddings = DummyEmbeddings()

    async def fake_client():
        return DummyClient()

    monkeypatch.setattr("mapping._get_embed_client", fake_client)

    feature = PlateauFeature(
        feature_id="f1",
        name="Same",
        description="entry",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    result = await mapping._embedding_top_k_items([feature], "catalogue", k=2)
    assert [item.id for item in result] == ["A", "B"]


@pytest.mark.asyncio
async def test_feature_embedding_cache(monkeypatch) -> None:
    """Repeated feature lookups should reuse cached embeddings."""

    mapping._FEATURE_EMBED_CACHE.clear()

    async def fake_catalogue_embeddings(dataset: str):
        return np.array([[1.0, 0.0]]), [MappingItem(id="A", name="One", description="")]

    monkeypatch.setattr("mapping._catalogue_embeddings", fake_catalogue_embeddings)

    calls = {"create": 0}

    class DummyEmbeddings:
        async def create(self, model: str, input: list[str]):
            calls["create"] += 1
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
            )

    class DummyClient:
        embeddings = DummyEmbeddings()

    async def fake_client():
        return DummyClient()

    monkeypatch.setattr("mapping._get_embed_client", fake_client)

    feature = PlateauFeature(
        feature_id="f1",
        name="One",
        description="",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="learners",
    )

    await mapping._embedding_top_k_items([feature], "catalogue", k=1)
    await mapping._embedding_top_k_items([feature], "catalogue", k=1)

    assert calls["create"] == 1


@pytest.mark.asyncio
async def test_init_embeddings_populates_cache(monkeypatch) -> None:
    """Initializer should preload embeddings for all datasets."""

    mapping._EMBED_CACHE.clear()
    configs = {
        "a": MappingTypeConfig(label="A", dataset="ds1"),
        "b": MappingTypeConfig(label="B", dataset="ds2"),
    }
    monkeypatch.setattr("mapping.load_mapping_type_config", lambda: configs)

    seen: list[str] = []

    async def fake_catalogue_embeddings(name: str):
        seen.append(name)
        mapping._EMBED_CACHE[name] = (np.zeros((1, 1)), [])
        return mapping._EMBED_CACHE[name]

    monkeypatch.setattr("mapping._catalogue_embeddings", fake_catalogue_embeddings)

    await mapping.init_embeddings()

    assert set(seen) == {"ds1", "ds2"}
    assert set(mapping._EMBED_CACHE) == {"ds1", "ds2"}


@pytest.mark.asyncio
async def test_init_embeddings_handles_errors(monkeypatch) -> None:
    """Initializer should log and skip datasets that fail."""

    mapping._EMBED_CACHE.clear()
    configs = {
        "a": MappingTypeConfig(label="A", dataset="good"),
        "b": MappingTypeConfig(label="B", dataset="bad"),
    }
    monkeypatch.setattr("mapping.load_mapping_type_config", lambda: configs)

    async def fake_catalogue_embeddings(name: str):
        if name == "bad":
            raise RuntimeError("boom")
        mapping._EMBED_CACHE[name] = (np.zeros((1, 1)), [])
        return mapping._EMBED_CACHE[name]

    monkeypatch.setattr("mapping._catalogue_embeddings", fake_catalogue_embeddings)

    await mapping.init_embeddings()

    assert "good" in mapping._EMBED_CACHE
    assert "bad" not in mapping._EMBED_CACHE
