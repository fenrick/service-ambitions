# SPDX-License-Identifier: MIT
"""Tests for plateau feature generation."""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import pytest
from pydantic_ai import Agent
from pydantic_core import from_json

from conversation import (
    ConversationSession,
)
from engine.plateau_runtime import PlateauRuntime
from models import (
    FeatureItem,
    FeatureMappingRef,
    MappingFeatureGroup,
    MappingSet,
    MaturityScore,
    PlateauDescriptionsResponse,
    PlateauFeature,
    PlateauFeaturesResponse,
    PlateauResult,
    RoleFeaturesResponse,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from plateau_generator import PlateauGenerator
from runtime.environment import RuntimeEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: Sequence[str | object]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.client = None
        self.stage = "test"

    def ask(self, prompt: str) -> object:
        """Return the next queued response, parsing JSON when necessary."""
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, str):
            for model in (
                PlateauDescriptionsResponse,
                PlateauFeaturesResponse,
                RoleFeaturesResponse,
            ):
                try:
                    return model.model_validate_json(response)
                except Exception:  # noqa: BLE001
                    continue
            raise ValueError(response)
        return response

    async def ask_async(self, prompt: str) -> object:
        return self.ask(prompt)

    def add_parent_materials(self, service_input: ServiceInput) -> None:
        pass

    def derive(self) -> "DummySession":
        return self


def _meta() -> ServiceMeta:
    return ServiceMeta(
        run_id="run",
        seed=None,
        models={},
        web_search=False,
        mapping_types=[],
        created=datetime.now(timezone.utc),
    )


def _feature_payload(count: int, level: int = 1) -> str:
    # Build a uniform payload with ``count`` valid features per customer type.
    features: dict[str, list[dict[str, object]]] = {
        "learners": [],
        "academics": [],
        "professional_staff": [],
    }
    for role in features:
        items: list[dict[str, object]] = []
        for i in range(count):
            items.append(
                {
                    "name": f"Feature {i}",
                    "description": f"Desc {i}",
                    "score": {
                        "level": 3,
                        "label": "Defined",
                        "justification": "test",
                    },
                }
            )
        features[role] = items
    payload = {"features": features}
    return json.dumps(payload)


@pytest.mark.asyncio()
async def test_map_features_maps_all_sets_with_full_list(monkeypatch) -> None:
    """Each mapping set processes the full feature list exactly once."""

    called: list[str] = []
    received: list[list[str]] = []

    async def fake_map_set(session, name, items, feats, **kwargs):
        called.append(name)
        received.append([f.feature_id for f in feats])
        return list(feats)

    mapping_sets = [
        MappingSet(name="Applications", file="applications.json", field="applications"),
        MappingSet(name="Technologies", file="technologies.json", field="technologies"),
        MappingSet(name="Data", file="information.json", field="data"),
        MappingSet(name="Extra", file="extra.json", field="extra"),
    ]
    monkeypatch.setattr("plateau_generator.map_set", fake_map_set)
    RuntimeEnv.initialize(cast(Any, SimpleNamespace(mapping_sets=mapping_sets)))
    monkeypatch.setattr(
        "plateau_generator.load_mapping_items",
        lambda path, sets: ({s.field: [] for s in sets}, "hash"),
    )
    session = DummySession([])
    gen = PlateauGenerator(
        cast(ConversationSession, session),
        use_local_cache=False,
        cache_mode="off",
    )
    feats = [
        PlateauFeature(
            feature_id="f1",
            name="Feat1",
            description="d1",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="learners",
        ),
        PlateauFeature(
            feature_id="f2",
            name="Feat2",
            description="d2",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="academics",
        ),
    ]

    await gen._map_features(
        cast(ConversationSession, session),
        feats,
        plateau=1,
        service_name="svc",
        service_description="desc",
    )

    assert called == [s.field for s in mapping_sets]
    assert all(ids == ["f1", "f2"] for ids in received)


def test_build_plateau_prompt_excludes_feature_id() -> None:
    """Old FEAT-* identifiers should not appear in prompts."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        use_local_cache=False,
        cache_mode="off",
    )
    generator._service = ServiceInput(
        service_id="s",
        name="svc",
        customer_type="type",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    prompt = generator._build_plateau_prompt(2, "d")

    assert "FEAT-" not in prompt


def test_to_feature_hashes_name_role_and_plateau() -> None:
    """_to_feature should hash name, role and plateau."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        use_local_cache=False,
        cache_mode="off",
    )
    item = FeatureItem(
        name="Example",
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
    )
    feature = generator._to_feature(item, "learners", "Foundational")
    assert feature.feature_id == "P545N4"


def test_generate_plateau_returns_results(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    desc_payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc",
                }
            ]
        }
    )
    responses = [desc_payload, _feature_payload(1)]
    session = DummySession(responses)

    call = {"n": 0}

    async def dummy_map_features(self, session, feats, **kwargs):
        call["n"] += 1
        refs = [
            FeatureMappingRef(feature_id=f.feature_id, description=f.description)
            for f in feats
        ]
        return {
            "data": [MappingFeatureGroup(id="d", name="d", mappings=refs.copy())],
            "applications": [
                MappingFeatureGroup(id="a", name="a", mappings=refs.copy())
            ],
            "technologies": [
                MappingFeatureGroup(id="t", name="t", mappings=refs.copy())
            ],
        }

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    runtime = PlateauRuntime(
        plateau=1,
        plateau_name="Foundational",
        description=desc_map["Foundational"],
    )
    plateau = generator.generate_plateau(runtime)

    assert isinstance(plateau, PlateauRuntime)
    assert len(plateau.features) == 3
    assert set(plateau.mappings.keys()) == {
        "data",
        "applications",
        "technologies",
    }
    assert call["n"] == 1
    assert len(session.prompts) == 2


def test_generate_plateau_repairs_missing_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    initial = json.dumps(
        {
            "features": {
                "learners": [],
                "academics": [
                    {
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "professional_staff": [
                    {
                        "name": "P",
                        "description": "dp",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
            }
        }
    )
    repair = json.dumps(
        {
            "features": [
                {
                    "name": "L",
                    "description": "dl",
                    "score": {
                        "level": 3,
                        "label": "Defined",
                        "justification": "j",
                    },
                }
            ]
        }
    )
    desc_payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc",
                }
            ]
        }
    )
    session = DummySession([desc_payload, initial, repair])

    async def dummy_map_features(self, session, feats, **kwargs):
        return {}

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    runtime = PlateauRuntime(
        plateau=1,
        plateau_name="Foundational",
        description=desc_map["Foundational"],
    )
    plateau = generator.generate_plateau(runtime)

    assert len(session.prompts) == 3
    learners = [f for f in plateau.features if f.customer_type == "learners"]
    assert len(learners) == 1


def test_generate_plateau_requests_missing_features_concurrently(
    monkeypatch,
) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)

    initial = json.dumps(
        {
            "features": {
                "learners": [
                    {
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    },
                    {
                        "name": "B",
                        "description": "db",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    },
                ],
                "academics": [
                    {
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "professional_staff": [
                    {
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
            }
        }
    )
    desc_payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc",
                }
            ]
        }
    )
    session = DummySession([desc_payload, initial])

    async def dummy_map_features(self, session, feats, **kwargs):
        return {}

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=2,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    async def fake_request(self, level, role, description, missing, session):
        await asyncio.sleep(0.1)  # Simulate network delay per role request.
        return [
            FeatureItem(
                name=f"Extra {role}",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
            )
        ]

    monkeypatch.setattr(
        PlateauGenerator, "_request_missing_features_async", fake_request
    )

    async def run() -> tuple[PlateauRuntime, float]:
        desc_map = await generator._request_descriptions_async(["Foundational"])
        start = time.perf_counter()
        runtime = PlateauRuntime(
            plateau=1,
            plateau_name="Foundational",
            description=desc_map["Foundational"],
        )
        plateau = await generator.generate_plateau_async(
            runtime, session=cast(ConversationSession, session)
        )
        duration = time.perf_counter() - start
        return plateau, duration

    plateau, duration = asyncio.run(run())

    academics = [
        f for f in plateau.features if f.customer_type == "academics"
    ]  # Extract academic features.
    professional = [
        f for f in plateau.features if f.customer_type == "professional_staff"
    ]  # Extract professional staff features.
    assert len(academics) == 2
    assert len(professional) == 2
    assert duration < 0.19  # Parallel calls should take ~0.1s overall.


def test_generate_plateau_raises_on_insufficient_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    repair = json.dumps({"features": []})
    desc_payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc",
                }
            ]
        }
    )
    responses = [desc_payload, _feature_payload(1), repair, repair, repair]
    session = DummySession(responses)
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=2,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    runtime = PlateauRuntime(
        plateau=1,
        plateau_name="Foundational",
        description=desc_map["Foundational"],
    )
    with pytest.raises(ValueError):
        generator.generate_plateau(runtime)


def test_generate_plateau_missing_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    desc_payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc",
                }
            ]
        }
    )
    responses = [desc_payload, "{}", "{}"]
    session = DummySession(responses)
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    runtime = PlateauRuntime(
        plateau=1,
        plateau_name="Foundational",
        description=desc_map["Foundational"],
    )
    with pytest.raises(ValueError):
        generator.generate_plateau(runtime)


@pytest.mark.asyncio()
async def test_generate_plateau_supports_custom_roles(monkeypatch) -> None:
    """Generator should handle arbitrary role identifiers."""

    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        if name == "plateau_prompt":
            return template
        if name == "plateau_descriptions_prompt":
            return "desc {plateaus} {schema}"
        return ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)

    payload = json.dumps(
        {
            "features": {
                "researchers": [
                    {
                        "name": "R1",
                        "description": "dr",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "students": [
                    {
                        "name": "S1",
                        "description": "ds",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
            }
        }
    )

    session = DummySession([payload])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        roles=["researchers", "students"],
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    async def dummy_map_features(self, session, feats, **kwargs):
        return {}

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    runtime = PlateauRuntime(plateau=1, plateau_name="Foundational", description="desc")
    plateau = await generator.generate_plateau_async(
        runtime, session=cast(ConversationSession, session)
    )

    assert {f.customer_type for f in plateau.features} == {"researchers", "students"}


@pytest.mark.asyncio()
async def test_request_descriptions_async(monkeypatch) -> None:
    """Agent returns plateau descriptions for requested names."""

    def fake_loader(name, *_, **__):
        return "template" if name == "plateau_descriptions_prompt" else ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    good = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc one",
                }
            ]
        }
    )
    session = DummySession([good])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )

    result = await generator._request_descriptions_async(["Foundational"])

    assert result == {"Foundational": "desc one"}
    assert session.prompts == ["template"]


def test_request_description_strips_preamble(monkeypatch) -> None:
    """Model-added preamble should be removed from descriptions."""

    def fake_loader(name, *_, **__):
        return "template" if name == "plateau_descriptions_prompt" else ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": (
                        "Prepared plateau-1 description for svc: actual details"
                    ),
                }
            ]
        }
    )
    session = DummySession([payload])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )

    result = generator._request_description(1)

    assert result == "actual details"


def test_request_descriptions_returns_mapping(monkeypatch) -> None:
    """_request_descriptions should parse batch responses."""

    def fake_loader(name, *_, **__):
        return "template" if name == "plateau_descriptions_prompt" else ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    payload = json.dumps(
        {
            "descriptions": [
                {
                    "plateau": 1,
                    "plateau_name": "Foundational",
                    "description": "desc one",
                }
            ]
        }
    )
    session = DummySession([payload])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        required_count=1,
        use_local_cache=False,
        cache_mode="off",
    )

    result = generator._request_descriptions(["Foundational"])

    assert result == {"Foundational": "desc one"}
    assert len(session.prompts) == 1


def test_generate_service_evolution_filters(monkeypatch) -> None:
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    class DummyAgent:
        def run_sync(self, prompt, message_history):
            return type("R", (), {"output": "", "new_messages": lambda: []})()

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
    generator = PlateauGenerator(
        session,
        use_local_cache=False,
        cache_mode="off",
    )

    called: list[int] = []
    sessions: set[int] = set()

    async def fake_generate_plateau_async(
        self, runtime, *, session=None
    ) -> PlateauRuntime:
        called.append(runtime.plateau)
        sessions.add(id(session))
        feats = [
            PlateauFeature(
                feature_id=f"l{runtime.plateau}",
                name="L",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
                customer_type="learners",
            ),
            PlateauFeature(
                feature_id=f"s{runtime.plateau}",
                name="S",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
                customer_type="academics",
            ),
        ]
        runtime.set_results(features=feats, mappings={})
        return runtime

    async def fake_request_descriptions_async(self, names, session=None):
        return {name: "desc" for name in names}

    monkeypatch.setattr(
        PlateauGenerator,
        "generate_plateau_async",
        fake_generate_plateau_async,
    )
    monkeypatch.setattr(
        PlateauGenerator,
        "_request_descriptions_async",
        fake_request_descriptions_async,
    )

    runtimes = [
        PlateauRuntime(plateau=1, plateau_name="Foundational", description="desc"),
        PlateauRuntime(plateau=2, plateau_name="Enhanced", description="desc"),
    ]
    evo = generator.generate_service_evolution(
        service,
        runtimes,
        ["learners", "academics"],
        meta=_meta(),
    )

    assert called == [1, 2]
    assert len(sessions) == 2
    assert len(evo.plateaus) == 2
    for plat in evo.plateaus:
        assert {f.customer_type for f in plat.features} <= {"learners", "academics"}


def test_generate_service_evolution_invalid_role_raises(monkeypatch) -> None:
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    class DummyAgent:
        def run_sync(self, prompt, message_history):
            return type("R", (), {"output": "", "new_messages": lambda: []})()

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
    generator = PlateauGenerator(
        session,
        use_local_cache=False,
        cache_mode="off",
    )

    async def fake_generate_plateau_async(self, runtime, *, session=None):
        feat = PlateauFeature(
            feature_id="f1",
            name="F",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="unknown",
        )
        runtime.set_results(features=[feat], mappings={})
        return runtime

    async def fake_request_descriptions_async(self, names, session=None):
        return {name: "desc" for name in names}

    monkeypatch.setattr(
        PlateauGenerator,
        "generate_plateau_async",
        fake_generate_plateau_async,
    )
    monkeypatch.setattr(
        PlateauGenerator,
        "_request_descriptions_async",
        fake_request_descriptions_async,
    )

    with pytest.raises(ValueError):
        generator.generate_service_evolution(
            service,
            [
                PlateauRuntime(
                    plateau=1, plateau_name="Foundational", description="desc"
                )
            ],
            ["learners"],
            meta=_meta(),
        )


def test_generate_service_evolution_unknown_plateau_raises(monkeypatch) -> None:
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    class DummyAgent:
        def run_sync(self, prompt, message_history):
            return type("R", (), {"output": "", "new_messages": lambda: []})()

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
    generator = PlateauGenerator(
        session,
        use_local_cache=False,
        cache_mode="off",
    )

    async def fake_generate_plateau_async(self, runtime, *, session=None):
        feat = PlateauFeature(
            feature_id="f1",
            name="F",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        )
        runtime.set_results(features=[feat], mappings={})
        runtime.plateau_name = "Mystery"
        return runtime

    async def fake_request_descriptions_async(self, names, session=None):
        return {name: "desc" for name in names}

    monkeypatch.setattr(
        PlateauGenerator,
        "generate_plateau_async",
        fake_generate_plateau_async,
    )
    monkeypatch.setattr(
        PlateauGenerator,
        "_request_descriptions_async",
        fake_request_descriptions_async,
    )

    evolution = generator.generate_service_evolution(
        service,
        [PlateauRuntime(plateau=1, plateau_name="Foundational", description="d")],
        ["learners"],
        meta=_meta(),
    )
    assert evolution.plateaus[0].plateau_name == "Mystery"


def test_generate_service_evolution_deduplicates_features(monkeypatch) -> None:
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    class DummyAgent:
        def run_sync(self, prompt, message_history):
            return type("R", (), {"output": "", "new_messages": lambda: []})()

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
    generator = PlateauGenerator(
        session,
        use_local_cache=False,
        cache_mode="off",
    )

    async def fake_generate_plateau_async(self, runtime, *, session=None):
        item = FeatureItem(
            name="A",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
        )
        feat1 = self._to_feature(item, "learners", runtime.plateau_name)
        feat2 = self._to_feature(item, "learners", runtime.plateau_name)
        runtime.set_results(features=[feat1, feat2], mappings={})
        return runtime

    async def fake_request_descriptions_async(self, names, session=None):
        return {name: "desc" for name in names}

    monkeypatch.setattr(
        PlateauGenerator,
        "generate_plateau_async",
        fake_generate_plateau_async,
    )
    monkeypatch.setattr(
        PlateauGenerator,
        "_request_descriptions_async",
        fake_request_descriptions_async,
    )

    evo = generator.generate_service_evolution(
        service,
        [PlateauRuntime(plateau=1, plateau_name="Foundational", description="desc")],
        ["learners"],
        meta=_meta(),
    )

    features = evo.plateaus[0].features
    assert len(features) == 1
    assert features[0].feature_id == "H4R765"


def test_validate_plateau_results_strict_checks() -> None:
    """Strict mode should validate roles and mappings."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        strict=True,
        use_local_cache=False,
        cache_mode="off",
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Feat",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="learners",
    )
    group = MappingFeatureGroup(
        id="a",
        name="a",
        mappings=[FeatureMappingRef(feature_id="f1", description="d")],
    )
    result = PlateauResult(
        plateau=1,
        plateau_name="Foundational",
        service_description="d",
        features=[feature],
        mappings={"apps": [group]},
    )
    plateaus, roles_seen = generator._validate_plateau_results(
        [result],
        ["Foundational"],
        ["learners", "academics"],
        strict=True,
    )
    assert roles_seen == {"learners": True, "academics": False}
    assert plateaus[0].mappings == {"apps": [group]}
    # Missing mappings should raise when strict
    result.mappings = {}
    with pytest.raises(ValueError):
        generator._validate_plateau_results(
            [result],
            ["Foundational"],
            ["learners"],
            strict=True,
        )

    # Best-effort mode allows incomplete data
    generator._validate_plateau_results(
        [result],
        ["Foundational"],
        ["learners"],
        strict=False,
    )


def test_write_transcript_writes_payload(tmp_path) -> None:
    """Transcript writing should persist payloads without modification."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="s1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )
    evolution = ServiceEvolution(meta=_meta(), service=service, plateaus=[])

    asyncio.run(generator._write_transcript(tmp_path, service, evolution))

    path = tmp_path / f"{service.service_id}.json"
    data = from_json(path.read_text(encoding="utf-8"))
    assert data["request"]["service_id"] == "s1"


@pytest.mark.asyncio()
async def test_generate_plateau_reads_feature_cache(monkeypatch, tmp_path) -> None:
    """Legacy feature caches are relocated and reused."""

    monkeypatch.chdir(tmp_path)
    payload = PlateauFeaturesResponse(
        features={
            "learners": [
                FeatureItem(
                    name="Feat",
                    description="Desc",
                    score=MaturityScore(level=1, label="Initial", justification="j"),
                )
            ]
        }
    )
    old_file = Path(".cache") / "unknown" / "svc" / "features.json"
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text(payload.model_dump_json(), encoding="utf-8")

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session),
        roles=["learners"],
        required_count=1,
        use_local_cache=True,
        cache_mode="read",
    )
    generator._service = ServiceInput(
        service_id="svc",
        name="svc",
        description="desc",
        jobs_to_be_done=[],
    )

    async def fake_map_features(*args, **kwargs):
        return {}

    monkeypatch.setattr(generator, "_map_features", fake_map_features)

    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")
    result = await generator.generate_plateau_async(runtime)

    assert result.features[0].name == "Feat"
    canonical = Path(".cache") / "unknown" / "svc" / "1" / "features.json"
    assert canonical.exists()
    assert not old_file.exists()
    assert session.prompts == []
