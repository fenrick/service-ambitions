# SPDX-License-Identifier: MIT
"""Tests for plateau feature generation."""

import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import Agent

from conversation import (
    ConversationSession,
)
from models import (
    Contribution,
    FeatureItem,
    MappingSet,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from plateau_generator import PlateauGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []
        self.client = None
        self.stage = "test"

    def ask(self, prompt: str, output_type=None) -> object:
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if output_type is None:
            return response
        return output_type.model_validate_json(response)

    async def ask_async(self, prompt: str, output_type=None) -> object:
        return self.ask(prompt, output_type)

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
        MappingSet(name="Technologies", file="technologies.json", field="technology"),
        MappingSet(name="Data", file="information.json", field="data"),
        MappingSet(name="Extra", file="extra.json", field="extra"),
    ]
    monkeypatch.setattr("plateau_generator.map_set", fake_map_set)
    monkeypatch.setattr(
        "plateau_generator.load_settings",
        lambda: SimpleNamespace(mapping_sets=mapping_sets),
    )
    monkeypatch.setattr(
        "plateau_generator.load_mapping_items",
        lambda path, sets: {s.field: [] for s in sets},
    )
    session = DummySession([])
    gen = PlateauGenerator(cast(ConversationSession, session))
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

    await gen._map_features(cast(ConversationSession, session), feats)

    assert called == [s.field for s in mapping_sets]
    assert all(ids == ["f1", "f2"] for ids in received)


def test_build_plateau_prompt_excludes_feature_id() -> None:
    """Old FEAT-* identifiers should not appear in prompts."""

    session = DummySession([])
    generator = PlateauGenerator(cast(ConversationSession, session))
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
    generator = PlateauGenerator(cast(ConversationSession, session))
    item = FeatureItem(
        name="Example",
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
    )
    feature = generator._to_feature(item, "learners", "Foundational")
    expected = hashlib.sha1("Example|learners|Foundational".encode()).hexdigest()
    assert feature.feature_id == expected


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

    async def dummy_map_features(self, session, feats):
        call["n"] += 1
        for feat in feats:
            feat.mappings["data"] = [Contribution(item="d")]
            feat.mappings["applications"] = [Contribution(item="a")]
            feat.mappings["technology"] = [Contribution(item="t")]
        return feats

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    plateau = generator.generate_plateau(
        1, "Foundational", description=desc_map["Foundational"]
    )

    assert isinstance(plateau, PlateauResult)
    assert len(plateau.features) == 3
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
                    "score": {"level": 3, "label": "Defined", "justification": "j"},
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

    async def dummy_map_features(self, session, feats):
        return feats

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    plateau = generator.generate_plateau(
        1, "Foundational", description=desc_map["Foundational"]
    )

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

    async def dummy_map_features(self, session, feats):
        return feats

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=2)
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

    async def run() -> tuple[PlateauResult, float]:
        desc_map = await generator._request_descriptions_async(["Foundational"])
        start = time.perf_counter()
        plateau = await generator.generate_plateau_async(
            1,
            "Foundational",
            session=cast(ConversationSession, session),
            description=desc_map["Foundational"],
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


def test_generate_plateau_repairs_invalid_role(monkeypatch) -> None:
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
                "learners": [{}],
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
                    "score": {"level": 3, "label": "Defined", "justification": "j"},
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

    async def dummy_map_features(self, session, feats):
        return feats

    monkeypatch.setattr(PlateauGenerator, "_map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    plateau = generator.generate_plateau(
        1, "Foundational", description=desc_map["Foundational"]
    )

    assert len(session.prompts) == 3
    learners = [f for f in plateau.features if f.customer_type == "learners"]
    assert len(learners) == 1


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
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=2)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    with pytest.raises(ValueError):
        generator.generate_plateau(
            1, "Foundational", description=desc_map["Foundational"]
        )


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
    responses = [desc_payload, "{}"]
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    desc_map = asyncio.run(generator._request_descriptions_async(["Foundational"]))
    with pytest.raises(ValueError) as exc:
        generator.generate_plateau(
            1, "Foundational", description=desc_map["Foundational"]
        )

    assert "invalid JSON" in str(exc.value)


def test_request_description_invalid_json(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    """Invalid description payloads should raise ``ValueError``."""
    session = DummySession(["not json"])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    with pytest.raises(ValueError):
        generator._request_description(1)
    assert len(session.prompts) == 1
    assert session.prompts[0].startswith("desc 1")


def test_request_description_strips_preamble(monkeypatch) -> None:
    """Model-added preamble should be removed from descriptions."""

    def fake_loader(name, *_, **__):
        return "desc {plateau}" if name == "description_prompt" else ""

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    payload = json.dumps(
        {
            "description": "Prepared plateau-1 description for svc: actual details",
        }
    )
    session = DummySession([payload])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)

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
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)

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

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    generator = PlateauGenerator(session)

    called: list[int] = []
    sessions: set[int] = set()

    async def fake_generate_plateau_async(
        self, level, plateau_name, *, session=None, description
    ) -> PlateauResult:
        called.append(level)
        sessions.add(id(session))
        feats = [
            PlateauFeature(
                feature_id=f"l{level}",
                name="L",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
                customer_type="learners",
            ),
            PlateauFeature(
                feature_id=f"s{level}",
                name="S",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
                customer_type="academics",
            ),
        ]
        return PlateauResult(
            plateau=level,
            plateau_name=plateau_name,
            service_description="d",
            features=feats,
        )

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
        ["Foundational", "Enhanced"],
        ["learners", "academics"],
        meta=_meta(),
    )

    assert called == [1, 2]
    assert len(sessions) == 1
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

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    generator = PlateauGenerator(session)

    async def fake_generate_plateau_async(
        self, level, plateau_name, *, session=None, description
    ):
        feat = PlateauFeature(
            feature_id="f1",
            name="F",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="unknown",
        )
        return PlateauResult(
            plateau=level,
            plateau_name=plateau_name,
            service_description="d",
            features=[feat],
        )

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
            ["Foundational"],
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

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    generator = PlateauGenerator(session)

    async def fake_generate_plateau_async(
        self, level, plateau_name, *, session=None, description
    ):
        feat = PlateauFeature(
            feature_id="f1",
            name="F",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        )
        return PlateauResult(
            plateau=level,
            plateau_name="Mystery",
            service_description="d",
            features=[feat],
        )

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
            ["Foundational"],
            ["learners"],
            meta=_meta(),
        )


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

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    generator = PlateauGenerator(session)

    async def fake_generate_plateau_async(
        self, level, plateau_name, *, session=None, description
    ):
        feat1 = PlateauFeature(
            feature_id="a",
            name="A",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        )
        feat2 = PlateauFeature(
            feature_id="b",
            name="A",
            description="d",
            score=MaturityScore(level=3, label="Defined", justification="j"),
            customer_type="learners",
        )
        return PlateauResult(
            plateau=level,
            plateau_name=plateau_name,
            service_description="d",
            features=[feat1, feat2],
        )

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
        ["Foundational"],
        ["learners"],
        meta=_meta(),
    )

    features = evo.plateaus[0].features
    assert len(features) == 1
    expected = hashlib.sha1("A|learners|Foundational".encode()).hexdigest()
    assert features[0].feature_id == expected


def test_validate_plateau_results_strict_checks() -> None:
    """Strict mode should validate roles and mappings."""

    session = DummySession([])
    generator = PlateauGenerator(cast(ConversationSession, session), strict=True)
    feature = PlateauFeature(
        feature_id="f1",
        name="Feat",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="learners",
        mappings={"apps": [Contribution(item="a")]},
    )
    result = PlateauResult(
        plateau=1,
        plateau_name="Foundational",
        service_description="d",
        features=[feature],
    )
    plateaus, roles_seen = generator._validate_plateau_results(
        [result],
        ["Foundational"],
        ["learners", "academics"],
        strict=True,
    )
    assert roles_seen == {"learners": True, "academics": False}
    assert plateaus[0].features == [feature]

    # Missing mappings should raise when strict
    feature.mappings = {}
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


def test_write_transcript_writes_payload(tmp_path, monkeypatch) -> None:
    """Transcript writing should persist redacted payloads."""

    session = DummySession([])
    generator = PlateauGenerator(cast(ConversationSession, session))
    service = ServiceInput(
        service_id="s1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )
    evolution = ServiceEvolution(meta=_meta(), service=service, plateaus=[])
    monkeypatch.setattr("plateau_generator.redact_pii", lambda x: x)

    asyncio.run(generator._write_transcript(tmp_path, service, evolution))

    path = tmp_path / f"{service.service_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["request"]["service_id"] == "s1"
