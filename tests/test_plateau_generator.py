"""Tests for plateau feature generation."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent

from conversation import (
    ConversationSession,
)
from models import (
    Contribution,
    FeatureItem,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceInput,
)
from plateau_generator import (
    PlateauGenerator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []
        self.client = None

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


def _feature_payload(count: int, level: int = 1) -> str:
    # Build a uniform payload with ``count`` valid features per customer type.
    features = {"learners": [], "academics": [], "professional_staff": []}
    for role in features:
        items = []
        for i in range(count):
            items.append(
                {
                    "feature_id": f"FEAT-{level}-{role}-f{i}",
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


def test_predict_token_load_uses_estimate_tokens(monkeypatch) -> None:
    """Token load predictions should call ``estimate_tokens``."""

    called: dict[str, tuple[str, int]] = {}

    def fake_estimate(text: str, expected: int) -> int:
        called["args"] = (text, expected)
        return 9

    monkeypatch.setattr("plateau_generator.estimate_tokens", fake_estimate)
    assert PlateauGenerator._predict_token_load("hello") == 9
    assert called["args"] == ("hello", 0)


def test_build_plateau_prompt_preserves_role_placeholder(monkeypatch) -> None:
    """_build_plateau_prompt should retain role placeholders."""

    template = 'Format: "FEAT-{plateau}-{{role}}-{{kebab-case-short-name}}"'
    monkeypatch.setattr(
        "plateau_generator.load_prompt_text", lambda *args, **kwargs: template
    )
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

    assert "FEAT-2-{role}-{kebab-case-short-name}" in prompt


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

    async def dummy_map_features(sess, feats):
        call["n"] += 1
        for feat in feats:
            feat.mappings["data"] = [Contribution(item="d", contribution=0.5)]
            feat.mappings["applications"] = [Contribution(item="a", contribution=0.5)]
            feat.mappings["technology"] = [Contribution(item="t", contribution=0.5)]
        return feats

    monkeypatch.setattr("plateau_generator.map_features_async", dummy_map_features)

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
                        "feature_id": "FEAT-1-academics-a",
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "professional_staff": [
                    {
                        "feature_id": "FEAT-1-professional_staff-p",
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
                    "feature_id": "FEAT-1-learners-l",
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

    async def dummy_map_features(sess, feats):
        return feats

    monkeypatch.setattr("plateau_generator.map_features_async", dummy_map_features)

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
                        "feature_id": "FEAT-1-learners-a",
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    },
                    {
                        "feature_id": "FEAT-1-learners-b",
                        "name": "B",
                        "description": "db",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    },
                ],
                "academics": [
                    {
                        "feature_id": "FEAT-1-academics-a",
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "professional_staff": [
                    {
                        "feature_id": "FEAT-1-professional_staff-a",
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

    async def dummy_map_features(sess, feats):
        return feats

    monkeypatch.setattr("plateau_generator.map_features_async", dummy_map_features)

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
                feature_id=f"FEAT-1-{role}-extra",
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
                "learners": [{"feature_id": "FEAT-1-learners-bad"}],
                "academics": [
                    {
                        "feature_id": "FEAT-1-academics-a",
                        "name": "A",
                        "description": "da",
                        "score": {"level": 3, "label": "Defined", "justification": "j"},
                    }
                ],
                "professional_staff": [
                    {
                        "feature_id": "FEAT-1-professional_staff-p",
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
                    "feature_id": "FEAT-1-learners-l",
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

    async def dummy_map_features(sess, feats):
        return feats

    monkeypatch.setattr("plateau_generator.map_features_async", dummy_map_features)

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
            PlateauFeature(
                feature_id=f"c{level}",
                name="C",
                description="d",
                score=MaturityScore(level=3, label="Defined", justification="j"),
                customer_type="professional_staff",
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
        ["learners", "academic"],
    )

    assert called == [1, 2]
    assert len(sessions) == 2
    assert len(evo.plateaus) == 2
    for plat in evo.plateaus:
        assert {f.customer_type for f in plat.features} <= {"learners", "academic"}
