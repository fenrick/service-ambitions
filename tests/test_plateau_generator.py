"""Tests for plateau feature generation."""

import json
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent  # noqa: E402  pylint: disable=wrong-import-position

from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    Contribution,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceInput,
)
from plateau_generator import (
    PlateauGenerator,
)  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def ask(
        self, prompt: str, output_type=None
    ) -> object:  # pragma: no cover - simple proxy
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if output_type is None:
            return response
        return output_type.model_validate_json(response)

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover - simple stub
        pass


def _feature_payload(count: int) -> str:
    # Build a uniform payload with ``count`` features per customer type.
    items = [
        {
            "feature_id": f"f{i}",
            "name": f"Feature {i}",
            "description": f"Desc {i}",
            "score": {
                "level": 3,
                "label": "Defined",
                "justification": "test",
            },
        }
        for i in range(count)
    ]
    payload = {
        "features": {
            "learners": items,
            "academics": items,
            "professional_staff": items,
        }
    }
    return json.dumps(payload)


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
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    responses = [json.dumps({"description": "desc"}), _feature_payload(1)]
    session = DummySession(responses)

    call = {"n": 0}

    def dummy_map_features(sess, feats):
        call["n"] += 1
        for feat in feats:
            feat.mappings["data"] = [Contribution(item="d", contribution="c")]
            feat.mappings["applications"] = [Contribution(item="a", contribution="c")]
            feat.mappings["technology"] = [Contribution(item="t", contribution="c")]
        return feats

    monkeypatch.setattr("plateau_generator.map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    generator._service = service

    plateau = generator.generate_plateau(1, "Foundational")

    assert isinstance(plateau, PlateauResult)
    assert len(plateau.features) == 3
    assert call["n"] == 1
    assert len(session.prompts) == 2


def test_generate_plateau_raises_on_insufficient_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    responses = [json.dumps({"description": "desc"}), _feature_payload(1)]
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

    with pytest.raises(ValueError):
        generator.generate_plateau(1, "Foundational")


def test_generate_plateau_missing_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau} {roles}"

    def fake_loader(name, *_, **__):
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    responses = [json.dumps({"description": "desc"}), "{}"]
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

    with pytest.raises(ValueError) as exc:
        generator.generate_plateau(1, "Foundational")

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
        def run_sync(self, prompt, message_history):  # pragma: no cover - stub
            return type("R", (), {"output": "", "new_messages": lambda: []})()

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    generator = PlateauGenerator(session)

    called: list[int] = []
    sessions: set[int] = set()

    def fake_generate_plateau(
        self, level, plateau_name, session=None, description=None
    ):
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

    monkeypatch.setattr(
        PlateauGenerator, "generate_plateau", fake_generate_plateau, raising=False
    )
    monkeypatch.setattr(
        PlateauGenerator,
        "_request_descriptions",
        lambda self, names, session=None: {name: "desc" for name in names},
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
