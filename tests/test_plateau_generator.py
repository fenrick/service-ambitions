"""Tests for plateau feature generation."""

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    Contribution,
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

    async def ask(self, prompt: str) -> str:  # pragma: no cover - simple proxy
        self.prompts.append(prompt)
        # Pop from the front so responses are returned in the order queued.
        return self._responses.pop(0)

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
            "score": 0.5,
        }
        for i in range(count)
    ]
    payload = {"learners": items, "academics": items, "professional_staff": items}
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_generate_plateau_returns_results(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"

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

    plateau = await generator.generate_plateau(1, "Foundational")

    assert isinstance(plateau, PlateauResult)
    assert len(plateau.features) == 3
    assert call["n"] == 1
    assert len(session.prompts) == 2


@pytest.mark.asyncio
async def test_generate_plateau_raises_on_insufficient_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"

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
        await generator.generate_plateau(1, "Foundational")


@pytest.mark.asyncio
async def test_request_description_invalid_json(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"

    def fake_loader(name, *_, **__):
        return template if name == "plateau_prompt" else "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    """Invalid description payloads should raise ``ValueError``."""
    session = DummySession(["not json"])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    with pytest.raises(ValueError):
        await generator._request_description(1)
    assert len(session.prompts) == 1
    assert session.prompts[0].startswith("desc 1")


@pytest.mark.asyncio
async def test_request_description_strips_code_fence(monkeypatch) -> None:
    """The generator should parse JSON wrapped in Markdown fences."""

    def fake_loader(name, *_, **__):
        return "desc {plateau}"

    monkeypatch.setattr("plateau_generator.load_prompt_text", fake_loader)
    response = '```json\n{"description": "hello"}\n```'
    session = DummySession([response])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)

    description = await generator._request_description(1)

    assert description == "hello"
    assert session.prompts[0].startswith("desc 1")


def test_parse_feature_payload_strips_code_fence() -> None:
    """Feature payload parsing should ignore surrounding code fences."""

    payload = _feature_payload(1)
    fenced = f"```json\n{payload}\n```"

    result = PlateauGenerator._parse_feature_payload(fenced)

    assert len(result.learners) == 1


@pytest.mark.asyncio
async def test_generate_service_evolution_filters(monkeypatch) -> None:
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )
    session = DummySession([])
    generator = PlateauGenerator(cast(ConversationSession, session))

    called: list[int] = []

    def fake_generate_plateau(self, level, plateau_name):
        called.append(level)
        feats = [
            PlateauFeature(
                feature_id=f"l{level}",
                name="L",
                description="d",
                score=0.5,
                customer_type="learners",
            ),
            PlateauFeature(
                feature_id=f"s{level}",
                name="S",
                description="d",
                score=0.5,
                customer_type="academics",
            ),
            PlateauFeature(
                feature_id=f"c{level}",
                name="C",
                description="d",
                score=0.5,
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

    evo = await generator.generate_service_evolution(
        service,
        ["Foundational", "Enhanced"],
        ["learners", "academic"],
    )

    assert called == [1, 2]
    assert len(evo.plateaus) == 2
    for plat in evo.plateaus:
        assert {f.customer_type for f in plat.features} <= {"learners", "academic"}
