"""Tests for plateau feature generation."""

import json
import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import (
    Contribution,
    PlateauFeature,
    PlateauResult,
    ServiceInput,
)  # noqa: E402  pylint: disable=wrong-import-position
from plateau_generator import (
    PlateauGenerator,
)  # noqa: E402  pylint: disable=wrong-import-position


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:  # pragma: no cover - simple proxy
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover - simple stub
        pass


def _feature_payload(count: int) -> str:
    items = [
        {
            "feature_id": f"f{i}",
            "name": f"Feature {i}",
            "description": f"Desc {i}",
            "score": 0.5,
        }
        for i in range(count)
    ]
    payload = {"learners": items, "staff": items, "community": items}
    return json.dumps(payload)


def test_generate_plateau_returns_results(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )
    responses = [json.dumps({"description": "desc"}), _feature_payload(1)]
    session = DummySession(responses)

    call = {"n": 0}

    def dummy_map_features(sess, feats):
        call["n"] += 1
        for feat in feats:
            feat.data = [Contribution(item="d", contribution="c")]
            feat.applications = [Contribution(item="a", contribution="c")]
            feat.technology = [Contribution(item="t", contribution="c")]
        return feats

    monkeypatch.setattr("plateau_generator.map_features", dummy_map_features)

    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    generator._service = service  # type: ignore[attr-defined]

    plateau = generator.generate_plateau(cast(ConversationSession, session), 1)

    assert isinstance(plateau, PlateauResult)
    assert len(plateau.features) == 3
    assert call["n"] == 1
    assert len(session.prompts) == 2


def test_generate_plateau_raises_on_insufficient_features(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )
    responses = [json.dumps({"description": "desc"}), _feature_payload(1)]
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=2)
    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    generator._service = service  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        generator.generate_plateau(cast(ConversationSession, session), 1)


def test_request_description_invalid_json(monkeypatch) -> None:
    template = "{required_count} {service_name} {service_description} {plateau}"
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )
    """Invalid description payloads should raise ``ValueError``."""
    session = DummySession(["not json"])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    with pytest.raises(ValueError):
        generator._request_description(cast(ConversationSession, session), 1)


def test_generate_service_evolution_filters(monkeypatch) -> None:
    service = ServiceInput(name="svc", customer_type="retail", description="d")
    session = DummySession([])
    generator = PlateauGenerator(cast(ConversationSession, session))

    called: list[int] = []

    def fake_generate_plateau(self, sess, level):
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
                customer_type="staff",
            ),
            PlateauFeature(
                feature_id=f"c{level}",
                name="C",
                description="d",
                score=0.5,
                customer_type="community",
            ),
        ]
        return PlateauResult(plateau=level, service_description="d", features=feats)

    monkeypatch.setattr(
        PlateauGenerator, "generate_plateau", fake_generate_plateau, raising=False
    )

    evo = generator.generate_service_evolution(
        service,
        ["Foundational", "Enhanced"],
        ["learners", "staff"],
    )

    assert called == [1, 2]
    assert len(evo.plateaus) == 2
    for plat in evo.plateaus:
        assert {f.customer_type for f in plat.features} <= {"learners", "staff"}
