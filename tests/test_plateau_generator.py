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
from models import ServiceInput  # noqa: E402  pylint: disable=wrong-import-position
from plateau_generator import (
    PlateauGenerator,
)  # noqa: E402  pylint: disable=wrong-import-position


class DummySession:
    """Conversation session returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def ask(self, prompt: str) -> str:  # pragma: no cover - simple proxy
        return self._responses.pop(0)


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
    return json.dumps({"features": items})


def test_generate_plateau_returns_results(monkeypatch) -> None:
    template = (
        "{required_count} {service_name} {service_description} "
        "{plateau} {customer_type}"
    )
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )
    mapping_template = (
        "{feature_name} {feature_description} {category_label} "
        "{category_items} {category_key}"
    )
    monkeypatch.setattr("mapping.load_mapping_prompt", lambda *a, **k: mapping_template)
    responses = [json.dumps({"description": "desc"})]
    for _ in range(3):
        responses.append(_feature_payload(1))
        responses.extend(
            [
                json.dumps({"data": [{"type": "d", "contribution": "c"}]}),
                json.dumps({"applications": [{"type": "a", "contribution": "c"}]}),
                json.dumps({"technology": [{"type": "t", "contribution": "c"}]}),
            ]
        )
    session = DummySession(responses)
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    generator._service = service  # type: ignore[attr-defined]

    results = generator.generate_plateau(cast(ConversationSession, session), 1)

    assert len(results) == 3


def test_generate_plateau_raises_on_insufficient_features(monkeypatch) -> None:
    template = (
        "{required_count} {service_name} {service_description} "
        "{plateau} {customer_type}"
    )
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
    template = (
        "{required_count} {service_name} {service_description} "
        "{plateau} {customer_type}"
    )
    monkeypatch.setattr(
        "plateau_generator.load_plateau_prompt", lambda *a, **k: template
    )
    """Invalid description payloads should raise ``ValueError``."""
    session = DummySession(["not json"])
    generator = PlateauGenerator(cast(ConversationSession, session), required_count=1)
    with pytest.raises(ValueError):
        generator._request_description(cast(ConversationSession, session), 1)
