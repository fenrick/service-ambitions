import json
from typing import Sequence, cast

import pytest

from conversation import ConversationSession
from mapping import MappingError, map_set
from models import MappingItem, MaturityScore, PlateauFeature


class DummySession:
    """Session returning queued responses for ``ask_async``."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def ask_async(self, prompt: str, output_type=None) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def _feature() -> PlateauFeature:
    return PlateauFeature(
        feature_id="f1",
        name="Feat",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="learners",
    )


def _item() -> MappingItem:
    return MappingItem(id="a", name="A", description="d")


@pytest.mark.asyncio()
async def test_map_set_retries_on_failure(monkeypatch) -> None:
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    valid = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    session = DummySession(["bad", valid])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service="svc",
    )
    assert session.prompts == ["PROMPT", "PROMPT\nReturn valid JSON only."]
    assert mapped[0].mappings["applications"][0].item == "a"


@pytest.mark.asyncio()
async def test_map_set_quarantines_on_double_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    session = DummySession(["bad", "still bad"])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service="svc",
    )
    assert mapped[0].mappings["applications"] == []
    qfile = tmp_path / "quarantine" / "mappings" / "svc" / "applications.txt"
    assert qfile.read_text() == "still bad"


@pytest.mark.asyncio()
async def test_map_set_strict_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    session = DummySession(["bad", "still bad"])
    with pytest.raises(MappingError):
        await map_set(
            cast(ConversationSession, session),
            "applications",
            [_item()],
            [_feature()],
            service="svc",
            strict=True,
        )
    qfile = tmp_path / "quarantine" / "mappings" / "svc" / "applications.txt"
    assert qfile.exists()
