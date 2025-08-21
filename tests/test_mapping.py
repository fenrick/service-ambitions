import json
from typing import Sequence, cast

import pytest

import mapping
from conversation import ConversationSession
from mapping import MappingError, map_set
from models import (
    Contribution,
    MappingFeature,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    MaturityScore,
    PlateauFeature,
)


class DummySession:
    """Session returning queued responses for ``ask_async``."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def ask_async(self, prompt: str, output_type=None) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def _feature(feature_id: str = "f1") -> PlateauFeature:
    return PlateauFeature(
        feature_id=feature_id,
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


def test_merge_mapping_results_aggregates_unknown_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    features = [_feature("f1"), _feature("f2")]
    payload = MappingResponse(
        features=[
            MappingFeature(
                feature_id="f1",
                applications=[
                    Contribution(item="a"),
                    Contribution(item="x"),
                ],
                technologies=[
                    Contribution(item="t1"),
                    Contribution(item="y1"),
                ],
            ),
            MappingFeature(
                feature_id="f2",
                applications=[Contribution(item="x2")],
                technologies=[
                    Contribution(item="t1"),
                    Contribution(item="y2"),
                ],
            ),
        ]
    )
    mapping_types = {
        "applications": MappingTypeConfig(dataset="applications", label="Applications"),
        "technologies": MappingTypeConfig(dataset="technologies", label="Technologies"),
    }
    catalogue = {
        "applications": [MappingItem(id="a", name="A", description="d")],
        "technologies": [MappingItem(id="t1", name="T1", description="d")],
    }

    warnings: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        mapping.logfire, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )

    merged = mapping._merge_mapping_results(
        features, payload, mapping_types, catalogue_items=catalogue
    )

    assert [c.item for c in merged[0].mappings["applications"]] == ["a"]
    assert merged[1].mappings["applications"] == []
    assert [c.item for c in merged[0].mappings["technologies"]] == ["t1"]
    assert [c.item for c in merged[1].mappings["technologies"]] == ["t1"]

    qfile = tmp_path / "quarantine" / "mappings" / "unknown_ids.json"
    data = json.loads(qfile.read_text())
    assert set(data["applications"]) == {"x", "x2"}
    assert set(data["technologies"]) == {"y1", "y2"}

    aggregated = [w for w in warnings if "count" in w[1]]
    counts = {kw["key"]: kw["count"] for _, kw in aggregated}
    assert counts == {"applications": 2, "technologies": 2}
