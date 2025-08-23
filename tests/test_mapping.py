import json
from pathlib import Path
from typing import Any, Sequence, cast

import pytest

import mapping
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
async def test_map_set_successful_mapping(monkeypatch) -> None:
    """Agent response is retried once then merged into features."""

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
async def test_map_set_quarantines_unknown_ids(monkeypatch, tmp_path) -> None:
    """Unknown IDs are dropped and quarantined for later review."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    # Response contains one valid and one unknown ID.
    response = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "applications": [{"item": "a"}, {"item": "x"}],
                }
            ]
        }
    )
    session = DummySession([response])
    paths: list[Path] = []
    monkeypatch.setattr(
        "telemetry.record_quarantine", lambda p: paths.append(p.resolve())
    )
    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mapping.logfire, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service="svc",
    )
    assert [c.item for c in mapped[0].mappings["applications"]] == ["a"]
    qdir = tmp_path / "quarantine" / "svc" / "applications"
    qfile = qdir / "unknown_ids_1.json"
    assert json.loads(qfile.read_text()) == ["x"]
    manifest = json.loads((qdir / "manifest.json").read_text())
    assert manifest["unknown_ids"]["count"] == 1
    assert manifest["unknown_ids"]["examples"] == [["x"]]
    assert paths == [qfile.resolve()]
    dropped = [kw for msg, kw in warnings if msg == "Dropped unknown mapping IDs"]
    assert dropped[0]["examples"] == ["x"]


@pytest.mark.asyncio()
async def test_quarantine_separates_unknown_ids_by_service(
    monkeypatch, tmp_path
) -> None:
    """Unknown IDs are quarantined per service ID."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "x"}]}]}
    )
    session = DummySession([response, response])
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service="svc1",
    )
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service="svc2",
    )
    qfile1 = tmp_path / "quarantine" / "svc1" / "applications" / "unknown_ids_1.json"
    qfile2 = tmp_path / "quarantine" / "svc2" / "applications" / "unknown_ids_1.json"
    assert json.loads(qfile1.read_text()) == ["x"]
    assert json.loads(qfile2.read_text()) == ["x"]


@pytest.mark.asyncio()
async def test_map_set_strict_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    session = DummySession(["bad", "still bad"])
    paths: list[Path] = []
    monkeypatch.setattr(
        "telemetry.record_quarantine", lambda p: paths.append(p.resolve())
    )
    with pytest.raises(MappingError):
        await map_set(
            cast(ConversationSession, session),
            "applications",
            [_item()],
            [_feature()],
            service="svc",
            strict=True,
        )
    qfile = tmp_path / "quarantine" / "svc" / "applications" / "json_parse_error_1.json"
    assert qfile.exists()
    assert paths == [qfile.resolve()]


@pytest.mark.asyncio()
async def test_map_set_strict_unknown_ids(monkeypatch, tmp_path) -> None:
    """Strict mode raises when the agent invents mapping identifiers."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "x"}]}]}
    )
    session = DummySession([response])
    paths: list[Path] = []
    monkeypatch.setattr(
        "telemetry.record_quarantine", lambda p: paths.append(p.resolve())
    )
    with pytest.raises(MappingError):
        await map_set(
            cast(ConversationSession, session),
            "applications",
            [_item()],
            [_feature()],
            service="svc",
            strict=True,
        )
    qfile = tmp_path / "quarantine" / "svc" / "applications" / "unknown_ids_1.json"
    assert json.loads(qfile.read_text()) == ["x"]
    assert paths == [qfile.resolve()]


@pytest.mark.asyncio()
async def test_map_set_diagnostics_includes_rationale(monkeypatch) -> None:
    """Diagnostics responses with rationales are accepted."""

    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    response = json.dumps(
        {
            "features": [
                {
                    "feature_id": "f1",
                    "applications": [{"item": "a", "rationale": "match"}],
                }
            ]
        }
    )
    session = DummySession([response])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        diagnostics=True,
    )
    assert mapped[0].mappings["applications"][0].item == "a"


@pytest.mark.asyncio()
async def test_map_set_writes_cache(monkeypatch, tmp_path) -> None:
    """Cache miss writes the model response to the filesystem."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    session = DummySession([response])
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        use_local_cache=True,
    )
    cache_file = Path(".cache") / "mapping" / "applications" / f"{hash('PROMPT')}.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text()) == json.loads(response)


@pytest.mark.asyncio()
async def test_map_set_reads_cache(monkeypatch, tmp_path) -> None:
    """Cache hit bypasses the network call."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    cached = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    cache_dir = Path(".cache") / "mapping" / "applications"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{hash('PROMPT')}.json").write_text(cached)
    session = DummySession([cached])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        use_local_cache=True,
    )
    assert session.prompts == []
    assert mapped[0].mappings["applications"][0].item == "a"
