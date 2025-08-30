# SPDX-License-Identifier: MIT
import json
from pathlib import Path
from typing import Any, Sequence, cast

import pytest

import mapping
from conversation import ConversationSession
from mapping import MappingError, cache_write_json_atomic, map_set
from models import (
    Contribution,
    FeatureMappingRef,
    MappingFeature,
    MappingFeatureGroup,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    MaturityScore,
    PlateauFeature,
)


class DummySession:
    """Session returning queued responses for ``ask_async``."""

    def __init__(
        self,
        responses: Sequence[Any],
        *,
        diagnostics: bool = False,
        log_prompts: bool = False,
    ) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.diagnostics = diagnostics
        self.log_prompts = log_prompts
        self.last_tokens = 0

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


def test_cache_write_json_atomic_rejects_invalid_json(tmp_path) -> None:
    """Invalid JSON content is not persisted to disk."""

    path = tmp_path / "file.json"
    with pytest.raises(json.JSONDecodeError):
        cache_write_json_atomic(path, "{bad")
    assert not path.exists()


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
        service_name="svc",
        service_description="desc",
        plateau=1,
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
        service_name="svc",
        service_description="desc",
        plateau=1,
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
        service_name="svc1",
        service_description="desc",
        plateau=1,
        service="svc1",
    )
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc2",
        service_description="desc",
        plateau=1,
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
            service_name="svc",
            service_description="desc",
            plateau=1,
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
            service_name="svc",
            service_description="desc",
            plateau=1,
            service="svc",
            strict=True,
        )
    qfile = tmp_path / "quarantine" / "svc" / "applications" / "unknown_ids_1.json"
    assert json.loads(qfile.read_text()) == ["x"]
    assert paths == [qfile.resolve()]


def test_merge_mapping_missing_feature(monkeypatch, tmp_path) -> None:
    """Missing features result in empty mappings and a warning."""

    monkeypatch.chdir(tmp_path)
    features = [_feature("f1"), _feature("f2")]
    payload = MappingResponse(
        features=[
            MappingFeature(
                feature_id="f1", mappings={"applications": [Contribution(item="a")]}
            )
        ]
    )
    mapping_types = {
        "applications": MappingTypeConfig(dataset="applications", label="Applications")
    }
    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mapping.logfire, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )
    merged, unknown = mapping._merge_mapping_results(
        features,
        payload,
        mapping_types,
        catalogue_items={"applications": [_item()]},
        service="svc",
    )
    assert unknown == 0
    assert merged[1].mappings["applications"] == []
    assert (
        "missing mapping",
        {"feature_id": "f2"},
    ) in warnings


def test_merge_mapping_missing_feature_strict(monkeypatch, tmp_path) -> None:
    """Strict mode raises when features are missing from the response."""

    monkeypatch.chdir(tmp_path)
    features = [_feature("f1"), _feature("f2")]
    payload = MappingResponse(
        features=[
            MappingFeature(
                feature_id="f1", mappings={"applications": [Contribution(item="a")]}
            )
        ]
    )
    mapping_types = {
        "applications": MappingTypeConfig(dataset="applications", label="Applications")
    }
    with pytest.raises(MappingError):
        mapping._merge_mapping_results(
            features,
            payload,
            mapping_types,
            catalogue_items={"applications": [_item()]},
            service="svc",
            strict=True,
        )


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
        service_name="svc",
        service_description="desc",
        plateau=1,
        diagnostics=True,
    )
    assert mapped[0].mappings["applications"][0].item == "a"


@pytest.mark.asyncio()
async def test_map_set_writes_cache(monkeypatch, tmp_path) -> None:
    """Cache miss writes compact JSON to the filesystem."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(mapping, "_build_cache_key", lambda *a, **k: "key")
    response = MappingResponse.model_validate(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    session = DummySession([response])
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode="read",
    )
    cache_file = (
        Path(".cache")
        / "unknown"
        / "unknown"
        / "mappings"
        / "f1"
        / "applications"
        / "key.json"
    )
    assert cache_file.exists()
    content = cache_file.read_text()
    assert json.loads(content) == response.model_dump()
    assert ": " not in content and ", " not in content


@pytest.mark.asyncio()
async def test_map_set_reads_cache(monkeypatch, tmp_path) -> None:
    """Cache hit bypasses the network call."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(mapping, "_build_cache_key", lambda *a, **k: "key")
    cached = {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    cache_dir = (
        Path(".cache") / "unknown" / "unknown" / "mappings" / "f1" / "applications"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "key.json").open("w", encoding="utf-8") as fh:
        json.dump(cached, fh)

    class NoCallSession(DummySession):
        async def ask_async(self, prompt: str, output_type=None) -> str:
            raise AssertionError("ask_async should not be called")

    session = NoCallSession([])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode="read",
    )
    assert session.prompts == []
    assert mapped[0].mappings["applications"][0].item == "a"


@pytest.mark.asyncio()
async def test_map_set_bad_cache_renamed(monkeypatch, tmp_path) -> None:
    """Corrupt cache files are renamed and the request retried."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(mapping, "_build_cache_key", lambda *a, **k: "key")
    cache_dir = (
        Path(".cache") / "unknown" / "unknown" / "mappings" / "f1" / "applications"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    bad_file = cache_dir / "key.json"
    bad_file.write_text("not json", encoding="utf-8")
    valid = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    session = DummySession([valid])
    mapped = await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode="read",
    )
    assert mapped[0].mappings["applications"][0].item == "a"
    assert bad_file.exists()
    assert (cache_dir / "key.bad.json").exists()
    assert session.prompts == ["PROMPT"]


@pytest.mark.asyncio()
@pytest.mark.parametrize("change", ["catalogue", "template", "feature"])
async def test_map_set_cache_invalidation(monkeypatch, tmp_path, change) -> None:
    """Cache key changes when inputs differ."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    if change == "template":  # Template text changes between calls
        versions = iter(["v1", "v2"])
        monkeypatch.setattr(mapping, "load_prompt_text", lambda _: next(versions))
    else:  # Template remains constant
        monkeypatch.setattr(mapping, "load_prompt_text", lambda _: "v1")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    session = DummySession([response, response])
    features1 = [_feature()]
    features2 = features1
    cat_hash1 = "h1"
    cat_hash2 = "h1"
    if change == "catalogue":  # Different catalogue content
        cat_hash2 = "h2"
    elif change == "feature":  # Feature description changed
        feat = features1[0].model_copy(update={"description": "d2"})
        features2 = [feat]
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        features1,
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode="read",
        catalogue_hash=cat_hash1,
    )
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        features2,
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode="read",
        catalogue_hash=cat_hash2,
    )
    cache_dir1 = (
        Path(".cache") / "unknown" / "unknown" / "mappings" / "f1" / "applications"
    )
    cache_dir2 = (
        Path(".cache") / "unknown" / "unknown" / "mappings" / "f2" / "applications"
    )
    assert (cache_dir1 / "key.json").exists()
    assert (cache_dir2 / "key.json").exists()
    assert len(session.prompts) == 2


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "mode,prepopulate,expected",
    [
        ("read", False, "miss"),
        ("read", True, "hit"),
        ("refresh", False, "refresh"),
    ],
)
async def test_map_set_logs_cache_status(
    monkeypatch, tmp_path, mode, prepopulate, expected
) -> None:
    """Cache operations emit a single status log line."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(mapping, "_build_cache_key", lambda *a, **k: "key")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    if prepopulate:
        cache_dir = (
            Path(".cache") / "unknown" / "unknown" / "mappings" / "f1" / "applications"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / "key.json").open("w", encoding="utf-8") as fh:
            json.dump(json.loads(response), fh)
    session = DummySession([response])
    logs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mapping.logfire, "info", lambda msg, **kw: logs.append((msg, kw))
    )
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode=mode,
    )
    assert logs[0][1]["cache"] == expected
    assert logs[0][1]["cache_key"] == "key"
    assert logs[0][1]["features"] == 1


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "mode,prepopulate,expected_prompts,expected_content",
    [
        ("off", False, 1, None),
        ("read", False, 1, "response"),
        ("refresh", True, 1, "response"),
        ("write", True, 1, "cached"),
    ],
)
async def test_map_set_cache_modes(
    monkeypatch, tmp_path, mode, prepopulate, expected_prompts, expected_content
) -> None:
    """Different cache modes control read/write behaviour."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mapping.render_set_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(mapping, "_build_cache_key", lambda *a, **k: "key")
    response = json.dumps(
        {"features": [{"feature_id": "f1", "applications": [{"item": "a"}]}]}
    )
    cache_file = (
        Path(".cache")
        / "unknown"
        / "unknown"
        / "mappings"
        / "f1"
        / "applications"
        / "key.json"
    )
    if prepopulate:  # Seed cache file when required
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("cached")
    session = DummySession([response])
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
        cache_mode=mode,
    )
    assert len(session.prompts) == expected_prompts
    if expected_content is None:  # off mode does not write cache
        assert not cache_file.exists()
    else:  # remaining modes leave an on-disk file
        assert cache_file.read_text() == (
            response if expected_content == "response" else "cached"
        )


@pytest.mark.asyncio()
async def test_map_set_prompt_logging_respects_flags(monkeypatch) -> None:
    """Prompt logging only occurs when permitted and excludes catalogue items."""

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
    logs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mapping.logfire, "debug", lambda msg, **kw: logs.append((msg, kw))
    )
    session = DummySession([response], diagnostics=True, log_prompts=True)
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
    )
    assert logs
    features_json = logs[0][1]["features"]
    assert "Feat" in features_json  # feature name present
    assert "A" not in features_json  # catalogue item excluded


@pytest.mark.asyncio()
async def test_map_set_prompt_logging_skipped(monkeypatch) -> None:
    """Prompt logging is skipped when not explicitly allowed."""

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
    logs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mapping.logfire, "debug", lambda msg, **kw: logs.append((msg, kw))
    )
    session = DummySession([response], diagnostics=True, log_prompts=False)
    await map_set(
        cast(ConversationSession, session),
        "applications",
        [_item()],
        [_feature()],
        service_name="svc",
        service_description="desc",
        plateau=1,
    )
    assert logs == []


def test_group_features_by_mapping() -> None:
    """Features are grouped under mapping items."""

    feat1 = _feature("f1")
    feat2 = _feature("f2")
    feat1.mappings["applications"] = [Contribution(item="a")]
    feat2.mappings["applications"] = [Contribution(item="a")]
    feat3 = _feature("f3")  # Unmapped feature

    groups = mapping.group_features_by_mapping(
        [feat1, feat2, feat3], "applications", [_item()]
    )
    assert groups == [
        MappingFeatureGroup(
            id="a",
            name="A",
            mappings=[
                FeatureMappingRef(feature_id="f1", description="d"),
                FeatureMappingRef(feature_id="f2", description="d"),
            ],
        )
    ]
