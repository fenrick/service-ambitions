from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from models import (
    FeatureItem,
    MappingFeatureGroup,
    MappingSet,
    MaturityScore,
    PlateauFeature,
    PlateauFeaturesResponse,
)
from runtime.environment import RuntimeEnv


def _dummy_feature(name: str) -> FeatureItem:
    return FeatureItem(
        name=name,
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
    )


def _dummy_plateau_feature(name: str) -> PlateauFeature:
    return PlateauFeature(
        feature_id=name,
        name=name,
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="r",
    )


class DummySession:
    def __init__(self) -> None:
        self.seen: str | None = None

    async def ask_async(self, prompt: str) -> PlateauFeaturesResponse:
        self.seen = prompt
        return PlateauFeaturesResponse(features={})


class MapDummySession:
    def __init__(self) -> None:
        self.stage = ""

    def derive(self) -> "MapDummySession":
        clone = MapDummySession()
        clone.stage = self.stage
        return clone


def test_load_cached_payload_reads_valid_file(tmp_path, monkeypatch) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="desc")
    payload = PlateauFeaturesResponse(features={"r": [_dummy_feature("f")]})
    cache_file = tmp_path / "features.json"
    cache_file.write_text(payload.model_dump_json())

    monkeypatch.setattr(
        runtime, "_discover_feature_cache", lambda service: (cache_file, cache_file)
    )

    result, path = runtime._load_cached_payload("svc", True, "read")

    assert result == payload
    assert path == cache_file


@pytest.mark.asyncio
async def test_dispatch_features_caches_payload(monkeypatch, tmp_path) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")

    payload = PlateauFeaturesResponse(features={"r": [_dummy_feature("f")]})
    monkeypatch.setattr(runtime, "_build_plateau_prompt", lambda **_: "PROMPT")
    session = DummySession()

    async def fake_ask_async(prompt: str) -> PlateauFeaturesResponse:
        session.seen = prompt
        return payload

    monkeypatch.setattr(session, "ask_async", fake_ask_async)

    cache_file = tmp_path / "features.json"

    result = await runtime._dispatch_features(
        cast(ConversationSession, session),
        service_id="svc",
        service_name="svc",
        roles=["r"],
        cache_file=cache_file,
        use_local_cache=True,
        cache_mode="write",
    )

    assert result == payload
    assert cache_file.exists()
    assert session.seen == "PROMPT"


@pytest.mark.asyncio
async def test_run_mapping_set_invokes_map_and_groups(monkeypatch) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")
    runtime.features = [_dummy_plateau_feature("f1")]
    cfg = MappingSet(name="Apps", file="apps.json", field="applications")
    items: dict[str, list[Any]] = {cfg.field: []}
    groups = [MappingFeatureGroup(id="x", name="X", mappings=[])]
    called: dict[str, object] = {}

    async def fake_map_set(
        session, name, items_list, feats, params, **kwargs
    ):  # noqa: ANN001
        called["set_name"] = name
        called["stage"] = session.stage
        return list(feats)

    def fake_group(features, mapping_type, catalogue):  # noqa: ANN001
        called["mapping_type"] = mapping_type
        called["catalogue"] = catalogue
        return groups

    monkeypatch.setattr("engine.plateau_runtime.map_set", fake_map_set)
    monkeypatch.setattr("engine.plateau_runtime.group_features_by_mapping", fake_group)

    session = MapDummySession()
    result = await runtime._run_mapping_set(
        cast(ConversationSession, session),
        cfg,
        items=items,
        service_name="svc",
        service_id="svc",
        service_description="desc",
        strict=False,
        use_local_cache=True,
        cache_mode="read",
        catalogue_hash="hash",
    )

    assert result == groups
    assert called["set_name"] == cfg.field
    assert called["mapping_type"] == cfg.field
    assert called["stage"] == f"mapping_{cfg.field}"


@pytest.mark.asyncio
async def test_generate_mappings_collects_from_helper(monkeypatch) -> None:
    mapping_sets = [
        MappingSet(name="Apps", file="apps.json", field="applications"),
        MappingSet(name="Tech", file="tech.json", field="technologies"),
    ]
    RuntimeEnv.reset()
    RuntimeEnv.initialize(
        cast(
            Any,
            SimpleNamespace(
                mapping_sets=mapping_sets,
                mapping_data_dir=Path("data"),
                prompt_dir=Path("prompts"),
            ),
        )
    )
    monkeypatch.setattr(
        "engine.plateau_runtime.load_mapping_items",
        lambda sets: ({s.field: [] for s in sets}, "hash"),
    )
    calls: list[str] = []

    async def fake_run(self, session, cfg, **kwargs):  # noqa: ANN001
        calls.append(cfg.field)
        return [MappingFeatureGroup(id=cfg.field, name=cfg.field, mappings=[])]

    monkeypatch.setattr(PlateauRuntime, "_run_mapping_set", fake_run)

    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")
    runtime.features = [_dummy_plateau_feature("f1")]

    session = MapDummySession()
    await runtime.generate_mappings(
        cast(ConversationSession, session),
        service_name="svc",
        service_id="svc",
        service_description="desc",
        strict=False,
        use_local_cache=False,
        cache_mode="off",
    )

    assert calls == [s.field for s in mapping_sets]
    assert set(runtime.mappings) == {s.field for s in mapping_sets}
    assert runtime.success is True
