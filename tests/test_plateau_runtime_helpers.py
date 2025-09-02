from typing import cast

import pytest

from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from models import FeatureItem, MaturityScore, PlateauFeaturesResponse


def _dummy_feature(name: str) -> FeatureItem:
    return FeatureItem(
        name=name,
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
    )


class DummySession:
    def __init__(self) -> None:
        self.seen: str | None = None

    async def ask_async(self, prompt: str) -> PlateauFeaturesResponse:
        self.seen = prompt
        return PlateauFeaturesResponse(features={})


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
async def test_dispatch_feature_prompt_builds_prompt(monkeypatch) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")
    calls: dict[str, object] = {}

    def fake_build(*, service_name, description, roles, required_count):
        calls.update(
            {
                "service_name": service_name,
                "description": description,
                "roles": roles,
                "required_count": required_count,
            }
        )
        return "PROMPT"

    monkeypatch.setattr(runtime, "_build_plateau_prompt", fake_build)

    session = DummySession()
    payload = await runtime._dispatch_feature_prompt(
        cast(ConversationSession, session),
        service_id="svc",
        service_name="svc",
        roles=["r"],
        required_count=1,
    )

    assert calls["service_name"] == "svc"
    assert calls["roles"] == ["r"]
    assert calls["required_count"] == 1
    assert session.seen == "PROMPT"
    assert isinstance(payload, PlateauFeaturesResponse)


@pytest.mark.asyncio
async def test_recover_feature_shortfalls(monkeypatch) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")
    valid = {"r1": [_dummy_feature("f1")], "r2": []}
    invalid = ["r2"]
    missing = {"r1": 1, "r2": 1}

    async def fake_recover_invalid_roles(
        self, invalid_roles, *, level, description, session, required_count
    ):
        return {"r2": [_dummy_feature("f2")]} if "r2" in invalid_roles else {}

    async def fake_request_missing_features_async(
        self, level, role, description, missing, session
    ):
        return [_dummy_feature(f"extra_{role}")]

    monkeypatch.setattr(
        PlateauRuntime, "_recover_invalid_roles", fake_recover_invalid_roles
    )
    monkeypatch.setattr(
        PlateauRuntime,
        "_request_missing_features_async",
        fake_request_missing_features_async,
    )

    result = await runtime._recover_feature_shortfalls(
        valid,
        invalid,
        missing,
        level=1,
        description="d",
        session=cast(ConversationSession, DummySession()),
        required_count=2,
        roles=["r1", "r2"],
    )

    assert len(result["r1"]) == 2
    assert len(result["r2"]) == 2


@pytest.mark.asyncio
async def test_dispatch_and_cache_features(monkeypatch, tmp_path) -> None:
    runtime = PlateauRuntime(plateau=1, plateau_name="p1", description="d")

    payload = PlateauFeaturesResponse(features={"r": [_dummy_feature("f")]})

    async def fake_dispatch(*args, **kwargs):  # noqa: ANN001
        return payload

    def fake_validate(role_data, *, roles, required_count):  # noqa: ANN001
        return role_data, [], {}

    async def fake_recover(*args, **kwargs):  # noqa: ANN001
        return payload.features

    monkeypatch.setattr(runtime, "_dispatch_feature_prompt", fake_dispatch)
    monkeypatch.setattr(runtime, "_validate_roles", fake_validate)
    monkeypatch.setattr(runtime, "_recover_feature_shortfalls", fake_recover)

    cache_file = tmp_path / "features.json"

    result = await runtime._dispatch_and_cache_features(
        cast(ConversationSession, DummySession()),
        service_id="svc",
        service_name="svc",
        roles=["r"],
        required_count=1,
        cache_file=cache_file,
        use_local_cache=True,
        cache_mode="write",
    )

    assert result == payload
    assert cache_file.exists()
