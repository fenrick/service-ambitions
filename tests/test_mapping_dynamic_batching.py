import pytest

from mapping import map_features_async
from models import MappingTypeConfig, MaturityScore, PlateauFeature


class DummySession:
    """Minimal session used for mapping tests."""


@pytest.mark.asyncio
async def test_map_features_downsizes_batches(monkeypatch) -> None:
    """Mapping should split batches when the prompt exceeds ``token_cap``."""

    # Prompt length scales with number of features to trigger downsizing.
    monkeypatch.setattr(
        "mapping._build_mapping_prompt",
        lambda feats, *_, **__: "x" * (len(feats) * 10),
    )
    monkeypatch.setattr(
        "mapping.estimate_tokens", lambda prompt, expected_output: len(prompt)
    )

    captured: list[list[str]] = []

    async def fake_map_parallel(session, batches, mapping_types, **kwargs):
        captured.extend([[f.feature_id for f in batch] for batch in batches])
        return {f.feature_id: f for batch in batches for f in batch}

    monkeypatch.setattr("mapping._map_parallel", fake_map_parallel)
    logs: list[str] = []
    monkeypatch.setattr(
        "mapping_utils.logfire.info",
        lambda msg, *a, **k: logs.append(msg % a if a else msg),
    )

    features = [
        PlateauFeature(
            feature_id=f"F{i}",
            name="N",
            description="D",
            score=MaturityScore(level=1, label="L", justification="J"),
            customer_type="learners",
        )
        for i in range(5)
    ]
    mapping_types = {"apps": MappingTypeConfig(dataset="applications", label="Apps")}
    await map_features_async(
        DummySession(),
        features,
        mapping_types=mapping_types,
        batch_size=5,
        token_cap=15,
    )

    assert captured == [[f"F{i}"] for i in range(5)]
    assert any("Reduced mapping batch size" in msg for msg in logs)
