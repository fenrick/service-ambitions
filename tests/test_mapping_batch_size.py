import pytest

from models import MaturityScore, PlateauFeature, PlateauResult, ServiceInput
from plateau_generator import (
    DEFAULT_PLATEAU_NAMES,
    PlateauGenerator,
)


class DummySession:
    """Minimal session stub for testing."""

    def __init__(self, client=None, stage=None):
        self.client = client
        self.stage = stage

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover - no-op
        pass

    def derive(self):  # pragma: no cover - no-op
        return self


@pytest.mark.asyncio
async def test_long_description_reduces_batch_size(monkeypatch) -> None:
    """A long description should trigger a reduced mapping batch size."""

    # Predict tokens as 1 per line to make calculations predictable.
    monkeypatch.setattr(
        "plateau_generator.estimate_tokens",
        lambda prompt, expected_output: prompt.count("\n") + 1,
    )

    captured: dict[str, object] = {}

    async def dummy_map_features(
        session, features, mapping_types=None, *, batch_size, parallel_types=True
    ):
        captured["batch_size"] = batch_size
        captured["order"] = [f.feature_id for f in features]
        return list(features)

    monkeypatch.setattr("plateau_generator.map_features_async", dummy_map_features)

    session = DummySession()
    gen = PlateauGenerator(session, mapping_batch_size=5, mapping_token_cap=3)
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
    await gen._map_features(session, "desc", features)

    assert captured["batch_size"] == 2
    assert captured["order"] == [f"F{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_scheduler_orders_by_predicted_tokens(monkeypatch) -> None:
    """TokenScheduler should process plateau tasks in ascending token order."""

    monkeypatch.setattr(
        "plateau_generator.estimate_tokens", lambda text, expected_output: len(text)
    )

    class SyncScheduler:
        def __init__(self, max_workers: int = 4) -> None:
            self._queue = []

        def submit(self, func, tokens: int) -> None:
            self._queue.append((tokens, func))

        async def run(self):
            self._queue.sort(key=lambda item: item[0])
            results = []
            for _, func in self._queue:
                results.append(await func())
            return results

    monkeypatch.setattr("plateau_generator.TokenScheduler", SyncScheduler)
    monkeypatch.setattr("plateau_generator.ConversationSession", DummySession)

    order: list[str] = []

    async def dummy_generate(self, level, plateau_name, session=None, description=""):
        order.append(plateau_name)
        return PlateauResult(
            plateau=level,
            plateau_name=plateau_name,
            service_description=description,
            features=[],
        )

    monkeypatch.setattr(PlateauGenerator, "generate_plateau_async", dummy_generate)

    session = DummySession()
    gen = PlateauGenerator(session)
    service_input = ServiceInput(
        service_id="S1",
        name="svc",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )
    gen._prepare_sessions(service_input)

    names = DEFAULT_PLATEAU_NAMES[:2]
    desc_map = {names[0]: "short", names[1]: "x" * 1000}
    await gen._schedule_plateaus(names, desc_map, service_input)

    assert order == names
