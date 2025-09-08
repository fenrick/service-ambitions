"""Tests enforcing prompt call contracts and no-parallel policies."""

from typing import cast

import pytest
from pydantic import ValidationError

from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from generation.plateau_generator import PlateauGenerator
from models import (
    FeatureItem,
    MaturityScore,
    PlateauFeaturesResponse,
    ServiceInput,
    ServiceMeta,
)
from utils import ShortCodeRegistry


class CountingSession:
    def __init__(
        self,
        payload: PlateauFeaturesResponse | None = None,
        *,
        raise_invalid: bool = False,
    ) -> None:
        self.calls = 0
        self.payload = payload
        self.raise_invalid = raise_invalid

    async def ask_async(self, prompt: str):  # noqa: ANN001
        self.calls += 1
        if self.raise_invalid:
            return 123  # invalid type for model parsing
        assert self.payload is not None
        return self.payload


@pytest.mark.asyncio()
async def test_single_ask_async_call_per_plateau(monkeypatch) -> None:
    """Feature generation should prompt exactly once per plateau regardless of roles."""
    roles = ["learners", "academics", "professional_staff"]
    item = FeatureItem(
        name="F",
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
    )
    payload = PlateauFeaturesResponse(features={r: [item] for r in roles})
    session = CountingSession(payload)

    runtime = PlateauRuntime(plateau=1, plateau_name="Foundational", description="desc")
    await runtime.generate_features(
        cast(ConversationSession, session),
        service_id="svc",
        service_name="Service",
        roles=roles,
        code_registry=ShortCodeRegistry(),
        use_local_cache=False,
        cache_mode="off",
    )

    assert session.calls == 1


@pytest.mark.asyncio()
async def test_invalid_response_raises_without_retry(monkeypatch) -> None:
    """Invalid model responses should raise immediately without retries."""
    session = CountingSession(None, raise_invalid=True)
    runtime = PlateauRuntime(plateau=1, plateau_name="Foundational", description="desc")

    with pytest.raises(ValidationError):
        await runtime.generate_features(
            cast(ConversationSession, session),
            service_id="svc",
            service_name="Service",
            roles=["learners"],
            code_registry=ShortCodeRegistry(),
            use_local_cache=False,
            cache_mode="off",
        )

    assert session.calls == 1


def test_no_parallel_model_calls(monkeypatch) -> None:
    """Generator should not use asyncio.gather/create_task around model calls."""
    # Fail fast if any parallel helpers are invoked
    monkeypatch.setattr(
        "asyncio.gather",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("gather not allowed")),
    )
    monkeypatch.setattr(
        "asyncio.create_task",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("create_task not allowed")
        ),
    )

    # Simple session that records prompts
    class DummySession:
        def __init__(self, responses):
            self._responses = list(responses)
            self.stage = "test"
            self.client = None

        def ask(self, prompt: str):  # noqa: ANN001
            return self._responses.pop(0)

        async def ask_async(self, prompt: str):  # noqa: ANN001
            return self.ask(prompt)

        def add_parent_materials(
            self, service_input: ServiceInput
        ) -> None:  # noqa: D401
            return None

        def derive(self):  # noqa: D401
            return self

    # Prepare minimal inputs for a two-plateau evolution
    from models import PlateauDescriptionsResponse

    desc_payload = PlateauDescriptionsResponse.model_validate(
        {
            "descriptions": [
                {"plateau": 1, "plateau_name": "Foundational", "description": "d1"},
                {"plateau": 2, "plateau_name": "Enhanced", "description": "d2"},
            ]
        }
    )

    # Features payload will be provided by PlateauRuntime. We bypass it by
    # stubbing generate_plateau_async to avoid model calls and focus on
    # scheduling logic.
    session = DummySession([desc_payload])
    gen = PlateauGenerator(
        cast(ConversationSession, session), use_local_cache=False, cache_mode="off"
    )
    service = ServiceInput(
        service_id="svc",
        name="Service",
        customer_type="retail",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
    )

    async def fake_generate_plateau_async(
        self, runtime, *, session=None
    ):  # noqa: ANN001
        runtime.set_results(features=[], mappings={})
        return runtime

    monkeypatch.setattr(
        PlateauGenerator, "generate_plateau_async", fake_generate_plateau_async
    )

    meta = ServiceMeta(run_id="r", models={}, web_search=False, mapping_types=[])
    evo = gen.generate_service_evolution(service, None, ["learners"], meta=meta)
    assert evo is not None
