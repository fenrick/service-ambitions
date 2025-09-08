import asyncio
from types import SimpleNamespace

import pytest

from core.conversation import ConversationSession
from engine.plateau_runtime import PlateauRuntime
from generation.plateau_generator import PlateauGenerator
from models import ServiceInput
from runtime.environment import RuntimeEnv


class _Result:
    def __init__(self, output: object) -> None:
        self.output = output

    def new_messages(self):  # pragma: no cover - simple stub
        return []

    def usage(self):  # pragma: no cover - simple stub
        return SimpleNamespace(total_tokens=1)


class _Agent:
    async def run(self, user_prompt: str):  # pragma: no cover - simple stub
        return _Result(output={"ok": True, "prompt": user_prompt})


class _FakeQueue:
    def __init__(self, delays: dict[str, float] | None = None):
        self.events: list[str] = []
        self.delays = delays or {}

    async def submit(self, factory, meta=None):  # noqa: ANN001
        stage = getattr(meta, "stage", None)
        if stage:
            self.events.append(stage)
        delay = self.delays.get(stage or "", 0.0)
        if delay:
            await asyncio.sleep(delay)
        return await factory()


@pytest.mark.asyncio()
async def test_conversation_routes_via_queue(monkeypatch):
    env = RuntimeEnv.instance()
    env.settings.llm_queue_enabled = True
    fakeq = _FakeQueue()
    # Inject the fake queue directly
    monkeypatch.setattr(env, "_llm_queue", fakeq, raising=False)

    session = ConversationSession(_Agent(), stage="features_1")
    out = await session.ask_async("hello")
    assert out == {"ok": True, "prompt": "hello"}
    assert fakeq.events == ["features_1"]


@pytest.mark.asyncio()
async def test_plateau_pipeline_overlaps_when_enabled(monkeypatch):
    env = RuntimeEnv.instance()
    env.settings.llm_queue_enabled = True
    # Delay features_2 so mapping for plateau 1 can be scheduled between them
    fakeq = _FakeQueue({"features_1": 0.05, "features_2": 0.2, "mapping": 0.1})
    monkeypatch.setattr(env, "_llm_queue", fakeq, raising=False)

    # Monkeypatch runtime methods to avoid heavy validation and focus on scheduling
    async def fake_generate_features(self, session, **kwargs):  # noqa: ANN001
        await session.ask_async("features")
        self.success = True

    async def fake_generate_mappings(self, session, **kwargs):  # noqa: ANN001
        await session.ask_async("mapping")
        self.success = True

    monkeypatch.setattr(PlateauRuntime, "generate_features", fake_generate_features)
    monkeypatch.setattr(PlateauRuntime, "generate_mappings", fake_generate_mappings)

    # Build generator with simple sessions
    session = ConversationSession(_Agent(), stage="features")
    mapping_session = ConversationSession(_Agent(), stage="mapping")
    generator = PlateauGenerator(session, roles=["role"], mapping_session=mapping_session, description_session=session)

    svc = ServiceInput(
        service_id="svc1",
        name="Service",
        description="d",
        jobs_to_be_done=["job"],
    )
    runtimes = [
        PlateauRuntime(plateau=1, plateau_name="P1", description="d1"),
        PlateauRuntime(plateau=2, plateau_name="P2", description="d2"),
    ]
    await generator._schedule_plateaus(runtimes, svc)
    # Expect that mapping is submitted before features_2 finishes (so appears
    # after features_1 and features_2 submits).
    assert fakeq.events[0:3] == ["features_1", "features_2", "mapping"]


@pytest.mark.asyncio()
async def test_llm_queue_limits_concurrency():
    from llm.queue import LLMQueue

    q = LLMQueue(max_concurrency=2)
    starts: list[int] = []

    async def worker(i: int):
        async def _run():
            starts.append(i)
            await asyncio.sleep(0.05)
            return i

        return await q.submit(_run)

    # Submit three tasks; only two should start before any finishes
    res = await asyncio.gather(worker(1), worker(2), worker(3))
    # The first two started before the third
    assert starts[:2] == [1, 2]
    assert sorted(res) == [1, 2, 3]


@pytest.mark.asyncio()
async def test_plateau_sequential_when_disabled(monkeypatch):
    # Ensure feature flag is off
    env = RuntimeEnv.instance()
    env.settings.llm_queue_enabled = False

    events: list[str] = []

    async def fake_generate_features(self, session, **kwargs):  # noqa: ANN001
        events.append(f"features_{self.plateau}")

    async def fake_generate_mappings(self, session, **kwargs):  # noqa: ANN001
        events.append(f"mapping_{self.plateau}")

    monkeypatch.setattr(PlateauRuntime, "generate_features", fake_generate_features)
    monkeypatch.setattr(PlateauRuntime, "generate_mappings", fake_generate_mappings)

    session = ConversationSession(_Agent(), stage="features")
    mapping_session = ConversationSession(_Agent(), stage="mapping")
    generator = PlateauGenerator(session, roles=["role"], mapping_session=mapping_session, description_session=session)

    svc = ServiceInput(
        service_id="svc2",
        name="Service",
        description="d",
        jobs_to_be_done=["job"],
    )
    runtimes = [
        PlateauRuntime(plateau=1, plateau_name="P1", description="d1"),
        PlateauRuntime(plateau=2, plateau_name="P2", description="d2"),
    ]
    await generator._schedule_plateaus(runtimes, svc)
    # With the flag off, execution is sequential per plateau
    assert events == ["features_1", "mapping_1", "features_2", "mapping_2"]


@pytest.mark.asyncio()
async def test_global_queue_caps_across_sessions(monkeypatch):
    # Enable the real queue with concurrency 2
    from llm.queue import LLMQueue

    env = RuntimeEnv.instance()
    env.settings.llm_queue_enabled = True
    env._llm_queue = LLMQueue(max_concurrency=2)  # type: ignore[attr-defined]

    current = 0
    max_seen = 0

    class _AgentCapped:
        async def run(self, user_prompt: str):  # pragma: no cover - simple stub
            nonlocal current, max_seen
            current += 1
            max_seen = max(max_seen, current)
            try:
                await asyncio.sleep(0.05)
                return _Result(output={"ok": True, "prompt": user_prompt})
            finally:
                current -= 1

    # Create multiple sessions and fire a bunch of calls concurrently
    s1 = ConversationSession(_AgentCapped(), stage="features_1")
    s2 = ConversationSession(_AgentCapped(), stage="features_2")
    s3 = ConversationSession(_AgentCapped(), stage="mapping")

    await asyncio.gather(
        s1.ask_async("a"),
        s2.ask_async("b"),
        s3.ask_async("c"),
        s1.ask_async("d"),
        s2.ask_async("e"),
    )

    assert max_seen <= 2
