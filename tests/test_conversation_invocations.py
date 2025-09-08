# SPDX-License-Identifier: MIT
"""Tests for ConversationSession invocation counters and one-shot behaviour."""

from __future__ import annotations

from types import SimpleNamespace

from core.conversation import ConversationSession
from models import ServiceInput


class _DummyResult:
    def __init__(self) -> None:
        self.output = "ok"

    def usage(self):  # noqa: D401 - minimal stub
        return SimpleNamespace(total_tokens=1)

    def new_messages(self):  # noqa: D401 - minimal stub
        return []


class _DummyAgent:
    def __init__(self) -> None:
        self.model = SimpleNamespace(model_name="dummy")
        self.output_type = None
        self.calls = 0

    async def run(self, user_prompt: str, *, message_history):  # type: ignore[no-untyped-def]
        self.calls += 1
        return _DummyResult()

    def run_sync(self, user_prompt: str, *, message_history):  # type: ignore[no-untyped-def]
        self.calls += 1
        return _DummyResult()


def test_invocations_increment_on_call_but_not_on_cache(tmp_path) -> None:
    agent = _DummyAgent()
    session = ConversationSession(
        agent,
        stage="features_1",
        use_local_cache=True,
        cache_mode="read",
    )
    svc = ServiceInput(
        service_id="svc-1",
        name="alpha",
        description="d",
        jobs_to_be_done=[],
    )
    session.add_parent_materials(svc)

    # First call writes cache
    out1 = session.ask("PROMPT")
    # Second call should hit cache and not increment invocations
    out2 = session.ask("PROMPT")

    assert out1 == "ok" and out2 == "ok"
    assert agent.calls == 1
    assert session.invocations_by_stage.get("features_1", 0) == 1
