"""Parity tests verifying queue does not alter outputs."""

from types import SimpleNamespace

import pytest

from core.conversation import ConversationSession
from llm.queue import LLMQueue
from runtime.environment import RuntimeEnv


class _Agent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, user_prompt: str, message_history=None):  # noqa: ANN001
        self.calls += 1
        return SimpleNamespace(
            output={"prompt": user_prompt},
            new_messages=lambda: [],
            usage=lambda: SimpleNamespace(total_tokens=1),
        )


async def _run_with_queue(enabled: bool) -> str:
    env = RuntimeEnv.instance()
    env.settings.llm_queue_enabled = enabled
    env._llm_queue = LLMQueue(max_concurrency=1) if enabled else None  # type: ignore[attr-defined]
    session = ConversationSession(_Agent(), stage="descriptions")
    return await session.ask_async("hello")


@pytest.mark.asyncio()
async def test_results_equal_with_and_without_queue():
    without_q = await _run_with_queue(False)
    with_q = await _run_with_queue(True)
    assert without_q == with_q
