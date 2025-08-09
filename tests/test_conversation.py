"""Unit tests for the :mod:`conversation` module."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import (  # noqa: E402  pylint: disable=wrong-import-position
    Agent,
    messages,
)

from conversation import (  # noqa: E402  pylint: disable=wrong-import-position
    ConversationSession,
)
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    ServiceFeature,
    ServiceInput,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Minimal stand-in for a Pydantic-AI agent."""

    def __init__(self) -> None:  # pragma: no cover - trivial init
        self.called_with: list[str] = []

    async def run(
        self, prompt: str, message_history: list
    ):  # pragma: no cover - simple stub
        self.called_with.append(prompt)
        return SimpleNamespace(output="pong", new_messages=lambda: ["msg"])


def test_add_parent_materials_records_history() -> None:
    """``add_parent_materials`` should append service info to history."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type=None,
        description="desc",
        jobs_to_be_done=["job1", "job2"],
    )
    session.add_parent_materials(service)

    assert len(session._history) == 1  # noqa: SLF001 - accessing test-only attribute
    assert isinstance(session._history[0], messages.ModelRequest)
    material = session._history[0].parts[0].content  # noqa: SLF001
    assert "Service ID: svc-1" in material
    assert "Jobs to be done: job1, job2" in material


def test_add_parent_materials_includes_features() -> None:
    """Seed materials should list existing service features when provided."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type=None,
        description="desc",
        jobs_to_be_done=[],
        features=[
            ServiceFeature(
                feature_id="F1",
                name="Feat",
                description="D",
            )
        ],
    )
    session.add_parent_materials(service)

    material = cast(
        messages.TextPart, session._history[0].parts[0]
    ).content  # noqa: SLF001 - test helper
    assert "Existing features: F1: Feat" in material


@pytest.mark.asyncio
async def test_ask_adds_responses_to_history() -> None:
    """``ask`` should forward prompts and store new messages."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    reply = await session.ask("ping")

    assert reply == "pong"
    assert session._history[-1] == "msg"  # noqa: SLF001 - accessing test-only attribute


@pytest.mark.asyncio
async def test_ask_forwards_prompt_to_agent() -> None:
    """``ask`` should delegate to the underlying agent."""
    agent = DummyAgent()
    session = ConversationSession(cast(Agent[None, str], agent))
    await session.ask("hello")
    assert agent.called_with == ["hello"]
