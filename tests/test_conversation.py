"""Unit tests for the :mod:`conversation` module."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from typing import cast

from pydantic_ai import (
    Agent,
    messages,
)  # noqa: E402  pylint: disable=wrong-import-position

from conversation import (
    ConversationSession,
)  # noqa: E402  pylint: disable=wrong-import-position
from models import ServiceInput  # noqa: E402  pylint: disable=wrong-import-position


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
    service = ServiceInput(name="svc", customer_type=None, description="desc")
    session.add_parent_materials(service)

    assert len(session._history) == 1  # noqa: SLF001 - accessing test-only attribute
    assert isinstance(session._history[0], messages.ModelRequest)


def test_ask_adds_responses_to_history() -> None:
    """``ask`` should forward prompts and store new messages."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    reply = session.ask("ping")

    assert reply == "pong"
    assert session._history[-1] == "msg"  # noqa: SLF001 - accessing test-only attribute
