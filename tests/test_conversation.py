"""Unit tests for the :mod:`conversation` module."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pydantic_ai import Agent, messages

import conversation
from conversation import ConversationSession
from models import (
    ServiceFeature,
    ServiceInput,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Minimal stand-in for a Pydantic-AI agent."""

    def __init__(self) -> None:
        self.called_with: list[str] = []

    def run_sync(self, prompt: str, message_history: list[str], output_type=None):
        """Synchronously return a canned response."""

        self.called_with.append(prompt)
        return SimpleNamespace(
            output="pong",
            new_messages=lambda: ["msg"],
            usage=lambda: SimpleNamespace(total_tokens=0),
        )

    async def run(self, prompt: str, message_history: list[str], output_type=None):
        self.called_with.append(prompt)
        return SimpleNamespace(
            output="pong",
            new_messages=lambda: ["msg"],
            usage=lambda: SimpleNamespace(total_tokens=0),
        )


def test_add_parent_materials_records_history() -> None:
    """``add_parent_materials`` should append service info to history."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type=None,
        description="desc",
        jobs_to_be_done=[{"name": "job1"}, {"name": "job2"}],
    )
    session.add_parent_materials(service)

    assert len(session._history) == 1
    assert isinstance(session._history[0], messages.ModelRequest)
    part = session._history[0].parts[0]
    assert isinstance(part, messages.UserPromptPart)
    material = cast(str, part.content)
    assert material.startswith("SERVICE_CONTEXT:\n")
    data = json.loads(material.split("SERVICE_CONTEXT:\n", 1)[1])
    assert data["service_id"] == "svc-1"
    assert data["jobs_to_be_done"] == [
        {"name": "job1"},
        {"name": "job2"},
    ]


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

    part = cast(messages.UserPromptPart, session._history[0].parts[0])
    material = cast(str, part.content)
    data = json.loads(material.split("SERVICE_CONTEXT:\n", 1)[1])
    assert data["features"] == [
        {
            "feature_id": "F1",
            "name": "Feat",
            "description": "D",
        }
    ]


def test_ask_adds_responses_to_history() -> None:
    """``ask`` should forward prompts and store new messages."""

    session = ConversationSession(cast(Agent[None, str], DummyAgent()))
    reply = session.ask("ping")

    assert reply == "pong"
    assert session._history[-1] == "msg"


def test_ask_forwards_prompt_to_agent() -> None:
    """``ask`` should delegate to the underlying agent."""
    agent = DummyAgent()
    session = ConversationSession(cast(Agent[None, str], agent))
    session.ask("hello")
    assert agent.called_with == ["hello"]


def test_prompt_token_estimate_metric(monkeypatch) -> None:
    """``ask`` should record a token estimate metric labelled by stage."""

    agent = DummyAgent()
    session = ConversationSession(cast(Agent[None, str], agent), stage="stage")

    calls: list[tuple[int, dict[str, str]]] = []

    class DummyMetric:
        def add(
            self, value: int, *, attributes: dict[str, str] | None = None
        ) -> None:  # pragma: no cover - simple stub
            calls.append((value, attributes or {}))

    monkeypatch.setattr(conversation, "PROMPT_TOKEN_ESTIMATE", DummyMetric())
    monkeypatch.setattr(
        conversation, "estimate_tokens", lambda prompt, expected_output=0: 5
    )

    session.ask("ping")

    assert calls == [(5, {"stage": "stage"})]
