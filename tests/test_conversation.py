"""Unit tests for the :mod:`conversation` module."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import Agent, messages

import conversation
from backpressure import RollingMetrics
from conversation import ConversationSession
from models import ServiceFeature, ServiceInput
from stage_metrics import iter_stage_totals, reset_stage_totals

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Minimal stand-in for a Pydantic-AI agent."""

    def __init__(self) -> None:
        self.called_with: list[str] = []

    def run_sync(self, prompt: str, message_history: list[str], output_type=None):
        self.called_with.append(prompt)
        return SimpleNamespace(
            output="pong",
            new_messages=lambda: ["msg"],
            usage=lambda: SimpleNamespace(total_tokens=5),
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


def test_stage_metrics_accumulate() -> None:
    """Conversation calls should update stage metrics."""

    reset_stage_totals()
    session = ConversationSession(cast(Agent[None, str], DummyAgent()), stage="test")
    session.ask("ping")
    stage, totals = next(iter_stage_totals())
    assert stage == "test"
    assert totals.prompts == 1
    assert totals.total_tokens == 5


def test_ask_omits_prompt_logging_when_disabled(monkeypatch) -> None:
    """Prompts should not be logged when logging is disabled."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent), diagnostics=True, log_prompts=False
    )
    calls: list[str] = []
    monkeypatch.setattr(conversation.logfire, "debug", lambda msg: calls.append(msg))

    session.ask("hello")

    assert calls == []


def test_ask_redacts_prompt_when_enabled(monkeypatch) -> None:
    """Prompt text should pass through ``redact_pii`` before logging."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        diagnostics=True,
        log_prompts=True,
        redact_prompts=True,
    )
    monkeypatch.setattr(conversation, "redact_pii", lambda s: "<redacted>")
    calls: list[str] = []
    monkeypatch.setattr(conversation.logfire, "debug", lambda msg: calls.append(msg))

    session.ask("secret")

    assert calls == ["Sending prompt: <redacted>"]


def test_add_parent_materials_redacts_when_enabled(monkeypatch) -> None:
    """Service material logging should redact PII when enabled."""

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        diagnostics=True,
        log_prompts=True,
        redact_prompts=True,
    )
    monkeypatch.setattr(conversation, "redact_pii", lambda s: "<redacted>")
    calls: list[str] = []
    monkeypatch.setattr(conversation.logfire, "debug", lambda msg: calls.append(msg))

    session.add_parent_materials(
        ServiceInput(
            service_id="svc-1", name="svc", description="x", jobs_to_be_done=[]
        )
    )

    assert calls == ["Adding service material to history: <redacted>"]
    part = cast(messages.UserPromptPart, session._history[0].parts[0])
    assert cast(str, part.content) == "<redacted>"


def test_ask_records_redacted_history(monkeypatch) -> None:
    """History should store redacted prompts when enabled."""

    class EchoAgent(DummyAgent):
        def run_sync(self, prompt: str, message_history: list[str], output_type=None):
            req = messages.ModelRequest(parts=[messages.UserPromptPart(prompt)])
            return SimpleNamespace(
                output="pong",
                new_messages=lambda: [req],
                usage=lambda: SimpleNamespace(total_tokens=5),
            )

    session = ConversationSession(
        cast(Agent[None, str], EchoAgent()), redact_prompts=True
    )
    monkeypatch.setattr(conversation, "redact_pii", lambda s: "<redacted>")

    session.ask("topsecret")

    part = cast(messages.ModelRequest, session._history[-1])
    prompt_part = cast(messages.UserPromptPart, part.parts[0])
    assert cast(str, prompt_part.content) == "<redacted>"


def test_metrics_recorded_on_success() -> None:
    """Metrics should capture request and token usage on success."""

    metrics = RollingMetrics(window=1)
    session = ConversationSession(cast(Agent[None, str], DummyAgent()), metrics=metrics)
    session.ask("ping")

    assert len(metrics._requests) == 1
    assert len(metrics._latencies) == 1
    assert len(metrics._tokens) == 1
    assert metrics._in_flight == 0


def test_metrics_recorded_on_error() -> None:
    """Metrics should record errors including 429 hints."""

    class ErrorAgent(DummyAgent):
        def run_sync(self, prompt: str, message_history: list[str], output_type=None):
            raise RuntimeError("429 rate limit")

    metrics = RollingMetrics(window=1)
    session = ConversationSession(cast(Agent[None, str], ErrorAgent()), metrics=metrics)
    with pytest.raises(RuntimeError):
        session.ask("ping")

    assert len(metrics._requests) == 1
    assert len(metrics._errors) == 1
    assert len(metrics._errors_429) == 1
    assert len(metrics._latencies) == 1
