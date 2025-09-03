# SPDX-License-Identifier: MIT
"""Unit tests for the :mod:`conversation` module."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai import Agent, messages
from pydantic_core import from_json

from core import conversation
from core.conversation import ConversationSession
from models import ServiceFeature, ServiceInput
from runtime.environment import RuntimeEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DummyAgent:
    """Minimal stand-in for a Pydantic-AI agent."""

    def __init__(self) -> None:
        self.called_with: list[str] = []

    def run_sync(self, prompt: str, message_history: list[str]):
        self.called_with.append(prompt)
        return SimpleNamespace(
            output="pong",
            new_messages=lambda: ["msg"],
            usage=lambda: SimpleNamespace(total_tokens=5),
        )


def test_add_parent_materials_records_history() -> None:
    """``add_parent_materials`` should append service info to history."""

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
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
    data = from_json(material.split("SERVICE_CONTEXT:\n", 1)[1])
    assert data["service_id"] == "svc-1"
    assert data["jobs_to_be_done"] == [
        {"name": "job1"},
        {"name": "job2"},
    ]


def test_add_parent_materials_includes_features() -> None:
    """Seed materials should list existing service features when provided."""

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
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
    data = from_json(material.split("SERVICE_CONTEXT:\n", 1)[1])
    assert data["features"] == [
        {
            "feature_id": "F1",
            "name": "Feat",
            "description": "D",
        }
    ]


def test_ask_adds_responses_to_history() -> None:
    """``ask`` should forward prompts and store new messages."""

    session = ConversationSession(
        cast(Agent[None, str], DummyAgent()),
        use_local_cache=False,
        cache_mode="off",
    )
    reply = session.ask("ping")

    assert reply == "pong"
    assert session._history[-1] == "msg"


def test_ask_forwards_prompt_to_agent() -> None:
    """``ask`` should delegate to the underlying agent."""
    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        use_local_cache=False,
        cache_mode="off",
    )
    session.ask("hello")
    assert agent.called_with == ["hello"]


def test_ask_omits_prompt_logging_when_disabled(tmp_path, monkeypatch) -> None:
    """Prompts should not be logged when logging is disabled."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        diagnostics=True,
        log_prompts=False,
        transcripts_dir=tmp_path,
        use_local_cache=False,
        cache_mode="off",
    )
    calls: list[str] = []
    monkeypatch.setattr(conversation.logfire, "debug", lambda msg: calls.append(msg))

    session.ask("hello")

    assert calls == []


def test_catalogue_strings_not_logged_by_default(monkeypatch) -> None:
    """Catalogue details should not leak to logs without explicit opt-in."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        diagnostics=True,
        log_prompts=False,
        use_local_cache=False,
        cache_mode="off",
    )
    logged: list[str] = []
    monkeypatch.setattr(conversation.logfire, "debug", lambda msg: logged.append(msg))

    prompt = "## Available widgets\nW1\tWidget One\tDesc"
    session.ask(prompt)

    assert not any("Widget One" in msg for msg in logged)


def test_diagnostics_writes_transcript(tmp_path) -> None:
    """Diagnostics mode should persist prompt/response transcripts."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        stage="stage",
        diagnostics=True,
        transcripts_dir=tmp_path,
        use_local_cache=False,
        cache_mode="off",
    )
    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type=None,
        description="d",
        jobs_to_be_done=[],
    )
    session.add_parent_materials(service)
    session.ask("ping")

    path = tmp_path / "svc-1" / "stage.json"
    assert path.exists()
    data = from_json(path.read_text(encoding="utf-8"))
    assert data == {"prompt": "ping", "response": "pong"}


def test_ask_uses_cache_when_available(tmp_path, monkeypatch) -> None:
    """Existing cache entries should bypass agent invocation."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        stage="stage",
        use_local_cache=True,
        cache_mode="read",
    )
    RuntimeEnv.initialize(
        cast(Any, SimpleNamespace(cache_dir=tmp_path, context_id="ctx"))
    )
    key = conversation._prompt_cache_key("hello", "", "stage")
    path = conversation._prompt_cache_path("unknown", "stage", key)
    path.write_text(json.dumps("cached"), encoding="utf-8")

    reply = session.ask("hello")

    assert reply == "cached"
    assert agent.called_with == []


def test_ask_writes_cache_on_miss(tmp_path, monkeypatch) -> None:
    """Cache misses should persist responses for future calls."""

    agent = DummyAgent()
    session = ConversationSession(
        cast(Agent[None, str], agent),
        stage="stage",
        use_local_cache=True,
        cache_mode="read",
    )
    RuntimeEnv.initialize(
        cast(Any, SimpleNamespace(cache_dir=tmp_path, context_id="ctx"))
    )

    session.ask("hello")

    key = conversation._prompt_cache_key("hello", "", "stage")
    path = conversation._prompt_cache_path("unknown", "stage", key)
    assert path.exists()
