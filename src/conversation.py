"""Conversational session wrapper for LLM interactions.

This module exposes :class:`ConversationSession`, a light abstraction over a
Pydantic-AI ``Agent``. The session records message history so that each prompt
retains prior context and can be seeded with service details via
``add_parent_materials``. The :meth:`ask` method delegates to the underlying
agent without relying on asynchronous execution.
"""

from __future__ import annotations

import logging
from typing import TypeVar, overload

import logfire
from pydantic_ai import Agent, messages

from models import ServiceInput

logger = logging.getLogger(__name__)


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI agent.

    The session stores message history so that subsequent prompts include
    previous context. Additional materials about the service under discussion
    may be seeded using :meth:`add_parent_materials`.
    """

    @logfire.instrument()
    def __init__(self, client: Agent) -> None:
        """Initialise the session with a configured LLM client.

        Args:
            client: Pydantic-AI ``Agent`` used for exchanges with the model.
        """

        self.client = client
        self._history: list[messages.ModelMessage] = []

    @logfire.instrument()
    def add_parent_materials(self, service_input: ServiceInput) -> None:
        """Seed the conversation with details about the target service.

        Args:
            service_input: Metadata describing the service being evaluated.

        Side Effects:
            Appends a user prompt containing the service context to the session
            history.
        """
        ctx = "SERVICE_CONTEXT:\n" + service_input.model_dump_json()
        logger.debug("Adding service material to history: %s", ctx)
        self._history.append(
            messages.ModelRequest(parts=[messages.UserPromptPart(ctx)])
        )

    T = TypeVar("T")

    @overload
    def ask(self, prompt: str) -> str: ...

    @overload
    def ask(self, prompt: str, output_type: type[T]) -> T: ...

    @logfire.instrument()
    def ask(self, prompt: str, output_type: type[T] | None = None) -> T | str:
        """Return the agent's response to ``prompt``.

        The prompt together with accumulated message history is forwarded to the
        underlying ``Agent``. When ``output_type`` is supplied, the model is
        asked to return a structured object which is validated and returned. Any
        new messages are recorded in the session history.

        Args:
            prompt: The user message to send to the model.
            output_type: Optional Pydantic model used to validate the response.

        Returns:
            Structured ``output_type`` instance when provided, otherwise the
            agent's raw response text.
        """

        logger.debug("Sending prompt: %s", prompt)
        result = self.client.run_sync(
            prompt, message_history=self._history, output_type=output_type
        )
        self._history.extend(result.new_messages())
        logger.debug("Received response: %s", result.output)
        return result.output


__all__ = ["ConversationSession"]
