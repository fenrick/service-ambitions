"""Conversational session wrapper for LLM interactions.

This module exposes :class:`ConversationSession` which provides a thin
abstraction over a Pydantic-AI ``Agent``.  The session maintains the message
history so that each prompt sent to the model retains prior context.  The
session can be seeded with details from a :class:`models.ServiceInput` via
``add_parent_materials``.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent, messages

from models import ServiceInput

logger = logging.getLogger(__name__)


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI agent.

    The session stores message history so that subsequent prompts include
    previous context. Additional materials about the service under discussion
    may be seeded using :meth:`add_parent_materials`.
    """

    def __init__(self, client: Agent) -> None:
        """Initialise the session with a configured LLM client.

        Args:
            client: Pydantic-AI ``Agent`` used for exchanges with the model.
        """

        self.client = client
        self._history: list[messages.ModelMessage] = []

    def add_parent_materials(self, service_input: ServiceInput) -> None:
        """Seed the conversation with details about the target service.

        Args:
            service_input: Metadata describing the service being evaluated.
        """

        material = (
            f"Service name: {service_input.name}\n"
            f"Customer type: {service_input.customer_type or 'N/A'}\n"
            f"Description: {service_input.description}"
        )
        logger.debug("Adding service material to history: %s", material)
        self._history.append(
            messages.ModelRequest(parts=[messages.SystemPromptPart(material)])
        )

    def ask(self, prompt: str) -> str:
        """Send ``prompt`` to the agent and return the textual response.

        The prompt along with accumulated message history is forwarded to the
        underlying ``Agent``.  The response and any new messages are recorded in
        the session history.

        Args:
            prompt: The user message to send to the model.

        Returns:
            The agent's response text.
        """

        logger.info("Sending prompt: %s", prompt)
        result = asyncio.run(self.client.run(prompt, message_history=self._history))
        self._history.extend(result.new_messages())
        logger.info("Received response: %s", result.output)
        return result.output


__all__ = ["ConversationSession"]
