"""Conversation session management for Pydantic AI clients."""

from __future__ import annotations

import logging
from typing import Iterable

from pydantic_ai import Agent, messages

logger = logging.getLogger(__name__)


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI Agent.

    The session maintains message history so that subsequent requests include
    prior context. Additional materials can be injected as system prompts via
    :meth:`add_parent_materials`.
    """

    def __init__(self, client: Agent) -> None:
        """Initialize the session.

        Args:
            client: Configured Pydantic-AI agent used for exchanges.
        """

        self.client = client
        self._history: list[messages.ModelMessage] = []

    def add_parent_materials(self, materials: Iterable[str]) -> None:
        """Append ``materials`` to the conversation as system prompts.

        Args:
            materials: Iterable of additional context strings.
        """

        for material in materials:
            logger.debug("Adding parent material: %s", material)
            self._history.append(
                messages.ModelRequest(parts=[messages.SystemPromptPart(material)])
            )

    async def ask(self, prompt: str) -> str:
        """Send ``prompt`` to the agent and return the text response.

        Args:
            prompt: The user message to send.

        Returns:
            The agent's response as text.
        """

        logger.info("Sending prompt: %s", prompt)
        result = await self.client.run(prompt, message_history=self._history)
        self._history.extend(result.new_messages())
        logger.info("Received response: %s", result.output)
        return result.output
