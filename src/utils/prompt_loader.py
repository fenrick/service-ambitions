"""Prompt loading abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import logfire


class PromptLoader(ABC):
    """Interface for retrieving prompt templates.

    Implementations should cache aggressively as prompts are immutable and may
    be requested repeatedly. Reads must avoid blocking I/O where possible and
    complete within a few milliseconds for warm cache hits.
    """

    @abstractmethod
    def load(self, name: str) -> str:
        """Return the text for ``name``.

        Args:
            name: Identifier of the prompt template without extension.

        Returns:
            The template text.
        """


class FilePromptLoader(PromptLoader):
    """Load prompts from the local file system."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def load(self, name: str) -> str:  # noqa: D401 - short delegation
        with logfire.span("prompt_loader.load", attributes={"name": name}):
            path = self._base_dir / (name if name.endswith(".md") else f"{name}.md")
            with path.open("r", encoding="utf-8") as file:
                text = file.read().strip()
            return text
