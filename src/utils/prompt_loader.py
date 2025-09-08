"""Prompt loading abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import logfire


class PromptLoader(ABC):
    """Interface for retrieving prompt templates.

    Implementations should cache aggressively as prompts are immutable and may
    be requested repeatedly. Reads must avoid blocking I/O where possible and
    complete within a few milliseconds for warm cache hits. Call
    :meth:`clear_cache` to reset state when templates change.
    """

    @abstractmethod
    def load(self, name: str) -> str:
        """Return the text for ``name``.

        Args:
            name: Identifier of the prompt template without extension.

        Returns:
            The template text.
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached prompt text."""


class FilePromptLoader(PromptLoader):
    """Load prompts from the local file system with memoisation."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Retrieve the prompt template for ``name`` from disk.

        Args:
            name: Identifier of the prompt template without file extension.

        Returns:
            Template text with surrounding whitespace removed.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
            OSError: If an error occurs while reading the file.
        """
        with logfire.span("prompt_loader.load", attributes={"name": name}):
            if name in self._cache:
                return self._cache[name]
            path = self._base_dir / (name if name.endswith(".md") else f"{name}.md")
            with path.open("r", encoding="utf-8") as file:
                text = file.read().strip()
            self._cache[name] = text
            return text

    def clear_cache(self) -> None:
        """Reset memoised prompt text."""
        self._cache.clear()
