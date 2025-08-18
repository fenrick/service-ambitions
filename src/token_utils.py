"""Utility helpers for estimating token counts.

This module exposes :func:`estimate_tokens`, which uses ``tiktoken`` when
available and otherwise falls back to a simple heuristic.
"""

from __future__ import annotations

try:
    import tiktoken
except Exception:  # pragma: no cover - in case of unexpected import errors
    tiktoken = None


def estimate_tokens(prompt: str, expected_output: int) -> int:
    """Estimate the total tokens needed for a prompt and its response.

    Args:
        prompt: The textual prompt that will be sent to the model.
        expected_output: The anticipated number of tokens in the model's reply.

    Returns:
        The estimated total number of tokens for the prompt plus the expected
        output.
    """
    if tiktoken is not None:
        # Use the installed ``tiktoken`` package for accurate token counts.
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(prompt)) + expected_output

    # Fallback to a rough heuristic when ``tiktoken`` is unavailable.
    return len(prompt) // 4 + expected_output
