"""Utility helpers for estimating token usage and cost.

This module exposes :func:`estimate_tokens`, which uses ``tiktoken`` when
available and otherwise falls back to a simple heuristic. It also provides
``estimate_cost`` for rough per-model pricing based on total tokens consumed.
"""

from __future__ import annotations

try:
    import tiktoken
except Exception:
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


# Approximate USD pricing per 1K tokens for known models.
_MODEL_PRICING: dict[str, float] = {
    "gpt-5": 0.01,
    "o4-mini": 0.005,
    "gpt-4o-search-preview": 0.007,
}


def estimate_cost(model_name: str, tokens: int) -> float:
    """Return a rough USD cost estimate for ``tokens`` used by ``model_name``.

    Args:
        model_name: Identifier of the model, optionally with ``<provider>:``
            prefix.
        tokens: Total tokens consumed by the request.

    Returns:
        Approximate cost in USD based on a static price table.
    """

    key = model_name.split(":", 1)[-1]
    rate = _MODEL_PRICING.get(key, 0.01)
    return tokens / 1000 * rate


__all__ = ["estimate_tokens", "estimate_cost"]
