"""Shared utilities for mapping operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import logfire

T = TypeVar("T")


def fit_batch_to_token_cap(
    items: Sequence[T],
    target_size: int,
    token_cap: int,
    token_counter: Callable[[Sequence[T]], int],
    *,
    label: str,
) -> int:
    """Return the largest batch size within ``token_cap``.

    The ``token_counter`` callable is invoked on progressively smaller slices of
    ``items`` until the resulting token count is within ``token_cap`` or the
    batch is reduced to a single item. A reduction is logged when the final
    size is smaller than ``target_size``.

    Args:
        items: Complete sequence of items under consideration.
        target_size: Initial batch size to attempt.
        token_cap: Maximum permitted token count.
        token_counter: Function returning the token usage for a given slice of
            ``items``.
        label: Descriptor used in log messages to identify the batch type.

    Returns:
        The adjusted batch size, guaranteed to be at least ``1``.
    """

    size = min(target_size, len(items))
    initial = size
    tokens = token_counter(items[:size])
    # Reduce the batch until it fits within ``token_cap`` or only one item
    # remains. The loop handles the edge case where even a single item exceeds
    # the cap by returning a size of 1.
    while tokens > token_cap and size > 1:
        size -= 1
        tokens = token_counter(items[:size])
    if size < initial:
        logfire.info(
            f"Reduced {label} batch size from {initial} to {size} "
            f"({tokens} tokens > cap {token_cap})"
        )
    return max(1, size)
