from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator


@dataclass
class StageTotals:
    """Aggregate metrics for a conversation stage."""

    total_tokens: int = 0
    prompt_tokens_estimate: int = 0
    estimated_cost: float = 0.0
    total_duration: float = 0.0
    prompts: int = 0
    errors_429: int = 0


_stage_totals: dict[str, StageTotals] = defaultdict(StageTotals)


def record_stage_metrics(
    stage: str,
    tokens: int,
    cost: float,
    duration: float,
    is_429: bool,
    prompt_tokens: int,
) -> None:
    """Update aggregated metrics for ``stage``.

    Args:
        stage: Name of the conversation stage.
        tokens: Actual tokens consumed by the request.
        cost: Estimated USD cost for the request.
        duration: Request latency in seconds.
        is_429: Whether the request resulted in a rate limit error.
        prompt_tokens: Estimated prompt tokens before the request was sent.
    """

    totals = _stage_totals[stage]
    totals.total_tokens += tokens
    totals.prompt_tokens_estimate += prompt_tokens
    totals.estimated_cost += cost
    totals.total_duration += duration
    totals.prompts += 1
    if is_429:
        totals.errors_429 += 1


def iter_stage_totals() -> Iterator[tuple[str, StageTotals]]:
    """Yield accumulated stage metrics."""

    return iter(_stage_totals.items())


def reset_stage_totals() -> None:
    """Clear all recorded stage metrics."""

    _stage_totals.clear()


__all__ = [
    "StageTotals",
    "record_stage_metrics",
    "iter_stage_totals",
    "reset_stage_totals",
]
