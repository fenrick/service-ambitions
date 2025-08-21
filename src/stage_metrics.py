"""Utilities for aggregating and reporting per-stage metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from monitoring import logfire


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


def log_stage_totals() -> None:
    """Emit aggregated metrics for each recorded stage."""

    for stage, totals in iter_stage_totals():
        tokens_sec = (
            totals.total_tokens / totals.total_duration
            if totals.total_duration
            else 0.0
        )
        avg_latency = totals.total_duration / totals.prompts if totals.prompts else 0.0
        rate_429 = totals.errors_429 / totals.prompts if totals.prompts else 0.0
        logfire.info(
            "Stage totals",
            stage=stage,
            total_tokens=totals.total_tokens,
            prompt_tokens_estimate=totals.prompt_tokens_estimate,
            estimated_cost=totals.estimated_cost,
            tokens_per_sec=tokens_sec,
            avg_latency=avg_latency,
            rate_429=rate_429,
        )


__all__ = [
    "StageTotals",
    "record_stage_metrics",
    "iter_stage_totals",
    "reset_stage_totals",
    "log_stage_totals",
]
