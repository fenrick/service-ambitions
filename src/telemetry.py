# SPDX-License-Identifier: MIT
"""Aggregate mapping metrics for end-of-run reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, List


@dataclass
class SetMetrics:
    """Metrics collected for a single mapping set."""

    features: int = 0
    mapped_ids: int = 0
    unknown_ids: int = 0
    retries: int = 0
    latencies: List[float] = field(default_factory=list)
    tokens: int = 0
    cost: float = 0.0

    def add(
        self,
        *,
        features: int,
        mapped_ids: int,
        unknown_ids: int,
        retries: int,
        latency: float,
        tokens: int,
        cost: float,
    ) -> None:
        """Update metrics with a new mapping result."""

        self.features += features
        self.mapped_ids += mapped_ids
        self.unknown_ids += unknown_ids
        self.retries += retries
        self.latencies.append(latency)
        self.tokens += tokens
        self.cost += cost

    @property
    def average_latency(self) -> float:
        """Return the average latency for the mapping set."""

        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


_metrics: DefaultDict[str, SetMetrics] = DefaultDict(SetMetrics)
_quarantine_paths: List[Path] = []


def record_mapping_set(
    set_name: str,
    *,
    features: int,
    mapped_ids: int,
    unknown_ids: int,
    retries: int,
    latency: float,
    tokens: int,
    cost: float,
) -> None:
    """Record metrics for a mapping ``set_name``."""

    _metrics[set_name].add(
        features=features,
        mapped_ids=mapped_ids,
        unknown_ids=unknown_ids,
        retries=retries,
        latency=latency,
        tokens=tokens,
        cost=cost,
    )


def record_quarantine(path: Path) -> None:
    """Track creation of a quarantine ``path``."""

    _quarantine_paths.append(path)


def has_quarantines() -> bool:
    """Return ``True`` when any quarantine files were created."""

    return bool(_quarantine_paths)


def reset() -> None:
    """Clear all recorded metrics and quarantine paths."""

    _metrics.clear()
    _quarantine_paths.clear()


def print_summary() -> None:
    """Write a summary of collected metrics to ``stdout``."""

    if not _metrics:
        return
    for set_name, data in _metrics.items():
        avg_latency = data.average_latency
        print(
            f"{set_name}: features={data.features} mapped_ids={data.mapped_ids} "
            f"unknown_ids={data.unknown_ids} retries={data.retries} "
            f"avg_latency={avg_latency:.2f}s tokens={data.tokens} "
            f"cost=${data.cost:.4f}"
        )
    total_tokens = sum(d.tokens for d in _metrics.values())
    total_cost = sum(d.cost for d in _metrics.values())
    print(f"Totals: tokens={total_tokens} cost=${total_cost:.4f}")


__all__ = [
    "record_mapping_set",
    "record_quarantine",
    "print_summary",
    "has_quarantines",
    "reset",
]
