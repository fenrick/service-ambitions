"""Utility for scheduling plateau tasks by predicted token usage.

The :class:`TokenScheduler` sorts submitted coroutine factories by their
predicted token load before dispatching them across a fixed-size worker pool.
This favours shorter workloads so they complete before longer running jobs
when resources are constrained.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, List, Tuple


class TokenScheduler:
    """Schedule coroutines according to predicted token consumption."""

    def __init__(self, max_workers: int = 4) -> None:
        """Create a scheduler.

        Args:
            max_workers: Maximum number of concurrent tasks.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        # queue holds (tokens, submission index, coroutine factory)
        self._queue: List[Tuple[int, int, Callable[[], Awaitable[Any]]]] = []
        self._next_index = 0

    def submit(self, func: Callable[[], Awaitable[Any]], tokens: int) -> None:
        """Queue a coroutine factory for execution.

        Args:
            func: A parameterless callable returning an awaitable.
            tokens: Predicted token count for the task.
        """
        self._queue.append((tokens, self._next_index, func))
        self._next_index += 1

    async def run(self) -> List[Any]:
        """Execute queued tasks respecting token ordering.

        Returns:
            List of results from executed coroutines in submission order.
        """
        # Sort first by token count then by submission index to keep relative
        # ordering for equal token predictions.
        self._queue.sort(key=lambda item: (item[0], item[1]))
        semaphore = asyncio.Semaphore(self._max_workers)

        async def runner(func: Callable[[], Awaitable[Any]]) -> Any:
            async with semaphore:
                return await func()

        indices = [index for _, index, _ in self._queue]
        coros = [runner(func) for _, _, func in self._queue]
        results = await asyncio.gather(*coros)
        # Pair results with their original submission indices so callers receive
        # outputs in the order tasks were submitted.
        ordered = [
            result
            for _, result in sorted(
                zip(indices, results, strict=False), key=lambda item: item[0]
            )
        ]
        self._queue.clear()
        return ordered
