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
        self._queue: List[Tuple[int, Callable[[], Awaitable[Any]]]] = []

    def submit(self, func: Callable[[], Awaitable[Any]], tokens: int) -> None:
        """Queue a coroutine factory for execution.

        Args:
            func: A parameterless callable returning an awaitable.
            tokens: Predicted token count for the task.
        """
        self._queue.append((tokens, func))

    async def run(self) -> List[Any]:
        """Execute queued tasks respecting token ordering.

        Returns:
            List of results from executed coroutines in submission order.
        """
        self._queue.sort(key=lambda item: item[0])
        semaphore = asyncio.Semaphore(self._max_workers)

        async def runner(func: Callable[[], Awaitable[Any]]) -> Any:
            async with semaphore:
                return await func()

        coros = [runner(func) for _, func in self._queue]
        return await asyncio.gather(*coros)
