"""Adaptive backpressure utilities.

This module provides helpers for rate limiting and telemetry. The
``AdaptiveSemaphore`` dynamically reduces concurrency when upstream services
signal throttling via ``Retry-After`` hints and gradually restores capacity.
``RollingMetrics`` exposes simple request and error counters to Logfire for
observability.
"""

from __future__ import annotations

import asyncio
import time
from asyncio import Lock, Semaphore
from collections import deque
from contextlib import asynccontextmanager
from types import ModuleType
from typing import AsyncContextManager, AsyncIterator, Deque

try:
    import logfire as _logfire
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _logfire = None

if _logfire and hasattr(_logfire, "metric"):
    logfire = _logfire
else:  # pragma: no cover - default stub for metrics
    logfire = ModuleType("logfire")
    logfire.metric = lambda name, value: None  # type: ignore[attr-defined]


class AdaptiveSemaphore:
    """Semaphore that reacts to rate limit signals.

    The limiter supports weighted permits so tasks can reserve multiple units
    of capacity at once. When a ``Retry-After`` hint is observed the semaphore
    temporarily reduces the number of available permits to half the current
    limit. After the hinted delay it releases one permit at a time until the
    original concurrency is restored. This helps avoid cascading failures when
    providers apply throttling.
    """

    def __init__(self, permits: int, *, ramp_interval: float = 1.0) -> None:
        self._sem = Semaphore(permits)
        self._max = permits
        self._current = permits
        self._ramp_interval = ramp_interval
        self._lock = Lock()

    async def acquire(self, weight: int = 1) -> None:
        """Acquire one or more permits.

        Args:
            weight: Number of permits to reserve. Must be positive.

        Raises:
            ValueError: If ``weight`` is less than one.
        """

        if weight < 1:
            raise ValueError("weight must be positive")

        for _ in range(weight):
            await self._sem.acquire()

    def release(self, weight: int = 1) -> None:
        """Release one or more permits back to the semaphore.

        Args:
            weight: Number of permits to release. Must be positive.

        Raises:
            ValueError: If ``weight`` is less than one.
        """

        if weight < 1:
            raise ValueError("weight must be positive")

        for _ in range(weight):
            self._sem.release()

    async def __aenter__(self) -> "AdaptiveSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard
        self.release()

    def __call__(self, weight: int = 1) -> AsyncContextManager["AdaptiveSemaphore"]:
        """Return a context manager acquiring ``weight`` permits.

        Example:
            ```python
            async with limiter(2):
                ...  # uses two permits
            ```
        """

        @asynccontextmanager
        async def _manager() -> AsyncIterator["AdaptiveSemaphore"]:
            await self.acquire(weight)
            try:
                yield self
            finally:
                self.release(weight)

        return _manager()

    @property
    def limit(self) -> int:
        """Return the current concurrency limit."""

        return self._current

    def throttle(self, delay: float) -> None:
        """Reduce available permits and schedule gradual recovery.

        Args:
            delay: Number of seconds suggested by ``Retry-After`` before
                beginning to restore capacity.
        """

        asyncio.create_task(self._throttle(delay))

    async def _throttle(self, delay: float) -> None:
        async with self._lock:
            if self._current <= 1:
                # Already at minimum concurrency; nothing to adjust.
                return
            target = max(1, self._current // 2)
            reduction = self._current - target
            # Each acquire reduces the available permits without blocking new
            # tasks beyond the desired limit.
            await self.acquire(reduction)
            self._current = target

        # Wait for the provider hint before slowly ramping back up.
        await asyncio.sleep(delay)
        while self._current < self._max:
            await asyncio.sleep(self._ramp_interval)
            self.release()
            self._current += 1


class RollingMetrics:
    """Track rolling request rates and error ratios."""

    def __init__(self, window: float = 60.0) -> None:
        self._window = window
        self._requests: Deque[float] = deque()
        self._errors: Deque[float] = deque()

    def _trim(self, buf: Deque[float], now: float) -> None:
        while buf and now - buf[0] > self._window:
            buf.popleft()

    def record_request(self) -> None:
        """Record a request and emit updated metrics."""

        now = time.monotonic()
        self._requests.append(now)
        self._trim(self._requests, now)
        self._trim(self._errors, now)
        rps = len(self._requests) / self._window
        error_rate = len(self._errors) / len(self._requests) if self._requests else 0.0
        logfire.metric("requests_per_second", rps)
        logfire.metric("error_rate", error_rate)

    def record_error(self) -> None:
        """Record an error occurrence."""

        now = time.monotonic()
        self._errors.append(now)
        self._trim(self._errors, now)


__all__ = ["AdaptiveSemaphore", "RollingMetrics"]
