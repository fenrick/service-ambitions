"""Adaptive backpressure utilities.

This module provides helpers for rate limiting and telemetry. The
``AdaptiveSemaphore`` is a weighted semaphore tuned for language model
workloads. Tasks may reserve multiple permits proportional to their estimated
output tokens so large responses cannot monopolise concurrency. The
``Generator`` class derives these weights from its
``expected_output_tokens`` configuration.

When upstream services send ``Retry-After`` hints, the semaphore halves the
available permits and enters a slow-start recovery. After the hinted delay it
releases one permit at a time, doubling the interval between releases on each
consecutive throttle. ``RollingMetrics`` exposes request and error counters to
Logfire for observability.

Example:
    ```python
    from service_ambitions.backpressure import AdaptiveSemaphore
    import math

    expected_output_tokens = 256
    limiter = AdaptiveSemaphore(permits=5, ramp_interval=1.0)

    token_estimate = 800
    weight = math.ceil(token_estimate / expected_output_tokens)

    async with limiter(weight):
        ...
    ```
"""

from __future__ import annotations

import asyncio
import time
from asyncio import Lock, Semaphore
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Deque, Optional

import logfire

REQUESTS_TOTAL = logfire.metric_counter("requests_total")
"""Counter for total requests processed."""

ERRORS_TOTAL = logfire.metric_counter("errors_total")
"""Counter for total errors encountered."""

TOKENS_IN_FLIGHT = logfire.metric_gauge("tokens_in_flight")
"""Gauge tracking tokens currently being processed."""

REQUESTS_PER_SECOND = logfire.metric_gauge("requests_per_second")

ERROR_RATE = logfire.metric_gauge("error_rate")

TOKENS_PER_SECOND = logfire.metric_gauge("tokens_per_second")

RATE_429 = logfire.metric_gauge("rate_429")

AVG_LATENCY = logfire.metric_gauge("avg_latency")


class AdaptiveSemaphore:
    """Semaphore that reacts to rate limit signals with weighted permits.

    Tasks may acquire multiple permits to reflect anticipated token usage,
    typically derived from ``expected_output_tokens``. On ``Retry-After`` the
    limiter halves its concurrency and then gradually restores capacity using a
    ramp strategy: permits are released one at a time with an exponentially
    increasing delay after consecutive throttles. This slow-start recovery helps
    avoid cascading failures when providers apply throttling.
    """

    def __init__(
        self,
        permits: int,
        *,
        ramp_interval: float = 1.0,
        grace_period: float = 60.0,
    ) -> None:
        """Create the semaphore.

        Args:
            permits: Initial maximum concurrency.
            ramp_interval: Base delay between permit releases during recovery;
                consecutive throttles double this interval to slow-start.
            grace_period: Time window in seconds after which slow-start resets if
                no further throttling occurs.
        """

        self._sem = Semaphore(permits)
        self._max = permits
        self._current = permits
        self._base_ramp = ramp_interval
        self._ramp_interval = ramp_interval
        self._grace_period = grace_period
        self._lock = Lock()
        self._consecutive = 0
        self._last_throttle = 0.0
        self._reset_task: Optional[asyncio.Task[None]] = None
        self._tasks: set[asyncio.Task[None]] = set()

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

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __call__(self, weight: int = 1) -> AsyncContextManager["AdaptiveSemaphore"]:
        """Return a context manager acquiring ``weight`` permits.

        Use this to weight concurrency by estimated tokens.

        Example:
            ```python
            import math
            token_estimate = 800
            weight = math.ceil(token_estimate / expected_output_tokens)
            async with limiter(weight):
                ...
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

        task = asyncio.create_task(self._throttle(delay))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _schedule_reset(self) -> None:
        """Reset slow-start state after the grace period."""

        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()
        self._reset_task = asyncio.create_task(self._reset_after_grace())

    async def _reset_after_grace(self) -> None:
        await asyncio.sleep(self._grace_period)
        self._consecutive = 0
        self._ramp_interval = self._base_ramp

    async def _throttle(self, delay: float) -> None:
        async with self._lock:
            now = time.monotonic()
            # Reset slow-start counters if enough time has elapsed.
            if now - self._last_throttle > self._grace_period:
                self._consecutive = 0
                self._ramp_interval = self._base_ramp

            self._consecutive += 1
            self._last_throttle = now
            if self._consecutive > 1:
                # Exponential backoff of the ramp interval during slow-start.
                self._ramp_interval = self._base_ramp * (2 ** (self._consecutive - 1))

            if self._current <= 1:
                # Already at minimum concurrency; nothing to adjust.
                self._schedule_reset()
                return
            target = max(1, self._current // 2)
            reduction = self._current - target
            # Each acquire reduces the available permits without blocking new
            # tasks beyond the desired limit.
            await self.acquire(reduction)
            self._current = target

        # Reset counters once no additional throttles occur for the grace
        # window.
        self._schedule_reset()

        # Wait for the provider hint before slowly ramping back up.
        await asyncio.sleep(delay)
        while self._current < self._max:
            await asyncio.sleep(self._ramp_interval)
            self.release()
            self._current += 1


class RollingMetrics:
    """Track rolling request, error and token rates."""

    def __init__(self, window: float = 60.0) -> None:
        self._window = window
        self._requests: Deque[float] = deque()
        self._errors: Deque[float] = deque()
        self._errors_429: Deque[float] = deque()
        self._latencies: Deque[tuple[float, float]] = deque()
        self._tokens: Deque[tuple[float, int]] = deque()
        self._in_flight = 0

    def _trim(self, buf: Deque[float], now: float) -> None:
        while buf and now - buf[0] > self._window:
            buf.popleft()

    def _trim_tokens(self, now: float) -> None:
        while self._tokens and now - self._tokens[0][0] > self._window:
            self._tokens.popleft()

    def _trim_latencies(self, now: float) -> None:
        while self._latencies and now - self._latencies[0][0] > self._window:
            self._latencies.popleft()

    def record_request(self) -> None:
        """Record a request start."""

        now = time.monotonic()
        self._requests.append(now)
        self._trim(self._requests, now)
        self._trim(self._errors, now)
        self._trim(self._errors_429, now)
        self._trim_latencies(now)
        REQUESTS_TOTAL.add(1)

    def record_error(self, is_429: bool = False) -> None:
        """Record an error occurrence.

        Args:
            is_429: Mark the error as a rate limit (HTTP 429) event.
        """

        now = time.monotonic()
        self._errors.append(now)
        if is_429:
            self._errors_429.append(now)
        self._trim(self._errors, now)
        self._trim(self._errors_429, now)
        ERRORS_TOTAL.add(1)

    def record_latency(self, duration: float) -> None:
        """Record request latency in seconds."""

        now = time.monotonic()
        self._latencies.append((now, duration))
        self._trim_latencies(now)

    def record_start_tokens(self, count: int) -> None:
        """Increment in-flight tokens and emit the current count."""

        self._in_flight += count
        TOKENS_IN_FLIGHT.set(self._in_flight)

    def record_end_tokens(self, count: int) -> None:
        """Decrement in-flight tokens and emit updated metrics."""

        now = time.monotonic()
        self._in_flight = max(self._in_flight - count, 0)
        TOKENS_IN_FLIGHT.set(self._in_flight)
        self._tokens.append((now, count))
        self._trim_tokens(now)
        self._trim(self._requests, now)
        self._trim(self._errors, now)
        self._trim(self._errors_429, now)
        self._trim_latencies(now)
        total_tokens = sum(t for _, t in self._tokens)
        tps = total_tokens / self._window
        TOKENS_PER_SECOND.set(tps)
        rps = len(self._requests) / self._window
        REQUESTS_PER_SECOND.set(rps)
        error_rate = len(self._errors) / len(self._requests) if self._requests else 0.0
        ERROR_RATE.set(error_rate)
        rate_429 = (
            len(self._errors_429) / len(self._requests) if self._requests else 0.0
        )
        RATE_429.set(rate_429)
        avg_latency = (
            sum(d for _, d in self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )
        AVG_LATENCY.set(avg_latency)
        logfire.info(
            "Rolling metrics updated",
            tokens_per_sec=tps,
            rps=rps,
            error_rate=error_rate,
            rate_429=rate_429,
            avg_latency=avg_latency,
            in_flight=self._in_flight,
        )

    def record_tokens(self, count: int) -> None:
        """Alias for :meth:`record_end_tokens` for backwards compatibility."""

        self.record_end_tokens(count)


__all__ = ["AdaptiveSemaphore", "RollingMetrics"]
