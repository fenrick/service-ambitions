"""Backpressure utilities for rolling metrics.

Provides :class:`RollingMetrics` to monitor request and token rates, emitting
metrics through Logfire for observability.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

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


__all__ = ["RollingMetrics"]
