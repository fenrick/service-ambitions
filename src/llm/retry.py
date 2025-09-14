# SPDX-License-Identifier: MIT
"""Retry helpers for LLM invocations.

This module centralises exponential backoff, transient exception handling and
circuit breaking so call sites can share consistent logic.
"""

from __future__ import annotations

import asyncio
import os
import random
from importlib import import_module
from typing import Awaitable, Callable, TypeVar

import logfire

T = TypeVar("T")


# -- Transient exception handling -------------------------------------------------

PROVIDER_EXCEPTION_MAP: dict[str, tuple[str, ...]] = {
    "openai": (
        "openai.APIConnectionError",
        "openai.RateLimitError",
        "openai.APITimeoutError",
    )
}


def _import_exception(path: str) -> type[BaseException] | None:
    """Return exception type from ``path`` or ``None`` if unavailable."""
    try:
        module_name, exc_name = path.rsplit(".", 1)
        module = import_module(module_name)
        exc = getattr(module, exc_name)
    except (ImportError, AttributeError, ValueError):  # pragma: no cover - defensive
        return None
    return exc if isinstance(exc, type) and issubclass(exc, BaseException) else None


def _load_transient_exceptions() -> tuple[type[BaseException], ...]:
    """Construct the transient exception set based on configuration."""
    exceptions: list[type[BaseException]] = [asyncio.TimeoutError, ConnectionError]
    provider = os.getenv("SA_LLM_PROVIDER", "openai")
    for path in PROVIDER_EXCEPTION_MAP.get(provider, ()):  # pragma: no branch
        exc = _import_exception(path)
        if exc is not None:
            exceptions.append(exc)
    extra = os.getenv("SA_ADDITIONAL_TRANSIENT_EXCEPTIONS")
    if extra:
        for path in extra.split(","):
            exc = _import_exception(path.strip())
            if exc is not None:
                exceptions.append(exc)
    return tuple(exceptions)


TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = _load_transient_exceptions()


def extend_transient_exceptions(*exception_types: type[BaseException]) -> None:
    """Extend ``TRANSIENT_EXCEPTIONS`` with additional types."""
    global TRANSIENT_EXCEPTIONS
    TRANSIENT_EXCEPTIONS = tuple({*TRANSIENT_EXCEPTIONS, *exception_types})


# -- Circuit breaker --------------------------------------------------------------


class CircuitBreaker:
    """Simple circuit breaker pausing after repeated failures."""

    def __init__(self, threshold: int = 5, timeout: float = 30.0) -> None:
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self._lock = asyncio.Lock()

    async def record_failure(self) -> None:
        """Record a failure and pause if the threshold is exceeded."""
        async with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                logfire.warning(
                    "Circuit breaker activated",
                    pause=self.timeout,
                    failures=self.failures,
                )
                await asyncio.sleep(self.timeout)
                self.failures = 0

    async def record_success(self) -> None:
        """Reset the failure count after a successful call."""
        async with self._lock:
            self.failures = 0


# -- Retry computation ------------------------------------------------------------

try:  # pragma: no cover - optional dependency
    from openai import RateLimitError as OpenAIRateLimitError
except Exception:  # pragma: no cover - fallback when dependency absent

    class OpenAIRateLimitError(Exception):  # type: ignore[no-redef]
        """Fallback when OpenAI SDK is absent."""


RateLimitError = OpenAIRateLimitError


def _parse_retry_datetime(retry_after: str) -> float | None:
    """Return seconds until ``retry_after`` datetime or ``None``."""
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(retry_after)
        if dt is None:
            return None
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (ImportError, ValueError, TypeError):
        return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Return ``Retry-After`` hint in seconds if available."""
    if RateLimitError is None or not isinstance(exc, RateLimitError):
        return None
    headers = getattr(getattr(exc, "response", None), "headers", {})
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return _parse_retry_datetime(retry_after)


def _compute_backoff_delay(
    exc: BaseException,
    attempt: int,
    *,
    attempts: int,
    base: float,
    cap: float,
) -> float:
    if attempt + 1 >= attempts:
        raise exc
    delay = min(cap, base * (2**attempt))
    delay *= 1 + random.random() * 0.25  # nosec B311 - jitter
    hint = _retry_after_seconds(exc)
    if hint is not None:
        delay = max(delay, hint)
    return float(delay)


async def _record_and_wait(
    exc: BaseException,
    attempt: int,
    *,
    attempts: int,
    base: float,
    cap: float,
    circuit_breaker: CircuitBreaker | None,
) -> None:
    if circuit_breaker:
        await circuit_breaker.record_failure()
    delay = _compute_backoff_delay(exc, attempt, attempts=attempts, base=base, cap=cap)
    logfire.warning("Retrying request", attempt=attempt + 1, backoff_delay=delay)
    await asyncio.sleep(delay)


async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    request_timeout: float,
    attempts: int = 6,
    base: float = 1.0,
    cap: float = 30.0,
    circuit_breaker: CircuitBreaker | None = None,
) -> tuple[T, int]:
    """Execute ``coro_factory`` with exponential backoff and jitter."""
    for attempt in range(attempts):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=request_timeout)
        except TRANSIENT_EXCEPTIONS as exc:
            await _record_and_wait(
                exc,
                attempt,
                attempts=attempts,
                base=base,
                cap=cap,
                circuit_breaker=circuit_breaker,
            )
            continue
        if circuit_breaker:
            await circuit_breaker.record_success()
        return result, attempt
    raise RuntimeError("Unreachable retry state")


__all__ = [
    "CircuitBreaker",
    "TRANSIENT_EXCEPTIONS",
    "extend_transient_exceptions",
    "with_retry",
]
