"""Shared retry helpers for transient API failures."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

import logfire

from backpressure import RollingMetrics

T = TypeVar("T")

if TYPE_CHECKING:
    from openai import APIConnectionError as OpenAIAPIConnectionError
    from openai import RateLimitError as OpenAIRateLimitError
else:
    try:
        from openai import APIConnectionError as OpenAIAPIConnectionError
        from openai import RateLimitError as OpenAIRateLimitError
    except Exception:  # pragma: no cover - optional dependency

        class OpenAIAPIConnectionError(Exception):
            """Fallback when OpenAI SDK is absent."""

        class OpenAIRateLimitError(Exception):
            """Fallback when OpenAI SDK is absent."""


RateLimitError = OpenAIRateLimitError

TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    OpenAIAPIConnectionError,
    OpenAIRateLimitError,
)


def _retry_after_seconds(
    exc: BaseException, rate_limit_error: type[BaseException] | None
) -> float | None:
    """Return ``Retry-After`` hint in seconds if provided by ``exc``."""

    if rate_limit_error is not None and isinstance(exc, rate_limit_error):
        headers = getattr(getattr(exc, "response", None), "headers", {})
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            try:
                from datetime import datetime, timezone
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(retry_after)
                if dt is None:
                    return None
                return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                return None
    return None


async def _with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    request_timeout: float,
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
    on_retry_after: Callable[[float], None] | None = None,
    metrics: RollingMetrics | None = None,
    transient_exceptions: tuple[type[BaseException], ...] = TRANSIENT_EXCEPTIONS,
    rate_limit_error: type[BaseException] | None = RateLimitError,
) -> T:
    """Execute ``coro_factory`` with exponential backoff and jitter."""

    def _handle_retry(exc: BaseException, attempt: int) -> float:
        if metrics:
            metrics.record_error()
        if attempt == attempts - 1:
            raise
        delay = min(cap, base * (2**attempt))
        delay *= 1 + random.random() * 0.25
        hint = _retry_after_seconds(exc, rate_limit_error)
        if hint is not None:
            delay = max(delay, hint)
            if on_retry_after:
                on_retry_after(hint)
        return delay

    for attempt in range(attempts):
        if metrics:
            metrics.record_request()
        try:
            return await asyncio.wait_for(coro_factory(), timeout=request_timeout)
        except transient_exceptions as exc:
            delay = _handle_retry(exc, attempt)
            logfire.warning(
                "Retrying request", attempt=attempt + 1, backoff_delay=delay
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable retry state")
