# SPDX-License-Identifier: MIT
"""Lightweight global LLM execution queue.

This module provides a minimal async queue abstraction to centralise
concurrency control for LLM calls. It is intentionally simple for the initial
scaffold and can be extended to include retries, circuit breaking and richer
observability.

Usage is feature-flagged via settings and integrated in ConversationSession so
existing call sites do not need to change. When disabled, the queue is not
used and execution proceeds as before.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Protocol, TypeVar

import logfire

from .retry import CircuitBreaker, with_retry

T = TypeVar("T", covariant=True)


class _CoroFactory(Protocol[T]):
    def __call__(self) -> Awaitable[T]: ...


@dataclass
class LLMTaskMeta:
    """Optional metadata for tracing/metrics."""

    stage: str | None = None
    model_name: str | None = None
    service_id: str | None = None


class LLMQueue:
    """Bounded concurrency queue for LLM invocations.

    In addition to a semaphore limiting concurrent executions, ``LLMQueue`` can
    optionally retry transient failures with exponential backoff and a
    circuit-breaker to pause after repeated faults.
    """

    def __init__(self, max_concurrency: int = 3) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrency)
        self._inflight = logfire.metric_gauge("llm_queue_inflight")
        self._submitted = logfire.metric_counter("llm_queue_submitted")
        self._completed = logfire.metric_counter("llm_queue_completed")
        self._breaker = CircuitBreaker()

    @asynccontextmanager
    async def _slot(self) -> Any:
        await self._sem.acquire()
        try:
            self._inflight.set(self._sem._value)
            yield
        finally:
            self._sem.release()
            self._inflight.set(self._sem._value)

    async def submit(
        self,
        factory: _CoroFactory[T],
        *,
        meta: LLMTaskMeta | None = None,
        retry: bool = False,
        attempts: int = 6,
        base: float = 1.0,
        cap: float = 30.0,
    ) -> T:
        """Run ``factory`` under the queue's concurrency gate and return its result.

        Args:
            factory: Zero-arg coroutine factory for the actual LLM call.
            meta: Optional metadata recorded for tracing.
            retry: Enable retry/backoff on transient errors when ``True``.
            attempts: Maximum retry attempts when ``retry`` is enabled.
            base: Initial delay in seconds for backoff when ``retry`` is enabled.
            cap: Maximum backoff delay in seconds when ``retry`` is enabled.
        """
        self._submitted.add(1)
        stage = getattr(meta, "stage", None)
        model_name = getattr(meta, "model_name", None)
        service_id = getattr(meta, "service_id", None)
        span_attrs = {
            k: v
            for k, v in {
                "stage": stage,
                "model_name": model_name,
                "service_id": service_id,
            }.items()
            if v is not None
        }
        with logfire.span("llm_queue.submit", attributes=span_attrs):
            async with self._slot():
                if retry:
                    result, _ = await with_retry(
                        factory,
                        request_timeout=60.0,
                        attempts=attempts,
                        base=base,
                        cap=cap,
                        circuit_breaker=self._breaker,
                    )
                else:
                    result = await factory()
                self._completed.add(1)
                return result


__all__ = ["LLMQueue", "LLMTaskMeta"]
