# SPDX-License-Identifier: MIT
"""Async ambition generation using Pydantic AI.

This module orchestrates concurrent requests to an LLM and streams the
resulting ambitions directly to disk.  Concurrency is carefully controlled to
avoid overwhelming upstream APIs while still maximising throughput.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from asyncio import Semaphore, TaskGroup
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, TextIO, TypeVar

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from tqdm import tqdm

from canonical import canonicalise_record
from models import ReasoningConfig, ServiceInput

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
SERVICES_FAILED = logfire.metric_counter("services_failed")

T = TypeVar("T")


# Transient failures that warrant a retry. Provider specific errors are optional
# to avoid hard dependencies when the SDK is absent.
if TYPE_CHECKING:
    from openai import APIConnectionError as OpenAIAPIConnectionError
    from openai import RateLimitError as OpenAIRateLimitError
else:
    try:
        from openai import APIConnectionError as OpenAIAPIConnectionError
        from openai import RateLimitError as OpenAIRateLimitError
    except Exception:

        class OpenAIAPIConnectionError(Exception):
            """Fallback when OpenAI SDK is absent."""

        class OpenAIRateLimitError(Exception):
            """Fallback when OpenAI SDK is absent."""


TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    OpenAIAPIConnectionError,
    OpenAIRateLimitError,
)

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
    except Exception:
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


async def _with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    request_timeout: float,
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
    on_retry_after: Callable[[float], None] | None = None,
) -> tuple[T, int]:
    """Execute ``coro_factory`` with exponential backoff and jitter.

    Each attempt is wrapped in ``asyncio.wait_for`` to enforce a sixty second
    timeout. Only transient network or provider errors listed in
    ``TRANSIENT_EXCEPTIONS`` trigger a retry with exponential backoff capped at
    ``cap`` seconds and up to twenty-five percent random jitter.

    Args:
        coro_factory: Factory returning the coroutine to execute.
        attempts: Maximum number of attempts before giving up.
        base: Initial delay in seconds used for backoff.
        cap: Maximum backoff in seconds.

    Returns:
        The result of the successful coroutine.

    Raises:
        Exception: Propagates the last exception if all attempts fail or if the
            error is not transient.
    """

    def _handle_retry(exc: BaseException, attempt: int) -> float:
        if attempt == attempts - 1:
            raise
        delay = min(cap, base * (2**attempt))
        delay *= 1 + random.random() * 0.25
        hint = _retry_after_seconds(exc)
        if hint is not None:
            delay = max(delay, hint)
            if on_retry_after:
                on_retry_after(hint)
        return delay

    for attempt in range(attempts):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=request_timeout)
        except TRANSIENT_EXCEPTIONS as exc:
            delay = _handle_retry(exc, attempt)
            logfire.warning(
                "Retrying request",
                attempt=attempt + 1,
                backoff_delay=delay,
            )
            await asyncio.sleep(delay)
            continue
        return result, attempt
    raise RuntimeError("Unreachable retry state")


class AmbitionModel(BaseModel):
    """Structured ambitions response allowing arbitrary keys.

    The model mirrors the flexible JSON returned by the agent.  Allowing extra
    fields means callers are free to introduce new ambition categories without
    having to update the schema.
    """

    model_config = {"extra": "allow"}


class ServiceAmbitionGenerator:
    """Generate ambitions for services using a Pydantic AI model.

    Instances manage a bounded pool of concurrent workers and reuse the input
    model across requests. Prompts are provided per ``generate`` invocation to
    keep the class stateless between runs.
    """

    def __init__(
        self,
        model: Model,
        concurrency: int = 5,
        request_timeout: float = 60,
        retries: int = 5,
        retry_base_delay: float = 0.5,
    ) -> None:
        """Initialize the generator.

        Args:
            model: Model used for ambition generation.
            concurrency: Maximum number of concurrent requests.
            request_timeout: Per-request timeout in seconds.
            retries: Number of retry attempts on failure.
            retry_base_delay: Initial backoff delay in seconds.

        Raises:
            ValueError: If ``concurrency`` is less than one.
        """

        if concurrency < 1:
            # Guard against a misconfiguration that would deadlock the semaphore.
            raise ValueError("concurrency must be a positive integer")
        self.model = model
        self.concurrency = concurrency
        self.request_timeout = request_timeout
        self.retries = retries
        self.retry_base_delay = retry_base_delay
        self._prompt: str | None = None
        self._limiter: Semaphore | None = None

    def _setup_controls(self) -> None:
        """Ensure rate limiter is initialised."""

        if self._limiter is None:
            self._limiter = Semaphore(self.concurrency)

    async def _process_service_line(
        self, service: ServiceInput
    ) -> tuple[dict[str, Any] | None, str, int, int, str]:
        """Return ambitions payload and metrics or ``None`` on failure."""

        logfire.info(f"Processing service {service.name}")
        try:
            result, tokens, retries = await self.process_service(service)
        except Exception as exc:
            msg = f"Failed to process service {service.name}: {exc}"
            logfire.error(msg)
            return None, service.service_id, 0, self.retries - 1, "error"
        return result, service.service_id, tokens, retries, "success"

    async def _run_one(
        self,
        service: ServiceInput,
        handle: TextIO,
        lock: asyncio.Lock,
        processed: set[str],
        progress: tqdm[Any] | None,
        transcripts_dir: Path | None,
    ) -> None:
        """Process a single service and write its result to ``handle``.

        Writes are protected by ``lock`` to ensure each JSON line is appended
        atomically.
        """
        limiter = self._limiter
        if limiter is None:  # pragma: no cover - defensive
            raise RuntimeError("Limiter not initialized")
        with logfire.span("service") as span:
            span.set_attribute("service.id", service.service_id)
            tokens, retries, status = await self._execute_service(
                limiter,
                service,
                transcripts_dir,
                handle,
                lock,
                processed,
                progress,
            )
            span.set_attribute("tokens.total", tokens)
            span.set_attribute("retries", retries)
            span.set_attribute("status", status)

    async def _execute_service(
        self,
        limiter: Semaphore,
        service: ServiceInput,
        transcripts_dir: Path | None,
        handle: TextIO,
        lock: asyncio.Lock,
        processed: set[str],
        progress: tqdm[Any] | None,
    ) -> tuple[int, int, str]:
        """Process ``service`` and append its result to ``handle``.

        ``process_service`` is rate limited via ``limiter`` while the final
        JSONL write is protected by ``lock`` to keep output consistent.
        """

        async with limiter:
            payload, svc_id, tokens, retries, status = await self._process_service_line(
                service
            )

        # Token usage is tracked via logfire metrics.

        if payload is not None:
            record = canonicalise_record(
                AmbitionModel.model_validate(payload).model_dump(mode="json")
            )
            line = json.dumps(
                record, separators=(",", ":"), ensure_ascii=False, sort_keys=True
            )
            if transcripts_dir is not None:
                transcript = {
                    "request": service.model_dump(),
                    "response": record,
                }
                data = json.dumps(
                    transcript,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                path = transcripts_dir / f"{svc_id}.json"
                await asyncio.to_thread(path.write_text, data, encoding="utf-8")

            async with lock:
                await asyncio.to_thread(handle.write, f"{line}\n")
                processed.add(svc_id)
                if progress:
                    progress.update(1)
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())

            SERVICES_PROCESSED.add(1)
        else:
            SERVICES_FAILED.add(1)
        return tokens, retries, status

    async def process_service(
        self,
        service: ServiceInput,
        prompt: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Return ambitions and token usage for ``service``.

        Args:
            service: Structured representation of the service under analysis.
            prompt: Optional instructions overriding the instance-wide prompt.

        Returns:
            Tuple of the generated ambitions and the total tokens consumed.

        Raises:
            ValueError: If no prompt was supplied via ``prompt`` or ``generate``.

        The function spins up a fresh agent for each call to avoid sharing
        mutable state across tasks. The JSON representation of ``service`` is
        passed as the prompt input and the model response is converted back
        into a standard dictionary for downstream processing.
        """

        # Prefer the per-call prompt but fall back to the cached version.
        instructions = prompt if prompt is not None else self._prompt
        if instructions is None:
            # Without instructions the agent cannot operate.
            raise ValueError("prompt must be provided")
        agent = Agent(self.model, instructions=instructions)
        service_details = service.model_dump_json()
        result, retries = await _with_retry(
            lambda: agent.run(service_details, output_type=AmbitionModel),
            request_timeout=self.request_timeout,
            attempts=self.retries,
            base=self.retry_base_delay,
            on_retry_after=getattr(self._limiter, "throttle", None),
        )
        usage = result.usage()
        tokens = usage.total_tokens or 0
        return result.output.model_dump(), tokens, retries

    async def _process_all(
        self,
        services: Iterable[ServiceInput],
        out_path: str,
        progress: tqdm[Any] | None = None,
        transcripts_dir: Path | None = None,
    ) -> set[str]:
        """Process ``services`` and stream results to ``out_path``.

        Args:
            services: Collection of services requiring ambition generation.
            out_path: Destination path for the JSONL results.
            progress: Optional progress bar updated as services complete.
            transcripts_dir: Directory to store per-service transcripts. ``None``
                disables transcript persistence.
        """
        with logfire.span("process_all") as span:
            span.set_attribute("concurrency", self.concurrency)
            self._setup_controls()
            processed: set[str] = set()
            lock = asyncio.Lock()
            handle = await asyncio.to_thread(open, out_path, "a", encoding="utf-8")
            try:
                async with TaskGroup() as tg:
                    for service in services:
                        tg.create_task(
                            self._run_one(
                                service,
                                handle,
                                lock,
                                processed,
                                progress,
                                transcripts_dir,
                            )
                        )
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())
            finally:
                await asyncio.to_thread(handle.close)
            return processed

    async def generate_async(
        self,
        services: Iterable[ServiceInput],
        prompt: str,
        output_path: str,
        progress: tqdm[Any] | None = None,
        transcripts_dir: Path | None = None,
    ) -> set[str]:
        """Process ``services`` lazily and write ambitions to ``output_path``.

        Args:
            services: Collection of services requiring ambition generation.
            prompt: Instructions guiding the model's output.
            output_path: Destination path for the JSONL results.
            progress: Optional progress bar updated as services complete.
            transcripts_dir: Optional directory used to persist per-service
                request/response transcripts.

        Side Effects:
            Creates/overwrites ``output_path`` with one JSON record per line and
            optionally stores transcripts in ``transcripts_dir``.

        Raises:
            OSError: Propagated if writing to ``output_path`` fails.
        """

        with logfire.span("generate_async") as span:
            span.set_attribute("concurrency", self.concurrency)
            self._prompt = prompt
            try:
                self._limiter = Semaphore(self.concurrency)
                processed = await self._process_all(
                    services, output_path, progress, transcripts_dir=transcripts_dir
                )
                span.set_attribute("service_ids", list(processed))
                return processed
            except Exception as exc:
                logfire.error(f"Failed to write results to {output_path}: {exc}")
                raise
            finally:
                self._prompt = None
                self._limiter = None

    async def generate(
        self,
        services: Iterable[ServiceInput],
        prompt: str,
        output_path: str,
    ) -> set[str]:
        """Backward compatible wrapper for ``generate_async``."""

        return await self.generate_async(services, prompt, output_path)


def build_model(
    model_name: str,
    api_key: str,
    *,
    seed: int | None = None,
    reasoning: ReasoningConfig | None = None,
    web_search: bool = False,
) -> Model:
    """Return a configured Pydantic AI model.

    Args:
        model_name: Identifier of the OpenAI model to use.
        api_key: Optional API key for authenticating with OpenAI.
        seed: Optional seed for deterministic model responses.
        reasoning: Optional reasoning configuration passed through to the model.
        web_search: Enable OpenAI web search tooling for model browsing. Defaults to
            ``False``.

    Returns:
        A ready-to-use ``Model`` instance.

    Side Effects:
        Sets ``OPENAI_API_KEY`` in the environment if ``api_key`` is provided.
    """

    if api_key:
        # Expose the key via environment variables for model libraries that
        # expect it there rather than accepting it directly.
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    # Allow callers to pass provider-prefixed names such as ``openai:gpt-4``.
    model_name = model_name.split(":", 1)[-1]
    settings: OpenAIResponsesModelSettings = {
        "temperature": 0,
        "top_p": 1,
        "seed": seed if seed is not None else 0,
    }
    if web_search:
        # Allow optional access to the ``web_search_preview`` tool which provides
        # browsing capabilities. Disabling keeps runs deterministic and avoids
        # additional cost for schema-only generation.
        settings["openai_builtin_tools"] = [{"type": "web_search_preview"}]
    if reasoning:
        # Map each supported reasoning field to the ``openai_reasoning_*``
        # parameter. Extra fields allowed by ``ReasoningConfig`` are ignored to
        # keep type checking strict.
        if reasoning.effort is not None:
            settings["openai_reasoning_effort"] = reasoning.effort
        if reasoning.summary is not None:
            settings["openai_reasoning_summary"] = reasoning.summary
    return OpenAIResponsesModel(model_name, settings=settings)


__all__ = ["AmbitionModel", "ServiceAmbitionGenerator", "build_model"]
