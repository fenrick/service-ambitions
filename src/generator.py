"""Async ambition generation using Pydantic AI.

This module orchestrates concurrent requests to an LLM and streams the
resulting ambitions directly to disk.  Concurrency is carefully controlled to
avoid overwhelming upstream APIs while still maximising throughput.
"""

from __future__ import annotations

import asyncio
import json
import os
import random  # noqa: F401
from asyncio import TaskGroup
from itertools import islice
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, TypeVar

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from tqdm import tqdm

from backpressure import AdaptiveSemaphore, RollingMetrics
from models import ReasoningConfig, ServiceInput
from retry_utils import (
    TRANSIENT_EXCEPTIONS as DEFAULT_TRANSIENT_EXCEPTIONS,
)
from retry_utils import (  # noqa: F401
    RateLimitError,
)
from retry_utils import (
    _with_retry as shared_with_retry,
)
from token_utils import estimate_tokens

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
SERVICES_FAILED = logfire.metric_counter("services_failed")
TOKENS_IN_FLIGHT = logfire.metric_counter("tokens_in_flight")

T = TypeVar("T")

# Local copy so tests can monkeypatch ``TRANSIENT_EXCEPTIONS`` without affecting
# other modules importing the shared implementation.
TRANSIENT_EXCEPTIONS = DEFAULT_TRANSIENT_EXCEPTIONS


async def _with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    request_timeout: float,
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
    on_retry_after: Callable[[float], None] | None = None,
    metrics: RollingMetrics | None = None,
) -> T:
    """Delegate to shared retry helper using module-level exceptions."""

    return await shared_with_retry(
        coro_factory,
        request_timeout=request_timeout,
        attempts=attempts,
        base=base,
        cap=cap,
        on_retry_after=on_retry_after,
        metrics=metrics,
        transient_exceptions=TRANSIENT_EXCEPTIONS,
        rate_limit_error=RateLimitError,
    )


async def _write_queue(
    out_path: str,
    queue: asyncio.Queue[tuple[str, str] | None],
    processed: set[str],
    progress: tqdm[Any] | None,
    flush_interval: int,
) -> None:
    """Consume lines from ``queue`` and write them to ``out_path``."""

    with open(out_path, "a", encoding="utf-8") as handle:
        line_count = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            line, svc_id = item
            await asyncio.to_thread(handle.write, f"{line}\n")
            processed.add(svc_id)
            if progress:
                progress.update(1)
            line_count += 1
            if line_count % flush_interval == 0:
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())
        await asyncio.to_thread(handle.flush)
        await asyncio.to_thread(os.fsync, handle.fileno())


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
    model across requests.  Prompts are provided per ``generate`` invocation to
    keep the class stateless between runs.  Concurrency permits can optionally
    be weighted by estimated token usage to smooth throughput across varying
    request sizes.
    """

    def __init__(
        self,
        model: Model,
        concurrency: int = 5,
        batch_size: int | None = None,
        request_timeout: float = 60,
        retries: int = 5,
        retry_base_delay: float = 0.5,
        expected_output_tokens: int = 256,
        flush_interval: int = 100,
        token_weighting: bool = True,
    ) -> None:
        """Initialize the generator.

        Args:
            model: Model used for ambition generation.
            concurrency: Maximum number of concurrent requests.
            batch_size: Number of services scheduled per batch. ``None``
                schedules all services at once.
            request_timeout: Per-request timeout in seconds.
            retries: Number of retry attempts on failure.
            retry_base_delay: Initial backoff delay in seconds.
            expected_output_tokens: Anticipated tokens in each response used to
                weight concurrency limits.
            flush_interval: Number of lines to write before forcing a flush and
                ``os.fsync`` to ensure results are durably persisted.
            token_weighting: Apply token-based weighting to semaphore permits.

        Raises:
            ValueError: If ``concurrency`` is less than one.
        """

        if concurrency < 1:
            # Guard against a misconfiguration that would deadlock the semaphore.
            raise ValueError("concurrency must be a positive integer")
        self.model = model
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.retries = retries
        self.retry_base_delay = retry_base_delay
        self.expected_output_tokens = expected_output_tokens
        self.flush_interval = flush_interval
        self.token_weighting = token_weighting
        self._prompt: str | None = None
        self._limiter: AdaptiveSemaphore | None = None
        self._metrics: RollingMetrics | None = None

    def _setup_controls(self) -> None:
        """Ensure rate limiter and metrics are initialised."""

        if self._limiter is None:
            self._limiter = AdaptiveSemaphore(self.concurrency)
        if self._metrics is None:
            self._metrics = RollingMetrics()

    def _estimate_weight(self, service: ServiceInput) -> int:
        """Return estimated token weight for ``service``."""

        service_json = service.model_dump_json()
        prompt_payload = (self._prompt or "") + service_json
        return estimate_tokens(prompt_payload, self.expected_output_tokens)

    async def _process_service_line(
        self, service: ServiceInput, transcripts_dir: Path | None
    ) -> tuple[str | None, str]:
        """Return JSON line for ``service`` or ``None`` on failure."""

        with logfire.span("process_service") as span:
            span.set_attribute("service.id", service.service_id)
            if service.customer_type:
                span.set_attribute("customer_type", service.customer_type)
            logfire.info(f"Processing service {service.name}")
            try:
                result = await self.process_service(service)
            except Exception as exc:
                msg = f"Failed to process service {service.name}: {exc}"
                logfire.error(msg)
                return None, service.service_id
            line = AmbitionModel.model_validate(result).model_dump_json()
            if transcripts_dir is not None:
                payload = {
                    "request": service.model_dump(),
                    "response": json.loads(line),
                }
                path = transcripts_dir / f"{service.service_id}.json"
                await asyncio.to_thread(
                    path.write_text,
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            return line, service.service_id

    async def _run_one(
        self,
        service: ServiceInput,
        queue: asyncio.Queue[tuple[str, str] | None],
        transcripts_dir: Path | None,
    ) -> None:
        """Process a single service and enqueue its result."""
        with logfire.span("run_one") as span:
            span.set_attribute("service.id", service.service_id)
            span.set_attribute("concurrency", self.concurrency)
            span.set_attribute("batch_size", self.batch_size)
            span.set_attribute("queue_length_before", queue.qsize())
            weight_estimate = (
                self._estimate_weight(service) if self.token_weighting else 1
            )
            limiter = self._limiter
            if limiter is None:  # pragma: no cover - defensive
                raise RuntimeError("Limiter not initialized")
            weight = weight_estimate if self.token_weighting else 1
            async with limiter(weight):
                if self._metrics and self.token_weighting:
                    self._metrics.record_start_tokens(weight_estimate)
                TOKENS_IN_FLIGHT.add(weight_estimate)
                try:
                    line, svc_id = await self._process_service_line(
                        service, transcripts_dir
                    )
                    if line is not None:
                        await queue.put((line, svc_id))
                        SERVICES_PROCESSED.add(1)
                        span.set_attribute("queue_length_after", queue.qsize())
                    else:
                        SERVICES_FAILED.add(1)
                finally:
                    if self._metrics and self.token_weighting:
                        self._metrics.record_end_tokens(weight_estimate)
                    TOKENS_IN_FLIGHT.add(-weight_estimate)

    @logfire.instrument()
    async def process_service(
        self,
        service: ServiceInput,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Return ambitions for ``service``.

        Args:
            service: Structured representation of the service under analysis.
            prompt: Optional instructions overriding the instance-wide prompt.

        Returns:
            A dictionary containing the generated ambitions.

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
        result = await _with_retry(
            lambda: agent.run(service_details, output_type=AmbitionModel),
            request_timeout=self.request_timeout,
            attempts=self.retries,
            base=self.retry_base_delay,
            on_retry_after=self._limiter.throttle if self._limiter else None,
            metrics=self._metrics,
        )
        return result.output.model_dump()

    async def _process_all(
        self,
        services: Iterable[ServiceInput],
        out_path: str,
        progress: tqdm[Any] | None = None,
        transcripts_dir: Path | None = None,
    ) -> set[str]:
        """Process ``services`` and stream results to ``out_path``.

        Args:
            services: Collection of services requiring ambition generation. The
                iterable is consumed incrementally in batches.
            out_path: Destination path for the JSONL results.
            progress: Optional progress bar updated as services complete.
            transcripts_dir: Directory to store per-service transcripts. ``None``
                disables transcript persistence.
        """
        with logfire.span("process_all") as span:
            span.set_attribute("concurrency", self.concurrency)
            span.set_attribute("batch_size", self.batch_size)
            self._setup_controls()
            queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
            processed: set[str] = set()
            writer_task = asyncio.create_task(
                _write_queue(out_path, queue, processed, progress, self.flush_interval)
            )

            services_iter = iter(services)

            while True:
                batch = list(islice(services_iter, self.batch_size))
                if not batch:
                    break
                span.set_attribute("service_ids", [svc.service_id for svc in batch])
                span.set_attribute("queue_length", queue.qsize())
                async with TaskGroup() as tg:
                    for service in batch:
                        tg.create_task(self._run_one(service, queue, transcripts_dir))

            await queue.put(None)
            await writer_task
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
            services: Collection of services requiring ambition generation. The
                iterable is consumed incrementally in batches to keep memory
                usage low.
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
            span.set_attribute("batch_size", self.batch_size)
            span.set_attribute("queue_length", 0)
            self._prompt = prompt
            try:
                self._limiter = AdaptiveSemaphore(self.concurrency)
                self._metrics = RollingMetrics()
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
                self._metrics = None

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
    settings: OpenAIResponsesModelSettings = {}
    if seed is not None:
        settings["seed"] = seed
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
