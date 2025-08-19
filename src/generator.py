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
from asyncio import TaskGroup
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, TypeVar

import logfire  # type: ignore[import-not-found]
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from tqdm import tqdm

from backpressure import AdaptiveSemaphore, RollingMetrics
from models import ReasoningConfig, ServiceInput
from token_utils import estimate_tokens

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


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Return ``Retry-After`` hint in seconds if available.

    Providers may include a ``Retry-After`` header on rate limit errors to
    indicate how long clients should pause before retrying.  When present, this
    delay is returned in seconds; otherwise ``None`` is returned.

    Args:
        exc: Exception raised by the provider.

    Returns:
        Suggested delay in seconds or ``None`` if unavailable.
    """

    if RateLimitError is not None and isinstance(exc, RateLimitError):
        headers = getattr(getattr(exc, "response", None), "headers", {})
        # Header keys may vary in capitalisation
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
) -> T:
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

    for attempt in range(attempts):
        if metrics:
            metrics.record_request()
        try:
            return await asyncio.wait_for(coro_factory(), timeout=request_timeout)
        except TRANSIENT_EXCEPTIONS as exc:
            if metrics:
                metrics.record_error()
            if attempt == attempts - 1:
                raise
            delay = min(cap, base * (2**attempt))
            delay *= 1 + random.random() * 0.25
            hint = _retry_after_seconds(exc)
            if hint is not None:
                delay = max(delay, hint)
                if on_retry_after:
                    on_retry_after(hint)
            await asyncio.sleep(delay)
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
    model across requests.  Prompts are provided per ``generate`` invocation to
    keep the class stateless between runs.
    """

    @logfire.instrument()
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

    @logfire.instrument()
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
        if self._limiter is None:
            self._limiter = AdaptiveSemaphore(self.concurrency)
        if self._metrics is None:
            self._metrics = RollingMetrics()
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        processed: set[str] = set()

        async def writer() -> None:
            """Consume lines from ``queue`` and write them to disk.

            After every ``flush_interval`` lines the file handle is flushed and
            synced to disk to reduce the risk of data loss if the process
            crashes.  This provides stronger durability guarantees than relying
            solely on the OS to flush buffers asynchronously.
            """

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
                        # Updating the progress bar can block, so keep it in writer.
                        progress.update(1)
                    line_count += 1
                    if line_count % self.flush_interval == 0:
                        # ``flush`` pushes Python's buffers to the OS and
                        # ``fsync`` asks the OS to commit those bytes to disk.
                        await asyncio.to_thread(handle.flush)
                        await asyncio.to_thread(os.fsync, handle.fileno())
                # Final sync in case the last batch didn't align with
                # ``flush_interval``.
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())

        async def run_one(service: ServiceInput) -> None:
            """Process a single service and enqueue its JSON line."""

            weight_estimate = 1
            if self.token_weighting:
                # Estimate the total tokens for the request and anticipated
                # response so larger requests reserve more semaphore capacity.
                service_json = service.model_dump_json()
                prompt_payload = (self._prompt or "") + service_json
                weight_estimate = estimate_tokens(
                    prompt_payload, self.expected_output_tokens
                )

            limiter = self._limiter
            if limiter is None:  # pragma: no cover - defensive
                raise RuntimeError("Limiter not initialized")
            weight = weight_estimate if self.token_weighting else 1
            async with limiter(weight):
                if self._metrics and self.token_weighting:
                    self._metrics.record_start_tokens(weight_estimate)
                try:
                    # Record service metadata on the span so that traces include the
                    # originating service identifier and customer segment.
                    with logfire.span("process_service") as span:
                        span.set_attribute("service.id", service.service_id)
                        if service.customer_type:
                            span.set_attribute("customer_type", service.customer_type)
                        logfire.info(f"Processing service {service.name}")
                        try:
                            result = await self.process_service(service)
                        except Exception as exc:
                            # Continue processing other services but record the failure.
                            msg = f"Failed to process service {service.name}: {exc}"
                            logfire.error(msg)
                            return
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
                        await queue.put((line, service.service_id))
                finally:
                    if self._metrics and self.token_weighting:
                        self._metrics.record_end_tokens(weight_estimate)

        writer_task = asyncio.create_task(writer())
        services_iter = iter(services)

        while True:
            # Process services in bounded batches to avoid creating excessive
            # pending tasks when the input iterable is large.
            batch = list(islice(services_iter, self.batch_size))
            if not batch:
                break  # Exhausted the input iterator
            async with TaskGroup() as tg:
                for service in batch:
                    tg.create_task(run_one(service))

        await queue.put(None)
        await writer_task
        return processed

    @logfire.instrument()
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

        self._prompt = prompt
        try:
            self._limiter = AdaptiveSemaphore(self.concurrency)
            self._metrics = RollingMetrics()
            return await self._process_all(
                services, output_path, progress, transcripts_dir=transcripts_dir
            )
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


@logfire.instrument()
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
