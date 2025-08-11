"""Async ambition generation using Pydantic AI.

This module orchestrates concurrent requests to an LLM and streams the
resulting ambitions directly to disk.  Concurrency is carefully controlled to
avoid overwhelming upstream APIs while still maximising throughput.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from asyncio import Lock, Semaphore, TaskGroup
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, TypeVar

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from tqdm import tqdm

from models import ServiceInput

logger = logging.getLogger(__name__)


T = TypeVar("T")


# Transient failures that warrant a retry. Provider specific errors are optional
# to avoid hard dependencies when the SDK is absent.
if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from openai import (
        APIConnectionError as OpenAIAPIConnectionError,
    )
    from openai import (
        RateLimitError as OpenAIRateLimitError,
    )
else:  # pragma: no cover - openai may be missing
    try:
        from openai import (
            APIConnectionError as OpenAIAPIConnectionError,
        )
        from openai import (
            RateLimitError as OpenAIRateLimitError,
        )
    except Exception:

        class OpenAIAPIConnectionError(Exception):  # type: ignore[empty-body]
            """Fallback when OpenAI SDK is absent."""

        class OpenAIRateLimitError(Exception):  # type: ignore[empty-body]
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
            except Exception:  # pragma: no cover - best effort parsing
                return None
    return None


async def _with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    request_timeout: float,
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
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
        try:
            return await asyncio.wait_for(coro_factory(), timeout=request_timeout)
        except TRANSIENT_EXCEPTIONS as exc:  # pragma: no cover - handled retries
            if attempt == attempts - 1:
                raise
            delay = min(cap, base * (2**attempt))
            delay *= 1 + random.random() * 0.25
            hint = _retry_after_seconds(exc)
            if hint is not None:
                delay = max(delay, hint)
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
        )
        return result.output.model_dump()

    @logfire.instrument()
    async def _process_all(
        self,
        services: Iterable[ServiceInput],
        out_path: str,
        progress: tqdm[Any] | None = None,
    ) -> set[str]:
        """Process ``services`` and stream results to ``out_path``.

        Args:
            services: Collection of services requiring ambition generation.
            out_path: Destination path for the JSONL results.
            progress: Optional progress bar updated as services complete.
        """
        sem = Semaphore(self.concurrency)
        lock = Lock()
        processed: set[str] = set()

        async def run_one(service: ServiceInput) -> None:
            """Process a single service and append its JSON line to disk.

            Args:
                service: Service to analyse.
            """

            async with sem:
                # Record service metadata on the span so that traces include the
                # originating service identifier and customer segment.
                with logfire.span("process_service") as span:
                    span.set_attribute("service.id", service.service_id)
                    if service.customer_type:
                        span.set_attribute("customer_type", service.customer_type)
                    logger.info("Processing service %s", service.name)
                    try:
                        result = await self.process_service(service)
                    except Exception as exc:  # pylint: disable=broad-except
                        # Continue processing other services but record the failure.
                        logger.error(
                            "Failed to process service %s: %s",
                            service.name,
                            exc,
                        )
                        return
                    line = AmbitionModel.model_validate(result).model_dump_json()
                    async with lock:
                        handle.write(f"{line}\n")
                        processed.add(service.service_id)
                    if progress:
                        # Advance the progress bar for each completed service.
                        progress.update(1)

        with open(out_path, "a", encoding="utf-8") as handle:
            async with TaskGroup() as tg:
                for service in services:
                    tg.create_task(run_one(service))
            handle.flush()
        return processed

    @logfire.instrument()
    async def generate_async(
        self,
        services: Iterable[ServiceInput],
        prompt: str,
        output_path: str,
        progress: tqdm[Any] | None = None,
    ) -> set[str]:
        """Process ``services`` lazily and write ambitions to ``output_path``.

        Args:
            services: Collection of services requiring ambition generation. The
                iterable is consumed incrementally to keep memory usage low.
            prompt: Instructions guiding the model's output.
            output_path: Destination path for the JSONL results.
            progress: Optional progress bar updated as services complete.

        Side Effects:
            Creates/overwrites ``output_path`` with one JSON record per line.

        Raises:
            OSError: Propagated if writing to ``output_path`` fails.
        """

        self._prompt = prompt
        try:
            return await self._process_all(services, output_path, progress)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to write results to %s: %s", output_path, exc)
            raise
        finally:
            self._prompt = None

    async def generate(
        self,
        services: Iterable[ServiceInput],
        prompt: str,
        output_path: str,
    ) -> set[str]:
        """Backward compatible wrapper for ``generate_async``."""

        return await self.generate_async(services, prompt, output_path)


@logfire.instrument()
def build_model(model_name: str, api_key: str, *, seed: int | None = None) -> Model:
    """Return a configured Pydantic AI model.

    Args:
        model_name: Identifier of the OpenAI model to use.
        api_key: Optional API key for authenticating with OpenAI.
        seed: Optional seed for deterministic model responses.

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
    extra = {"seed": seed} if seed is not None else {}
    settings = OpenAIResponsesModelSettings(
        openai_builtin_tools=[{"type": "web_search_preview"}],
        openai_reasoning_summary="concise",
        openai_reasoning_effort="medium",
        **extra,  # type: ignore[typeddict-item]
    )
    return OpenAIResponsesModel(model_name, settings=settings)


__all__ = ["ServiceAmbitionGenerator", "build_model"]
