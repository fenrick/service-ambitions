"""Conversational session wrapper for LLM interactions.

This module exposes :class:`ConversationSession`, a light abstraction over a
Pydantic-AI ``Agent``. The session records message history so that each prompt
retains prior context and can be seeded with service details via
``add_parent_materials``. The :meth:`ask` method delegates to the underlying
agent without relying on asynchronous execution.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Awaitable, Callable, TypeVar, overload

import logfire
from pydantic_ai import Agent, messages

from backpressure import RollingMetrics
from models import ServiceInput
from redaction import redact_pii
from stage_metrics import record_stage_metrics
from token_utils import estimate_cost, estimate_tokens

PROMPTS_SENT = logfire.metric_counter("prompts_sent")
TOKENS_CONSUMED = logfire.metric_counter("tokens_consumed")


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI agent.

    The session stores message history so that subsequent prompts include
    previous context. Additional materials about the service under discussion
    may be seeded using :meth:`add_parent_materials`.
    """

    def __init__(
        self,
        client: Agent,
        *,
        stage: str | None = None,
        diagnostics: bool = False,
        log_prompts: bool = False,
        redact_prompts: bool = False,
        metrics: RollingMetrics | None = None,
    ) -> None:
        """Initialise the session with a configured LLM client.

        Args:
            client: Pydantic-AI ``Agent`` used for exchanges with the model.
            stage: Optional name of the generation stage for observability.
            diagnostics: Enable detailed logging and span creation.
            log_prompts: Debug log prompt text when ``diagnostics`` is ``True``.
            redact_prompts: Redact prompt text before logging when ``log_prompts``.
            metrics: Optional rolling metrics recorder for request telemetry.
        """

        self.client = client
        self.stage = stage
        self.diagnostics = diagnostics
        self.log_prompts = log_prompts if diagnostics else False
        self.redact_prompts = redact_prompts
        self.metrics = metrics
        self._history: list[messages.ModelMessage] = []

    def add_parent_materials(self, service_input: ServiceInput) -> None:
        """Seed the conversation with details about the target service.

        Args:
            service_input: Metadata describing the service being evaluated.

        Side Effects:
            Appends a user prompt containing the service context to the session
            history.
        """
        ctx = "SERVICE_CONTEXT:\n" + service_input.model_dump_json()
        stored_ctx = redact_pii(ctx) if self.redact_prompts else ctx
        if self.diagnostics and self.log_prompts:
            logfire.debug(f"Adding service material to history: {stored_ctx}")
        self._history.append(
            messages.ModelRequest(parts=[messages.UserPromptPart(stored_ctx)])
        )

    def derive(self) -> "ConversationSession":
        """Return a new session copying the current history."""

        clone = ConversationSession(
            self.client,
            stage=self.stage,
            diagnostics=self.diagnostics,
            log_prompts=self.log_prompts,
            redact_prompts=self.redact_prompts,
            metrics=self.metrics,
        )
        clone._history = list(self._history)
        return clone

    def _record_new_messages(self, msgs: list[messages.ModelMessage]) -> None:
        """Append ``msgs`` to history, redacting user prompts when enabled."""

        if not self.redact_prompts:
            self._history.extend(msgs)
            return
        sanitised: list[messages.ModelMessage] = []
        for msg in msgs:
            if isinstance(msg, messages.ModelRequest):
                parts: list[Any] = []
                for part in msg.parts:
                    if isinstance(part, messages.UserPromptPart):
                        content = redact_pii(str(part.content))
                        parts.append(messages.UserPromptPart(content))
                    else:
                        parts.append(part)
                sanitised.append(messages.ModelRequest(parts=parts))
            else:
                sanitised.append(msg)
        self._history.extend(sanitised)

    @contextmanager
    def _prepare_span(
        self,
        span_name: str,
        stage: str,
        model_name: str,
        prompt_token_estimate: int,
    ) -> Any:
        """Create a logging span and attach common attributes."""

        span_ctx = logfire.span(span_name) if self.diagnostics else nullcontext()
        with span_ctx as span:
            if span and self.diagnostics:
                # annotate span for observability when diagnostics enabled
                span.set_attribute("stage", stage)
                span.set_attribute("model_name", model_name)
                span.set_attribute("prompt_token_estimate", prompt_token_estimate)
            yield span

    def _log_prompt(self, prompt: str) -> None:
        """Optionally log the prompt text for debugging."""

        if self.diagnostics and self.log_prompts:
            # redact prompt if requested before logging
            logged_prompt = redact_pii(prompt) if self.redact_prompts else prompt
            logfire.debug(f"Sending prompt: {logged_prompt}")

    def _handle_success(
        self,
        result: Any,
        stage: str,
        model_name: str,
        prompt_token_estimate: int,
    ) -> tuple[Any, int, float]:
        """Process a successful model invocation."""

        self._record_new_messages(list(result.new_messages()))
        usage = result.usage()
        tokens = usage.total_tokens or 0
        cost = estimate_cost(model_name, tokens)
        logfire.info(
            "Prompt succeeded",
            stage=stage,
            model_name=model_name,
            total_tokens=tokens,
            prompt_token_estimate=prompt_token_estimate,
            estimated_cost=cost,
        )
        return result.output, tokens, cost

    def _handle_failure(
        self,
        exc: Exception,
        stage: str,
        model_name: str,
        prompt_token_estimate: int,
        tokens: int,
        cost: float,
    ) -> bool:
        """Log failure details and record metrics."""

        error_429 = "429" in getattr(exc, "args", ("",))[0]
        if self.metrics:
            # tag metrics when the model rate limits
            self.metrics.record_error(is_429=error_429)
        logfire.error(
            "Prompt failed",
            stage=stage,
            model_name=model_name,
            total_tokens=tokens,
            prompt_token_estimate=prompt_token_estimate,
            estimated_cost=cost,
            error=str(exc),
        )
        return error_429

    def _finalise_metrics(
        self,
        span: Any,
        stage: str,
        tokens: int,
        cost: float,
        start: float,
        error_429: bool,
        prompt_token_estimate: int,
    ) -> None:
        """Record latency, token usage and stage metrics."""

        duration = time.monotonic() - start
        if self.metrics:
            # persist timing and token consumption metrics
            self.metrics.record_latency(duration)
            self.metrics.record_end_tokens(tokens)
        if span and self.diagnostics:
            span.set_attribute("total_tokens", tokens)
            span.set_attribute("estimated_cost", cost)
        TOKENS_CONSUMED.add(tokens)
        record_stage_metrics(
            stage, tokens, cost, duration, error_429, prompt_token_estimate
        )

    T = TypeVar("T")

    async def _ask_common(
        self,
        prompt: str,
        output_type: type[T] | None,
        runner: Callable[
            [str, list[messages.ModelMessage], type[T] | None], Awaitable[Any]
        ],
        span_name: str,
    ) -> T | str:
        stage = self.stage or "unknown"
        model_name = getattr(getattr(self.client, "model", None), "model_name", "")
        PROMPTS_SENT.add(1)
        tokens = 0
        cost = 0.0
        error_429 = False
        prompt_token_estimate = estimate_tokens(prompt, 0)
        start = time.monotonic()
        if self.metrics:
            # record request initiation metrics
            self.metrics.record_request()
        with self._prepare_span(
            span_name, stage, model_name, prompt_token_estimate
        ) as span:
            try:
                self._log_prompt(prompt)
                result = await runner(prompt, self._history, output_type)
                output, tokens, cost = self._handle_success(
                    result, stage, model_name, prompt_token_estimate
                )
                return output
            except Exception as exc:  # pragma: no cover - defensive logging
                error_429 = self._handle_failure(
                    exc,
                    stage,
                    model_name,
                    prompt_token_estimate,
                    tokens,
                    cost,
                )
                raise
            finally:
                self._finalise_metrics(
                    span,
                    stage,
                    tokens,
                    cost,
                    start,
                    error_429,
                    prompt_token_estimate,
                )

    @overload
    def ask(self, prompt: str) -> str: ...

    @overload
    def ask(self, prompt: str, output_type: type[T]) -> T: ...

    def ask(self, prompt: str, output_type: type[T] | None = None) -> T | str:
        """Return the agent's response to ``prompt``."""

        async def runner(
            user_prompt: str,
            message_history: list[messages.ModelMessage],
            out_type: type[Any] | None,
        ) -> Any:
            return self.client.run_sync(
                user_prompt, message_history=message_history, output_type=out_type
            )

        return asyncio.run(
            self._ask_common(
                prompt, output_type, runner, self.stage or "ConversationSession.ask"
            )
        )

    @overload
    async def ask_async(self, prompt: str) -> str: ...

    @overload
    async def ask_async(self, prompt: str, output_type: type[T]) -> T: ...

    async def ask_async(
        self, prompt: str, output_type: type[T] | None = None
    ) -> T | str:
        """Asynchronously return the agent's response to ``prompt``."""

        async def runner(
            user_prompt: str,
            message_history: list[messages.ModelMessage],
            out_type: type[Any] | None,
        ) -> Any:
            return await self.client.run(
                user_prompt, message_history=message_history, output_type=out_type
            )

        return await self._ask_common(
            prompt,
            output_type,
            runner,
            self.stage or "ConversationSession.ask_async",
        )


__all__ = ["ConversationSession"]
