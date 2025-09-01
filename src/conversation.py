# SPDX-License-Identifier: MIT
"""Conversational session wrapper for LLM interactions.

This module exposes :class:`ConversationSession`, a light abstraction over a
Pydantic-AI ``Agent``. The session records message history so that each prompt
retains prior context and can be seeded with service details via
``add_parent_materials``. The :meth:`ask` method delegates to the underlying
agent without relying on asynchronous execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar, cast, overload

import logfire
from pydantic_ai import Agent, messages
from pydantic_core import from_json, to_json

from mapping import cache_write_json_atomic
from models import ServiceInput
from settings import load_settings


def _prompt_cache_key(prompt: str, model: str, stage: str) -> str:
    """Return a stable cache key for ``prompt`` and ``model``."""

    data = to_json([prompt, model, stage]).decode()
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]


def _prompt_cache_path(
    service: str, stage: str, key: str, feature_id: str | None = None
) -> Path:
    """Return cache path for ``key`` grouped by context and identifiers."""

    try:
        settings = load_settings()
        cache_root = settings.cache_dir
        context = settings.context_id
    except Exception:  # pragma: no cover - fallback when settings unavailable
        cache_root = Path(".cache")
        context = "unknown"

    if stage.startswith("mapping_"):
        _, mapping_type = stage.split("_", 1)
        if feature_id:
            subdir = Path("mappings") / feature_id / mapping_type
        else:
            subdir = Path("mappings") / mapping_type
    elif stage in {"descriptions", "description"}:
        subdir = Path("description")
    elif stage == "features" and feature_id:
        subdir = Path("features") / feature_id
    elif stage == "features":
        subdir = Path("features")
    else:
        subdir = Path(stage)

    path = cache_root / context / service / subdir / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI agent.

    The session stores message history so that subsequent prompts include
    previous context. Additional materials about the service under discussion
    may be seeded using :meth:`add_parent_materials`. Local caching is enabled
    by default in read-only mode.
    """

    def __init__(
        self,
        client: Agent,
        *,
        stage: str | None = None,
        diagnostics: bool = False,
        log_prompts: bool = False,
        transcripts_dir: Path | None = None,
        use_local_cache: bool = True,
        cache_mode: Literal["off", "read", "refresh", "write"] = "read",
    ) -> None:
        """Initialise the session with a configured LLM client.

        Args:
            client: Pydantic-AI ``Agent`` used for exchanges with the model.
            stage: Optional name of the generation stage for observability.
            diagnostics: Enable detailed logging and span creation.
            log_prompts: Debug log prompt text when ``diagnostics`` is ``True``.
            transcripts_dir: Directory used to store prompt/response transcripts
                when diagnostics mode is enabled.
            use_local_cache: Read from and write to the local cache when ``True``.
                Caching is enabled by default.
            cache_mode: Behaviour when interacting with the cache. Defaults to
                ``"read"`` for read-only access.
        """

        self.client = client
        self.stage = stage
        self.diagnostics = diagnostics
        self.log_prompts = log_prompts
        self._history: list[messages.ModelMessage] = []
        self.transcripts_dir = (
            transcripts_dir
            if transcripts_dir is not None
            else (Path("transcripts") if diagnostics else None)
        )
        self._service_id: str | None = None
        # Token usage for the most recent request
        self.last_tokens: int = 0
        self.use_local_cache = use_local_cache
        self.cache_mode: Literal["off", "read", "refresh", "write"] = cache_mode

    def add_parent_materials(self, service_input: ServiceInput) -> None:
        """Seed the conversation with details about the target service.

        Args:
            service_input: Metadata describing the service being evaluated.

        Side Effects:
            Appends a user prompt containing the service context to the session
            history.
        """
        ctx = "SERVICE_CONTEXT:\n" + service_input.model_dump_json()
        if self.diagnostics and self.log_prompts:
            logfire.debug(f"Adding service material to history: {ctx}")
        self._history.append(
            messages.ModelRequest(parts=[messages.UserPromptPart(ctx)])
        )
        self._service_id = service_input.service_id

    def derive(self) -> "ConversationSession":
        """Return a new session copying the current history."""

        clone = ConversationSession(
            self.client,
            stage=self.stage,
            diagnostics=self.diagnostics,
            log_prompts=self.log_prompts,
            transcripts_dir=self.transcripts_dir,
            use_local_cache=self.use_local_cache,
            cache_mode=self.cache_mode,
        )
        clone._history = list(self._history)
        clone._service_id = self._service_id
        return clone

    def _record_new_messages(self, msgs: list[messages.ModelMessage]) -> None:
        """Append ``msgs`` to history."""

        self._history.extend(msgs)

    @contextmanager
    def _prepare_span(
        self,
        span_name: str,
        stage: str,
        model_name: str,
    ) -> Any:
        """Create a logging span and attach common attributes."""

        span_ctx = logfire.span(span_name) if self.diagnostics else nullcontext()
        with span_ctx as span:
            if span and self.diagnostics:
                # annotate span for observability when diagnostics enabled
                span.set_attribute("stage", stage)
                span.set_attribute("model_name", model_name)
            yield span

    def _log_prompt(self, prompt: str) -> None:
        """Optionally log the prompt text for debugging."""

        if self.diagnostics and self.log_prompts:
            logfire.debug(f"Sending prompt: {prompt}")

    async def _write_transcript(self, prompt: str, response: Any) -> None:
        """Persist ``prompt`` and ``response`` when diagnostics are enabled."""

        if not (
            self.diagnostics and self.transcripts_dir and self._service_id is not None
        ):
            return
        svc_dir = self.transcripts_dir / self._service_id
        try:
            await asyncio.to_thread(svc_dir.mkdir, parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive
            return
        stage_name = self.stage or "unknown"
        payload = {"prompt": prompt, "response": str(response)}
        data = to_json(payload).decode()
        path = svc_dir / f"{stage_name}.json"
        await asyncio.to_thread(path.write_text, data, encoding="utf-8")

    def _handle_success(
        self,
        result: Any,
        stage: str,
        model_name: str,
    ) -> tuple[Any, int]:
        """Process a successful model invocation."""

        self._record_new_messages(list(result.new_messages()))
        usage = result.usage()
        tokens = usage.total_tokens or 0
        logfire.info(
            "Prompt succeeded",
            stage=stage,
            model_name=model_name,
            total_tokens=tokens,
        )
        return result.output, tokens

    def _handle_failure(
        self,
        exc: Exception,
        stage: str,
        model_name: str,
        tokens: int,
    ) -> None:
        """Log failure details."""

        logfire.error(
            "Prompt failed",
            stage=stage,
            model_name=model_name,
            total_tokens=tokens,
            error=str(exc),
        )

    def _finalise_metrics(
        self,
        span: Any,
        tokens: int,
        start: float,
    ) -> None:
        """Record latency and token usage."""

        duration = time.monotonic() - start
        if span and self.diagnostics:
            span.set_attribute("total_tokens", tokens)
            span.set_attribute("duration", duration)

    T = TypeVar("T")

    async def _ask_common(
        self,
        prompt: str,
        output_type: type[T] | None,
        runner: Callable[
            [str, list[messages.ModelMessage], type[T] | None], Awaitable[Any]
        ],
        span_name: str,
        feature_id: str | None = None,
    ) -> T | str:
        stage = self.stage or "unknown"
        model_name = getattr(getattr(self.client, "model", None), "model_name", "")
        cache_file: Path | None = None
        write_after_call = False
        if self.use_local_cache and self.cache_mode != "off":
            key = _prompt_cache_key(prompt, model_name, stage)
            svc = self._service_id or "unknown"
            cache_file = _prompt_cache_path(svc, stage, key, feature_id)
            exists_before = cache_file.exists()
            if self.cache_mode == "read" and exists_before:
                try:
                    with cache_file.open("rb") as fh:
                        data = from_json(fh.read())
                    if output_type and hasattr(output_type, "model_validate"):
                        payload = cast(Any, output_type).model_validate(data)
                    else:
                        payload = data
                    self.last_tokens = 0
                    return payload
                except Exception:  # pragma: no cover - invalid cache content
                    cache_file.replace(cache_file.with_suffix(".bad.json"))
                    exists_before = False
            if self.cache_mode == "refresh":
                write_after_call = True
            elif self.cache_mode == "write" and not exists_before:
                write_after_call = True
            elif self.cache_mode == "read" and not exists_before:
                write_after_call = True

        tokens = 0
        start = time.monotonic()
        with self._prepare_span(span_name, stage, model_name) as span:
            try:
                self._log_prompt(prompt)
                result = await runner(prompt, self._history, output_type)
                output, tokens = self._handle_success(result, stage, model_name)
                if cache_file and write_after_call:
                    content = (
                        output.model_dump() if hasattr(output, "model_dump") else output
                    )
                    cache_write_json_atomic(cache_file, content)
                self.last_tokens = tokens
                await self._write_transcript(prompt, output)
                return output
            except Exception as exc:  # pragma: no cover - defensive logging
                self._handle_failure(
                    exc,
                    stage,
                    model_name,
                    tokens,
                )
                raise
            finally:
                self._finalise_metrics(span, tokens, start)

    @overload
    def ask(self, prompt: str) -> str: ...

    @overload
    def ask(self, prompt: str, output_type: type[T]) -> T: ...

    def ask(
        self,
        prompt: str,
        output_type: type[T] | None = None,
        *,
        feature_id: str | None = None,
    ) -> T | str:
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
                prompt,
                output_type,
                runner,
                self.stage or "ConversationSession.ask",
                feature_id,
            )
        )

    @overload
    async def ask_async(self, prompt: str) -> str: ...

    @overload
    async def ask_async(self, prompt: str, output_type: type[T]) -> T: ...

    async def ask_async(
        self,
        prompt: str,
        output_type: type[T] | None = None,
        *,
        feature_id: str | None = None,
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
            feature_id,
        )


__all__ = ["ConversationSession"]
