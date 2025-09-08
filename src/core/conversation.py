# SPDX-License-Identifier: MIT
"""Conversational session wrapper for LLM interactions.

This module exposes :class:`ConversationSession`, a light abstraction over a
Pydantic-AI ``Agent``. The session records message history so that each prompt
retains prior context and can be seeded with service details via
``add_parent_materials``. The :meth:`ask` method delegates to the underlying
agent without relying on asynchronous execution.
"""

from __future__ import annotations

# mypy: ignore-errors
import asyncio
import hashlib
import json
import time
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast
from uuid import uuid4

import logfire
from pydantic import ValidationError
from pydantic_ai import Agent, messages
from pydantic_core import from_json, to_json

from constants import DEFAULT_CACHE_DIR
from models import ServiceInput
from runtime.environment import RuntimeEnv

from .dry_run import DryRunInvocation
from .mapping import cache_write_json_atomic


def _prompt_cache_key(prompt: str, model: str, stage: str) -> str:
    """Return a stable cache key for ``prompt`` and ``model``."""
    data = to_json([prompt, model, stage]).decode()
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]


def _prompt_cache_path(
    service: str, stage: str, key: str, feature_id: str | None = None
) -> Path:
    """Return cache path for ``key`` grouped by context and identifiers."""
    try:
        settings = RuntimeEnv.instance().settings
        cache_root = settings.cache_dir
        context = settings.context_id
    except RuntimeError:  # pragma: no cover - fallback when settings unavailable
        cache_root = DEFAULT_CACHE_DIR
        context = "unknown"

    if stage.startswith("mapping_"):
        _, mapping_type = stage.split("_", 1)
        subdir = Path("mappings") / mapping_type
    elif stage.startswith("features_"):
        plateau = stage.split("_", 1)[1]
        subdir = Path(plateau)
    elif stage in {"descriptions", "description"}:
        subdir = Path()
    elif stage == "features" and feature_id:
        subdir = Path(feature_id)
    else:
        subdir = Path(stage)

    base = cache_root / context / service
    path = base / subdir / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _service_cache_root(service: str) -> Path:
    """Return cache root for ``service`` using runtime settings."""
    try:
        settings = RuntimeEnv.instance().settings
        root = settings.cache_dir / settings.context_id / service
    except RuntimeError:  # pragma: no cover - fallback when settings unavailable
        root = DEFAULT_CACHE_DIR / "unknown" / service
    return root


def _find_cache_file(service_root: Path, key: str, cache_file: Path) -> Path | None:
    """Return existing cache file matching ``key`` under ``service_root``."""
    for path in service_root.glob(f"**/{key}.json"):
        return path
    return cache_file if cache_file.exists() else None


class ConversationSession:
    """Manage a conversational interaction with a Pydantic-AI agent.

    The session stores message history so that subsequent prompts include
    previous context. Additional materials about the service under discussion
    may be seeded using :meth:`add_parent_materials`. Local caching is enabled
    by default in read-only mode.
    """

    def __init__(
        self,
        client: Agent[Any, Any],
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
        # Track model invocation counts per stage for observability/tests.
        self._invocations_by_stage: dict[str, int] = {}
        # Metric counter for actual model calls (excludes cache hits).
        self._prompt_calls = logfire.metric_counter("prompt_calls")
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
        request_id: str,
    ) -> Any:
        """Create a logging span and attach common attributes."""
        span_ctx = logfire.span(span_name) if self.diagnostics else nullcontext()
        with span_ctx as span:
            if span and self.diagnostics:
                # annotate span for observability when diagnostics enabled
                span.set_attribute("stage", stage)
                span.set_attribute("model_name", model_name)
                span.set_attribute("request_id", request_id)
                if self._service_id is not None:
                    span.set_attribute("service_id", self._service_id)
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
        except OSError:  # pragma: no cover - defensive
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
        request_id: str,
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
            request_id=request_id,
            service_id=self._service_id,
        )
        return result.output, tokens

    def _handle_failure(
        self,
        exc: Exception,
        stage: str,
        model_name: str,
        tokens: int,
        request_id: str,
    ) -> None:
        """Log failure details."""
        logfire.error(
            "Prompt failed",
            stage=stage,
            model_name=model_name,
            total_tokens=tokens,
            error=str(exc),
            request_id=request_id,
            service_id=self._service_id,
        )

    def _finalise_metrics(
        self,
        span: Any,
        tokens: int,
        start: float,
        _request_id: str,
    ) -> None:
        """Record latency and token usage."""
        duration = time.monotonic() - start
        if span and self.diagnostics:
            span.set_attribute("total_tokens", tokens)
            span.set_attribute("duration", duration)

    def _load_cache_payload(
        self,
        path_to_use: Path,
        cache_file: Path,
        out_type: type[Any] | None,
    ) -> Any:
        """Return cached payload validating against ``out_type``."""
        try:
            with path_to_use.open("rb") as fh:
                data = from_json(fh.read())
            if out_type and hasattr(out_type, "model_validate"):
                payload = cast(Any, out_type).model_validate(data)
                dump = payload.model_dump()
            else:
                payload = data
                dump = data
            if path_to_use != cache_file:
                cache_write_json_atomic(
                    cache_file,
                    dump if not isinstance(dump, str) else json.dumps(dump),
                )
                with suppress(OSError):
                    path_to_use.unlink()
            self.last_tokens = 0
            return payload
        except (ValidationError, ValueError, OSError) as exc:
            raise RuntimeError(f"Invalid cache file: {path_to_use}") from exc

    @dataclass
    class CacheResult:
        """Capture cache lookup details."""

        payload: Any | None
        cache_file: Path | None
        write_after_call: bool

    def _try_cache_read(
        self,
        prompt: str,
        model_name: str,
        stage: str,
        feature_id: str | None,
        out_type: type[Any] | None,
    ) -> CacheResult:
        """Return cached payload and write flag if applicable."""
        if not self.use_local_cache:
            return ConversationSession.CacheResult(None, None, False)
        if self.cache_mode == "off":
            return ConversationSession.CacheResult(None, None, False)
        key = _prompt_cache_key(prompt, model_name, stage)
        svc = self._service_id or "unknown"
        cache_file = _prompt_cache_path(svc, stage, key, feature_id)
        service_root = _service_cache_root(svc)
        path_to_use = _find_cache_file(service_root, key, cache_file)
        exists_before = path_to_use is not None
        if self.cache_mode == "read" and path_to_use:
            payload = self._load_cache_payload(path_to_use, cache_file, out_type)
            return ConversationSession.CacheResult(payload, cache_file, False)
        write_after_call = self.cache_mode == "refresh" or (
            self.cache_mode in {"write", "read"} and not exists_before
        )
        return ConversationSession.CacheResult(None, cache_file, write_after_call)

    async def _invoke_runner(
        self,
        prompt: str,
        runner: Callable[[str, list[messages.ModelMessage]], Awaitable[Any]],
        stage: str,
        model_name: str,
        request_id: str,
    ) -> tuple[Any, int]:
        """Execute ``runner`` and return output and token count."""
        self._log_prompt(prompt)
        result = await runner(prompt, self._history)
        return self._handle_success(result, stage, model_name, request_id)

    def _write_cache_result(self, cache_file: Path, output: Any) -> None:
        """Persist ``output`` to ``cache_file`` as JSON."""
        content = output.model_dump() if hasattr(output, "model_dump") else output
        cache_write_json_atomic(
            cache_file,
            content if not isinstance(content, str) else json.dumps(content),
        )

    def _session_details(self) -> tuple[str, str, type[Any] | None]:
        """Return stage, model name and output type for the session."""
        stage = self.stage or "unknown"
        model_name = getattr(getattr(self.client, "model", None), "model_name", "")
        out_type = cast(type[Any] | None, getattr(self.client, "output_type", None))
        return stage, model_name, out_type

    async def _execute_with_cache(
        self,
        prompt: str,
        runner: Callable[[str, list[messages.ModelMessage]], Awaitable[Any]],
        span_name: str,
        stage: str,
        model_name: str,
        cache: CacheResult,
    ) -> Any:
        tokens = 0
        start = time.monotonic()
        request_id = uuid4().hex
        with self._prepare_span(span_name, stage, model_name, request_id) as span:
            try:
                output, tokens = await self._invoke_runner(
                    prompt, runner, stage, model_name, request_id
                )
                if cache.cache_file and cache.write_after_call:
                    self._write_cache_result(cache.cache_file, output)
                self.last_tokens = tokens
                await self._write_transcript(prompt, output)
                return output
            except (
                ValidationError,
                ValueError,
                OSError,
                RuntimeError,
            ) as exc:  # pragma: no cover - defensive logging
                self._handle_failure(exc, stage, model_name, tokens, request_id)
                raise
            finally:
                self._finalise_metrics(span, tokens, start, request_id)

    async def _ask_common(
        self,
        prompt: str,
        runner: Callable[[str, list[messages.ModelMessage]], Awaitable[Any]],
        span_name: str,
        feature_id: str | None = None,
    ) -> Any:
        stage, model_name, out_type = self._session_details()
        cache = self._try_cache_read(prompt, model_name, stage, feature_id, out_type)
        if cache.payload is not None:
            return cache.payload
        # Dry-run: stop before making an agent call when cache is missing.
        try:
            settings = RuntimeEnv.instance().settings
        except RuntimeError:
            settings = None  # pragma: no cover - defensive fallback
        if getattr(settings, "dry_run", False):
            cache_file = cache.cache_file
            raise DryRunInvocation(
                stage=stage,
                model=model_name,
                cache_file=cache_file,
                service_id=self._service_id,
            )
        # Record a single invocation per actual model call (no cache).
        self._invocations_by_stage[stage] = self._invocations_by_stage.get(stage, 0) + 1
        self._prompt_calls.add(1)
        return await self._execute_with_cache(
            prompt, runner, span_name, stage, model_name, cache
        )

    def ask(
        self,
        prompt: str,
        *,
        feature_id: str | None = None,
    ) -> Any:
        """Return the agent's response to ``prompt``."""

        async def runner(
            user_prompt: str,
            message_history: list[messages.ModelMessage],
        ) -> Any:
            return self.client.run_sync(user_prompt, message_history=message_history)

        return asyncio.run(
            self._ask_common(
                prompt,
                runner,
                self.stage or "ConversationSession.ask",
                feature_id,
            )
        )

    @property
    def invocations_by_stage(self) -> dict[str, int]:
        """Return a mapping of stage name to actual model invocation count.

        Cache hits do not increment this counter. Useful for tests asserting
        one-shot behaviour (e.g., at most one call per plateau stage).
        """
        return dict(self._invocations_by_stage)

    async def ask_async(
        self,
        prompt: str,
        *,
        feature_id: str | None = None,
    ) -> Any:
        """Asynchronously return the agent's response to ``prompt``."""

        async def runner(
            user_prompt: str,
            message_history: list[messages.ModelMessage],
        ) -> Any:
            return await self.client.run(user_prompt, message_history=message_history)

        return await self._ask_common(
            prompt,
            runner,
            self.stage or "ConversationSession.ask_async",
            feature_id,
        )


__all__ = ["ConversationSession"]
