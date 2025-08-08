"""Async ambition generation using Pydantic AI.

This module orchestrates concurrent requests to an LLM and streams the
resulting ambitions directly to disk.  Concurrency is carefully controlled to
avoid overwhelming upstream APIs while still maximising throughput.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Iterable

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from models import ServiceInput

logger = logging.getLogger(__name__)


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
    def __init__(self, model: Model, concurrency: int = 5) -> None:
        """Initialize the generator.

        Args:
            model: Model used for ambition generation.
            concurrency: Maximum number of concurrent requests.

        Raises:
            ValueError: If ``concurrency`` is less than one.
        """

        if concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.model = model
        self.concurrency = concurrency
        self._prompt: str | None = None

    @logfire.instrument()
    async def process_service(
        self,
        service: ServiceInput,
        prompt: str | None = None,
    ) -> Dict[str, Any]:
        """Return ambitions for ``service``.

        The function spins up a fresh agent for each call to avoid sharing
        mutable state across tasks. The JSON representation of ``service`` is
        passed as the prompt input and the model response is converted back
        into a standard dictionary for downstream processing.
        """

        instructions = prompt if prompt is not None else self._prompt
        if instructions is None:
            raise ValueError("prompt must be provided")
        agent = Agent(self.model, instructions=instructions)
        service_details = service.model_dump_json()
        result = await agent.run(service_details, output_type=AmbitionModel)
        return result.output.model_dump()

    @logfire.instrument()
    async def _worker(
        self,
        service: ServiceInput,
        output_file,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Write ambitions for a single ``service`` to ``output_file``.

        The coroutine acquires ``semaphore`` to ensure only a limited number of
        requests run concurrently.  Any errors from the model are logged and the
        service is skipped rather than aborting the entire batch.
        """

        async with semaphore:
            # The semaphore keeps the number of in-flight requests under the
            # configured threshold, protecting the API from overload.
            logger.info("Processing service %s", service.name)
            try:
                # Delegate to ``process_service`` for the actual model call.
                result = await self.process_service(service)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Failed to process service %s: %s",
                    service.name,
                    exc,
                )
                return
            # Persist each result on a single line so the file can be consumed
            # using standard ``jsonlines`` tooling.
            output_file.write(
                f"{AmbitionModel.model_validate(result).model_dump_json()}\n"
            )

    @logfire.instrument()
    async def _process_all(
        self,
        services: Iterable[ServiceInput],
        output_file,
    ) -> None:
        """Dispatch worker tasks for ``services`` and await completion."""

        # Semaphore coordinates concurrent worker tasks so only ``concurrency``
        # requests run at any given time.
        semaphore = asyncio.Semaphore(self.concurrency)
        await asyncio.gather(
            *(self._worker(service, output_file, semaphore) for service in services)
        )

    @logfire.instrument()
    def generate(
        self,
        services: Iterable[ServiceInput],
        prompt: str,
        output_path: str,
    ) -> None:
        """Process ``services`` and write ambitions to ``output_path``.

        Side Effects:
            Creates/overwrites ``output_path`` with one JSON record per line.
        """

        self._prompt = prompt
        try:
            with open(output_path, "w", encoding="utf-8") as output_file:
                # Run the async processing loop and stream results directly to
                # disk to avoid storing large intermediate data sets.
                asyncio.run(self._process_all(services, output_file))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to write results to %s: %s", output_path, exc)
            raise
        finally:
            self._prompt = None


@logfire.instrument()
def build_model(model_name: str, api_key: str) -> Model:
    """Return a configured Pydantic AI model.

    The OpenAI client reads its API key from the environment.  Supplying the key
    explicitly here ensures downstream libraries can locate it without coupling
    callers to environment management.
    """

    if api_key:
        # Expose the key via environment variables for model libraries that
        # expect it there rather than accepting it directly.
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    model_name = model_name.split(":", 1)[-1]
    settings = OpenAIResponsesModelSettings(
        openai_builtin_tools=[{"type": "web_search"}],
        openai_reasoning_summary="concise",
        openai_reasoning_effort="medium",
    )
    return OpenAIResponsesModel(model_name, settings=settings)


__all__ = ["ServiceAmbitionGenerator", "build_model"]
