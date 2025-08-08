"""Async ambition generation using Pydantic AI."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Iterable

import logfire
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

logger = logging.getLogger(__name__)


class AmbitionModel(BaseModel):
    """Structured ambitions response allowing arbitrary keys."""

    model_config = {"extra": "allow"}


class ServiceAmbitionGenerator:
    """Generate ambitions for services using a Pydantic AI model."""

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
        self, service: Dict[str, Any], prompt: str | None = None
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
        service_details = TypeAdapter(Dict[str, Any]).dump_json(service).decode()
        result = await agent.run(service_details, output_type=AmbitionModel)
        return result.output.model_dump()

    @logfire.instrument()
    async def _worker(
        self,
        service: Dict[str, Any],
        output_file,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            # The semaphore keeps the number of in-flight requests under the
            # configured threshold, protecting the API from overload.
            logger.info("Processing service %s", service.get("name", "unknown"))
            try:
                # Delegate to ``process_service`` for the actual model call.
                result = await self.process_service(service)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Failed to process service %s: %s",
                    service.get("name", "unknown"),
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
        services: Iterable[Dict[str, Any]],
        output_file,
    ) -> None:
        # Semaphore coordinates concurrent worker tasks so only ``concurrency``
        # requests run at any given time.
        semaphore = asyncio.Semaphore(self.concurrency)
        await asyncio.gather(
            *(self._worker(service, output_file, semaphore) for service in services)
        )

    @logfire.instrument()
    def generate(
        self,
        services: Iterable[Dict[str, Any]],
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
    """Return a configured Pydantic AI model."""

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
