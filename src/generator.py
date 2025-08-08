"""Async ambition generation using Pydantic AI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Iterable

import logfire
from pydantic import BaseModel
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

    @logfire.instrument()
    async def process_service(
        self, service: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Generate ambitions for ``service`` asynchronously."""

        agent = Agent(self.model, instructions=prompt)
        service_details = json.dumps(service)
        result = await agent.run(service_details, output_type=AmbitionModel)
        return result.output.model_dump()

    @logfire.instrument()
    async def _worker(
        self,
        service: Dict[str, Any],
        prompt: str,
        output_file,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            logger.info("Processing service %s", service.get("name", "unknown"))
            try:
                result = await self.process_service(service, prompt)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Failed to process service %s: %s",
                    service.get("name", "unknown"),
                    exc,
                )
                return
            output_file.write(f"{json.dumps(result)}\n")

    @logfire.instrument()
    async def _process_all(
        self,
        services: Iterable[Dict[str, Any]],
        prompt: str,
        output_file,
    ) -> None:
        semaphore = asyncio.Semaphore(self.concurrency)
        await asyncio.gather(
            *(
                self._worker(service, prompt, output_file, semaphore)
                for service in services
            )
        )

    @logfire.instrument()
    def generate(
        self,
        services: Iterable[Dict[str, Any]],
        prompt: str,
        output_path: str,
    ) -> None:
        """Process ``services`` and write ambitions to ``output_path``."""

        try:
            with open(output_path, "w", encoding="utf-8") as output_file:
                asyncio.run(self._process_all(services, prompt, output_file))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to write results to %s: %s", output_path, exc)
            raise


@logfire.instrument()
def build_model(model_name: str, api_key: str) -> Model:
    """Return a configured Pydantic AI model."""

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    model_name = model_name.split(":", 1)[-1]
    settings = OpenAIResponsesModelSettings(
        openai_builtin_tools=[{"type": "web_search"}],
        openai_reasoning_summary="concise",
        openai_reasoning_effort="medium",
    )
    return OpenAIResponsesModel(model_name, settings=settings)
