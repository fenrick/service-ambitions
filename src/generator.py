"""Async ambition generation using Pydantic AI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Iterable

from pydantic import BaseModel
from pydantic_ai import Agent, models

logger = logging.getLogger(__name__)


class AmbitionModel(BaseModel):
    """Structured ambitions response allowing arbitrary keys."""

    model_config = {"extra": "allow"}


class ServiceAmbitionGenerator:
    """Generate ambitions for services using a Pydantic AI model."""

    def __init__(self, model: models.Model, concurrency: int = 5) -> None:
        self.model = model
        self.concurrency = concurrency

    async def process_service(
        self, service: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Generate ambitions for ``service`` asynchronously."""

        agent = Agent(self.model, instructions=prompt)
        service_details = json.dumps(service)
        result = await agent.run(service_details, output_type=AmbitionModel)
        return result.output.model_dump()

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


def build_model(model_name: str, api_key: str) -> models.Model:
    """Return a configured Pydantic AI model."""

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    return models.infer_model(model_name)
