"""Async ambition generation using chat models."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Iterable, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AmbitionModel(BaseModel):
    """Structured ambitions response allowing arbitrary keys."""

    model_config = {"extra": "allow"}


class ServiceAmbitionGenerator:
    """Generate ambitions for services using a chat model."""

    def __init__(self, model: BaseChatModel, concurrency: int = 5) -> None:
        self.model = model
        self.concurrency = concurrency

    async def process_service(
        self, service: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Generate ambitions for ``service`` asynchronously."""

        prompt_template = ChatPromptTemplate(
            [("system", "{system_prompt}"), ("user", "{user_prompt}")]
        )
        chain = cast(
            Any, prompt_template | self.model.with_structured_output(AmbitionModel)
        )
        service_details = json.dumps(service)
        result: BaseModel = await asyncio.to_thread(
            chain.invoke,
            {"system_prompt": prompt, "user_prompt": service_details},
        )
        return result.model_dump()

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


def build_model(
    model_name: str, api_key: str, response_format: str | None
) -> ChatOpenAI:
    """Return a configured :class:`ChatOpenAI` instance."""

    model_kwargs: Dict[str, Any] = {"model": model_name, "api_key": api_key}
    if response_format:
        model_kwargs["response_format"] = response_format
    return ChatOpenAI(**model_kwargs)
