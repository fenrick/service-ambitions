"""Plateau feature generation and service evolution utilities."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable

from conversation import ConversationSession
from loader import load_plateau_prompt
from mapping import MappedPlateauFeature, map_feature
from models import PlateauFeature, PlateauResult, ServiceEvolution, ServiceInput

logger = logging.getLogger(__name__)


class PlateauGenerator:
    """Generate plateau features and service evolution summaries."""

    def __init__(
        self,
        session: ConversationSession,
        prompt_dir: str = "prompts",
        required_count: int = 5,
    ) -> None:
        """Initialize the generator.

        Args:
            session: Active conversation session for agent queries.
            prompt_dir: Directory containing prompt templates.
            required_count: Minimum number of features per plateau.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.prompt_dir = prompt_dir
        self.required_count = required_count

    async def generate_plateau(
        self, service: ServiceInput, plateau: str, customer_type: str
    ) -> list[PlateauResult]:
        """Return plateau results for ``service`` and ``customer_type``.

        Requests at least ``required_count`` features for the specified plateau
        and customer type, mapping each feature using :func:`map_feature`.
        """

        template = load_plateau_prompt(self.prompt_dir)
        prompt = template.format(
            required_count=self.required_count,
            service_name=service.name,
            service_description=service.description,
            plateau=plateau,
            customer_type=customer_type,
        )
        logger.info(
            "Requesting features for service=%s plateau=%s customer=%s",
            service.name,
            plateau,
            customer_type,
        )
        response = await asyncio.to_thread(self.session.ask, prompt)
        logger.debug("Raw feature response: %s", response)

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from feature response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

        raw_features = payload.get("features")
        if not isinstance(raw_features, list):
            raise ValueError("'features' key missing or not a list")
        if len(raw_features) < self.required_count:
            logger.error(
                "Expected at least %s features, received %s",
                self.required_count,
                len(raw_features),
            )
            raise ValueError("Insufficient number of features returned")

        results: list[PlateauResult] = []
        for item in raw_features:
            feature = PlateauFeature(
                feature_id=item["feature_id"],
                name=item["name"],
                description=item["description"],
            )
            mapped: MappedPlateauFeature = await map_feature(
                self.session, feature, self.prompt_dir
            )
            results.append(PlateauResult(feature=mapped, score=float(item["score"])))
        return results

    async def generate_service_evolution(
        self,
        service: ServiceInput,
        plateaus: Iterable[str],
        customer_types: Iterable[str],
    ) -> ServiceEvolution:
        """Return aggregated service evolution across ``plateaus``.

        Each plateau is processed for every customer type using
        :meth:`generate_plateau`.
        """

        all_results: list[PlateauResult] = []
        for plateau in plateaus:
            for customer in customer_types:
                logger.debug("Processing plateau=%s for customer=%s", plateau, customer)
                try:
                    plateau_results = await self.generate_plateau(
                        service, plateau, customer
                    )
                except ValueError as exc:
                    logger.error(
                        "Failed to generate plateau %s for customer %s: %s",
                        plateau,
                        customer,
                        exc,
                    )
                    raise
                all_results.extend(plateau_results)
        return ServiceEvolution(service=service, results=all_results)


__all__ = ["PlateauGenerator"]
