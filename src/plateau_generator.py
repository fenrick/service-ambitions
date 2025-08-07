"""Plateau feature generation and service evolution utilities."""

from __future__ import annotations

import json
import logging

from conversation import ConversationSession
from loader import load_plateau_prompt
from mapping import map_features
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
        """Initialise the generator.

        Args:
            session: Active conversation session for agent queries.
            prompt_dir: Directory containing prompt templates.
            required_count: Minimum number of features per customer type.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.prompt_dir = prompt_dir
        self.required_count = required_count
        self._service: ServiceInput | None = None

    def _request_description(self, session: ConversationSession, level: int) -> str:
        """Return the service description for ``level``.

        The agent must respond with JSON containing a ``description`` field.
        """
        prompt = (
            "Provide JSON with a 'description' field describing the service "
            f"at plateau level {level}."
        )
        response = session.ask(prompt)
        try:
            payload = json.loads(response)
            description = payload["description"]
        except (json.JSONDecodeError, KeyError) as exc:  # pragma: no cover - logging
            logger.error("Invalid plateau description: %s", exc)
            raise ValueError("Agent returned invalid plateau description") from exc
        if not isinstance(description, str) or not description:
            raise ValueError("'description' must be a non-empty string")
        return description

    def generate_plateau(
        self, session: ConversationSession, level: int
    ) -> PlateauResult:
        """Return mapped plateau features for ``level``.

        The function requests a plateau-specific service description, then
        issues a single prompt asking for features for learners, staff and
        community. The response must provide at least ``required_count``
        features for each customer type. The list of features is enriched using
        :func:`map_features` before being returned as part of a
        :class:`PlateauResult`.
        """
        if self._service is None:
            raise ValueError(
                "ServiceInput not set. Call generate_service_evolution first."
            )

        description = self._request_description(session, level)
        template = load_plateau_prompt(self.prompt_dir)
        prompt = template.format(
            required_count=self.required_count,
            service_name=self._service.name,
            service_description=description,
            plateau=str(level),
        )
        logger.info("Requesting features for level=%s", level)
        response = session.ask(prompt)
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from feature response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

        features: list[PlateauFeature] = []
        for customer in ("learners", "staff", "community"):
            raw_features = payload.get(customer)
            if (
                not isinstance(raw_features, list)
                or len(raw_features) < self.required_count
            ):
                raise ValueError(
                    f"Insufficient number of features returned for {customer}"
                )
            for item in raw_features:
                features.append(
                    PlateauFeature(
                        feature_id=item["feature_id"],
                        name=item["name"],
                        description=item["description"],
                        score=float(item["score"]),
                        customer_type=customer,
                    )
                )
        mapped = map_features(session, features)
        return PlateauResult(
            plateau=level, service_description=description, features=mapped
        )

    def generate_service_evolution(
        self, service_input: ServiceInput
    ) -> ServiceEvolution:
        """Return aggregated service evolution across plateaus 1-4."""
        self._service = service_input
        self.session.add_parent_materials(service_input)

        plateaus: list[PlateauResult] = []
        for level in range(1, 5):
            plateaus.append(self.generate_plateau(self.session, level))
        return ServiceEvolution(service=service_input, plateaus=plateaus)


__all__ = ["PlateauGenerator"]
