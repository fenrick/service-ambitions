"""Plateau feature generation and service evolution utilities."""

from __future__ import annotations

import json
import logging

from conversation import ConversationSession
from loader import load_prompt_text
from mapping import map_features
from models import (
    DescriptionResponse,
    PlateauFeature,
    PlateauFeaturesResponse,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)

logger = logging.getLogger(__name__)


class PlateauGenerator:
    """Generate plateau features and service evolution summaries."""

    def __init__(
        self,
        session: ConversationSession,
        required_count: int = 5,
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for agent queries.
            required_count: Minimum number of features per customer type.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.required_count = required_count
        self._service: ServiceInput | None = None

    def _request_description(self, level: int) -> str:
        """Return the service description for ``level``.

        The agent must respond with JSON containing a ``description`` field.
        """
        template = load_prompt_text("description_prompt")
        schema = json.dumps(DescriptionResponse.model_json_schema(), indent=2)
        prompt = template.format(
            plateau=level,
            schema=str(schema),
        )
        # Query the model using the stored conversation session so the
        # description becomes part of the evolving chat history.
        response = self.session.ask(prompt)
        try:
            payload = json.loads(response)
            description = payload["description"]
        except (json.JSONDecodeError, KeyError) as exc:  # pragma: no cover - logging
            logger.error("Invalid plateau description: %s", exc)
            raise ValueError("Agent returned invalid plateau description") from exc
        if not isinstance(description, str) or not description:
            raise ValueError("'description' must be a non-empty string")
        return description

    def _to_feature(self, item: dict[str, str], customer: str) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``.

        Args:
            item: Raw feature dictionary supplied by the agent.
            customer: Customer type the feature addresses.
        """

        return PlateauFeature(
            feature_id=item["feature_id"],
            name=item["name"],
            description=item["description"],
            score=float(item["score"]),
            customer_type=customer,
        )

    def generate_plateau(self, level: int) -> PlateauResult:
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
        # Ask the model to describe the service at the specified plateau level.
        description = self._request_description(level)
        schema = json.dumps(PlateauFeaturesResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_prompt")
        prompt = template.format(
            required_count=self.required_count,
            service_name=self._service.name,
            service_description=description,
            plateau=str(level),
            schema=str(schema),
        )
        logger.info("Requesting features for level=%s", level)
        # Using the shared conversation session ensures features are generated
        # in the same context as previous interactions.
        response = self.session.ask(prompt)
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
                # Normalise the raw dictionary into a strongly typed feature.
                features.append(self._to_feature(item, customer))
        # Enrich the raw features with mapping information before returning.
        mapped = map_features(self.session, features)
        return PlateauResult(
            plateau=level, service_description=description, features=mapped
        )

    def generate_service_evolution(
        self,
        service_input: ServiceInput,
        plateau_names: list[str],
        customer_types: list[str],
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and customers.

        Args:
            service_input: Service under evaluation.
            plateau_names: Ordered plateau identifiers to process.
            customer_types: Customer segments to include in results.

        Returns:
            Combined evolution limited to the requested plateaus and customers.

        Side Effects:
            Stores ``service_input`` for subsequent plateau generation and seeds
            the conversation session with its details.
        """
        self._service = service_input
        # Seed the conversation so later model queries have the service context.
        self.session.add_parent_materials(service_input)

        plateaus: list[PlateauResult] = []
        for level, _name in enumerate(plateau_names, start=1):
            # Generate features for each plateau level in turn.
            result = self.generate_plateau(level)
            filtered = [
                feat for feat in result.features if feat.customer_type in customer_types
            ]
            plateaus.append(
                PlateauResult(
                    plateau=result.plateau,
                    service_description=result.service_description,
                    features=filtered,
                )
            )
        return ServiceEvolution(service=service_input, plateaus=plateaus)


__all__ = ["PlateauGenerator"]
