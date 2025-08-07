"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the CLI."""

    model: str = "o4-mini"
    log_level: str = "INFO"
    openai_api_key: str
    logfire_token: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_settings() -> Settings:
    """Load settings from the environment.

    Returns:
        Settings: Validated application configuration.

    Raises:
        RuntimeError: If required environment variables are missing or invalid.
    """

    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it via a .env file or a secret manager."
        ) from exc
