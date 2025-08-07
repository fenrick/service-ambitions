"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logfire
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the CLI."""

    model: str = Field(
        "openai:gpt-4o-mini",
        description="Chat model in '<provider>:<model>' format.",
    )
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
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
