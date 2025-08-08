"""Application configuration loaded from config files and environment variables."""

from __future__ import annotations

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from loader import load_app_config


class Settings(BaseSettings):
    """Application settings combining file-based and environment configuration."""

    model: str = Field(..., description="Chat model in '<provider>:<model>' format.")
    log_level: str = Field(..., description="Logging verbosity level.")
    openai_api_key: str = Field(..., description="OpenAI API access token.")
    logfire_token: str | None = Field(
        None, description="Logfire authentication token, if available."
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_settings() -> Settings:
    """Load settings from configuration files and environment variables.

    Returns:
        Settings: Validated application configuration.

    Raises:
        RuntimeError: If required configuration values are missing or invalid.
    """

    config = load_app_config()
    data = {"model": config.model, "log_level": config.log_level}
    try:
        return Settings(**data)  # type: ignore[arg-type]
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
