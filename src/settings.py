"""Centralised application configuration management.

This module exposes :class:`Settings`, a ``pydantic-settings`` model that
combines values sourced from JSON configuration files and environment
variables. Environment variables take precedence over file-based values and
the merged configuration is validated before use.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from loader import load_app_config
from models import ReasoningConfig


class Settings(BaseSettings):
    """Application settings combining file-based and environment configuration."""

    model: str = Field(..., description="Chat model in '<provider>:<model>' format.")
    reasoning: ReasoningConfig | None = Field(
        None, description="Optional reasoning configuration for the model."
    )
    log_level: str = Field(..., description="Logging verbosity level.")
    prompt_dir: Path = Field(..., description="Directory containing prompt components.")
    context_id: str = Field(..., description="Situational context identifier.")
    inspiration: str = Field(..., description="Inspirations identifier.")
    concurrency: int = Field(
        ..., ge=1, description="Number of services to process concurrently."
    )
    batch_size: int | None = Field(
        None, ge=1, description="Number of services to schedule per batch."
    )
    request_timeout: int = Field(
        60, gt=0, description="Per-request timeout in seconds."
    )
    retries: int = Field(5, ge=1, description="Number of retry attempts.")
    retry_base_delay: float = Field(
        0.5, gt=0, description="Initial backoff delay in seconds."
    )
    features_per_role: int = Field(
        5, ge=1, description="Required number of features per role."
    )
    openai_api_key: str = Field(..., description="OpenAI API access token.")
    logfire_token: str | None = Field(
        None, description="Logfire authentication token, if available."
    )

    model_config = SettingsConfigDict(extra="ignore")


def load_settings() -> Settings:
    """Load and validate application settings.

    Configuration values are read from the application configuration file and
    then merged with environment variables using ``pydantic-settings``. When a
    value is provided in both sources the environment variable wins. A ``.env``
    file in the working directory is loaded automatically when present. The
    final configuration is validated and returned to the caller.

    Returns:
        Settings: Fully validated application configuration.

    Raises:
        RuntimeError: If required configuration values are missing or invalid.
    """

    config = load_app_config()
    data = {
        "model": config.model,
        "reasoning": config.reasoning,
        "log_level": config.log_level,
        "prompt_dir": config.prompt_dir,
        "context_id": config.context_id,
        "inspiration": config.inspiration,
        "concurrency": config.concurrency,
        "batch_size": config.batch_size,
        "request_timeout": config.request_timeout,
        "retries": config.retries,
        "retry_base_delay": config.retry_base_delay,
        "features_per_role": config.features_per_role,
    }
    env_file = Path(".env")
    env_kwargs = {"_env_file": env_file} if env_file.exists() else {}
    try:
        # Validate and merge configuration from file, env file and environment.
        return Settings(**data, **env_kwargs)  # type: ignore[arg-type]
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        # Summarise validation issues so the caller receives clear feedback.
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
