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
from models import ReasoningConfig, StageModels


class Settings(BaseSettings):
    """Application settings combining file-based and environment configuration."""

    model: str = Field(..., description="Chat model in '<provider>:<model>' format.")
    models: StageModels | None = Field(None, description="Per-stage model overrides.")
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
    web_search: bool = Field(
        False, description="Enable OpenAI web search tooling for model browsing."
    )

    mapping_data_dir: Path = Field(
        Path("data"), description="Directory containing mapping reference data."
    )
    diagnostics: bool = Field(
        False, description="Enable verbose diagnostics and tracing."
    )
    strict_mapping: bool = Field(
        False, description="Fail when feature mappings are missing."
    )
    mapping_mode: str = Field("per_set", description="Mapping enrichment strategy")

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
    env_file_path = Path(".env")
    env_file = env_file_path if env_file_path.exists() else None
    try:
        # Validate and merge configuration from file, env file and environment.
        return Settings(
            model=config.model,
            models=config.models,
            reasoning=config.reasoning,
            log_level=config.log_level,
            prompt_dir=config.prompt_dir,
            context_id=config.context_id,
            inspiration=config.inspiration,
            concurrency=config.concurrency,
            request_timeout=config.request_timeout,
            retries=config.retries,
            retry_base_delay=config.retry_base_delay,
            features_per_role=config.features_per_role,
            web_search=config.web_search,
            mapping_data_dir=getattr(config, "mapping_data_dir", Path("data")),
            diagnostics=getattr(config, "diagnostics", False),
            strict_mapping=getattr(config, "strict_mapping", False),
            mapping_mode=getattr(config, "mapping_mode", "per_set"),
            _env_file=env_file,
        )
    except ValidationError as exc:
        # Summarise validation issues so the caller receives clear feedback.
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
