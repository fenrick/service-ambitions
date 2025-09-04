# SPDX-License-Identifier: MIT
"""Centralised application configuration management.

This module exposes :class:`Settings`, a ``pydantic-settings`` model that
combines values sourced from JSON configuration files and environment
variables. Environment variables take precedence over file-based values and
the merged configuration is validated before use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from io_utils.loader import load_app_config
from models import MappingSet, ReasoningConfig, StageModels


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
    use_local_cache: bool = Field(
        True, description="Enable reading and writing the cache directory."
    )
    cache_mode: Literal["off", "read", "refresh", "write"] = Field(
        "read", description="Caching strategy for local cache entries."
    )
    cache_dir: Path = Field(
        Path(".cache"), description="Directory to store cache files."
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
    mapping_sets: list[MappingSet] = Field(
        default_factory=list,
        description="Mapping dataset configurations.",
    )
    diagnostics: bool = Field(
        False, description="Enable verbose diagnostics and tracing."
    )
    strict: bool = Field(
        False, description="Fail when features or mappings are missing."
    )
    strict_mapping: bool = Field(
        False, description="Fail when feature mappings are missing."
    )
    mapping_mode: str = Field("per_set", description="Mapping enrichment strategy")

    model_config = SettingsConfigDict(extra="ignore", env_prefix="SA_")


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

    data = config.model_dump()
    prefix = "SA_"

    if env_file is not None:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith(prefix):
                field = key[len(prefix) :].lower()
                if field in Settings.model_fields:
                    data[field] = value.strip()

    for field in Settings.model_fields:
        env_name = f"{prefix}{field.upper()}"
        if env_name in os.environ:
            data[field] = os.environ[env_name]

    try:
        return Settings(**data, _env_file=None)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
