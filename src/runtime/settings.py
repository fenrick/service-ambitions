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

from constants import DEFAULT_CACHE_DIR
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
    use_local_cache: bool = Field(
        True, description="Enable reading and writing the cache directory."
    )
    cache_mode: Literal["off", "read", "refresh", "write"] = Field(
        "read", description="Caching strategy for local cache entries."
    )
    cache_dir: Path = Field(
        DEFAULT_CACHE_DIR, description="Directory to store cache files."
    )
    openai_api_key: str = Field(..., description="OpenAI API access token.", repr=False)
    logfire_token: str | None = Field(
        None, description="Logfire authentication token, if available.", repr=False
    )
    web_search: bool = Field(
        False, description="Enable OpenAI web search tooling for model browsing."
    )

    # LLM queue feature flag and concurrency
    llm_queue_enabled: bool = Field(
        False,
        description=(
            "Enable the global LLM execution queue to centralise concurrency."
        ),
    )
    llm_queue_concurrency: int = Field(
        3,
        ge=1,
        description="Maximum number of concurrent LLM calls across the app.",
    )

    # Dry-run mode: proceed through the pipeline but do not invoke agents.
    # When enabled, cached artifacts are read as usual; if a cache miss occurs
    # at any point where an agent call would be required, execution halts with
    # a clear error so users can see what would have been invoked.
    dry_run: bool = Field(
        False,
        description=(
            "Enable dry-run mode that reads from cache and halts before invoking"
            " any agent when a cache miss occurs."
        ),
    )

    mapping_data_dir: Path = Field(
        Path("data"), description="Directory containing mapping reference data."
    )
    roles_file: Path = Field(
        Path("data/roles.json"),
        description="Path to JSON file containing role identifiers.",
    )
    mapping_sets: list[MappingSet] = Field(
        default_factory=list,
        description="Mapping dataset configurations.",
    )
    diagnostics: bool = Field(
        False, description="Enable verbose diagnostics and tracing."
    )
    strict: bool = Field(
        False,
        description="Fail when features or mappings are missing.",
    )
    strict_mapping: bool = Field(
        False, description="Fail when feature mappings are missing."
    )
    mapping_mode: str = Field("per_set", description="Mapping enrichment strategy")

    model_config = SettingsConfigDict(env_prefix="SA_", extra="ignore")


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load and validate application settings.

    Configuration values are read from the application configuration file and
    then merged with environment variables using ``pydantic-settings``. When a
    value is provided in both sources the environment variable wins. A ``.env``
    file in the working directory is loaded automatically when present. The
    optional ``config_path`` parameter allows overriding the default
    ``config/app.yaml`` location. The final configuration is validated and
    returned to the caller.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        Settings: Fully validated application configuration.

    Raises:
        RuntimeError: If required configuration values are missing or invalid.
    """
    if config_path:
        cfg_path = Path(config_path)
        config = load_app_config(cfg_path.parent, cfg_path.name)
    else:
        config = load_app_config()
    env_file_path = Path(".env")
    env_file = env_file_path if env_file_path.exists() else None
    env_use_local_cache = os.getenv("SA_USE_LOCAL_CACHE")
    env_cache_mode = os.getenv("SA_CACHE_MODE")
    env_cache_dir = os.getenv("SA_CACHE_DIR")
    raw_cache_dir = env_cache_dir or str(
        getattr(config, "cache_dir", DEFAULT_CACHE_DIR)
    )
    expanded_cache_dir = os.path.expandvars(raw_cache_dir)
    if "$" in expanded_cache_dir:
        expanded_cache_dir = str(DEFAULT_CACHE_DIR)
    cache_dir = Path(expanded_cache_dir).expanduser()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_file = cache_dir / ".write_test"
        with test_file.open("w", encoding="utf-8"):
            pass
        test_file.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot access cache directory '{cache_dir}': {exc}"
        ) from exc
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
            use_local_cache=(
                env_use_local_cache.lower() in {"1", "true", "yes"}
                if env_use_local_cache is not None
                else getattr(config, "use_local_cache", True)
            ),
            cache_mode=env_cache_mode or getattr(config, "cache_mode", "read"),
            cache_dir=cache_dir,
            web_search=config.web_search,
            mapping_data_dir=getattr(config, "mapping_data_dir", Path("data")),
            roles_file=getattr(
                config,
                "roles_file",
                Path("data/roles.json"),
            ),
            mapping_sets=getattr(config, "mapping_sets", []),
            diagnostics=getattr(config, "diagnostics", False),
            strict=getattr(config, "strict", False),
            strict_mapping=getattr(config, "strict_mapping", False),
            mapping_mode=getattr(config, "mapping_mode", "per_set"),
            dry_run=getattr(config, "dry_run", False),
            llm_queue_enabled=getattr(config, "llm_queue_enabled", False),
            llm_queue_concurrency=getattr(config, "llm_queue_concurrency", 3),
            _env_file=env_file,
        )
    except ValidationError as exc:
        # Summarise validation issues so the caller receives clear feedback.
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        raise RuntimeError(f"Invalid configuration: {details}") from exc
