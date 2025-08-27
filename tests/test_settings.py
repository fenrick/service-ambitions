# SPDX-License-Identifier: MIT
"""Tests for configuration loading."""

import sys
from pathlib import Path

import pytest

from settings import load_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_settings_reads_env(monkeypatch) -> None:
    """Environment variables should populate the settings model."""

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("USE_LOCAL_CACHE", "1")
    monkeypatch.setenv("CACHE_MODE", "write")
    monkeypatch.setenv("CACHE_DIR", "/tmp/cache")
    settings = load_settings()
    assert settings.openai_api_key == "token"
    assert settings.model == "openai:gpt-5-mini"
    assert settings.models is not None
    assert settings.models.descriptions == "openai:o4-mini"
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 60
    assert settings.retries == 5
    assert settings.retry_base_delay == 0.5
    assert settings.features_per_role == 5
    assert settings.use_local_cache is True
    assert settings.cache_mode == "write"
    assert settings.cache_dir == Path("/tmp/cache")
    assert settings.mapping_data_dir == Path("data")
    assert settings.diagnostics is False
    assert settings.strict_mapping is False
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "medium"


def test_load_settings_requires_key(monkeypatch) -> None:
    """Missing API key should raise ``RuntimeError``."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_LOCAL_CACHE", raising=False)
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.delenv("CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()
