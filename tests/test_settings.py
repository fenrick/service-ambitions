"""Tests for configuration loading."""

import sys
from pathlib import Path

import pytest

from settings import load_settings  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_settings_reads_env(monkeypatch) -> None:
    """Environment variables should populate the settings model."""

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    settings = load_settings()
    assert settings.openai_api_key == "token"
    assert settings.model == "openai:gpt-4o-mini"
    assert settings.log_level == "INFO"


def test_load_settings_requires_key(monkeypatch) -> None:
    """Missing API key should raise ``RuntimeError``."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()
