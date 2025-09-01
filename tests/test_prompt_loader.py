# SPDX-License-Identifier: MIT
"""Tests for prompt loader caching."""

from pathlib import Path

from utils import FilePromptLoader


def test_file_prompt_loader_caches(tmp_path: Path) -> None:
    base = tmp_path
    prompt = base / "foo.md"
    prompt.write_text("one", encoding="utf-8")
    loader = FilePromptLoader(base)
    assert loader.load("foo") == "one"
    prompt.write_text("two", encoding="utf-8")
    assert loader.load("foo") == "one"
    loader.clear_cache()
    assert loader.load("foo") == "two"
