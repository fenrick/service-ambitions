"""Tests for resume invariants."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.processing_engine import ProcessingEngine
from runtime.environment import RuntimeEnv


def _make_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt",
        openai_api_key="key",
        models=None,
        reasoning=None,
        prompt_dir=tmp_path,
        mapping_data_dir=tmp_path,
        roles_file=tmp_path / "roles.json",
        context_id="ctx",
        inspiration=None,
        concurrency=1,
        web_search=False,
    )


def _make_args(tmp_path: Path, input_file: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_file=str(tmp_path / "out.jsonl"),
        resume=True,
        transcripts_dir=None,
        seed=0,
        roles_file=str(tmp_path / "roles.json"),
        input_file=str(input_file),
        max_services=None,
        progress=False,
        temp_output_dir=None,
        dry_run=False,
        allow_prompt_logging=False,
    )


def test_refuse_resume_on_input_changed(tmp_path: Path) -> None:
    """Engine should abort resume when the input file differs."""
    original = tmp_path / "services.jsonl"
    original.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    state_path = output.with_name("resume_state.json")
    processed_path = output.with_name("processed_ids.txt")
    processed_path.write_text("", encoding="utf-8")
    meta = {
        "input_hash": sha256(original.read_bytes()).hexdigest(),
        "output_path": str(output),
        "settings": {},
    }
    state_path.write_text(json.dumps(meta), encoding="utf-8")
    original.write_text("changed", encoding="utf-8")
    settings = _make_settings(tmp_path)
    RuntimeEnv.reset()
    RuntimeEnv.initialize(settings)
    args = _make_args(tmp_path, original)
    with pytest.raises(ValueError, match="input file has changed"):
        ProcessingEngine(args, None)
