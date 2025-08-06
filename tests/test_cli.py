import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def test_cli_generates_output(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are a helpful assistant.")

    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n{"name": "beta"}\n')

    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(main, "init_chat_model", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        main,
        "process_service",
        lambda service, model, prompt: {
            "service": service["name"],
            "prompt": prompt[:3],
        },
    )

    argv = [
        "main",
        "--prompt-file",
        str(prompt_file),
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main.main()

    lines = output_file.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"service": "alpha", "prompt": "You"},
        {"service": "beta", "prompt": "You"},
    ]


def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["main"])
    with pytest.raises(RuntimeError):
        main.main()


def test_cli_uses_prompt_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompt-special.md").write_text("Special prompt")
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n')
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(main, "init_chat_model", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        main,
        "process_service",
        lambda service, model, prompt: {"prompt": prompt},
    )

    argv = [
        "main",
        "--prompt-id",
        "special",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main.main()

    line = json.loads(output_file.read_text().strip())
    assert line["prompt"] == "Special prompt"


def test_cli_model_parameters_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompt.md").write_text("Prompt")
    input_file = tmp_path / "services.jsonl"
    input_file.write_text('{"name": "alpha"}\n')
    output_file = tmp_path / "output.jsonl"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("MODEL_PROVIDER", "acme")

    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(main, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(
        main, "process_service", lambda service, model, prompt: {"ok": True}
    )

    argv = [
        "main",
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main.main()

    assert captured["model"] == "test-model"
    assert captured["model_provider"] == "acme"
