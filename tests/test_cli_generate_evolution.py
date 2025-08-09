"""Tests for the generate-evolution CLI subcommand."""

import argparse
import json
from types import SimpleNamespace

from cli import _cmd_generate_evolution
from models import ServiceEvolution, ServiceInput


def test_generate_evolution_writes_results(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "svc-1",
                "name": "svc",
                "description": "desc",
                "customer_type": "retail",
                "jobs_to_be_done": ["job"],
            }
        )
        + "\n"
    )

    def fake_build_model(
        model_name: str, api_key: str
    ) -> object:  # pragma: no cover - stub
        return object()

    class DummyAgent:  # pragma: no cover - simple stub
        def __init__(self, model: object, instructions: str) -> None:
            self.model = model
            self.instructions = instructions

    def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="test-model",
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
        concurrency=2,
        prompt_dir="prompts",
        context_id="university",
        inspiration="general",
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
    )

    _cmd_generate_evolution(args, settings)

    payload = json.loads(output_path.read_text().strip())
    assert payload["service"]["name"] == "svc"
    assert payload["service"]["service_id"] == "svc-1"


def test_generate_evolution_uses_agent_model(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "s1",
                "name": "svc",
                "description": "d",
                "customer_type": "retail",
                "jobs_to_be_done": [],
            }
        )
        + "\n"
    )

    captured: dict[str, object] = {}

    def fake_build_model(model_name: str, api_key: str) -> object:
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        return "model"

    class DummyAgent:
        def __init__(
            self, model: object, instructions: str
        ) -> None:  # pragma: no cover - simple init
            captured["agent_model"] = model
            captured["instructions"] = instructions

    def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model=None,
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
        concurrency=1,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model="special",  # override default
        logfire_service=None,
        log_level=None,
        verbose=0,
    )

    _cmd_generate_evolution(args, settings)

    assert captured["model_name"] == "special"
    assert captured["agent_model"] == "model"


def test_generate_evolution_respects_concurrency(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "s1",
                "name": "svc",
                "description": "d",
                "customer_type": "retail",
                "jobs_to_be_done": [],
            }
        )
        + "\n"
    )

    class DummyAgent:
        def __init__(
            self, model: object, instructions: str
        ) -> None:  # pragma: no cover - simple init
            self.model = model
            self.instructions = instructions

    def fake_build_model(
        model_name: str, api_key: str
    ) -> object:  # pragma: no cover - stub
        return object()

    def fake_generate(
        self, service: ServiceInput
    ) -> ServiceEvolution:  # pragma: no cover - stub
        return ServiceEvolution(service=service, plateaus=[])

    captured: dict[str, int] = {}

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            captured["workers"] = max_workers

        def __enter__(self):  # pragma: no cover - simple context
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple context
            return False

        def map(self, func, iterable):  # pragma: no cover - sequential map
            for item in iterable:
                yield func(item)

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="m",
        log_level="INFO",
        openai_api_key="k",
        logfire_token=None,
        concurrency=3,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
    )

    _cmd_generate_evolution(args, settings)

    assert captured["workers"] == 3
