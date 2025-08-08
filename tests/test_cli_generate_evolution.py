"""Tests for the generate-evolution CLI subcommand."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    ) -> object:  # pragma: no cover - simple stub
        return object()

    class DummyAgent:  # pragma: no cover - simple stub
        def __init__(self, model: object) -> None:  # noqa: D401 - no behaviour
            self.model = model

    def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="test-model",
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        plateaus=["alpha"],
        customers=["retail"],
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
        def __init__(self, model: object) -> None:  # pragma: no cover - simple init
            captured["agent_model"] = model

    def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model=None,
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        plateaus=["alpha"],
        customers=["retail"],
        model="special",  # override default
        logfire_service=None,
        log_level=None,
        verbose=0,
    )

    _cmd_generate_evolution(args, settings)

    assert captured["model_name"] == "special"
    assert captured["agent_model"] == "model"
