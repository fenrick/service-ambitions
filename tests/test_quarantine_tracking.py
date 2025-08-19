import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import plateau_generator
from models import MaturityScore, PlateauFeature, ServiceInput


class DummySession:
    client = object()
    stage = "stage"

    def add_parent_materials(self, _service):
        pass

    def ask(self, prompt, output_type=None):
        raise RuntimeError("boom")

    async def ask_async(self, prompt, output_type=None):
        raise RuntimeError("boom")


def _service_input() -> ServiceInput:
    return ServiceInput(
        service_id="svc-1",
        name="svc",
        description="desc",
        customer_type="retail",
        jobs_to_be_done=[{"name": "job"}],
    )


def test_request_description_quarantines(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    gen = plateau_generator.PlateauGenerator(DummySession())
    gen._service = _service_input()
    plateau_generator.QUARANTINED_DESCRIPTIONS.clear()
    monkeypatch.setattr(
        plateau_generator, "load_prompt_text", lambda _name: "{plateau}{schema}"
    )

    with pytest.raises(ValueError):
        gen._request_description(1, session=DummySession())

    assert len(plateau_generator.QUARANTINED_DESCRIPTIONS) == 1
    path = Path(plateau_generator.QUARANTINED_DESCRIPTIONS[0])
    assert path.exists()


def test_map_features_quarantines(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    gen = plateau_generator.PlateauGenerator(
        DummySession(), mapping_session=DummySession()
    )
    gen._service = _service_input()
    feature = PlateauFeature(
        feature_id="f1",
        name="n",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="learners",
    )
    plateau_generator.QUARANTINED_MAPPING_PAYLOADS.clear()

    async def fake_map(*_args, **_kwargs):
        raise RuntimeError("mapping fail")

    monkeypatch.setattr(plateau_generator, "map_features_async", fake_map)

    with pytest.raises(RuntimeError):
        asyncio.run(gen._map_features_with_quarantine([feature], "p1"))

    assert len(plateau_generator.QUARANTINED_MAPPING_PAYLOADS) == 1
    path = Path(plateau_generator.QUARANTINED_MAPPING_PAYLOADS[0])
    assert path.exists()


def test_main_logs_quarantine_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "load_settings", lambda: SimpleNamespace(logfire_token=None)
    )
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli.logfire, "force_flush", lambda: None)

    plateau_generator.QUARANTINED_DESCRIPTIONS[:] = ["a.json"]
    plateau_generator.QUARANTINED_MAPPING_PAYLOADS[:] = ["b.json", "c.json"]

    def dummy_func(_args, _settings):
        return None

    ns = argparse.Namespace(func=dummy_func, seed=None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: ns)

    messages: list[str] = []
    monkeypatch.setattr(cli.logfire, "info", lambda m: messages.append(m))

    cli.main()

    assert messages[-1] == "Quarantined: 1 descriptions, 2 mapping payloads"
