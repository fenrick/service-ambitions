import sys
from types import SimpleNamespace

import monitoring


def test_init_logfire_configures_sdk(monkeypatch):
    called = {}

    def configure(**kwargs):
        called.update(kwargs)

    instruments = []

    def instrument_pydantic_ai():
        instruments.append("ai")

    def instrument_pydantic():
        instruments.append("pydantic")

    def instrument_openai():
        instruments.append("openai")

    info_logged = {"msg": None}

    def info(msg):
        info_logged["msg"] = msg

    dummy_module = SimpleNamespace(
        configure=configure,
        instrument_system_metrics=lambda **kw: called.setdefault("metrics", kw),
        instrument_pydantic_ai=instrument_pydantic_ai,
        instrument_pydantic=instrument_pydantic,
        instrument_openai=instrument_openai,
        info=info,
    )

    monkeypatch.setitem(sys.modules, "logfire", dummy_module)

    monitoring.init_logfire("token")

    assert called["token"] == "token"
    assert called["service_name"] == "service-ambition-generator"
    assert called["metrics"] == {"base": "full"}
    assert instruments == ["ai", "pydantic", "openai"]
    assert info_logged["msg"] == "Logfire telemetry enabled"


def test_init_logfire_without_token(monkeypatch):
    called = {}

    def configure(**kwargs):
        called.update(kwargs)

    dummy_module = SimpleNamespace(
        configure=configure,
        instrument_system_metrics=lambda **kw: None,
        info=lambda *a, **k: None,
    )

    monkeypatch.setitem(sys.modules, "logfire", dummy_module)
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)

    monitoring.init_logfire()

    assert "token" in called and called["token"] is None
