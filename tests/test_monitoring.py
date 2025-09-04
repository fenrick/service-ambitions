from types import SimpleNamespace

from observability import monitoring


def test_init_logfire_configures_and_instruments(monkeypatch):
    called: dict[str, object] = {}
    instruments: list[str] = []
    logs: list[tuple[str, dict[str, object]]] = []

    class ConsoleOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dummy_module = SimpleNamespace(
        ConsoleOptions=ConsoleOptions,
        configure=lambda **kwargs: called.update(kwargs),
        instrument_system_metrics=lambda **kw: called.setdefault("metrics", kw),
        instrument_pydantic_ai=lambda: instruments.append("ai"),
        instrument_pydantic=lambda: instruments.append("pydantic"),
        instrument_openai=lambda: instruments.append("openai"),
        debug=lambda msg, **kw: logs.append((msg, kw)),
        add_masking_rule=lambda value: called.setdefault("mask", value),
    )
    monkeypatch.setattr(monitoring, "logfire", dummy_module)

    monitoring.init_logfire("token", "info")

    assert called["token"] == "token"
    assert called["mask"] == "token"
    assert called["service_name"] == "service-ambition-generator"
    assert called["console"].min_log_level == "info"
    assert called["metrics"] == {"base": "full"}
    assert instruments == ["ai", "pydantic", "openai"]
    assert logs[0][1]["token"] == "toke..."


def test_init_logfire_without_token(monkeypatch):
    called: dict[str, object] = {}
    logs: list[tuple[str, dict[str, object]]] = []

    class ConsoleOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dummy_module = SimpleNamespace(
        ConsoleOptions=ConsoleOptions,
        configure=lambda **kwargs: called.update(kwargs),
        instrument_system_metrics=lambda **kw: None,
        instrument_pydantic_ai=lambda: None,
        instrument_pydantic=lambda: None,
        instrument_openai=lambda: None,
        debug=lambda msg, **kw: logs.append((msg, kw)),
        add_masking_rule=lambda value: None,
    )
    monkeypatch.setattr(monitoring, "logfire", dummy_module)
    monkeypatch.delenv("SA_LOGFIRE_TOKEN", raising=False)

    monitoring.init_logfire()

    assert "token" in called and called["token"] is None
    assert logs[0][1]["token"] is None


def test_init_logfire_masks_token(monkeypatch):
    logs: list[tuple[str, dict[str, object]]] = []

    class ConsoleOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dummy_module = SimpleNamespace(
        ConsoleOptions=ConsoleOptions,
        configure=lambda **kwargs: None,
        instrument_system_metrics=lambda **kw: None,
        instrument_pydantic_ai=lambda: None,
        instrument_pydantic=lambda: None,
        instrument_openai=lambda: None,
        debug=lambda msg, **kw: logs.append((msg, kw)),
        add_masking_rule=lambda value: None,
    )
    monkeypatch.setattr(monitoring, "logfire", dummy_module)

    monitoring.init_logfire("secret-token", "info")

    assert logs
    assert logs[0][1]["token"] == "secr..."
    assert "secret-token" not in logs[0][1]["token"]
