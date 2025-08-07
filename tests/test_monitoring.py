import logging
import sys
from types import SimpleNamespace

import monitoring


def test_init_logfire_replaces_root_handlers(monkeypatch):
    class DummyHandler(logging.Handler):
        def emit(self, record):  # pragma: no cover - no output
            pass

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(DummyHandler())

    class LFHandler(logging.Handler):
        def emit(self, record):  # pragma: no cover - no output
            pass

    installed = False

    def install() -> None:
        nonlocal installed
        installed = True

    dummy_module = SimpleNamespace(
        configure=lambda **kwargs: None,
        instrument_pydantic_ai=lambda: None,
        instrument_pydantic=lambda: None,
        instrument_openai=lambda: None,
        instrument_system_metrics=lambda **kwargs: None,
        install_auto_tracing=install,
        LogfireLoggingHandler=LFHandler,
    )
    monkeypatch.setitem(sys.modules, "logfire", dummy_module)
    monkeypatch.setenv("LOGFIRE_TOKEN", "token")

    monitoring.init_logfire("svc")

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], LFHandler)
    assert installed

    root_logger.handlers.clear()
