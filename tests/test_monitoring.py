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
    install_args: dict[str, object] = {}

    def install(
        modules, *, min_duration, check_imported_modules="error"
    ):  # type: ignore[no-untyped-def]
        nonlocal installed
        installed = True
        install_args["modules"] = modules
        install_args["min_duration"] = min_duration
        install_args["check"] = check_imported_modules

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
    assert install_args["modules"] == []
    assert install_args["min_duration"] == 0

    root_logger.handlers.clear()
