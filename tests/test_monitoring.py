import json
import logging
import sys
from types import SimpleNamespace

import monitoring


def test_init_logfire_replaces_root_handlers(monkeypatch):
    class DummyHandler(logging.Handler):
        def emit(self, record):
            pass

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = DummyHandler()
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    class LFHandler(logging.Handler):
        def emit(self, record):
            pass

    installed = False
    install_args: dict[str, object] = {}

    def install(modules, *, min_duration, check_imported_modules="error"):
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
    new_handler = handlers[0]
    assert isinstance(new_handler, LFHandler)
    assert new_handler.level == logging.WARNING
    assert isinstance(new_handler.formatter, logging.Formatter)
    assert new_handler.formatter._fmt == "%(message)s"
    assert installed
    assert install_args["modules"] == []
    assert install_args["min_duration"] == 0

    root_logger.handlers.clear()


def test_init_logfire_uses_json_fallback(tmp_path, monkeypatch):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    monitoring.init_logfire()
    logging.getLogger().info("hello")
    logging.getLogger().handlers[0].flush()
    with open(monitoring.LOG_FILE_NAME, encoding="utf-8") as fh:
        assert json.loads(fh.read())["message"] == "hello"

    root_logger.handlers.clear()
