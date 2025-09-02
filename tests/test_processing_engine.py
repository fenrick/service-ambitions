import argparse
import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from engine.processing_engine import ProcessingEngine
from engine.service_runtime import ServiceRuntime
from models import ServiceInput
from test_processing_engine_methods import _make_args
from utils import LoggingErrorHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    ["factory", "system_prompt", "role_ids", "sem", "error_handler"],
)
async def test_run_service_missing_dependency_raises(
    tmp_path: Path, missing: str
) -> None:
    args = cast(argparse.Namespace, _make_args(tmp_path))
    engine = ProcessingEngine(args, None)
    service = ServiceInput(
        service_id="svc",
        name="svc",
        description="d",
        jobs_to_be_done=[],
    )

    # Set up valid dependencies
    engine.factory = cast(Any, object())
    engine.system_prompt = "prompt"
    engine.role_ids = ["role"]
    engine.sem = asyncio.Semaphore(1)
    engine.error_handler = LoggingErrorHandler()

    # Remove the dependency under test
    setattr(engine, missing, None)

    expected = {
        "factory": "Model factory is not initialised",
        "system_prompt": "System prompt is not loaded",
        "role_ids": "Role identifiers are not loaded",
        "sem": "Concurrency semaphore is not configured",
        "error_handler": "Error handler is not configured",
    }
    with pytest.raises(RuntimeError) as exc_info:
        await engine._run_service(service)
    assert str(exc_info.value) == expected[missing]


@pytest.mark.asyncio
async def test_finalise_missing_line_raises(tmp_path: Path) -> None:
    args = cast(argparse.Namespace, _make_args(tmp_path))
    engine = ProcessingEngine(args, None)
    service = ServiceInput(
        service_id="svc",
        name="svc",
        description="d",
        jobs_to_be_done=[],
    )
    runtime = ServiceRuntime(service)
    runtime.success = True
    engine.runtimes.append(runtime)

    with pytest.raises(RuntimeError) as exc_info:
        await engine.finalise()
    assert "producing a line" in str(exc_info.value)
