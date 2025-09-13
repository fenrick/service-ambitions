# Implementation plan

This document lists code-quality improvements for the core system. Remove items once implemented.

## Engine orchestration & concurrency
- **Good**: Layered engine design and deterministic runtime architecture are documented in [runtime-architecture.md](docs/runtime-architecture.md).
- **Improvements**:
  - Plateau stages so feature generation overlaps with mapping as outlined in [llm-queue-migration.md](docs/llm-queue-migration.md).
    - Create an `asyncio.Queue` in [`src/engine/processing_engine.py`](src/engine/processing_engine.py) to buffer features.
    - Update `ServiceExecution.run` to start consumer tasks that map items from the queue.
    - Add tests `tests/test_processing_engine_queue.py::test_maps_features_concurrently` and `::test_mapping_error_propagates`.
  - Centralise error handling, respecting [coding-standards.md](docs/coding-standards.md#errors-and-exceptions).
    - Create `src/engine/errors.py` with `ProcessingError` and `MappingError`.
    - Wrap each stage call in `ProcessingEngine.run` with `try/except` to raise these errors.
    - Log errors with context using structured messages.
    - Add unit tests `tests/test_processing_engine_errors.py` verifying propagation.
  - Expose queue metrics for observability per [llm-queue-migration.md](docs/llm-queue-migration.md).
    - Instrument `ProcessingEngine` with `processing_queue_inflight`, `processing_queue_submitted_total`, and `processing_queue_completed_total` via `prometheus_client`.
    - Record metric documentation in `docs/logging-levels.md`.

## Runtime environment & configuration
- **Good**: `RuntimeEnv` offers a thread-safe singleton and lazily loads prompts and mappings.
- **Improvements**:
  - Allow dependency injection of prompt and mapping loaders as detailed in [runtime-architecture.md](docs/runtime-architecture.md).
    - Define `PromptLoader` and `MappingLoader` protocols in `src/runtime/loaders.py`.
    - Modify `RuntimeEnv.__init__` to accept these loader instances.
    - Replace direct file access in tests with injected fakes.
  - Extend settings validation following [AGENTS.md](AGENTS.md).
    - Add `src/runtime/settings.py` with a `RuntimeSettings` pydantic model for `config/app.yaml`.
    - Validate settings during `RuntimeEnv.initialise()` and raise `ValueError` on invalid config.
    - Document environment overrides in `README.md` and provide examples in `config/app.example.yaml`.

## Data loading & caching
- **Good**: Caching uses atomic writes and relocates legacy files.
- **Improvements**:
  - Add integrity checks for cached JSON, following [runtime-architecture.md](docs/runtime-architecture.md#lazy-loading-and-caching).
    - Introduce `src/runtime/cache.py::validate_cache_entry(path: Path) -> bool` to ensure files contain valid JSON.
    - Skip or rebuild invalid entries and log warnings with context.
    - Add integration tests in `tests/test_cache_integrity.py` covering corrupted cache scenarios.
  - Provide CLI tooling to purge or validate cache directories.
    - Implement `--purge` and `--validate` flags in `src/cli/cache.py`.
    - Document usage in `README.md` and `config/app.example.yaml`.

## Observability & diagnostics
- **Good**: Structured logging via Logfire with spans and metrics (see [logging-levels.md](docs/logging-levels.md)).
- **Improvements**:
  - Publish per-stage latency histograms and cache hit/miss metrics.
    - Record `engine_stage_duration_seconds` histograms in `src/engine/processing_engine.py`.
    - Track `cache_hit_total` counters in `src/runtime/cache.py`.
    - Expose metrics via the existing `/metrics` endpoint.
    - Add latency alerts in `docs/observability.md`.
  - Ensure logs avoid implying generation retries, per [logging-levels.md](docs/logging-levels.md#note-on-generation-policy).
    - Search existing log messages for retry language and revise phrasing.
    - Create `scripts/log_lint.py` to flag disallowed phrases; add to CI via `pre-commit`.

## Testing & QA
- **Good**: Automated checks in [AGENTS.md](AGENTS.md) enforce formatting, linting, typing, security and coverage gates.
- **Improvements**:
  - Resolve current CI failures so the baseline suite is clean.
    - Fix linting issues in [`src/cli/mapping.py`](src/cli/mapping.py) per [coding-standards.md](docs/coding-standards.md#formatting).
      - Sort imports (`ruff I001`).
      - Refactor `remap_features` to reduce cyclomatic complexity below 8 (`ruff C901`).
      - Wrap long lines to meet the 88‑char limit (`ruff E501`).
    - Rename [`src/llm/queue.py`](src/llm/queue.py) and update imports to avoid the standard-library name clash that triggers `mypy`'s "source file found twice" error.
    - Provide test configuration with a dummy `openai_api_key` so `tests/test_async_processing.py::test_process_service_async` no longer raises `RuntimeError` from [`src/runtime/settings.py`](src/runtime/settings.py).
    - Export `Agent` from [`src/generation/generator.py`](src/generation/generator.py) or adjust the test to patch the internal import so `tests/test_async_processing.py::test_process_service_async` can monkeypatch the agent.
    - Investigate `pip-audit` SSL certificate failures and configure trusted certificates or offline indexes.
  - Increase test coverage for the LLM queue and plateau pipelining once implemented, meeting thresholds in `AGENTS.md`.
    - Add `tests/test_processing_engine_queue.py::test_graceful_shutdown` for shutdown behaviour.
    - Use `pytest.mark.slow` for long-running queue tests.
  - Add integration tests for resume functionality and cache corruption failure modes from [runtime-architecture.md](docs/runtime-architecture.md).
    - Simulate interruption and resume using temporary directories in `tests/test_resume_flow.py`.
    - Verify cache rebuilds after corruption in `tests/test_cache_integrity.py`.
  - Replace ad-hoc `sys.path` manipulation in tests with standard package imports to reduce code smells and satisfy [`AGENTS.md`](AGENTS.md) guidance.

## Documentation
- **Good**: Detailed system design notes in `docs/` clarify architecture and migration paths.
- **Improvements**:
  - Document queue enablement and tuning in `README.md` with a link to [llm-queue-migration.md](docs/llm-queue-migration.md).
    - Add a "Queue" section with startup commands and configuration flags.
    - Reference related metrics and troubleshooting tips.
  - Include cache management examples in `config/app.example.yaml` and `README.md`.
    - Show `--purge` and `--validate` usage with sample output.
    - Cross-reference integrity checks and expected log output.

Ensure all work follows the repository's standards in [AGENTS.md](AGENTS.md) and [coding-standards.md](docs/coding-standards.md).
