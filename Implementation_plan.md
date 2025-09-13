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

# Highest-impact engineering work

## 1) LLM queue hardening (retry/backoff + breaker + parity)

* **Implement**

  * Extend `src/llm/queue.py` to optionally apply the same retry/backoff logic already used in `generation/generator.py` so all LLM calls are consistently protected.

    * Add optional params to `LLMQueue.submit(..., retry: bool=False, attempts:int=6, base:float=1.0, cap:float=30.0)`.
    * Reuse/port `TRANSIENT_EXCEPTIONS`, `_parse_retry_datetime`, and `CircuitBreaker` patterns from `generation/generator.py` into a small utility module `src/llm/retry.py` to avoid duplication.
  * When `Settings.llm_queue_enabled` is true, make `ConversationSession.ask_async` pass `retry=True` to the queue for “descriptions”, “features\_*”, and “mapping\_*” stages.
* **Files**

  * `src/llm/queue.py`, `src/llm/retry.py` (new), `src/core/conversation.py`
* **Tests**

  * `tests/test_llm_queue.py::test_submit_retries_on_transient`
  * `tests/test_llm_queue.py::test_retry_after_header_honoured`
  * `tests/test_conversation_queue_parity.py::test_results_equal_with_and_without_queue` (seeded)
* **Success criteria**

  * With queue on/off and fixed seed, outputs match (byte-for-byte) across the golden fixtures for a small sample; transient errors are retried; breaker pauses after N failures.

## 2) Graceful cancellation & shutdown path

* **Implement**

  * Add signal handling for `SIGINT/SIGTERM` in the CLI to cancel in-flight tasks cleanly.
  * Ensure `ProcessingEngine.run` and `ServiceExecution.run` propagate `asyncio.CancelledError` without masking, flush temp output, and persist `processed_ids.txt`.
* **Files**

  * `src/cli/main.py`, `src/engine/processing_engine.py`, `src/engine/service_execution.py`
* **Tests**

  * `tests/test_cancellation.py::test_ctrl_c_flushes_partial_output`
  * `tests/test_cancellation.py::test_resume_after_cancel_continues_from_processed_ids`
* **Success criteria**

  * Hitting cancel mid-run leaves a valid `.tmp.part` and `processed_ids.txt`; re-run with `--resume` appends without duplication.

## 3) Cache integrity & self-healing

* **Implement**

  * Add a manifest file with SHA-256 for each cache entry and validate on read; auto-purge corrupt entries when `cache_mode` allows writes.
  * Provide a CLI subcommand `cache validate|purge` for explicit maintenance.
* **Files**

  * `src/utils/cache_manager.py` (hashing, manifest IO)
  * `src/core/conversation.py` (validate hash on `_read_cache_result`)
  * `src/cli/main.py` (subcommand)
* **Tests**

  * `tests/test_cache_integrity.py::test_detects_and_purges_corrupt_cache`
  * `tests/test_cache_integrity.py::test_validate_reports_incorrect_hash`
* **Success criteria**

  * Corrupt cache files are detected deterministically; optional auto-repair works; CLI reports accurate counters.

# Engine, pipeline & models

## 4) Plateau pipelining guardrails & observability

* **Implement**

  * When queue is enabled, confirm features→mapping overlap is bounded: add a per-service plateau semaphore to cap concurrent plateaus (prevents bursty memory usage).
  * Expose queue/plateau gauges and counters (inflight, submitted, completed) consistently via logfire metrics in `ProcessingEngine`.
* **Files**

  * `src/engine/processing_engine.py`, `src/engine/service_execution.py`, `src/llm/queue.py`
* **Tests**

  * `tests/test_processing_engine_queue.py::test_plateau_concurrency_bound`
  * `tests/test_processing_engine_metrics.py::test_metrics_increment`
* **Success criteria**

  * Peak concurrent tasks per service respect the bound; metrics appear with expected names and values.

## 5) Strict mode ergonomics & failure shaping

* **Implement**

  * In strict and `strict_mapping` modes, standardise error types and messages (use `utils.error_handler` to emit one-line, structured errors; raise domain exceptions).
  * Add `MappingError` and `ProcessingError` exception classes (as your current plan notes) and ensure they carry `service_id` and `stage`.
* **Files**

  * `src/utils/error_handler.py`, `src/engine/errors.py` (new), `src/engine/processing_engine.py`, `src/engine/service_execution.py`
* **Tests**

  * `tests/test_processing_engine_errors.py::test_mapping_error_carries_context`
  * `tests/test_cli_modes.py::test_strict_mode_exits_nonzero_on_missing_mapping`
* **Success criteria**

  * Clear, structured error surfaces; CLI exit codes are deterministic.

# CLI, UX & developer experience

## 6) End-to-end resume UX polish

* **Implement**

  * Document and enforce invariants for `--resume`: same input file hash, same output path. Refuse resume if they differ (to prevent accidental data skew).
  * Persist input file SHA-256 and settings snapshot alongside `processed_ids.txt`.
* **Files**

  * `src/engine/processing_engine.py`, `src/cli/main.py`
* **Tests**

  * `tests/test_resume_guards.py::test_refuse_resume_on_input_changed`
* **Success criteria**

  * Resume is safe-by-default; users get a precise message on mismatch.

## 7) CLI: cache & diagnostics quality of life

* **Implement**

  * Flags: `--cache validate`, `--cache purge`, `--allow-prompt-logging` already exists—add `--trace-ids` to print request IDs for failed items.
  * Automatically suppress progress when not TTY (already hinted in README) and when `--json-logs` is on; assert with tests.
* **Files**

  * `src/cli/main.py`
* **Tests**

  * `tests/test_cli_modes.py::test_progress_suppressed_in_non_tty`
* **Success criteria**

  * Cleaner CI logs; faster triage of failures.

# Observability & docs

## 8) Metric & log naming consistency

* **Implement**

  * Standardise metric names across modules (e.g., `sa_llm_queue_*`, `sa_engine_*`, `sa_mapping_*`) and document in `docs/logging-levels.md`.
  * Ensure all spans include `{stage, model_name, service_id, request_id}` when `diagnostics=true`.
* **Files**

  * `src/core/conversation.py`, `src/llm/queue.py`, `src/engine/*`, `docs/logging-levels.md`
* **Tests**

  * `tests/test_observability.py::test_span_attributes_present`
* **Success criteria**

  * Uniform metric prefix; spans consistently annotated.

## 9) README & docs correctness pass

* **Implement**

  * Fix truncated code fences and tables (there are visible `...` truncations in `README.md` and `Implementation_plan.md` itself).
  * Add a short “Enable the queue” section with flags (`llm_queue_enabled`, `llm_queue_concurrency`), troubleshooting, and metric names.
* **Files**

  * `README.md`, `docs/llm-queue-migration.md`, `docs/index.md`
* **Success criteria**

  * Copy/pasteable commands; no truncated lines; queue feature documented end-to-end.

# Quality, CI & safety nets

## 10) Test stability outside Poetry

* **Implement**

  * Make tests resilient when `pytest` isn’t run via Poetry by ensuring `pythonpath = ["src"]` is respected in all environments (fallback `conftest` sys.path insert when env var `PYTEST_PYPROJECT=0`).
* **Files**

  * `tests/conftest.py`
* **Tests**

  * `tests/test_imports.py::test_modules_import_without_poetry`
* **Success criteria**

  * `pytest -q` works locally without Poetry bootstrap.

## 11) Migrations CLI (schema + cache)

* **Implement**

  * Add a small CLI entrypoint `service-ambitions migrate --from 1.0 --to 1.1 --input in.jsonl --output out.jsonl` that wraps `migrations.schema_migration.migrate_record`.
* **Files**

  * `src/cli/main.py`, `src/migrations/schema_migration.py`
* **Tests**

  * `tests/test_cli_migrate.py::test_schema_migrate_jsonl_roundtrip`
* **Success criteria**

  * Deterministic migration across versions; preserves ordering and metadata.

---

## Quick insert template for your `Implementation_plan.md`

You can paste this block as-is and tweak wording:

> ### LLM queue hardening
>
> * Add retry/backoff & circuit-breaker in `LLMQueue.submit` (new `src/llm/retry.py` utility).
> * Wire `ConversationSession.ask_async` to pass `retry=True` when queue is enabled.
> * **Tests:** `tests/test_llm_queue.py::{test_submit_retries_on_transient,test_retry_after_header_honoured}`; `tests/test_conversation_queue_parity.py::test_results_equal_with_and_without_queue`.
> * **Done when:** parity holds on golden fixtures with/without queue; transient errors are retried; breaker pauses after N failures.

> ### Graceful cancellation
>
> * Handle SIGINT/SIGTERM in `cli/main.py`; ensure safe flush of `.tmp.part` and `processed_ids.txt` in `ProcessingEngine`.
> * **Tests:** `tests/test_cancellation.py::{test_ctrl_c_flushes_partial_output,test_resume_after_cancel_continues_from_processed_ids}`.
> * **Done when:** resume appends without dupes after cancellation.

> ### Cache integrity & self-healing
>
> * Add SHA-256 manifest + validation on cache read; `cache validate|purge` subcommand.
> * **Tests:** `tests/test_cache_integrity.py::{test_detects_and_purges_corrupt_cache,test_validate_reports_incorrect_hash}`.
> * **Done when:** corrupt files detected and optionally repaired; CLI reports accurate counts.

> ### Plateau pipelining guardrails
>
> * Bound per-service plateau concurrency; add metrics.
> * **Tests:** `tests/test_processing_engine_queue.py::test_plateau_concurrency_bound`; `tests/test_processing_engine_metrics.py::test_metrics_increment`.
> * **Done when:** bound respected; metrics emit as `sa_engine_*`.

> ### Strict mode error shaping
>
> * Introduce `ProcessingError`/`MappingError` with context; standardise messages.
> * **Tests:** `tests/test_processing_engine_errors.py::test_mapping_error_carries_context`; `tests/test_cli_modes.py::test_strict_mode_exits_nonzero_on_missing_mapping`.
> * **Done when:** structured errors; deterministic exit codes.

> ### Resume UX guardrails
>
> * Persist input hash + settings snapshot; refuse resume on mismatch.
> * **Test:** `tests/test_resume_guards.py::test_refuse_resume_on_input_changed`.
> * **Done when:** safe resume with clear messages.

> ### CLI diagnostics QoL
>
> * Add `--cache validate|purge`, `--trace-ids`; suppress progress on non-TTY/`--json-logs`.
> * **Test:** `tests/test_cli_modes.py::test_progress_suppressed_in_non_tty`.
> * **Done when:** cleaner logs and faster triage.

> ### Metrics naming & span attributes
>
> * Prefix all metrics `sa_*`; ensure spans include `{stage, model_name, service_id, request_id}`.
> * **Test:** `tests/test_observability.py::test_span_attributes_present`.
> * **Done when:** names consistent; attributes present.

> ### Docs correctness & queue guide
>
> * Fix truncated lines; add queue enablement + troubleshooting section.
> * **Done when:** README sections copy/paste cleanly; links valid.

> ### Pytest outside Poetry
>
> * Fallback `sys.path` insert in `tests/conftest.py` when `PYTEST_PYPROJECT=0`.
> * **Test:** `tests/test_imports.py::test_modules_import_without_poetry`.
> * **Done when:** `pytest -q` works without Poetry.

> ### Migrations CLI
>
> * Add `service-ambitions migrate ...` wrapper over `migrations.schema_migration`.
> * **Test:** `tests/test_cli_migrate.py::test_schema_migrate_jsonl_roundtrip`.
> * **Done when:** JSONL migration is deterministic and validated.

Ensure all work follows the repository's standards in [AGENTS.md](AGENTS.md) and [coding-standards.md](docs/coding-standards.md).
