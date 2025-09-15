# Implementation plan

This document lists code-quality improvements for the core system. Remove items once implemented.

Wherever possible we configure existing frameworks or compose lightweight tools
rather than building new infrastructure from scratch. The plan below assumes
using off-the-shelf libraries (e.g., Tenacity for retries) to meet these goals.

# Highest-impact engineering work

## Cache integrity & self-healing

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

## Plateau pipelining guardrails & observability

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

## Strict mode ergonomics & failure shaping

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

## CLI: cache & diagnostics quality of life

* **Implement**

  * Flags: `--cache validate`, `--cache purge`, `--allow-prompt-logging` already exists—add `--trace-ids` to print request IDs for failed items.
* **Files**

  * `src/cli/main.py`
* **Tests**

  * `tests/test_cli_modes.py::test_trace_ids_reports_request_ids`
* **Success criteria**

  * Cleaner CI logs; faster triage of failures.

# Observability & docs

## Metric & log naming consistency

* **Implement**

  * Standardise metric names across modules (e.g., `sa_llm_queue_*`, `sa_engine_*`, `sa_mapping_*`) and document in `docs/logging-levels.md`.
  * Ensure all spans include `{stage, model_name, service_id, request_id}` when `diagnostics=true`.
* **Files**

  * `src/core/conversation.py`, `src/llm/queue.py`, `src/engine/*`, `docs/logging-levels.md`
* **Tests**

  * `tests/test_observability.py::test_span_attributes_present`
* **Success criteria**

  * Uniform metric prefix; spans consistently annotated.

## README & docs correctness pass

* **Implement**

  * Fix truncated code fences and tables (there are visible `...` truncations in `README.md` and `Implementation_plan.md` itself).
  * Add a short “Enable the queue” section with flags (`llm_queue_enabled`, `llm_queue_concurrency`), troubleshooting, and metric names.
* **Files**

  * `README.md`, `docs/llm-queue-migration.md`, `docs/index.md`
* **Success criteria**

  * Copy/pasteable commands; no truncated lines; queue feature documented end-to-end.

# Quality, CI & safety nets

## Migrations CLI (schema + cache)

* **Implement**

  * Add a small CLI entrypoint `service-ambitions migrate --from 1.0 --to 1.1 --input in.jsonl --output out.jsonl` that wraps `migrations.schema_migration.migrate_record`.
* **Files**

  * `src/cli/main.py`, `src/migrations/schema_migration.py`
* **Tests**

  * `tests/test_cli_migrate.py::test_schema_migrate_jsonl_roundtrip`
* **Success criteria**

  * Deterministic migration across versions; preserves ordering and metadata.

---

## Provider-aware rate limiting (RPM/TPM) in the global LLM queue

* **Implement**

  * Add optional rate limiters to `LLMQueue` for **requests/min** and **tokens/min** per provider/model.
  * Config via `settings.models_rate_limits: dict[str, dict[str, int]]` (e.g., `{"openai:o4-mini": {"rpm": 240, "tpm": 600000}}`).
  * Token accounting: expose a lightweight hook on `ConversationSession` to pass returned token counts back to the queue limiter.
* **Files**

  * `src/llm/queue.py` (token bucket / leaky bucket), `src/runtime/settings.py` (new field), `src/core/conversation.py` (return tokens to limiter).
* **Tests**

  * `tests/test_llm_queue.py::test_rpm_limiter_throttles`
  * `tests/test_llm_queue.py::test_tpm_limiter_throttles`
* **Success criteria**

  * Sustained load stays within configured RPM/TPM; no burst overrun beyond 10%.

## Cost tracking, budgets & end-of-run report

* **Implement**

  * Maintain per-stage token counts and **estimated cost** using a small pricing map (USD per 1K input/output tokens).
  * Add `Settings.run_budget_usd` and hard/soft budget modes (`"enforce"` aborts politely when exceeded).
  * Emit a final **cost summary** and per-service breakdown (when `diagnostics=true`).
* **Files**

  * `src/models/pricing.py` (pricing map), `src/observability/telemetry.py` (aggregate), `src/cli/main.py` (print summary), `src/runtime/settings.py`.
* **Tests**

  * `tests/test_costs.py::{test_cost_estimates_aggregate,test_budget_enforced_abort}`
* **Success criteria**

  * ±2% cost estimation accuracy on mocked fixtures; run aborts when over budget in “enforce” mode.

## Secrets & PII safety in logs and transcripts

* **Implement**

  * Add a **scrubber** that redacts common PII and known secret patterns in: prompt/response transcripts, diagnostic spans, and error messages.
  * Default to `--allow-prompt-logging` **off**; when on, scrub before write.
  * Add `SA_REDACTION_PATTERNS` to extend masks without code changes.
* **Files**

  * `src/utils/scrub.py` (regex set + unit tests), `src/core/conversation.py` (apply before transcript write), `src/observability/monitoring.py` (span/event scrub).
* **Tests**

  * `tests/test_redaction.py::{test_secrets_masked,test_pii_masked_in_transcripts}`
* **Success criteria**

  * No raw secrets/PII appear in transcripts or logs under test fixtures.

## Reproducibility: seeds, git SHA & config snapshot

* **Implement**

  * Add `--seed` (propagate to all RNG uses); record **git SHA**, **settings snapshot**, and **models in use** in `evolutions.jsonl` header record.
  * Ensure deterministic sort orders and stable serialisation (already partly done) across platforms.
* **Files**

  * `src/cli/main.py` (args + header writer), `src/engine/service_execution.py` (header emission), `src/io_utils/persistence.py` (helper).
* **Tests**

  * `tests/test_reproducibility.py::{test_same_seed_same_output,test_header_contains_metadata}`
* **Success criteria**

  * Byte-for-byte identical outputs with the same seed, SHA and config.

## Model fallback & chaos testing

* **Implement**

  * Add `Settings.model_fallbacks: dict[str, list[str]]` (e.g., primary → candidates).
  * In the queue wrapper, on specific non-transient failures (or model unavailability), **retry on next fallback** using a configured library rather than custom helpers; surface which model produced the result.
  * Provide a chaos flag `--inject-failures p=0.05` for tests to randomly raise retriable errors.
* **Files**

  * `src/llm/queue.py`, `src/generation/generator.py` (optional: reuse fallback path), `src/runtime/settings.py`, `src/cli/main.py`.
* **Tests**

  * `tests/test_fallbacks.py::{test_fallback_on_unavailable_model,test_inject_failures_exercises_fallbacks}`
* **Success criteria**

  * When primaries fail, success rate recovers using fallbacks; model attribution is logged.

## Cache eviction, TTLs & quotas

* **Implement**

  * Add **LRU size cap** (bytes) and optional **TTL** per stage; evict oldest or expired entries.
  * Extend `cache validate|purge` with `--older-than` and `--max-size` to trim.
* **Files**

  * `src/utils/cache_manager.py` (index + eviction), `src/cli/main.py` (flags), `src/runtime/settings.py` (caps).
* **Tests**

  * `tests/test_cache_eviction.py::{test_lru_under_size_cap,test_ttl_expiry_removes_entries}`
* **Success criteria**

  * Cache stays under configured limits; validate/purge reports accurate counts.

## Watchdog for stuck tasks + hard timeouts

* **Implement**

  * Add a **per-stage timeout** (settings: `descriptions_timeout`, `features_timeout`, `mapping_timeout`) enforced in `ConversationSession._ask_common` with `asyncio.wait_for`.
  * Add a watchdog in `ProcessingEngine` that logs any task active > 95% of timeout and cancels on overrun.
* **Files**

  * `src/core/conversation.py`, `src/engine/processing_engine.py`, `src/runtime/settings.py`.
* **Tests**

  * `tests/test_timeouts.py::{test_stage_timeout_cancels,test_watchdog_logs_slow_task}`
* **Success criteria**

  * Hung calls are cancelled and reported; pipeline progresses for the rest.

## Performance harness & golden-throughput target

* **Implement**

  * Add a **mock LLM** (fast deterministic agent) and `scripts/bench.py` to measure E2E throughput (svc/sec), p50/p95 latencies, queue depth.
  * Fail CI if regression >15% vs. golden baselines stored in repo.
* **Files**

  * `tests/perf/test_throughput.py`, `scripts/bench.py`, `docs/perf.md`.
* **Tests**

  * `tests/perf/test_throughput.py::test_throughput_regression_guard`
* **Success criteria**

  * Baseline recorded; subsequent PRs get automatic perf guardrails.

## Supply-chain hygiene: SBOM, licence scan, provenance

* **Implement**

  * Generate **CycloneDX SBOM** on CI, publish as artefact; add **licence allowlist** check.
  * Add SLSA provenance (GitHub Attestations) for release artefacts.
* **Files**

  * `.github/workflows/ci-main.yml` (add `cyclonedx-bom`, `pip-licenses`, attestation step), `docs/security.md`.
* **Tests**

  * CI-only; verify artefacts exist and jobs pass.
* **Success criteria**

  * SBOM attached to builds; CI fails on disallowed licences.

## CLI UX: `quarantine` triage & `--trace-ids`

* **Implement**

  * Add `quarantine ls|show|rm` subcommands; implement `--trace-ids` to print provider request IDs on failures (already suggested—wire it fully).
* **Files**

  * `src/cli/main.py`, `src/io_utils/quarantine.py`.
* **Tests**

  * `tests/test_quarantine_cli.py::{test_ls_lists_cases,test_show_displays_details}`
* **Success criteria**

  * Quarantined records easily discoverable and actionable from CLI.

---

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
## Quick insert template for your `Implementation_plan.md`

You can paste this block as-is and tweak wording:

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

> ### Migrations CLI
>
> * Add `service-ambitions migrate ...` wrapper over `migrations.schema_migration`.
> * **Test:** `tests/test_cli_migrate.py::test_schema_migrate_jsonl_roundtrip`.
> * **Done when:** JSONL migration is deterministic and validated.

Ensure all work follows the repository's standards in [AGENTS.md](AGENTS.md) and [coding-standards.md](docs/coding-standards.md).

---

### Ready-to-paste template

> ## Provider-aware rate limiting
>
> * Add RPM/TPM token buckets to `LLMQueue`; feed token counts from `ConversationSession`.
> * **Files:** `src/llm/queue.py`, `src/runtime/settings.py`, `src/core/conversation.py`.
> * **Tests:** `tests/test_llm_queue.py::{test_rpm_limiter_throttles,test_tpm_limiter_throttles}`.
> * **Done when:** sustained load stays within configured limits.

> ## Cost tracking & budgets
>
> * Track tokens → cost; `run_budget_usd` with `"enforce"` mode; print summary.
> * **Files:** `src/models/pricing.py`, `src/observability/telemetry.py`, `src/cli/main.py`, `src/runtime/settings.py`.
> * **Tests:** `tests/test_costs.py`.
> * **Done when:** costs reported and budget enforced.

> ## Secrets/PII redaction
>
> * Scrub logs/transcripts; default prompt logging off.
> * **Files:** `src/utils/scrub.py`, `src/core/conversation.py`, `src/observability/monitoring.py`.
> * **Tests:** `tests/test_redaction.py`.
> * **Done when:** no PII/secrets leak in fixtures.

> ## Reproducibility metadata
>
> * `--seed`, header with git SHA + config + models.
> * **Files:** CLI/engine/persistence.
> * **Tests:** `tests/test_reproducibility.py`.
> * **Done when:** byte-stable outputs with same seed.

> ## Model fallbacks & chaos
>
> * `model_fallbacks` + failure injection.
> * **Files:** `src/llm/queue.py`, `src/runtime/settings.py`.
> * **Tests:** `tests/test_fallbacks.py`.
> * **Done when:** fallbacks used on targeted failures.

> ## Cache eviction & TTL
>
> * LRU + TTL; CLI trim flags.
> * **Files:** `src/utils/cache_manager.py`, `src/cli/main.py`.
> * **Tests:** `tests/test_cache_eviction.py`.
> * **Done when:** cache stays within caps.

> ## Watchdog & stage timeouts
>
> * Per-stage `asyncio.wait_for`; watchdog logs/cancels.
> * **Files:** `src/core/conversation.py`, `src/engine/processing_engine.py`.
> * **Tests:** `tests/test_timeouts.py`.
> * **Done when:** hung calls are cancelled.

> ## Perf harness & guardrails
>
> * Mock LLM + bench script; fail CI on >15% regression.
> * **Files:** `tests/perf/`, `scripts/bench.py`.
> * **Tests:** `tests/perf/test_throughput.py`.
> * **Done when:** baseline + regression checks in CI.

> ## SBOM & provenance
>
> * CycloneDX, licence check, SLSA attestations in CI.
> * **Files:** `.github/workflows/ci-main.yml`.
> * **Done when:** SBOM attaches; CI blocks bad licences.
