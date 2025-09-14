# Implementation Plan

This document consolidates implementation workstreams into a single plan with clear objectives and consistent per‑item structure: What’s Needed, Where It’s Needed, and Definitions of Done. It currently incorporates the LLM Queue Migration workstream and can be extended with additional items as they are defined.

Conventions for each item in this plan:
- Objective: outcome this item delivers and why it matters.
- What’s Needed: concrete changes, tests, or decisions required.
- Where It’s Needed: exact code paths, configs, or docs to touch.
- Definition of Done: verifiable criteria that mark completion.
- Status/Notes: current state, decisions, or traceability items.

---

## Phase 0 — Foundations (Do First)

Early, cross‑cutting tasks to unblock safe implementation and rollout. Complete these before substantial feature work.

### Tooling and CI Parity

Objective
- Guarantee local and CI run the same gates with identical flags; minimise drift.

What’s Needed
- Confirm `pyproject.toml` tool configs match coding standards (Ruff selects incl. `C901`, docstring convention `google`, Black preview flags); ensure coverage branch tracking is enabled.
- Add convenience commands (scripts or Makefile targets) to run `fmt`, `lint`, `type`, `sec`, `audit`, `test` consistently.
- Ensure CI workflows call the same commands to avoid divergence.

Where It’s Needed
- `pyproject.toml`, `Makefile`, `.github/workflows/*.yml`.

Definition of Done
- Single entrypoints exist for each gate and are used by CI; branch coverage enabled; local and CI outputs match.

Status/Notes
- CI already runs the gates; add/confirm Makefile or Poetry script parity for developer convenience.

---

### Pre‑commit Bootstrap

Objective
- Catch issues before commits and keep diffs clean.

What’s Needed
- Ensure `.pre-commit-config.yaml` covers Black, Ruff, mypy, Bandit (already present) and document installation/usage.
- Optionally add a basic secret scanner.

Where It’s Needed
- `.pre-commit-config.yaml`, `CONTRIBUTING.md` (installation), `AGENTS.md` (local commands).

Definition of Done
- `poetry run pre-commit install` and `poetry run pre-commit run --all-files` pass; optional secret scan configured or explicitly deferred with rationale.

Status/Notes
- Hook set covers the core tools; secret scanning can be added as a follow‑up if desired.

---

### Architecture Doc (Queue)

Objective
- Document the LLM queue design, control flow, and telemetry for maintainers and operators.

What’s Needed
- Create `docs/runtime-architecture.md` with a high‑level diagram, queue lifecycle, concurrency model, and integration points; add tuning guidance.
- Add to MkDocs nav for discoverability.

Where It’s Needed
- `docs/runtime-architecture.md`, `mkdocs.yml` (nav update).

Definition of Done
- Page renders in `mkdocs build --strict`; includes diagram or ASCII sketch, sequence overview, metrics/spans table, and tuning notes.

Status/Notes
- Not yet present; referenced in this plan and should be added early.

---

### Runbook (Enablement / Rollback)

Objective
- Provide a concise operational runbook for enabling, tuning, and rolling back the queue.

What’s Needed
- Create `docs/runbook.md` detailing feature flag toggling, safe concurrency defaults by provider, monitoring checks, and rollback steps.
- Link from README and plan.

Where It’s Needed
- `docs/runbook.md`, `README.md` (links), this plan (Enablement section).

Definition of Done
- Runbook exists, is actionable, and validated by a dry‑run in staging; referenced from README and this plan.

Status/Notes
- To be authored alongside Observability tasks so guidance is grounded in actual metrics.

---

### Testing Scaffolds (Deterministic)

Objective
- Enable reliable async/concurrency tests without flakiness.

What’s Needed
- Provide test helpers for time control or fakes (avoid `time.sleep` in async tests), and an LLM provider stub with deterministic responses.
- Shared fixtures for event loop and queue setup/teardown; helpers to assert order and concurrency without relying on wall‑clock.

Where It’s Needed
- `tests/conftest.py`, `tests/utils/` (helpers), `tests/test_llm_queue.py` (usage).

Definition of Done
- New tests for cancellation/timeouts/drain pass deterministically in CI; no sleeps in async paths; fixtures reused across tests.

Status/Notes
- Build before adding more queue edge‑case tests.

---

### Error Taxonomy

Objective
- Standardise error types for queue and provider failures to simplify handling and testing.

What’s Needed
- Define domain exceptions and wrap provider errors with causes; document retryable vs fatal classifications.

Where It’s Needed
- `src/llm/errors.py` (new) and usages in `src/llm/queue.py`, `src/core/conversation.py`, `src/generation/*`.

Definition of Done
- Tests assert exception types/messages; callers handle classes appropriately; no silent swallowing.

Status/Notes
- Create a tracking issue and introduce incrementally to avoid large refactors.

---

### Telemetry Redaction

Objective
- Ensure logs, spans, and metrics never leak secrets or PII.

What’s Needed
- Implement and apply a redaction helper for telemetry attributes; audit current spans/metrics for sensitive fields.

Where It’s Needed
- `src/observability/` (helper), `src/llm/queue.py` and call sites; documentation in `SECURITY.md`.

Definition of Done
- Bandit passes; manual spot‑checks show no sensitive data; tests cover redaction behaviour where practical.

Status/Notes
- Align with existing Logfire configuration and attributes.

---

### Commands Unification (DX)

Objective
- Provide a single set of commands to run all gates locally and in CI.

What’s Needed
- Add Make targets or Poetry scripts: `fmt`, `lint`, `type`, `sec`, `audit`, `test`, `check` (all).

Where It’s Needed
- `Makefile` and/or `pyproject.toml`.

Definition of Done
- `make check` (or `poetry run task check`) runs all gates; CI uses the same.

Status/Notes
- Consider keeping both Makefile and Poetry scripts for flexibility.

---

### Coverage Config Hygiene

Objective
- Avoid ambiguity by using a single coverage configuration.

What’s Needed
- Consolidate to one coverage config file and ensure branch coverage and thresholds are set.

Where It’s Needed
- `.coveragerc` and `coveragerc` (dedupe to one), CI pytest step.

Definition of Done
- Only one coverage config remains; CI and local runs read the same file; thresholds enforced.

Status/Notes
- Both `.coveragerc` and `coveragerc` currently exist; remove one after confirming usage.

## Workstream: LLM Queue Migration (Progressive)

Objective
- Introduce a global LLM execution queue with bounded concurrency while keeping the main processing line simple and stable. Each step is feature‑flagged and independently shippable.

### Current Status (Summary)

- Done
  - Bounded queue: `src/llm/queue.py` introduces `LLMQueue` with concurrency gate and basic metrics (`llm_queue_inflight`, `llm_queue_submitted`, `llm_queue_completed`).
  - Feature flags: `llm_queue_enabled`, `llm_queue_concurrency` added to `src/runtime/settings.py` and surfaced via config (`config/app.example.yaml`).
  - Environment bootstrap: `RuntimeEnv` initialises a global `LLMQueue` when enabled (`src/runtime/environment.py`).
  - Session routing: `ConversationSession.ask_async` routes via the global queue when enabled (`src/core/conversation.py`).
  - Pipelined plateau stages: features for plateau N+1 start while mapping for plateau N runs, bounded by the global queue (`src/generation/plateau_generator.py`).
  - Unified ambitions generator: `ServiceAmbitionGenerator` uses the global queue when enabled and disables its local limiter to avoid double gating (`src/generation/generator.py`).

---

### Step 1 — Verify Parity (flag off by default)

Objective
- Maintain identical behaviour with the feature flag off; no functional regressions.

What’s Needed
- Run CI with `llm_queue_enabled = false` (default) and ensure all tests pass.
- Manual spot‑check locally: run a small dataset once with the flag off and once with it on; compare outputs for parity where deterministic (allow minor model variance).

Where It’s Needed
- Config defaults remain flag‑off (`src/runtime/settings.py` defaults; `config/app.yaml` omits the flag; `config/app.example.yaml` documents it).
- Test execution via `pytest` and existing coverage gates.

Definition of Done
- CI green with the flag off; no new failures introduced.
- Local run confirms comparable outputs and identical control‑flow (cache usage, transcripts) aside from expected model variance.

Status/Notes
- Default remains off; parity verified before enabling by default in any environment.

---

### Step 2 — Queue‑level Retry/Backoff and Circuit Breaker (Skipped)

Objective
- Keep the queue minimal for this iteration; rely on existing component‑level retry/backoff/breaker behaviour.

What’s Needed
- Document the decision and create a tracking issue for a future enhancement (cover jittered backoff, error classification, breaker thresholds, and provider‑specific retry hints).

Where It’s Needed
- Documentation: this file.
- Future code hooks (when implemented) would live in `src/llm/queue.py`, controlled via settings.

Definition of Done
- Decision recorded here and a tracking issue created with clear scope and acceptance criteria.

Status/Notes
- Decision: Skipped for this iteration to reduce complexity and risk; revisit post‑observability.

---

### Step 3 — Pipeline Plateau Stages

Objective
- Overlap plateau work to improve throughput: start features for plateau N+1 while mapping for plateau N is running; rely on the global queue to enforce the concurrency cap.

What’s Needed
- Implement pipelined scheduling guarded by the feature flag (reuse `llm_queue_enabled`).
- Verify with tests that overlapping occurs while respecting concurrency; ensure sequential fallback when the flag is off.

Where It’s Needed
- Scheduling logic in `PlateauGenerator._schedule_plateaus` (`src/generation/plateau_generator.py`).
- Tests validating overlap ordering and fallback behaviour (`tests/test_llm_queue.py`).
- Settings flag use in `RuntimeEnv` and consumers.

Definition of Done
- Pipelined schedule implemented and exercised when `llm_queue_enabled` is true.
- Tests assert that mapping for plateau 1 can be scheduled before features for plateau 2 complete, and that execution is sequential when the flag is false.
- Concurrency is globally bounded by `LLMQueue` across sessions and stages.

Status/Notes
- Implemented and covered by tests.
- Original TODOs (retained for traceability)
  - [x] Implement pipelined scheduling (guarded by a flag; reusing `llm_queue_enabled`).
  - [x] Add tests to assert overlapping occurs (order/timestamps) — see `tests/test_llm_queue.py::test_plateau_pipeline_overlaps_when_enabled`.

---

### Step 4 — Unify Other Generators

Objective
- Route all generator LLM calls through the shared queue to centralise concurrency and observability; disable local limiters when the global queue is active to avoid double gating.

What’s Needed
- Integrate ambitions generation with the global queue and remove or bypass local concurrency controls under the flag.

Where It’s Needed
- `ServiceAmbitionGenerator` in `src/generation/generator.py` (routing via `RuntimeEnv.instance().llm_queue` and disabling local limiter when the flag is on).

Definition of Done
- Ambitions generator uses the global queue when enabled and behaves identically when disabled.

Status/Notes
- Done.

---

### Step 5 — Observability and Tuning

Objective
- Provide actionable visibility into queue health and stage throughput to tune concurrency and respect provider rate limits.

What’s Needed
- Metrics: expand beyond the existing inflight/submitted/completed to include latency histograms and per‑stage counters (labels include `stage`, `model_name`, `service_id`).
- Spans: ensure queue and call spans include key attributes for easy correlation in Logfire.
- Guidance: add brief operational notes on sizing `llm_queue_concurrency` vs provider limits and expected throughput.

Where It’s Needed
- Metrics/span additions in `src/llm/queue.py` and call‑sites that set `LLMTaskMeta` (e.g., `src/core/conversation.py`, `src/generation/generator.py`).
- Documentation updates in this file and, optionally, `docs/runtime-architecture.md`.

Definition of Done
- Histogram metrics available per stage; counters reflect per‑stage volumes; spans carry attributes for root‑cause analysis.
- Short “tuning” section documented with recommendations and safe defaults.

Status/Notes
- Partially done (basic counters/gauge present); latency histograms and guidance to be added.

---

### Enablement Checklist (Rollout)

Objective
- Safely enable the global queue under real workloads while respecting rate limits and maintaining parity.

What’s Needed
- Turn on the feature flag and set a safe concurrency value for your provider/account limits.
- Validate rate‑limit compliance and end‑to‑end throughput under load.

Where It’s Needed
- Config: set in `config/app.yaml` (or environment via `SA_LLM_QUEUE_ENABLED`, `SA_LLM_QUEUE_CONCURRENCY`). See commented examples in `config/app.example.yaml`.
- Runtime: ensure `RuntimeEnv.initialize` is called with loaded settings (already in place).

Definition of Done
- `llm_queue_enabled: true` with tuned `llm_queue_concurrency` shipped.
- Provider rate limits observed (no throttling/HTTP 429s sustained during steady‑state).
- Throughput and error rates monitored; concurrency adjusted as needed.

Status/Notes
- Proceed via staged rollout: dev → staging → production, with observability checks at each stage.

---

## Coding Standards Integration (Cross‑cutting)

These items ensure the LLM Queue Migration aligns with repository‑wide standards and CI gates.

### Tooling Parity and Gates

Objective
- Enforce consistent local and CI execution of formatting, linting, typing, security, dependency audit, and tests per the coding standards.

What’s Needed
- Use Poetry scripts or equivalent to run: Black (with preview + `string_processing`), Ruff (imports, errors, complexity), mypy (strict), Bandit, pip‑audit, and pytest with coverage gates.
- Ensure pre‑commit runs these checks locally.

Where It’s Needed
- `pyproject.toml` (tool configs and optional scripts), `.pre-commit-config.yaml`, CI workflow under `.github/workflows/*`, `AGENTS.md` and `CONTRIBUTING.md` for docs.

Definition of Done
- The following commands pass locally and in CI using the same entry points:
  - `poetry run black --preview --enable-unstable-feature string_processing .`
  - `poetry run ruff check .`
  - `poetry run mypy src`
  - `poetry run bandit -r src -ll`
  - `poetry run pip-audit`
  - `poetry run pytest --maxfail=1 --disable-warnings -q --cov=src --cov-report=term-missing --cov-fail-under=85`
- Pre‑commit hooks configured and runnable: `poetry run pre-commit run --all-files`.

Status/Notes
- Tooling files present; ensure any new code paths introduced by the queue are covered by the tools.

---

### Complexity Budgets (< 8)

Objective
- Keep cyclomatic complexity under 8 for functions impacted by the queue and pipeline changes.

What’s Needed
- Audit queue and scheduling code; refactor conditional logic and extract helpers where complexity spikes.
- For temporary waivers, add `# noqa: C901  # reason + link to issue` and create a refactor issue.

Where It’s Needed
- `src/llm/queue.py`, `src/generation/plateau_generator.py`, `src/generation/generator.py`, `src/core/conversation.py`.

Definition of Done
- Ruff `C901` has zero violations in changed/added code; any waiver includes justification and a linked issue with owner and due date.

Status/Notes
- Review after adding observability and cancellation paths, which can increase branching.

---

### Docstrings and Public API Hygiene

Objective
- Ensure public interfaces are documented with Google‑style docstrings for readability and maintenance.

What’s Needed
- Add/align docstrings covering purpose, params, returns, raises, and side effects for public classes/methods.

Where It’s Needed
- Public APIs in `src/llm/queue.py`, `src/generation/plateau_generator.py`, `src/generation/generator.py`, `src/core/conversation.py`.

Definition of Done
- Docstrings present and accurate; docstring lints (if enabled) pass; reviewers can infer intent without reading implementation details.

Status/Notes
- Update examples and references if public method signatures change.

---

### Strict Typing (mypy)

Objective
- Maintain strict, precise types across queue tasks, metadata, and async boundaries.

What’s Needed
- Add concrete types for task payloads and results; use `TypedDict`/`Protocol` where appropriate; avoid `Any`; annotate async functions and context managers.

Where It’s Needed
- `src/llm/queue.py`, LLM call sites in `src/core/conversation.py` and `src/generation/*`; mypy settings in `pyproject.toml`.

Definition of Done
- `poetry run mypy src` passes in strict mode without new ignores; generics and concurrency primitives are properly annotated.

Status/Notes
- Consider lightweight helper types for common LLM task signatures to reduce duplication.

---

### Tests and Coverage (incl. branches)

Objective
- Prove correctness for concurrency, error handling, and parity; keep coverage thresholds green.

What’s Needed
- Add tests for cancellation, timeouts, error propagation, shutdown/drain behaviour, and “flag off” parity.
- Avoid flakiness by using fakes or controlled clocks; no `time.sleep` in async tests.

Where It’s Needed
- `tests/test_llm_queue.py` (new/expanded), existing async fixtures.

Definition of Done
- Coverage gates met (≥ 85% lines; ≥ 75% branches on changes); deterministic on CI; parity verified with the flag off.

Status/Notes
- Add branch‑coverage assertions around error/cancel paths.

---

### Security and Secrets Hygiene

Objective
- Prevent leakage of sensitive data via logs/spans/metrics and satisfy Bandit.

What’s Needed
- Redact PII and secrets in telemetry; avoid unsafe subprocess and eval patterns; triage Bandit findings.

Where It’s Needed
- Logging and metrics in `src/llm/queue.py` and LLM call sites; env/config handling; `SECURITY.md` if patterns need documenting.

Definition of Done
- `bandit -r src -ll` passes with no high/critical findings; telemetry verified free of secrets; redaction behaviour covered by tests or reviewed.

Status/Notes
- Ensure span attributes exclude prompts or responses unless explicitly whitelisted and anonymised.

---

### Dependency Policy and Audit

Objective
- Keep dependency footprint minimal and vulnerability‑free; make optional features extras.

What’s Needed
- Avoid adding new runtime deps unless justified; add observability deps as optional extras; run `pip-audit`; verify licenses.

Where It’s Needed
- `pyproject.toml` (optionally under `[tool.poetry.extras]`), `poetry.lock`, `README.md` install instructions.

Definition of Done
- `poetry run pip-audit` reports no vulnerabilities; optional extras documented; no unnecessary deps introduced.

Status/Notes
- If adding metrics exporters, isolate under an `observability` extra (already referenced in README).

---

### Logging and Observability Guardrails

Objective
- Standardise telemetry with useful, safe metrics and spans; provide tuning guidance.

What’s Needed
- Adopt consistent metric names/labels, add latency histograms, and ensure spans include correlating attributes without PII.

Where It’s Needed
- `src/llm/queue.py`, call sites setting task metadata; docs here and `docs/runtime-architecture.md` for tuning guidance.

Definition of Done
- Per‑stage histograms/counters emitted; spans correlate queue wait, run time, and outcomes; tuning notes present.

Status/Notes
- Complements “Step 5 — Observability and Tuning” with concrete guardrails.

---

### CI Parity and Pre‑commit

Objective
- Ensure CI runs the exact same commands as local development; enforce hooks early.

What’s Needed
- CI workflow invokes Poetry commands identical to local; pre‑commit set up for contributors.

Where It’s Needed
- `.github/workflows/*`, `.pre-commit-config.yaml`, `pyproject.toml` scripts.

Definition of Done
- CI passes using local‑parity commands; contributors can run `poetry run pre-commit run --all-files` cleanly.

Status/Notes
- If needed, add a Makefile alias to centralise commands used by CI and local runs.

---

### Concurrency and Cancellation Semantics

Objective
- Define and validate cancellation, timeout, and shutdown behaviours to avoid deadlocks and leaks.

What’s Needed
- Specify behaviour for task cancel and queue shutdown (graceful drain vs immediate cancel); ensure no blocking in async paths.

Where It’s Needed
- `src/llm/queue.py` (implementation and docs), tests exercising cancel/timeout/drain.

Definition of Done
- Cancellation semantics documented and tested; no deadlocks; no `time.sleep` in async code; graceful shutdown verified.

Status/Notes
- Include examples for recommended shutdown sequence in runbook docs if applicable.

---

### Error Taxonomy and Wrapping

Objective
- Provide a clear error model for queue and provider failures for easier handling upstream.

What’s Needed
- Define domain exceptions and wrap provider errors with causes; avoid silent swallowing.

Where It’s Needed
- `src/llm/errors.py` (if needed) and usages across queue and callers.

Definition of Done
- Tests assert exception types/messages; callers can distinguish retryable vs fatal errors; logging captures context without secrets.

Status/Notes
- Add a tracking issue if introducing a new error module to avoid scope creep.

---

### Config, Rollout, and Rollback

Objective
- Ship a safe default and a clear path to enable, tune, and, if necessary, roll back.

What’s Needed
- Validate feature flag toggling end‑to‑end; document staged rollout steps and rollback runbook.

Where It’s Needed
- `config/*`, `src/runtime/settings.py`, this plan, and optionally `docs/runbook.md`.

Definition of Done
- Toggling verified in staging; rollback is documented and tested; defaults are safe (flag off).

Status/Notes
- Keep defaults conservative; enable progressively with observability checkpoints.
