# LLM Queue Migration Plan (Progressive)

This document outlines progressive, shippable steps to introduce a global LLM queue with bounded concurrency while keeping the main processing line simple and stable. Each step is feature‑flagged. The plan below is structured per item with what’s needed, where it’s needed, and clear definitions of done.

## Current Status

- Done
  - Bounded queue: `src/llm/queue.py` introduces `LLMQueue` with concurrency gate and basic metrics (`llm_queue_inflight`, `llm_queue_submitted`, `llm_queue_completed`).
  - Feature flags: `llm_queue_enabled`, `llm_queue_concurrency` added to `src/runtime/settings.py` and surfaced via config (`config/app.example.yaml`).
  - Environment bootstrap: `RuntimeEnv` initialises a global `LLMQueue` when enabled (`src/runtime/environment.py`).
  - Session routing: `ConversationSession.ask_async` routes via the global queue when enabled (`src/core/conversation.py`).
  - Pipelined plateau stages: features for plateau N+1 start while mapping for plateau N runs, bounded by the global queue (`src/generation/plateau_generator.py`).
  - Unified ambitions generator: `ServiceAmbitionGenerator` uses the global queue when enabled and disables its local limiter to avoid double gating (`src/generation/generator.py`).

## Step 1 — Verify Parity (flag off by default)

- Objective
  - Maintain identical behaviour with the feature flag off; no functional regressions.
- What’s Needed
  - Run CI with `llm_queue_enabled = false` (default) and ensure all tests pass.
  - Manual spot‑check locally: run a small dataset once with the flag off and once with it on; compare outputs for parity where deterministic, accepting minor non‑determinism from models.
- Where It’s Needed
  - Config defaults remain flag‑off (`src/runtime/settings.py` default values; `config/app.yaml` has the flag omitted; `config/app.example.yaml` documents the flag).
  - Test execution via `pytest` and existing coverage gates.
- Definition of Done
  - CI green with the flag off; no new failures introduced.
  - Local run confirms comparable outputs and identical control‑flow (cache usage, transcripts written) aside from expected model variance.

## Step 2 — Queue‑level Retry/Backoff and Circuit Breaker (Skipped)

- Decision
  - Not required for this iteration; keep the queue minimal. Existing components retain their own retry/backoff/circuit‑breaker handling.
- What’s Needed
  - Document the decision and create a tracking issue for a future enhancement (covering jittered backoff, error classification, breaker thresholds, and provider‑specific retry hints).
- Where It’s Needed
  - Documentation here; future code hooks would live in `src/llm/queue.py` and would be opt‑in via settings.
- Definition of Done
  - Decision recorded in this plan and a tracking issue created with clear scope and acceptance criteria.

## Step 3 — Pipeline Plateau Stages

- Objective
  - Overlap plateau work to improve throughput: start features for plateau N+1 while mapping for plateau N is running; rely on the global queue to enforce the concurrency cap.
- What’s Needed
  - Implement pipelined scheduling guarded by the feature flag (reuse `llm_queue_enabled` rather than a new flag), and verify with tests that overlapping occurs while respecting concurrency.
  - Ensure sequential fallback when the flag is off.
- Where It’s Needed
  - Scheduling logic in `PlateauGenerator._schedule_plateaus` (`src/generation/plateau_generator.py`).
  - Tests validating overlap ordering and fallback behaviour (`tests/test_llm_queue.py`).
  - Settings flag use in `RuntimeEnv` and consumers.
- Definition of Done
  - Pipelined schedule implemented and exercised when `llm_queue_enabled` is true.
  - Tests pass asserting that mapping for plateau 1 can be scheduled before features for plateau 2 complete, and that execution is sequential when the flag is false.
  - Concurrency is globally bounded by `LLMQueue` across sessions and stages.
- Status
  - Implemented and covered by tests.
- Original TODOs (retained for traceability)
  - [x] Implement pipelined scheduling (guarded by a flag; reusing `llm_queue_enabled`).
  - [x] Add tests to assert overlapping occurs (order/timestamps) — see `tests/test_llm_queue.py::test_plateau_pipeline_overlaps_when_enabled`.

## Step 4 — Unify Other Generators

- Objective
  - Route all generator LLM calls through the shared queue to centralise concurrency and observability; disable local limiters when the global queue is active to avoid double gating.
- What’s Needed
  - Integrate ambitions generation with the global queue and remove or bypass local concurrency controls under the flag.
- Where It’s Needed
  - `ServiceAmbitionGenerator` in `src/generation/generator.py` (routing via `RuntimeEnv.instance().llm_queue` and disabling local limiter when the flag is on).
- Definition of Done
  - Ambitions generator uses the global queue when enabled and behaves identically when disabled.
- Status
  - Done.

## Step 5 — Observability and Tuning

- Objective
  - Provide actionable visibility into queue health and stage throughput to tune concurrency and respect provider rate limits.
- What’s Needed
  - Metrics: expand beyond the existing inflight/submitted/completed to include latency histograms and per‑stage counters (labels include `stage`, `model_name`, `service_id`).
  - Spans: ensure queue and call spans include key attributes for easy correlation in Logfire.
  - Guidance: add brief operational notes on sizing `llm_queue_concurrency` vs provider limits and expected throughput.
- Where It’s Needed
  - Metrics/span additions in `src/llm/queue.py` and call‑sites that set `LLMTaskMeta` (e.g., `src/core/conversation.py`, `src/generation/generator.py`).
  - Documentation updates in this file and, optionally, `docs/runtime-architecture.md`.
- Definition of Done
  - Histogram metrics available per stage; counters reflect per‑stage volumes; spans carry attributes for root‑cause analysis.
  - Short “tuning” section documented with recommendations and safe defaults.
- Status
  - Partially done (basic counters/gauge present); latency histograms and guidance to be added.

## Enablement Checklist

- What’s Needed
  - Turn on the feature flag and set a safe concurrency value for your provider/account limits.
  - Validate rate‑limit compliance and end‑to‑end throughput under load.
- Where It’s Needed
  - Config: set in `config/app.yaml` (or environment via `SA_LLM_QUEUE_ENABLED`, `SA_LLM_QUEUE_CONCURRENCY`). See commented examples in `config/app.example.yaml`.
  - Runtime: ensure `RuntimeEnv.initialize` is called with loaded settings (already in place).
- Definition of Done
  - `llm_queue_enabled: true` with tuned `llm_queue_concurrency` shipped.
  - Provider rate limits observed (no throttling/HTTP 429s sustained during steady‑state).
  - Throughput and error rates monitored; concurrency adjusted as needed.
