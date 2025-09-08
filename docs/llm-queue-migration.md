# LLM Queue Migration Plan (Progressive)

This document outlines the progressive steps to introduce a global LLM queue
with bounded concurrency while keeping the main processing line simple and
stable. Each step is independently shippable and guarded behind a feature flag.

## Status

- Done:
  - Added `LLMQueue` (src/llm/queue.py) with bounded concurrency.
  - Added `llm_queue_enabled` and `llm_queue_concurrency` settings.
  - Bootstrapped queue in `RuntimeEnv` when enabled.
  - Routed `ConversationSession.ask_async` through the queue when enabled.
  - Pipelined plateau stages (features overlapping with mappings) when flag is on.
  - Unified ambitions generator to route via the queue when flag is on (local limiter disabled).

## Step 1 – Verify parity (flag off by default)

- Ensure CI passes with the flag off. No behavior change expected.
- Manual check: enable the flag locally and run a small set; outputs should match.

## Step 2 – (Skipped) queue-level retry/backoff and circuit breaker

Decision: Not required for this iteration. Keep queue minimal; existing components may retain their own handling.

## Step 3 – Pipeline plateau stages

- Overlap:
  - While mapping for plateau N is running, start features for plateau N+1.
  - The global queue enforces the overall concurrency cap.
- Implementation sketch:
  - In `PlateauGenerator._schedule_plateaus`, create per-plateau tasks with
    dependencies: `features -> mappings`, schedule N+1 features on completion of
    N features, not waiting for N mapping to finish.

TODO:
- [ ] Implement pipelined scheduling (guarded by a new `llm_queue_pipeline` flag or reuse `llm_queue_enabled`).
- [ ] Add tests to assert overlapping occurs (e.g., by tracking stage invocation order/timestamps).

## Step 4 – Unify other generators

Status: Done for `ServiceAmbitionGenerator`.

## Step 5 – Observability and tuning

- Expose queue metrics (inflight, submitted, completed) already present.
- Add latency histograms and per-stage counters.
- Document best practices for provider rate limits and expected throughput.

## Enablement checklist

- [ ] Set `llm_queue_enabled: true` and tune `llm_queue_concurrency`.
- [ ] Confirm provider-side rate limits are respected.
- [ ] Monitor throughput and error rates; adjust concurrency accordingly.
