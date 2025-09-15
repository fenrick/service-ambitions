# LLM Queue (experimental)

The LLM Queue centralises concurrency for all model calls and enables a
plateau-first pipeline that overlaps feature generation with mapping per service.
It is off by default to preserve the original sequential behaviour.

## Enablement

- Config file (`config/app.yaml`):

  ```yaml
  llm_queue_enabled: true
  llm_queue_concurrency: 3  # total concurrent LLM calls across the app
  ```

- Environment variables (override config):

  ```bash
  export SA_LLM_QUEUE_ENABLED=true
  export SA_LLM_QUEUE_CONCURRENCY=3
  ```

## Behaviour

- When enabled, ConversationSession async calls route through a global
  bounded‑concurrency queue.
- Plateau generation overlaps with mapping (pipeline mode) per service; the
  queue caps overall throughput.
- When disabled, per‑plateau stages execute sequentially (features → mapping).

## Observability

- Metrics
  - `sa_llm_queue_inflight` (gauge): available queue slots.
  - `sa_llm_queue_submitted` (counter): tasks submitted to the queue.
  - `sa_llm_queue_completed` (counter): tasks completed by the queue.
- Tracing
  - Queue spans include `{stage, model_name, service_id, request_id}` attributes
    when `--trace` is active.

## Tuning and troubleshooting

- Start conservatively (`llm_queue_concurrency: 2–3`) to avoid provider rate
  limits; increase gradually.
- If overall throughput feels low, consider:
  - Increasing `llm_queue_concurrency` (global LLM slots), or
  - Increasing `--concurrency` (number of services processed concurrently).
- Use `--trace` and structured logs (`--json-logs`) to inspect spans and
  counters during runs.

## Roadmap

- Provider‑aware RPM/TPM limits per model.
- Per‑service plateau concurrency guardrails and queue depth histograms.
- Cost and token budgets with end‑of‑run reports.
