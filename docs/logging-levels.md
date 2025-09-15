# Logging levels

Logfire provides structured logging to describe application behaviour and
troubleshooting information. Log entries are emitted at different levels to
control verbosity. Each level builds on the ones before it, inheriting their
messages while adding more detail.

## TRACE

- **Purpose:** Lowest level used for extremely fine‑grained events such as
  variable assignments or internal library calls.
- **Use:** Enable only when diagnosing elusive issues where every step matters.

## DEBUG

- **Purpose:** Provide detailed, contextual information about the application's
  internal state. Debug logs capture variable values, decision paths and
  interactions with external systems.
- **Use:** Primarily during development and testing or when troubleshooting
  specific production problems. Verbose output can impact performance and
  storage.

## INFO

- **Purpose:** Record high‑level application progress and key state changes.
- **Use:** Enabled in most environments to show normal operation without
  excessive detail.

## NOTICE

- **Purpose:** Highlight significant events that are not errors but may require
  attention, such as configuration fallbacks.
- **Use:** Useful for operations teams monitoring long‑running processes.

## WARNING

- **Purpose:** Indicate potential problems or unexpected situations that do not
  stop execution.
- **Use:** Trigger investigation when observed; often precedes an error if left
  unaddressed.
  - Transport retries (e.g., timeouts, connection errors, rate limits) are
    logged here with backoff details. These are I/O safeguards, not content
    re‑generations.

## ERROR

- **Purpose:** Report failures that prevent an operation from completing.
- **Use:** Should be logged whenever an exception is caught and handled or when
  user action is required to resolve an issue.

## FATAL

- **Purpose:** Capture irrecoverable conditions that lead to application
  termination.
- **Use:** Log immediately before exiting the process to preserve diagnostic
  data.

## EXCEPTION

- **Purpose:** Emit stack traces for unhandled errors.
- **Use:** Generally produced automatically by the logging system when an
  exception propagates without being caught.

Use `-q` and `-v` flags on the CLI to adjust verbosity from `fatal` through
`trace`. Structured JSON output is available via `--json-logs`, and detailed
per-request spans can be enabled with `--trace`.

Note on generation policy

- Logs must not imply multi‑pass content retries. Generation is one‑shot by
  design; only transport‑level retries are performed.

## Metrics

All internal metrics use the `sa_` prefix for consistency. Current metrics
include:

- `sa_llm_queue_inflight`: current queue slots in use.
- `sa_llm_queue_submitted`: total LLM tasks submitted.
- `sa_llm_queue_completed`: total LLM tasks completed.
- `sa_mapping_set_summary`/`sa_mapping_totals`: mapping run summaries.

When diagnostics are enabled, spans are annotated with ``stage``,
``model_name``, ``service_id`` and ``request_id`` to aid troubleshooting.
