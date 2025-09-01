# Runtime architecture

This project evaluates services by composing a small set of engines.
Each layer has a single responsibility and retains its output until the
final aggregation step, enabling deterministic runs and comprehensive
telemetry.

## Runtime environment

`RuntimeEnv` is a thread-safe singleton initialised in `cli.main()`.
It loads configuration, exposes global settings and holds shared
in-memory state such as caches.  Modules access the singleton via
`RuntimeEnv.instance()` instead of repeatedly loading configuration
files. Tests or reconfigurations can clear the singleton via
`RuntimeEnv.reset()`.

## Processing engine

`ProcessingEngine` orchestrates the overall workflow.  It iterates over
services from the input file, constructs a `ServiceExecution` for each
and awaits their completion.  The engine only reports whether the batch
succeeded; individual artefacts remain inside the service objects until
`finalise()` is invoked.

## Service execution

A `ServiceExecution` handles one service.  It lazily loads plateau
information, spawns `PlateauRuntime` objects for each plateau and
delegates feature generation and mapping.  Results for successful
plateaus are stored on the execution instance and flushed to disk during
`finalise()`.

## Plateau runtime

`PlateauRuntime` encapsulates the per‑plateau state: description,
features and mapping results.  Each runtime exposes a simple
`status()` method used by the processing engine to determine overall
success.

## Telemetry and logging

Structured logging and spans are provided by
[Logfire](https://logfire.pydantic.dev/).  The runtime environment
initialiser, the processing engine and each plateau runtime emit
contextual `debug` and `info` events and wrap long‑running operations in
spans.  These spans enable fine‑grained tracing across services and
plateaus, while log levels allow operators to dial in the desired amount
of detail.

## Caching strategy

Caching is opt‑in and scoped by context, service and plateau:

```
.cache/<context>/<service_id>/<descriptions>.json
.cache/<context>/<service_id>/<plateau>/<features>.json
.cache/<context>/<service_id>/<plateau>/mappings/<set>/<file>.json
```

Legacy files are discovered and relocated to the canonical structure.
Caches are indented JSON dictionaries for easy inspection.  Invalid or
non‑dictionary content halts processing with a descriptive error.

Prompt templates are lazily loaded with `PromptLoader` and cached via an
LRU to avoid repeated disk access.  Tests can reset the cache using the
`clear_prompt_cache()` hook.

