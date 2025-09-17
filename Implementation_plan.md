# Implementation plan

This plan reflects the current state of the code base (September 17, 2025) and
focuses on the minimum work required to keep the service-ambitions CLI moving
forward. The guiding principles are:

- Lean on proven frameworks and libraries instead of hand-rolled plumbing.
- Make improvements that reinforce the core purpose: generating, mapping, and
  exporting service evolutions with deterministic, observable behaviour.
- Keep the backlog tight; retire items that are complete or no longer aligned
  with the design.

## Immediate priorities (Q4 2025)

### 1. Mapping CLI uses real LLM sessions

*Objective*
- Ensure the `map` subcommand can populate mappings end-to-end without test-only
  stubs by reusing the production `ModelFactory` and `ConversationSession`
  abstractions.

*What’s needed*
- Instantiate stage-specific agents via `ModelFactory` inside
  `cli.mapping.remap_features` instead of casting `object()` to
  `ConversationSession`.
- Route mapping requests through the existing caching hook so cache misses and
  refreshes work identically to the `run` pipeline.
- Update CLI tests to exercise the real wiring by injecting a fake
  `ConversationSession` fixture rather than monkeypatching `mapping.map_set`.
- Refresh CLI docs to describe how to provide API credentials and
  cache strategies for mapping runs.

*Where it’s needed*
- `src/cli/mapping.py`
- `src/models/factory.py`
- `tests/test_e2e_cli_mapping.py`
- `docs/index.md`, `README.md`

*Definition of done*
- `poetry run service-ambitions map` performs live mappings when cache entries
  are missing; existing golden tests updated to use the shared fixture.
- No direct casts to fake `ConversationSession` remain in production code.
- Documentation covers expected environment variables and cache behaviour for
  mapping-only runs.

### 2. Resilience via `pydantic_ai.retries` and usage limits

*Objective*
- Configure the built-in retry/backoff and throttling facilities shipped with
  Pydantic AI so we stop maintaining custom wrappers and stay aligned with the
  framework’s roadmap.

*What’s needed*
- Replace the bespoke `_with_retry` / `CircuitBreaker` flow in
  `generation/generator.py` with configuration of
  `pydantic_ai.retries.RetryConfig`/`wait_retry_after`, including jitter, caps
  and logging callbacks surfaced via the framework.
- Initialise `pydantic_ai.usage.UsageLimits` (request/response/tool-call limits)
  from settings and feed them through the queue + agent invocations so RPM/TPM
  and runaway loops are enforced without manual semaphores.
- Embrace `ModelRetry` in tools/output handlers to delegate retry decisions to
  the framework instead of raising bespoke exceptions.
- Surface retry and usage-limit metrics emitted by Pydantic AI in Logfire for
  observability (ensure spans/counters have descriptive names).
- Update tests to assert retry behaviour via the configured policies rather
  than hand-rolled counters; include cases exercising `UsageLimitExceeded`.

*Where it’s needed*
- `src/generation/generator.py`
- `src/llm/queue.py`
- `src/runtime/settings.py`
- `tests/test_llm_queue.py`, `tests/test_generator.py`

*Definition of done*
- Pydantic AI retry policies are loaded from settings and exercised in tests;
  transient failure simulations pass without custom retry helpers.
- Usage limits throttle calls according to configuration with <10% overshoot in
  the test harness and raise `UsageLimitExceeded` when appropriate.
- Legacy retry/circuit-breaker code is deleted.

### 3. Cache & transcript storage built on DiskCache

*Objective*
- Replace the home-grown JSON cache writer with `diskcache` (or similar) so we
  inherit atomic writes, eviction policies, and statistics without maintaining
  additional code.

*What’s needed*
- Introduce a thin adapter around `diskcache.Cache` implementing the existing
  `CacheManager` interface; support namespace segregation for contexts and
  stages.
- Migrate transcript persistence to the same backing store (filesystem-backed
  cache) to avoid bespoke path juggling.
- Provide migration logic that reads legacy JSON files into the new cache on
  first access; log counts of migrated entries.
- Update settings to expose cache size/TTL knobs instead of ad-hoc enums.
- Refresh cache-focused tests to assert behaviour via the adapter and
  verify migration of existing fixtures.

*Where it’s needed*
- `src/utils/cache_manager.py`
- `src/core/conversation.py`
- `src/runtime/settings.py`
- `tests/test_cache_paths.py`, `tests/test_cache_integrity.py`

*Definition of done*
- Cache reads/writes route through `diskcache`; legacy code paths deleted.
- Settings expose size/TTL controls that map directly to the library.
- Tests cover corruption handling and migration into the new store.

### 4. Usage & cost reporting via `pydantic_ai.usage`

*Objective*
- Leverage Pydantic AI’s built-in usage tracking to expose cached vs live token
  counts, per-stage spend, and budget enforcement without duplicating logic.

*What’s needed*
- Harvest `RunUsage` snapshots via `agent_run.usage()`/`result.usage()` inside
  every `ConversationSession` and persist them per stage.
- Configure pricing tables through `pydantic_ai.usage` (provider/model pricing
  maps) instead of hard-coded dictionaries; expose them via settings.
- Emit per-run summaries by reading the framework aggregates; include cached vs
  uncached token counts and request counts in diagnostics output plus Logfire
  metrics.
- Implement soft/hard budget thresholds using `UsageLimits` so overages trigger
  structured errors and graceful shutdowns.

*Where it’s needed*
- `src/core/conversation.py`
- `src/observability/telemetry.py`
- `src/runtime/settings.py`, `config/app.yaml`
- `tests/test_diagnostics.py`, `tests/test_costs.py`

*Definition of done*
- Diagnostics output shows live vs cached token counts, request counts and
  estimated spend sourced directly from `RunUsage`; budgets enforced via
  `UsageLimits` in tests.
- No parallel cost accounting utilities remain.

## Supporting backlog

### A. Operational docs stay in sync with code

*Objective*: Keep runtime architecture, queue tuning, and runbooks aligned with
  the code base after the resilience and cache changes ship.

*What’s needed*
- Fold `docs/runtime-architecture.md`, `docs/llm-queue.md`, and the runbook into
  a single "Operations" section in MkDocs; document new limiter/cost flags.
- Add troubleshooting entries for cache migration and rate limit breaches.

*Definition of done*: MkDocs builds cleanly; README points to the consolidated
  operations guide.

### B. CLI experience modernisation

*Objective*: Adopt a CLI framework (e.g. Typer) to simplify subcommand wiring
  and shared options while keeping behaviour backwards compatible.

*What’s needed*
- Wrap existing argparse commands with Typer, exposing the same flags.
- Share common options (input/output, cache) via Typer decorators instead of
  manual argument duplication.
- Update docs/tests to exercise the Typer entrypoint; keep the poetry script
  pointing to the new launcher.

*Definition of done*: `service-ambitions --help` renders via Typer; legacy tests
  pass without substantial rewrites.

### C. Provider plug-in surface

*Objective*: Formalise a provider plug-in interface so non-OpenAI backends can
  be registered without patching core modules.

*What’s needed*
- Define a lightweight protocol (`ProviderAdapter`) that supplies model IDs,
  retry/usage configuration hooks, and pricing metadata compatible with
  Pydantic AI’s adapters.
- Load adapters dynamically via entry points; ship an OpenAI implementation as
  the default.
- Extend settings to choose a provider and map stage-specific models to that
  provider.
- When multiple providers are configured, compose them via
  `pydantic_ai.models.FallbackModel` instead of bespoke fallback logic.
- Add integration tests using a dummy provider to confirm lifecycle hooks.

*Definition of done*: Switching providers happens via configuration; pipeline
tests using the dummy adapter succeed without touching core logic.

### D. Agent UI stretch goal

*Objective*: Explore replacing the CLI-first experience with an AG-UI powered
  front end that orchestrates the existing agents.

*What’s needed*
- Prototype an AG-UI deployment that surfaces the core CLI flows (preflight,
  run, map) through a web interface.
- Expose runtime state via APIs so the UI can monitor queue status, usage
  limits, and cache hits without scraping logs.
- Align authentication/authorisation with existing operational requirements.
- Document migration steps for teams moving from scripts to the UI, including
  how CI/CD integrates.

*Definition of done*: Proof-of-concept UI demonstrates the main flows end to
end with parity to scripting; follow-up RFC captures rollout plan and open
questions.

### E. Logfire instrumentation alignment

*Objective*: Let Logfire capture agent spans, metrics, and token usage without
maintaining bespoke instrumentation.

*What’s needed*
- Adopt `logfire.instrument_pydantic_ai()` (optionally scoped to specific
  agents) and `logfire.instrument_openai()` so tracing/metrics align with the
  upstream SDK.
- Remove redundant manual span creation where the instrumentation already
  emits structured data; keep custom spans only for domain-level events.
- Document configuration in `docs/runtime-architecture.md` + operations guide,
  including how to toggle instrumentation in dev vs. prod.
- Extend tests or smoke checks to ensure instrumentation hooks don’t regress
  during CLI runs (e.g., via Logfire test project or mocked collector).

*Definition of done*: CLI invocations emit the expected Logfire spans/metrics
via instrumentation helpers; documentation explains how to enable/disable the
hooks.

---

Every item above is meant to be small enough to land independently while keeping
momentum on the core application. Revisit the list after each release to drop
completed work and add newly discovered gaps.
