# Coding Standards

This project follows the principles outlined in *Clean Code: A Handbook of Agile Software Craftsmanship* and applies them explicitly to Python.

> **Scope**: These standards apply to all Python code in this repo (apps, libs, tests, tools, CI scripts).
> **Non-goals**: These rules aren’t academic ideals. Prefer pragmatic choices that improve readability, safety, and maintainability.

---

## Core Principles

- **Single Purpose**
  - Each module, class, and function **must** do one thing and do it well.
  - Names **must** be clear and descriptive (avoid abbreviations).
  - If a function name needs “and”, split it.
  - Example: `parse_config()` and `validate_config()` instead of `parse_and_validate_config()`.

- **Functions on Objects**
  - Prefer instance/class methods when behaviour depends on object state.
  - Prefer pure functions (module-level) for stateless transformations.
  - Avoid “anemic” models: if a class holds data, put core behaviour with it.

- **Singleton Shared State**
  - Default stance: **avoid singletons and global mutable state**.
  - If shared state is required:
    - Encapsulate behind a minimal, well-defined interface.
    - Provide dependency injection (pass it in) rather than importing globals.
    - Make thread/async safety explicit (document locking/awaiting rules).
    - Example pattern: a `Settings`/`Clock`/`Random` object passed where needed.

- **Readability Over Cleverness**
  - Prefer straightforward code to “smart” one-liners.
  - Add a comment or refactor rather than rely on implicit behaviour.

- **Fail Fast, Loud, and Use Exceptions**
  - Validate inputs early. Raise precise exceptions with helpful messages.
  - Do not hide errors with broad `except`.

---

## Function Arguments

- Prefer **niladic** (0) and **monadic** (1) functions. **Dyadic** (2) is acceptable. **Triadic** (3) only with clear justification. **Polyadic** (>3) is a design smell—refactor (e.g., dataclass/params object).
- Prefer **keyword-only** arguments for clarity in public APIs.
- Avoid boolean flags that change behaviour (`do_stuff(verbose=False)`)—split into separate functions or use a strategy object.
- Never use mutable objects as default values (use `None` and set inside).
- Prefer small, typed objects (e.g., `@dataclass`) to pass related parameters.

**Good**
```python
def resize(image: Image, *, max_width: int, max_height: int) -> Image:
    return image.resize((max_width, max_height))
```

**Refactor from this**

```python
def process(a, b, c, d):
    return (a + b) * (c + d)
```

---

## Formatting and Style

* Code **must** conform to PEP 8 and is autoformatted with **Black** (line length 88).
* **Ruff** runs for linting and import sorting (use its `isort` rules).
* Use **type hints** everywhere in non-trivial code. Treat type errors as build failures.
* Strings: prefer f-strings. Paths: use `pathlib`. Time: use timezone-aware `datetime` (UTC).
* Naming:

  * modules/packages: `snake_case`
  * classes: `PascalCase`
  * functions/vars: `snake_case`
  * constants: `UPPER_SNAKE_CASE`
* Do not commit commented-out code; use history.

---

## Types and Static Analysis

* Python ≥ 3.11 required.
* Use typing for all public functions, dataclasses, and attributes.
* Use `typing.Literal`, `TypedDict`, or `Protocol` where helpful.
* Avoid `Any`. If unavoidable, confine it and add a comment explaining why.
* Prefer immutability (`frozen=True` dataclasses) for value objects.

---

## Documentation and Comments

* Public modules, classes, and functions **must** have docstrings (Google style).
* Explain **why**, not **what**. The code shows “what”.
* Document preconditions, units, ranges, and side effects.
* Keep TODOs actionable: `# TODO(username): short description, link to issue #123`.

**Example**

```python
def load_users(path: Path) -> list[User]:
    """Load users from a JSON file.

    Args:
      path: Path to a UTF-8 JSON file of users.

    Returns:
      List of User objects.

    Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If the JSON is invalid or schema mismatches.
    """
```

---

## Errors and Exceptions

* Raise specific exceptions; create domain exceptions (e.g., `ConfigError`).
* Catch only what you can handle. Re-raise with `from` to keep context.
* No bare `except:`. No silent `pass`.
* Don’t return `None`/sentinel for error states in public APIs—use exceptions.

---

## Logging

* Use the `logging` library, not `print`.
* Log at appropriate levels: `debug` (dev detail), `info` (state changes), `warning` (unexpected but recoverable), `error` (failed operation), `critical` (service unusable).
* No secrets or PII in logs. Redact tokens/keys/emails.
* Structure logs where supported (JSON formatter in services).

---

## Testing

* Framework: **pytest**.
* Coverage: **≥ 85% lines** and **≥ 75% branches** on changed files.
* Test types:

  * Unit (fast, isolated, pure functions/classes).
  * Integration (real I/O, DB, network—with fixtures and markers).
  * Property-based for critical logic (e.g., with `hypothesis`).
* Tests must be deterministic. Control randomness (seed/`Random` injector).
* No live network in unit tests; use fakes/mocks or `responses`.
* Naming:

  * Test files: `test_*.py`
  * Test functions: `test_<behaviour>__<case>()`
* Given/When/Then comments for clarity in longer tests.

---

## Concurrency and Async

* I/O-bound concurrency: prefer **asyncio**. CPU-bound: use processes.
* Never block the event loop with CPU work; offload to executors.
* Make cancellation safe (use `asyncio.shield` only with caution).
* Protect shared mutable state (locks) or avoid it.
* Document whether a function is safe to call from threads/async.

---

## I/O and Side Effects

* Separate pure logic from side-effecting code. Keep I/O at the edges.
* Use context managers for files/network/locks.
* Stream large payloads; avoid loading entire files into memory unless trivial.
* Fail fast on I/O errors with explicit messages.

---

## Configuration and Secrets

* Read configuration from environment variables or a typed config object.
* Do not hardcode credentials, tokens, or endpoints.
* Use secret managers (e.g., AWS/GCP/HashiCorp Vault) in production.
* Validate configuration at startup; fail fast with a clear error.
* Provide safe defaults and document overrides.

---

## Data, Time, and Money

* Time: always timezone-aware `datetime` in **UTC**; convert at boundaries (API/UI).
* Durations: use `timedelta`. Avoid naive arithmetic.
* Money: use `Decimal` with explicit contexts; never `float`.
* IDs/keys: treat as opaque strings; don’t parse unless schema guarantees.

---

## API Design (Internal Libraries)

* Keep public surface area small and stable.
* Prefer explicit functions/classes over implicit magic.
* Backwards compatibility: follow semantic versioning within the repo’s packages.
* Validate inputs and raise clear exceptions rather than returning partial results.

---

## Dependencies and Packaging

* Single source of truth: `pyproject.toml`.
* Pin direct dependencies (use a lock file where supported).
* Avoid heavy dependencies for trivial tasks.
* Regularly update and remove unused packages.
* Use `importlib.resources` (not `pkg_resources`) for data files.

---

## Git, Commits, and Reviews

* Branching: **trunk-based** with short-lived feature branches.
* Commit messages: **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
* Small, atomic commits that pass tests.
* Pull Requests:

  * Must include a clear summary, rationale, and testing notes.
  * Require at least one review; reviewers use the checklist below.

**Reviewer Checklist**

* [ ] Names and responsibilities are clear and minimal.
* [ ] No global mutable state; dependencies injected.
* [ ] Type hints complete; no stray `Any`.
* [ ] Errors handled specifically; no bare `except`.
* [ ] Logging is appropriate and non-sensitive.
* [ ] Tests cover happy path, edge cases, and failures.
* [ ] No obvious performance or security issues.
* [ ] Docs and comments explain **why**.

---

## Performance

* Measure before optimising. Add micro-benchmarks when relevant.
* Prefer algorithmic improvements over micro-optimisations.
* Avoid N+1 calls; batch where possible.
* Use lazy evaluation only when it clearly helps and stays readable.
* Profile with `cProfile`/`pyinstrument` and include before/after notes in PRs when performance changes.

---

## Security

* Treat all inputs as untrusted; validate and sanitise.
* Use parameterised queries (never string-format SQL).
* Avoid eval/exec. Avoid deserialising untrusted pickles.
* Keep third-party libraries updated; respond to advisories promptly.
* Principle of least privilege for service accounts and tokens.

---

## Tooling and Automation

* **Pre-commit** hooks must run Black, Ruff, and basic security checks (e.g., `bandit`, secret scanners).
* CI must run: lint → type check → tests → (optional) build.
* Lint disables (`# noqa`, `# ruff: noqa`) require a short justification.

---

## Modules, Packages, and Imports

* Package layout should reflect domain boundaries, not technical layers alone.
* Avoid deep import chains and circular dependencies.
* Use absolute imports within the project.
* Public API of a package should be curated via `__all__` or an `api.py`.

---

## Examples

**Avoid mutable defaults**

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class Order:
    items: list[str] = field(default_factory=list)  # ✅

def add_tag(tags: list[str] | None = None) -> list[str]:
    tags = tags or []  # ✅
    tags.append("new")
    return tags
```

**Dependency injection over globals**

```python
class EmailSender:
    def __init__(self, client: Client):
        self._client = client

    def send_welcome(self, user: User) -> None:
        self._client.send(to=user.email, subject="Welcome", body="Hi!")

# In composition root:
sender = EmailSender(client=SES(region="us-east-1"))
```

**Async I/O separation**

```python
async def fetch_user(session: ClientSession, user_id: str) -> User:
    return await session.get_user(user_id)

def compute_score(user: User) -> int:
    return user.score
```

---

## Formatting and Style (Enforcement)

* Code is automatically formatted with [Black](https://black.readthedocs.io/).
* Linting is enforced with [Ruff](https://docs.astral.sh/ruff/).
* Type checking is enforced (e.g., `mypy` or `pyright`) in CI.
* CI must fail on formatting, lint, or type errors.
