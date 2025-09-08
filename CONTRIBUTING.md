# Contributing

Thanks for contributing! This repository enforces a strict set of quality gates across formatting, linting, typing, security, dependencies, and tests. Please read this guide before opening a PR.

## Local workflow

Run these commands before pushing changes:

```bash
# Format (idempotent)
poetry run black --preview --enable-unstable-feature string_processing .

# Lint & import sort (autofix where possible)
poetry run ruff check --fix .

# Types
poetry run mypy src

# Security & deps
poetry run bandit -r src -ll
poetry run pip-audit

# Tests with coverage gates
poetry run pytest --maxfail=1 --disable-warnings -q \
  --cov=src --cov-report=term-missing --cov-fail-under=85
```

Expectations:

- Coverage: ≥ 85% lines and ≥ 75% branches on changed files (PRs may use diff-coverage).
- Complexity: cyclomatic complexity < 8 per function; refactor before waiving.

See `.pre-commit-config.yaml` for the exact tools (Black, Ruff, mypy, Bandit). You are encouraged to install and run pre-commit locally:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Please adhere to the project [coding standards](docs/coding-standards.md) (Google-style docstrings for public APIs, strict typing, no broad exception handling without justification, etc.).

## Issues and PR hygiene

- Create a GitHub issue before starting work; use the repository templates and include acceptance criteria.
- Reference the issue from the PR description using an auto-close keyword (e.g., "Fixes #123").
- Keep PRs focused and commits descriptive.

## Continuous Integration (required checks)

Merges to `main` must pass all required jobs:

1. Format: `black --check` (fails on drift)
2. Lint: `ruff check` (no new violations; import order enforced)
3. Types: `mypy` (no errors)
4. Security: `bandit -r src -ll` (no high/critical findings)
5. Dependencies: `pip-audit` (no known vulnerabilities)
6. Tests: `pytest` with coverage gates (≥ 85% lines; branch coverage on)
7. Docs hygiene: public APIs have docstrings (spot-checked/enforced via Ruff D rules)
8. Diff coverage: PRs must meet coverage thresholds on the changed lines

> Tip: CI uses Python 3.11. Keep tool target versions aligned (Black/Ruff) and ensure Docker matches the runtime.

All contributors must follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).
