# Agent Workflow Guidance

This repository uses automated checks (“agents”) to enforce quality on all Python sources.

> **Source of truth**  
> - **Standards** live in [`coding_standards.md`](./coding_standards.md).  
> - **Process** lives in [`CONTRIBUTING.md`](./CONTRIBUTING.md).  
> - If this file conflicts with either, **`coding_standards.md` prevails**, then `CONTRIBUTING.md`.

> **Blocking policy**  
> Pull requests **must not** be merged unless all required checks in this document pass.

---

## What agents enforce

- **Formatting** (Black)
- **Linting & import order** (Ruff)
- **Static typing** (mypy)
- **Security scanning** (Bandit)
- **Dependency vulnerabilities** (pip-audit)
- **Tests & coverage gates** (pytest; see thresholds below)
- **Documentation hygiene** (docstrings present for public APIs)
- **Complexity limits** (cyclomatic complexity thresholds)

These map directly to rules in `coding_standards.md` and `CONTRIBUTING.md`.

---

## Local commands (run before you push)

```bash
# Format (idempotent)
poetry run black --preview --enable-unstable-feature string_processing .

# Lint & import sort (autofix where possible)
poetry run ruff check --fix .

# Types
poetry run mypy .

# Security & deps
poetry run bandit -r src -ll
poetry run pip-audit

# Tests with coverage gates (edit paths to suit)
poetry run pytest --maxfail=1 --disable-warnings -q \
  --cov=src --cov-report=term-missing --cov-fail-under=85
````

**Expectations**

* **Coverage**: ≥ **85% lines** and ≥ **75% branches** on changed files (PRs may use diff-coverage).
* **Complexity**: cyclomatic complexity **< 8** per function. Where needed, refactor before raising a waiver.

---

## CI pipeline (required checks)

1. **Format**: `black --check` (fails on drift).
2. **Lint**: `ruff check` (no new violations; import order enforced).
3. **Types**: `mypy` (no errors).
4. **Security**: `bandit -r src -ll` (no high/critical findings).
5. **Dependencies**: `pip-audit` (no known vulns).
6. **Tests**: `pytest` with coverage gates as above.
7. **Docs**: verify public APIs have docstrings (spot-checked or via linter if configured).

> Tip: Prefer configuring these in CI with the same Poetry scripts to keep parity with local runs.

---

## Development process (what reviewers look for)

* **Refactor for readability first**, then performance (measure if you optimise).
* **Keep functions small** (complexity < 8). If you must exceed it briefly, add a clear refactor TODO and create an issue.
* **Comment intent, not mechanics**. Inline comments for branching logic or non-obvious constraints.
* **Fix all issues** from formatting, linting, typing, and security scans before review.

---

## Documentation

* Keep `README.md` and related docs in sync with code changes.
* Maintain accurate **Google-style docstrings** for all public APIs (see `coding_standards.md`).

---

## Issue tracking and PR hygiene

* During review, capture bugs/enhancements/questions as GitHub issues using the repo template.
* Use:

  ```bash
  gh issue create --title "<title>" --body "<body>"
  ```

  Include summary, repro steps, expected vs actual, environment, and context.
* Link issues to PRs using **`Fixes #123`** / **`Closes #123`**.
* Keep PRs focused and commits descriptive.
* Include **clear acceptance criteria** in every issue.

---

## Waivers (exception process)

* Rare only. Add a short justification next to any suppression (e.g., `# noqa: <rule>  # reason + link to issue`).
* Track all waivers in an issue tagged **tech-debt** with an owner and due date.

---

## Configuration pointers (optional but recommended)

Ensure these are set in `pyproject.toml` to back the gates:

* **Ruff**: enable complexity check (`C901`) with threshold 8; enable import sort; enable relevant error/select rules.
* **Black**: keep the same preview flags locally and in CI.
* **mypy**: strict mode for project packages; ignore-missing-imports only for third-party libs you can’t type.
* **pytest**: add coverage config and test discovery patterns.

See `coding_standards.md` for detailed expectations and examples.
