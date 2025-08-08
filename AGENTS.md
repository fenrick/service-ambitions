# Agent Workflow Guidance

This repository uses automated code quality tooling for all Python sources.

## Formatting

- Run `poetry run isort --float-to-top --combine-star --order-by-type .` to sort import statements in the code base.
- Run `poetry run black --preview --enable-unstable-feature string_processing .` to auto-format the code base.

## Linting

- Execute `poetry run ruff check .` to check style and catch common bugs.
- Execute `poetry run flake8 .` to check style and catch common bugs.

## Static Analysis

- Run `poetry run mypy .` to perform type checking.
- Execute `poetry run bandit -r src -ll` to scan for security issues.
- Run `poetry run pip-audit` to check dependencies for vulnerabilities.

### Development Process

- Refactor for readability and performance.
- Keep functions small (cyclomatic complexity < 8).
- Comment and extensively document files, functions and logic flows.
- Provide inline comments for all branching logic and any logic with a cyclomatic complexity above 2.
- Fix all issues from linting and static analysis.

## Documentation

- Keep README and other docs in sync with code changes.
- Maintain accurate docstrings for all public APIs.
