# Agent Workflow Guidance

This repository uses automated code quality tooling for all Python sources.

## Formatting

- Run `black .` to auto-format the code base.

## Linting

- Execute `ruff .` to check style and catch common bugs.

## Static Analysis

- Run `mypy .` to perform type checking.
- Execute `bandit -r src -ll` to scan for security issues.
- Run `pip-audit` to check dependencies for vulnerabilities.

### Development Process

- Refactor for readability and performance.
- Keep functions small (cyclomatic complexity < 8).
- Comment and extensively document files, functions and logic flows.

## Documentation

- Keep README and other docs in sync with code changes.
- Maintain accurate docstrings for all public APIs.
