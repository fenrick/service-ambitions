# Agent Workflow Guidance

This repository uses automated code quality tooling for all Python sources.

## Formatting

- Run `poetry run black --preview --enable-unstable-feature string_processing .` to auto-format the code base.

## Linting

- Execute `poetry run ruff check --fix .` to check style, catch common bugs, and sort imports.

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

## Issue Tracking and PR Hygiene

- During code review, capture bugs, enhancements, and questions as GitHub issues using the repository's issue template.
- Use `gh issue create --title <title> --body <body>` and follow the template's sections for summary, reproduction steps, expected vs. actual behaviour, environment details, and additional context.
- Link issues to pull requests with keywords like `Fixes #123` or `Closes #123` in commit messages or the PR description.
- Ensure each pull request references relevant issues and keeps its commits focused and descriptive.
- Include clear acceptance criteria in every issue to define completion.
