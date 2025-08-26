# Contributing

Contributions are welcome! To keep the project consistent and secure, install
[pre-commit](https://pre-commit.com/#install) and run the configured hooks
locally before submitting a pull request:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
poetry run pip-audit
```

See `.pre-commit-config.yaml` for the exact tools (Black, Ruff, mypy and
Bandit). Please ensure all checks pass and include appropriate tests and
documentation for any new functionality.

## Continuous Integration

- The only required status check for merging into `main` is **CI — Quick**.
- GitHub Actions workflows do not run on pushes to non-`main` branches; open a
  pull request to trigger checks.

All contributors are expected to follow the project's
[Code of Conduct](CODE_OF_CONDUCT.md).
