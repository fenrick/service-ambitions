# Contributing

Contributions are welcome! To keep the project consistent and secure, run the
following checks before submitting a pull request:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```

Please ensure all checks pass and include appropriate tests and documentation
for any new functionality.
