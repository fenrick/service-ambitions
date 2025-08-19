# Contributing

Contributions are welcome! To keep the project consistent and secure, install
[pre-commit](https://pre-commit.com/) and run all checks locally before
submitting a pull request:

```bash
poetry run pre-commit run --all-files
poetry run pip-audit
```

Please ensure all checks pass and include appropriate tests and documentation
for any new functionality.
