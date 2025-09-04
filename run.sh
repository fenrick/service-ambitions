#!/usr/bin/env bash
# Launch the Service Ambitions CLI.
# Requires SA_OPENAI_API_KEY to be set in the environment.

set -euo pipefail

# Use the CLI module directly rather than relying on an installed
# entry point.  The project is configured with `package-mode = false`
# so `service-ambitions` is not available on the `PATH`.
poetry run python src/cli.py "$@"
