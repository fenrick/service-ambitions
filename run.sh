#!/usr/bin/env bash
# Launch the Service Ambitions CLI.
# Requires SA_OPENAI_API_KEY to be set in the environment.

set -euo pipefail

# Use the installed console script via Poetry's environment. The project
# exposes an entry point `service-ambitions = "cli:main"` and is packaged
# (package-mode = true), so the console script is available.
poetry run service-ambitions "$@"
