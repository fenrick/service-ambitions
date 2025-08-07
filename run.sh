#!/usr/bin/env bash
# Launch the Service Ambitions CLI.
# Requires OPENAI_API_KEY to be set in the environment.

set -euo pipefail

poetry run python -m service_ambitions.cli "$@"
