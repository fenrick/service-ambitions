#!/usr/bin/env bash
# Launch the Service Ambitions CLI.
# Requires OPENAI_API_KEY to be set in the environment.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

poetry run python -m service_ambitions.cli "$@"
