#!/usr/bin/env bash
set -euo pipefail

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry is required but not installed. Visit https://python-poetry.org/docs/" >&2
  exit 1
fi

poetry install

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

poetry run python main.py "$@"
