#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p \
  state/no_docker/mlflow/artifacts \
  state/no_docker/rating_exports \
  state/cv_splits \
  state/db_diagrams

if [[ ! -f .env ]]; then
  cp .env.nodocker.example .env
  echo "created .env from .env.nodocker.example"
else
  echo ".env already exists; leaving it unchanged"
fi

uv sync

echo
echo "Checking installed ODBC drivers:"
if command -v odbcinst >/dev/null 2>&1; then
  odbcinst -q -d || true
else
  echo "odbcinst not found. Install Microsoft ODBC Driver 18 for SQL Server before connecting."
fi

echo
echo "Edit .env for local runtime paths and create src/work_runtime/database.py, then start MLflow with:"
echo "  uv run python scripts/start_mlflow_local.py"
