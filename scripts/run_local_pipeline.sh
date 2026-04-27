#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export AIRFLOW_PROJ_DIR="${AIRFLOW_PROJ_DIR:-${PROJECT_ROOT}}"

mkdir -p \
  "${AIRFLOW_PROJ_DIR}/state/mssql/data" \
  "${AIRFLOW_PROJ_DIR}/state/mlflow/artifacts" \
  "${AIRFLOW_PROJ_DIR}/state/rating_exports" \
  "${AIRFLOW_PROJ_DIR}/logs" \
  "${AIRFLOW_PROJ_DIR}/config" \
  "${AIRFLOW_PROJ_DIR}/plugins"

docker compose build

docker compose up -d --wait \
  postgres \
  redis \
  mssql \
  mlflow \
  airflow-apiserver \
  airflow-scheduler \
  airflow-dag-processor \
  airflow-worker \
  airflow-triggerer

docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py

docker compose exec -T airflow-apiserver airflow dags unpause pricing_superglm_pipeline
docker compose exec -T airflow-apiserver airflow dags trigger pricing_superglm_pipeline
