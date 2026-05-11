#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export AIRFLOW_PROJ_DIR="${AIRFLOW_PROJ_DIR:-${PROJECT_ROOT}}"
DAG_ID="${DAG_ID:-pricing_mtpl_frequency}"

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
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/cleanup_airflow_examples.py

for attempt in {1..30}; do
  if docker compose exec -T airflow-apiserver airflow dags list | grep -qE "^${DAG_ID}[[:space:]]"; then
    break
  fi

  if [[ "${attempt}" == "30" ]]; then
    docker compose exec -T airflow-apiserver airflow dags list-import-errors || true
    echo "DAG ${DAG_ID} was not available after waiting for the Airflow DAG processor." >&2
    exit 1
  fi

  sleep 2
done

docker compose exec -T airflow-apiserver airflow dags unpause "${DAG_ID}"
docker compose exec -T airflow-apiserver airflow dags trigger "${DAG_ID}"
