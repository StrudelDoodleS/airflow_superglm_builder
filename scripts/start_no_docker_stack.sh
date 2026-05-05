#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DRY_RUN=0
SERVICES_CSV=""

SERVICES=(
  airflow
  mlflow
  migrate
  load-raw
  load-raw-replace
  pipeline
  diagrams
  seed-demo
  cloudbeaver
)

service_label() {
  case "$1" in
    airflow) echo "Airflow standalone (host Python, long-running)" ;;
    mlflow) echo "MLflow tracking server (host Python, long-running)" ;;
    migrate) echo "Apply SQL migrations" ;;
    load-raw) echo "Load freMTPL raw data if empty" ;;
    load-raw-replace) echo "Truncate and reload freMTPL raw data" ;;
    pipeline) echo "Run full pipeline directly without Airflow" ;;
    diagrams) echo "Generate ERD into state/db_diagrams" ;;
    seed-demo) echo "Seed demo model/package history" ;;
    cloudbeaver) echo "CloudBeaver SQL UI (Docker Compose local-only)" ;;
    *) echo "$1" ;;
  esac
}

service_command() {
  case "$1" in
    airflow) echo "uv run python scripts/start_airflow_local.py" ;;
    mlflow) echo "uv run python scripts/start_mlflow_local.py" ;;
    migrate) echo "uv run python scripts/apply_sql_migrations.py" ;;
    load-raw) echo "uv run python scripts/load_fremtpl_raw.py" ;;
    load-raw-replace) echo "uv run python scripts/load_fremtpl_raw.py --replace" ;;
    pipeline) echo "uv run python scripts/run_pipeline_no_airflow.py" ;;
    diagrams) echo "uv run python scripts/generate_db_diagrams.py --schemas pricing --output-dir state/db_diagrams" ;;
    seed-demo) echo "uv run python scripts/seed_demo_model_variants.py" ;;
    cloudbeaver) echo "docker compose --profile sql-ui up -d cloudbeaver" ;;
    *) return 1 ;;
  esac
}

is_long_running() {
  case "$1" in
    airflow|mlflow) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'EOF'
Start the local no-Docker runtime with a keyboard menu.

Usage:
  scripts/start_no_docker_stack.sh
  scripts/start_no_docker_stack.sh --services airflow,mlflow
  scripts/start_no_docker_stack.sh --dry-run --services migrate,load-raw,airflow

With no --services argument, the script opens a keyboard menu. Press a service
number to toggle it, r to run, or q to quit. Airflow is selected by default.

Services:
  airflow          host Airflow standalone
  mlflow           host MLflow tracking server
  migrate          apply SQL migrations
  load-raw         load freMTPL raw data if empty
  load-raw-replace truncate and reload freMTPL raw data
  pipeline         direct no-Airflow pipeline run
  diagrams         generate static ERD files
  seed-demo        seed simulated model/package history
  cloudbeaver      Docker Compose local-only CloudBeaver SQL UI

CloudBeaver note:
  cloudbeaver uses Docker Compose in this repo. Do not select it on locked-down
  work machines where Docker/Docker Hub access is blocked.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --services)
      SERVICES_CSV="${2:-}"
      if [[ -z "${SERVICES_CSV}" ]]; then
        echo "--services requires a comma-separated value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

declare -A SELECTED=()
SELECTED[airflow]=1

is_known_service() {
  local candidate="$1"
  local service
  for service in "${SERVICES[@]}"; do
    if [[ "${service}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

select_from_csv() {
  local csv="$1"
  local service
  SELECTED=()
  IFS=',' read -ra requested <<< "${csv}"
  for service in "${requested[@]}"; do
    service="${service//[[:space:]]/}"
    if ! is_known_service "${service}"; then
      echo "Unknown service: ${service}" >&2
      exit 2
    fi
    SELECTED["${service}"]=1
  done
}

render_menu() {
  local index=1
  local service marker
  echo
  echo "No-Docker local launcher"
  echo "Toggle services with number keys, then press r to run."
  echo
  for service in "${SERVICES[@]}"; do
    marker=" "
    if [[ -n "${SELECTED[${service}]:-}" ]]; then
      marker="x"
    fi
    printf "  %d) [%s] %-16s %s\n" "${index}" "${marker}" "${service}" "$(service_label "${service}")"
    index=$((index + 1))
  done
  echo
  echo "  r) run selected"
  echo "  q) quit"
  echo
}

interactive_select() {
  local choice service
  while true; do
    render_menu
    read -r -p "Choice: " choice
    case "${choice}" in
      r|R) break ;;
      q|Q) exit 0 ;;
      '' ) ;;
      *[!0-9]*)
        echo "Enter a service number, r, or q."
        ;;
      *)
        if (( choice < 1 || choice > ${#SERVICES[@]} )); then
          echo "Invalid service number: ${choice}"
        else
          service="${SERVICES[$((choice - 1))]}"
          if [[ -n "${SELECTED[${service}]:-}" ]]; then
            unset "SELECTED[${service}]"
          else
            SELECTED["${service}"]=1
          fi
        fi
        ;;
    esac
  done
}

selected_services() {
  local service
  for service in "${SERVICES[@]}"; do
    if [[ -n "${SELECTED[${service}]:-}" ]]; then
      echo "${service}"
    fi
  done
}

run_command() {
  local service="$1"
  local command
  command="$(service_command "${service}")"
  if [[ "${service}" == "cloudbeaver" ]]; then
    echo "cloudbeaver uses Docker Compose in this repo; skipping this on Docker-blocked machines."
  fi
  echo "==> ${service}: ${command}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    bash -lc "${command}"
  fi
}

PIDS=()
cleanup() {
  local pid
  if (( ${#PIDS[@]} > 0 )); then
    echo "stopping selected long-running services"
    for pid in "${PIDS[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}

run_long_running() {
  local service="$1"
  local command
  command="$(service_command "${service}")"
  echo "==> ${service}: ${command}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    bash -lc "${command}" &
    PIDS+=("$!")
  fi
}

if [[ -n "${SERVICES_CSV}" ]]; then
  select_from_csv "${SERVICES_CSV}"
else
  interactive_select
fi

mapfile -t RUN_LIST < <(selected_services)
if (( ${#RUN_LIST[@]} == 0 )); then
  echo "No services selected."
  exit 0
fi

for service in "${RUN_LIST[@]}"; do
  if ! is_long_running "${service}"; then
    run_command "${service}"
  fi
done

trap cleanup INT TERM EXIT
for service in "${RUN_LIST[@]}"; do
  if is_long_running "${service}"; then
    run_long_running "${service}"
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  echo "Long-running services started. Press Ctrl-C to stop them."
  wait "${PIDS[@]}"
fi
