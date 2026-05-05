#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${ROOT}/scripts/start_no_docker_stack.sh" "$@"
