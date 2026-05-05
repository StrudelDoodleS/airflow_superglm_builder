from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER_PROJECT_ROOT = Path("/opt/pricing")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import load_env  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    try:
        path = ROOT / path.relative_to(DOCKER_PROJECT_ROOT)
    except ValueError:
        pass
    if path.is_absolute():
        return path
    return ROOT / path


def _prepend_pythonpath(path: Path) -> None:
    existing = os.environ.get("PYTHONPATH")
    parts = [str(path)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "airflow_args",
        nargs="*",
        help="Airflow CLI args to run instead of the default `standalone` command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    load_env()

    airflow_home = _repo_path(os.environ.get("AIRFLOW_HOME", "state/airflow"))
    airflow_home.mkdir(parents=True, exist_ok=True)
    _repo_path("logs").mkdir(parents=True, exist_ok=True)

    dags_folder = _repo_path(os.environ.get("AIRFLOW__CORE__DAGS_FOLDER", "dags"))
    rating_export_root = _repo_path(
        os.environ.get("RATING_EXPORT_ROOT", "state/rating_exports")
    )
    rating_export_root.mkdir(parents=True, exist_ok=True)
    migrations_dir = _repo_path(os.environ.get("PRICING_MIGRATIONS_DIR", "db/migrations"))
    project_root = _repo_path(os.environ.get("PRICING_PROJECT_ROOT", str(ROOT)))

    os.environ["AIRFLOW_HOME"] = str(airflow_home)
    os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(dags_folder)
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
    os.environ.setdefault("AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION", "true")
    os.environ["PRICING_PROJECT_ROOT"] = str(project_root)
    os.environ["PRICING_MIGRATIONS_DIR"] = str(migrations_dir)
    os.environ["RATING_EXPORT_ROOT"] = str(rating_export_root)
    _prepend_pythonpath(ROOT)

    airflow_executable = shutil.which("airflow")
    if airflow_executable is None:
        raise SystemExit("airflow executable not found. Run `uv sync` first.")

    command = [airflow_executable, *(args.airflow_args or ["standalone"])]
    print(f"airflow_home={airflow_home}", flush=True)
    print(f"airflow_dags_folder={dags_folder}", flush=True)
    os.execv(airflow_executable, command)


if __name__ == "__main__":
    main()
