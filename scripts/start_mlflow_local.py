from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import load_env  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _normalise_sqlite_uri(uri: str) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri

    database_path = Path(uri.removeprefix(prefix))
    if not database_path.is_absolute():
        database_path = ROOT / database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{database_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="5000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    load_env()

    artifact_root = _repo_path(os.environ.get("MLFLOW_ARTIFACT_ROOT", "state/mlflow/artifacts"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    backend_store_uri = _normalise_sqlite_uri(
        os.environ.get("MLFLOW_BACKEND_STORE_URI", "sqlite:///state/mlflow/mlflow.db")
    )
    allowed_hosts = os.environ.get(
        "MLFLOW_ALLOWED_HOSTS",
        "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000",
    )

    mlflow_executable = shutil.which("mlflow")
    if mlflow_executable is None:
        raise SystemExit("mlflow executable not found. Run `uv sync` first.")

    command = [
        mlflow_executable,
        "server",
        "--host",
        args.host,
        "--port",
        args.port,
        "--backend-store-uri",
        backend_store_uri,
        "--serve-artifacts",
        "--artifacts-destination",
        str(artifact_root),
        "--allowed-hosts",
        allowed_hosts,
    ]
    print(f"mlflow_backend_store_uri={backend_store_uri}", flush=True)
    print(f"mlflow_artifact_root={artifact_root}", flush=True)
    os.execv(mlflow_executable, command)


if __name__ == "__main__":
    main()
