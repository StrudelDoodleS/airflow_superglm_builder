"""Apply versioned schema DDL files from db/migrations."""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_runtime, load_env  # noqa: E402


def _schema_dir() -> Path:
    path = Path(
        os.environ.get(
            "PRICING_SCHEMA_DIR",
            os.environ.get("PRICING_MIGRATIONS_DIR", ROOT / "db" / "migrations"),
        )
    )
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_repo_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=(
            "Importable Python module that provides get_engine(database=None), "
            "get_schema_names(), and optional get_runtime_settings()."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_root_on_path()
    from pricing_pipeline.infra.migrations import apply_migrations, migration_files

    load_env()
    schema_dir = _schema_dir()

    files = migration_files(schema_dir)
    if not files:
        raise RuntimeError(f"No schema DDL files found in {schema_dir}")

    engine = get_runtime(args.runtime_module).get_engine()
    applied = set(apply_migrations(engine, schema_dir))
    for path in files:
        verb = "apply" if path.name in applied else "skip"
        print(f"{verb} {path.name}")

    print("done")


if __name__ == "__main__":
    main()
