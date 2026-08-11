"""Apply versioned schema DDL files from db/migrations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_runtime, load_env


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=(
            "Importable Python module that provides get_engine(database=None), "
            "get_schema_names(), and optional get_runtime_settings()."
        ),
    )
    parser.add_argument(
        "--expected-database",
        default=None,
        help=(
            "Database name that must match DB_NAME() before migrations are applied. "
            "Defaults to the runtime's configured pricing_database."
        ),
    )
    return parser.parse_args(argv)


def verify_expected_database(connection, expected_database: str) -> str:
    """Refuse migration writes when the connection resolves another database."""
    expected = str(expected_database or "").strip()
    if not expected:
        raise ValueError("expected database name is required")
    actual = str(connection.execute(text("SELECT DB_NAME();")).scalar_one() or "").strip()
    if actual != expected:
        raise RuntimeError(
            f"Refusing to apply migrations to database {actual!r}; expected {expected!r}."
        )
    return actual


def main() -> None:
    args = parse_args()
    from pricing_pipeline.infra.migrations import apply_migrations, migration_files

    load_env()
    schema_dir = _schema_dir()

    files = migration_files(schema_dir)
    if not files:
        raise RuntimeError(f"No schema DDL files found in {schema_dir}")

    runtime = get_runtime(args.runtime_module)
    expected_database = args.expected_database or runtime.settings.pricing_database
    engine = runtime.get_engine()
    with engine.connect() as connection:
        actual_database = verify_expected_database(connection, expected_database)
    print(f"database={actual_database}")
    applied = set(apply_migrations(engine, schema_dir))
    for path in files:
        verb = "apply" if path.name in applied else "skip"
        print(f"{verb} {path.name}")

    print("done")


if __name__ == "__main__":
    main()
