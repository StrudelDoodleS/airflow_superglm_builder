"""Apply versioned SQL files from db/migrations.

This is a tiny Flyway-like runner for local testing. In production, use Flyway if you prefer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts.pricing_db import ROOT, get_engine, load_env


def _migrations_dir() -> Path:
    path = Path(os.environ.get("PRICING_MIGRATIONS_DIR", ROOT / "db" / "migrations"))
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_repo_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def main() -> None:
    _ensure_repo_root_on_path()
    from pricing_pipeline.migrations import apply_migrations, migration_files

    load_env()
    engine = get_engine()
    migrations_dir = _migrations_dir()

    files = migration_files(migrations_dir)
    if not files:
        raise RuntimeError(f"No migration files found in {migrations_dir}")

    applied = set(apply_migrations(engine, migrations_dir))
    for path in files:
        verb = "apply" if path.name in applied else "skip"
        print(f"{verb} {path.name}")

    print("done")


if __name__ == "__main__":
    main()
