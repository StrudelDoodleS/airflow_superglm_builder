"""Apply versioned SQL files from db/migrations.

This is a tiny Flyway-like runner for local testing. In production, use Flyway if you prefer.
"""
from __future__ import annotations

import sys

from pricing_db import ROOT, get_engine

MIGRATIONS_DIR = ROOT / "db" / "migrations"


def _ensure_repo_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def main() -> None:
    _ensure_repo_root_on_path()
    from pricing_pipeline.migrations import apply_migrations, migration_files

    engine = get_engine()

    files = migration_files(MIGRATIONS_DIR)
    if not files:
        raise RuntimeError(f"No migration files found in {MIGRATIONS_DIR}")

    applied = set(apply_migrations(engine, MIGRATIONS_DIR))
    for path in files:
        verb = "apply" if path.name in applied else "skip"
        print(f"{verb} {path.name}")

    print("done")


if __name__ == "__main__":
    main()
