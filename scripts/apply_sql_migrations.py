"""Apply versioned SQL files from db/migrations.

This is a tiny Flyway-like runner for local testing. In production, use Flyway if you prefer.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from pricing_db import ROOT, get_engine, run_sql_file

MIGRATIONS_DIR = ROOT / "db" / "migrations"


def main() -> None:
    engine = get_engine()

    with engine.begin() as con:
        con.execute(text("""
        IF OBJECT_ID('dbo.SCHEMA_MIGRATION', 'U') IS NULL
        CREATE TABLE dbo.SCHEMA_MIGRATION (
            version_file NVARCHAR(256) NOT NULL PRIMARY KEY,
            applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
        );
        """))

    files = sorted(MIGRATIONS_DIR.glob("V*.sql"))
    if not files:
        raise RuntimeError(f"No migration files found in {MIGRATIONS_DIR}")

    for path in files:
        with engine.begin() as con:
            already = con.execute(
                text("SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = :name"),
                {"name": path.name},
            ).scalar()

        if already:
            print(f"skip {path.name}")
            continue

        print(f"apply {path.name}")
        run_sql_file(engine, path)

        with engine.begin() as con:
            con.execute(
                text("INSERT INTO dbo.SCHEMA_MIGRATION(version_file) VALUES (:name)"),
                {"name": path.name},
            )

    print("done")


if __name__ == "__main__":
    main()
