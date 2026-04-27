from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine


def split_sql_server_batches(sql_text: str) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)
    batch = "\n".join(current).strip()
    if batch:
        batches.append(batch)
    return batches


def migration_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("V*.sql"))


def apply_migrations(engine: Engine, migrations_dir: Path) -> list[str]:
    with engine.begin() as con:
        con.execute(
            text(
                """
                IF OBJECT_ID('dbo.SCHEMA_MIGRATION', 'U') IS NULL
                CREATE TABLE dbo.SCHEMA_MIGRATION (
                    version_file NVARCHAR(256) NOT NULL PRIMARY KEY,
                    applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
                );
                """
            )
        )

    applied: list[str] = []
    for path in migration_files(migrations_dir):
        with engine.begin() as con:
            exists = con.execute(
                text("SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = :name"),
                {"name": path.name},
            ).scalar()
        if exists:
            continue
        sql_text = path.read_text(encoding="utf-8")
        with engine.begin() as con:
            for batch in split_sql_server_batches(sql_text):
                con.execute(text(batch))
            con.execute(
                text("INSERT INTO dbo.SCHEMA_MIGRATION(version_file) VALUES (:name)"),
                {"name": path.name},
            )
        applied.append(path.name)
    return applied
