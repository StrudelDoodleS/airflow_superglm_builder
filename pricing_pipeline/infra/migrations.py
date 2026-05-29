from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.schema import (
    SchemaNames,
    render_sql_schemas,
    schema_names_from_connectable,
)


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


def render_migration_sql(sql_text: str, schemas: SchemaNames) -> str:
    return render_sql_schemas(sql_text, schemas)


def _ensure_schema_configuration(con, schemas: SchemaNames) -> None:
    con.execute(
        text(
            """
            IF OBJECT_ID('dbo.SCHEMA_CONFIGURATION', 'U') IS NULL
            CREATE TABLE dbo.SCHEMA_CONFIGURATION (
                config_key NVARCHAR(128) NOT NULL PRIMARY KEY,
                config_value NVARCHAR(128) NOT NULL,
                created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
            );
            """
        )
    )
    expected = schemas.as_execution_options()
    rows = con.execute(
        text(
            """
            SELECT config_key, config_value
            FROM dbo.SCHEMA_CONFIGURATION
            WHERE config_key IN (
                'pricing_schema',
                'pricing_staging_schema',
                'mlops_schema'
            );
            """
        )
    ).all()
    existing = {row[0]: row[1] for row in rows}
    mismatches = [
        f"{key} existing={existing[key]!r} requested={value!r}"
        for key, value in expected.items()
        if key in existing and existing[key] != value
    ]
    if mismatches:
        raise RuntimeError(
            "Database was already initialized with different schema names: "
            + "; ".join(mismatches)
        )

    for key, value in expected.items():
        if key not in existing:
            con.execute(
                text(
                    """
                    INSERT INTO dbo.SCHEMA_CONFIGURATION(config_key, config_value)
                    VALUES (:key, :value);
                    """
                ),
                {"key": key, "value": value},
            )


def apply_migrations(engine: Engine, migrations_dir: Path) -> list[str]:
    schemas = schema_names_from_connectable(engine)
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
        _ensure_schema_configuration(con, schemas)

    applied: list[str] = []
    for path in migration_files(migrations_dir):
        with engine.begin() as con:
            exists = con.execute(
                text("SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = :name"),
                {"name": path.name},
            ).scalar()
        if exists:
            continue
        sql_text = render_migration_sql(path.read_text(encoding="utf-8"), schemas)
        with engine.begin() as con:
            for batch in split_sql_server_batches(sql_text):
                con.execute(text(batch))
            con.execute(
                text("INSERT INTO dbo.SCHEMA_MIGRATION(version_file) VALUES (:name)"),
                {"name": path.name},
            )
        applied.append(path.name)
    return applied
