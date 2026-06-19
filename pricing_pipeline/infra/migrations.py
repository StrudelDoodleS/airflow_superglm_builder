from __future__ import annotations

import getpass
import hashlib
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.schema import (
    SchemaNames,
    render_sql_schemas,
    schema_names_from_connectable,
)


_MIGRATION_LOCK_TIMEOUT_MS = 10_000


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


def migration_checksum(sql_text: str) -> str:
    return hashlib.sha256(sql_text.encode("utf-8")).hexdigest()


def _ensure_schema_migration_table(con) -> None:
    con.execute(
        text(
            """
            IF OBJECT_ID('dbo.SCHEMA_MIGRATION', 'U') IS NULL
            CREATE TABLE dbo.SCHEMA_MIGRATION (
                version_file NVARCHAR(256) NOT NULL PRIMARY KEY,
                checksum_sha256 NVARCHAR(64) NULL,
                applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
                applied_by NVARCHAR(128) NULL,
                status NVARCHAR(32) NOT NULL DEFAULT 'APPLIED',
                error_message NVARCHAR(4000) NULL
            );
            """
        )
    )
    columns = {
        "checksum_sha256": "NVARCHAR(64) NULL",
        "applied_by": "NVARCHAR(128) NULL",
        "status": "NVARCHAR(32) NOT NULL DEFAULT 'APPLIED'",
        "error_message": "NVARCHAR(4000) NULL",
    }
    for column_name, column_definition in columns.items():
        con.execute(
            text(
                f"""
                IF COL_LENGTH('dbo.SCHEMA_MIGRATION', '{column_name}') IS NULL
                    ALTER TABLE dbo.SCHEMA_MIGRATION
                    ADD {column_name} {column_definition};
                """
            )
        )


def acquire_migration_lock(con) -> None:
    lock_result = con.execute(
        text(
            """
            DECLARE @lock_result INT;
            EXEC @lock_result = sys.sp_getapplock
                @Resource = 'pricing_schema_migrations',
                @LockMode = 'Exclusive',
                @LockOwner = 'Transaction',
                @LockTimeout = :lock_timeout_ms;
            SELECT @lock_result;
            """
        ),
        {"lock_timeout_ms": _MIGRATION_LOCK_TIMEOUT_MS},
    ).scalar_one()
    if int(lock_result) < 0:
        raise RuntimeError("could not acquire pricing_schema_migrations lock")


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
            "Database was already initialized with different schema names: " + "; ".join(mismatches)
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


def apply_migrations_in_transaction(
    con,
    migrations_dir: Path,
    *,
    schemas: SchemaNames | None = None,
    acquire_lock: bool = True,
) -> list[str]:
    schemas = schemas or schema_names_from_connectable(con)
    applied_by = getpass.getuser()
    applied: list[str] = []
    _ensure_schema_migration_table(con)
    if acquire_lock:
        acquire_migration_lock(con)
    _ensure_schema_configuration(con, schemas)

    for path in migration_files(migrations_dir):
        sql_text = render_migration_sql(path.read_text(encoding="utf-8"), schemas)
        checksum = migration_checksum(sql_text)
        row = (
            con.execute(
                text(
                    """
                SELECT checksum_sha256, status
                FROM dbo.SCHEMA_MIGRATION
                WHERE version_file = :name
                """
                ),
                {"name": path.name},
            )
            .mappings()
            .one_or_none()
        )
        if row is not None:
            existing_checksum = row["checksum_sha256"]
            if existing_checksum is None:
                con.execute(
                    text(
                        """
                        UPDATE dbo.SCHEMA_MIGRATION
                        SET checksum_sha256 = :checksum,
                            status = 'APPLIED',
                            error_message = NULL
                        WHERE version_file = :name
                        """
                    ),
                    {"name": path.name, "checksum": checksum},
                )
                continue
            if existing_checksum != checksum:
                raise RuntimeError(
                    "Migration checksum mismatch for "
                    f"{path.name}: applied={existing_checksum} current={checksum}"
                )
            continue

        for batch in split_sql_server_batches(sql_text):
            con.execute(text(batch))
        con.execute(
            text(
                """
                IF EXISTS (
                    SELECT 1
                    FROM dbo.SCHEMA_MIGRATION
                    WHERE version_file = :name
                      AND checksum_sha256 IS NOT NULL
                      AND checksum_sha256 <> :checksum
                )
                BEGIN
                    THROW 51010, 'Migration checksum mismatch for existing tracking row.', 1;
                END;

                IF EXISTS (
                    SELECT 1
                    FROM dbo.SCHEMA_MIGRATION
                    WHERE version_file = :name
                      AND checksum_sha256 IS NULL
                )
                BEGIN
                    UPDATE dbo.SCHEMA_MIGRATION
                    SET checksum_sha256 = :checksum,
                        applied_by = COALESCE(applied_by, :applied_by),
                        status = 'APPLIED',
                        error_message = NULL
                    WHERE version_file = :name;
                END;
                ELSE IF NOT EXISTS (
                    SELECT 1
                    FROM dbo.SCHEMA_MIGRATION
                    WHERE version_file = :name
                )
                BEGIN
                    INSERT INTO dbo.SCHEMA_MIGRATION(
                        version_file,
                        checksum_sha256,
                        applied_by,
                        status
                    )
                    VALUES (
                        :name,
                        :checksum,
                        :applied_by,
                        'APPLIED'
                    );
                END;
                """
            ),
            {"name": path.name, "checksum": checksum, "applied_by": applied_by},
        )
        applied.append(path.name)
    return applied


def apply_migrations(engine: Engine, migrations_dir: Path) -> list[str]:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        return apply_migrations_in_transaction(
            con,
            migrations_dir,
            schemas=schemas,
        )
