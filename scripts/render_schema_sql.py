"""Render SQL Server schema DDL with configurable pricing/mlops schema names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra.migrations import (  # noqa: E402
    migration_checksum,
    migration_files,
    render_migration_sql,
)
from pricing_pipeline.infra.schema import SchemaNames, validate_schema_name  # noqa: E402


def _sql_string(value: str) -> str:
    return "N'" + value.replace("'", "''") + "'"


def _schema_guard_sql(schemas: SchemaNames) -> str:
    values = schemas.as_execution_options()
    value_rows = ",\n        ".join(
        f"('{key}', {_sql_string(value)})" for key, value in values.items()
    )
    return f"""\
IF OBJECT_ID('dbo.SCHEMA_CONFIGURATION', 'U') IS NULL
CREATE TABLE dbo.SCHEMA_CONFIGURATION (
    config_key NVARCHAR(128) NOT NULL PRIMARY KEY,
    config_value NVARCHAR(128) NOT NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

DECLARE @requested_schema_config TABLE (
    config_key NVARCHAR(128) NOT NULL PRIMARY KEY,
    config_value NVARCHAR(128) NOT NULL
);

INSERT INTO @requested_schema_config(config_key, config_value)
VALUES
        {value_rows};

IF EXISTS (
    SELECT 1
    FROM dbo.SCHEMA_CONFIGURATION AS existing
    JOIN @requested_schema_config AS requested
      ON requested.config_key = existing.config_key
WHERE existing.config_value <> requested.config_value
)
BEGIN;
    THROW 50010, 'Database was already initialized with different schema names', 1;
END;

INSERT INTO dbo.SCHEMA_CONFIGURATION(config_key, config_value)
SELECT requested.config_key, requested.config_value
FROM @requested_schema_config AS requested
WHERE NOT EXISTS (
    SELECT 1
    FROM dbo.SCHEMA_CONFIGURATION AS existing
    WHERE existing.config_key = requested.config_key
);
GO

IF OBJECT_ID('dbo.SCHEMA_MIGRATION', 'U') IS NULL
CREATE TABLE dbo.SCHEMA_MIGRATION (
    version_file NVARCHAR(256) NOT NULL PRIMARY KEY,
    checksum_sha256 NVARCHAR(64) NULL,
    applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    applied_by NVARCHAR(128) NULL,
    status NVARCHAR(32) NOT NULL DEFAULT 'APPLIED',
    error_message NVARCHAR(4000) NULL
);
GO

IF COL_LENGTH('dbo.SCHEMA_MIGRATION', 'checksum_sha256') IS NULL
    ALTER TABLE dbo.SCHEMA_MIGRATION ADD checksum_sha256 NVARCHAR(64) NULL;
GO

IF COL_LENGTH('dbo.SCHEMA_MIGRATION', 'applied_by') IS NULL
    ALTER TABLE dbo.SCHEMA_MIGRATION ADD applied_by NVARCHAR(128) NULL;
GO

IF COL_LENGTH('dbo.SCHEMA_MIGRATION', 'status') IS NULL
    ALTER TABLE dbo.SCHEMA_MIGRATION ADD status NVARCHAR(32) NOT NULL DEFAULT 'APPLIED';
GO

IF COL_LENGTH('dbo.SCHEMA_MIGRATION', 'error_message') IS NULL
    ALTER TABLE dbo.SCHEMA_MIGRATION ADD error_message NVARCHAR(4000) NULL;
GO
"""


def render_schema_sql(
    migrations_dir: Path,
    *,
    pricing_schema: str,
    pricing_staging_schema: str,
    mlops_schema: str,
) -> str:
    schemas = SchemaNames(
        pricing=validate_schema_name(pricing_schema, "pricing_schema"),
        pricing_staging=validate_schema_name(
            pricing_staging_schema,
            "pricing_staging_schema",
        ),
        mlops=validate_schema_name(mlops_schema, "mlops_schema"),
    )
    files = migration_files(migrations_dir)
    if not files:
        raise RuntimeError(f"No schema DDL files found in {migrations_dir}")

    parts = [
        "-- Rendered Airflow SuperGLM Builder schema DDL.",
        "-- Run this against the already-created target database.",
        _schema_guard_sql(schemas).rstrip(),
    ]
    for path in files:
        migration_name = path.name
        rendered = render_migration_sql(path.read_text(encoding="utf-8"), schemas)
        checksum = migration_checksum(rendered)
        parts.extend(
            [
                "",
                f"PRINT N'Applying {migration_name}';",
                "GO",
                rendered.rstrip(),
                "",
                f"IF NOT EXISTS (SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = {_sql_string(migration_name)})",
                "    INSERT INTO dbo.SCHEMA_MIGRATION("
                "version_file, checksum_sha256, applied_by, status"
                ") VALUES ("
                f"{_sql_string(migration_name)}, {_sql_string(checksum)}, "
                "SUSER_SNAME(), N'APPLIED');",
                "GO",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"
