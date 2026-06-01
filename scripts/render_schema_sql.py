"""Render SQL Server schema DDL with configurable pricing/mlops schema names."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra.migrations import migration_files, render_migration_sql  # noqa: E402
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
    THROW 50010, 'Database was already initialized with different schema names', 1;

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
    applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
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
        parts.extend(
            [
                "",
                f"PRINT N'Applying {migration_name}';",
                rendered.rstrip(),
                "",
                f"IF NOT EXISTS (SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = {_sql_string(migration_name)})",
                f"    INSERT INTO dbo.SCHEMA_MIGRATION(version_file) VALUES ({_sql_string(migration_name)});",
                "GO",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"
