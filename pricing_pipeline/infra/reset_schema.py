from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.migrations import apply_migrations, migration_files
from pricing_pipeline.infra.schema import SchemaNames, validate_schema_name


DEFAULT_RESET_SCHEMAS = ("pricing", "pricing_stg", "mlops")
CONFIRMATION_FLAG = "--i-understand-this-drops-pricing-objects"


@dataclass(frozen=True)
class ResetSchemaResult:
    dry_run: bool
    expected_database: str
    actual_database: str
    schemas: tuple[str, ...]
    drop_batch_count: int
    applied_migrations: tuple[str, ...]


def normalize_schema_names(schema_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    raw_names = tuple(schema_names) or DEFAULT_RESET_SCHEMAS
    normalized: list[str] = []
    for name in raw_names:
        normalized.append(validate_schema_name(name, "schema name"))
    if len(set(normalized)) != len(normalized):
        raise ValueError("schema names must be unique")
    return tuple(normalized)


def schema_config_from_reset_schemas(schema_names: tuple[str, ...]) -> SchemaNames:
    if len(schema_names) != 3:
        raise ValueError(
            "reset/reseed requires exactly three schemas in this order: pricing pricing_stg mlops"
        )
    return SchemaNames(
        pricing=schema_names[0],
        pricing_staging=schema_names[1],
        mlops=schema_names[2],
    )


def _sql_string(value: str) -> str:
    return "N'" + value.replace("'", "''") + "'"


def _schema_filter(schema_names: tuple[str, ...]) -> str:
    return ", ".join(_sql_string(name) for name in schema_names)


def build_drop_batches(schema_names: tuple[str, ...] | list[str]) -> list[str]:
    schemas = normalize_schema_names(tuple(schema_names))
    schema_filter = _schema_filter(schemas)
    return [
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'ALTER TABLE '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name)
    + N' DROP CONSTRAINT ' + QUOTENAME(fk.name) + N';' + CHAR(10)
FROM sys.foreign_keys AS fk
JOIN sys.tables AS t
  ON t.object_id = fk.parent_object_id
JOIN sys.schemas AS s
  ON s.schema_id = t.schema_id
WHERE s.name IN ({schema_filter})
ORDER BY s.name, t.name, fk.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'DROP TRIGGER '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(tr.name) + N';' + CHAR(10)
FROM sys.triggers AS tr
JOIN sys.objects AS parent_object
  ON parent_object.object_id = tr.parent_id
JOIN sys.schemas AS s
  ON s.schema_id = parent_object.schema_id
WHERE tr.parent_class = 1
  AND s.name IN ({schema_filter})
ORDER BY s.name, tr.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'DROP VIEW '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(v.name) + N';' + CHAR(10)
FROM sys.views AS v
JOIN sys.schemas AS s
  ON s.schema_id = v.schema_id
WHERE s.name IN ({schema_filter})
ORDER BY s.name, v.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'DROP PROCEDURE '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(p.name) + N';' + CHAR(10)
FROM sys.procedures AS p
JOIN sys.schemas AS s
  ON s.schema_id = p.schema_id
WHERE s.name IN ({schema_filter})
ORDER BY s.name, p.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'DROP FUNCTION '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(o.name) + N';' + CHAR(10)
FROM sys.objects AS o
JOIN sys.schemas AS s
  ON s.schema_id = o.schema_id
WHERE s.name IN ({schema_filter})
  AND o.type IN ('FN', 'IF', 'TF', 'FS', 'FT')
ORDER BY s.name, o.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        f"""
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + N'DROP TABLE '
    + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name) + N';' + CHAR(10)
FROM sys.tables AS t
JOIN sys.schemas AS s
  ON s.schema_id = t.schema_id
WHERE s.name IN ({schema_filter})
ORDER BY s.name, t.name;
IF @sql <> N'' EXEC sys.sp_executesql @sql;
""",
        """
DROP TABLE IF EXISTS dbo.SCHEMA_MIGRATION;
DROP TABLE IF EXISTS dbo.SCHEMA_CONFIGURATION;
""",
    ]


def verify_expected_database(con, expected_database: str) -> str:
    actual_database = str(con.execute(text("SELECT DB_NAME();")).scalar_one())
    if actual_database != expected_database:
        raise RuntimeError(
            f"Refusing to reset database {actual_database!r}; expected {expected_database!r}."
        )
    return actual_database


def reset_and_reseed_schema(
    engine: Engine,
    *,
    migrations_dir: Path,
    expected_database: str,
    schema_names: tuple[str, ...] | list[str] = (),
    execute: bool = False,
) -> ResetSchemaResult:
    schemas = normalize_schema_names(tuple(schema_names))
    schema_config = schema_config_from_reset_schemas(schemas)
    configured_engine = engine.execution_options(**schema_config.as_execution_options())
    drop_batches = build_drop_batches(schemas)
    if execute and not migration_files(migrations_dir):
        raise RuntimeError(f"No schema DDL files found in {migrations_dir}")

    with configured_engine.begin() as con:
        actual_database = verify_expected_database(con, expected_database)
        if execute:
            for batch in drop_batches:
                con.execute(text(batch))

    applied: tuple[str, ...] = ()
    if execute:
        applied = tuple(apply_migrations(configured_engine, migrations_dir))

    return ResetSchemaResult(
        dry_run=not execute,
        expected_database=expected_database,
        actual_database=actual_database,
        schemas=schemas,
        drop_batch_count=len(drop_batches),
        applied_migrations=applied,
    )
