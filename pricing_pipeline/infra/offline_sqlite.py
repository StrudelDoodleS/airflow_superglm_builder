"""Persistent attached-schema SQLite storage for local pricing workflows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.file_lock import exclusive_file_lock


OFFLINE_DDL_DIR = Path(__file__).resolve().parents[2] / "db" / "offline_sqlite"
COORDINATOR_DB_FILE = "coordinator.sqlite"
SCHEMA_DB_FILES = {
    "pricing": "pricing.sqlite",
    "pricing_stg": "pricing_stg.sqlite",
    "mlops": "mlops.sqlite",
}
_OFFLINE_COLUMN_UPGRADES = (
    ("pricing", "DATASET_MANIFEST", "model_frame_sha256", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "frame_hash_metadata_json", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "exposure_column", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "data_as_of_column", "TEXT"),
    (
        "pricing",
        "MODEL_RUN",
        "parent_model_run_id",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "rating_workbook_sha256",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_artifact_path",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_artifact_sha256",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_artifact_format",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_artifact_size_bytes",
        "INTEGER",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_python_version",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_superglm_version",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "model_source_sha256",
        "TEXT",
    ),
    (
        "pricing",
        "PRICING_RATE_PACKAGE",
        "staging_content_sha256",
        "TEXT",
    ),
    (
        "pricing_stg",
        "STG_RATING_EXPORT",
        "staging_content_sha256",
        "TEXT",
    ),
)
_OFFLINE_NULLABILITY_UPGRADES = (
    ("pricing", "MODEL_RUN", "effective_from"),
    ("pricing", "PRICING_RATE_PACKAGE", "effective_from_date"),
)


@contextmanager
def local_publish_lock(root: str | Path) -> Iterator[BinaryIO]:
    """Serialize local staging/publication across notebook processes."""
    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    lock_path = resolved / ".publish.lock"
    with exclusive_file_lock(lock_path) as handle:
        yield handle


def offline_database_paths(root: str | Path) -> dict[str, Path]:
    """Return the persistent database file for each emulated SQL schema."""
    resolved = Path(root).expanduser().resolve()
    return {schema: resolved / filename for schema, filename in SCHEMA_DB_FILES.items()}


def sqlite_engine_with_offline_schemas(
    db_paths: Mapping[str, Path],
) -> Engine:
    """Create an engine whose connections attach the three schema databases."""
    missing = set(SCHEMA_DB_FILES) - set(db_paths)
    extra = set(db_paths) - set(SCHEMA_DB_FILES)
    if missing or extra:
        raise ValueError(
            "offline SQLite database paths must contain exactly: " + ", ".join(SCHEMA_DB_FILES)
        )

    resolved_paths = {
        schema: Path(path).expanduser().resolve() for schema, path in db_paths.items()
    }
    for path in resolved_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    parent_directories = {path.parent for path in resolved_paths.values()}
    if len(parent_directories) != 1:
        raise ValueError("offline SQLite database files must share one directory")
    coordinator_path = parent_directories.pop() / COORDINATOR_DB_FILE

    engine = create_engine(
        f"sqlite:///{coordinator_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _attach_pricing_schemas(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA main.journal_mode=DELETE")
        for schema, path in resolved_paths.items():
            dbapi_connection.execute(
                f"ATTACH DATABASE ? AS {schema}",
                (str(path),),
            )
            dbapi_connection.execute(f"PRAGMA {schema}.journal_mode=DELETE")

    return engine


def _relax_offline_column_nullability(
    connection,
    *,
    schema: str,
    table: str,
    column: str,
) -> bool:
    columns = list(connection.execute(f"PRAGMA {schema}.table_info('{table}')").fetchall())
    target = next((row for row in columns if str(row[1]) == column), None)
    if target is None or int(target[3]) == 0:
        return False

    create_row = connection.execute(
        f"SELECT sql FROM {schema}.sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if create_row is None or not create_row[0]:
        raise RuntimeError(f"cannot rebuild missing offline table {schema}.{table}")

    nullable_sql, replacements = re.subn(
        rf"(\b{re.escape(column)}\b\s+[A-Z0-9_]+(?:\([^)]*\))?)\s+NOT\s+NULL",
        r"\1",
        str(create_row[0]),
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements != 1:
        raise RuntimeError(
            f"cannot relax offline column {schema}.{table}.{column}: "
            "stored CREATE TABLE statement is not recognized"
        )
    qualified_sql, replacements = re.subn(
        rf"^CREATE\s+TABLE\s+{re.escape(table)}\s*",
        f"CREATE TABLE {schema}.{table} ",
        nullable_sql,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements != 1:
        raise RuntimeError(
            f"cannot rebuild offline table {schema}.{table}: "
            "stored CREATE TABLE prefix is not recognized"
        )

    old_table = f"__offline_upgrade_{table.lower()}"
    quoted_columns = ", ".join(f'"{str(row[1])}"' for row in columns)
    connection.execute(f'ALTER TABLE {schema}."{table}" RENAME TO "{old_table}"')
    connection.execute(qualified_sql)
    connection.execute(
        f'INSERT INTO {schema}."{table}" ({quoted_columns}) '
        f'SELECT {quoted_columns} FROM {schema}."{old_table}"'
    )
    connection.execute(f'DROP TABLE {schema}."{old_table}"')
    return True


def apply_offline_ddl(engine: Engine) -> None:
    """Create any missing local tables without deleting existing data."""
    connection = engine.raw_connection()
    try:
        for schema in SCHEMA_DB_FILES:
            ddl_path = OFFLINE_DDL_DIR / f"{schema}.sql"
            connection.executescript(ddl_path.read_text(encoding="utf-8"))
        for schema, table, column, column_type in _OFFLINE_COLUMN_UPGRADES:
            existing_columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA {schema}.table_info('{table}')").fetchall()
            }
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE {schema}.{table} ADD COLUMN {column} {column_type}"
                )
        connection.execute(
            """
            UPDATE pricing.MODEL_RUN AS child_run
            SET parent_model_run_id = (
                SELECT parent_run.model_run_id
                FROM pricing.PRICING_RATE_PACKAGE AS child_package
                JOIN pricing.MODEL_RUN AS parent_run
                  ON parent_run.rate_package_id = child_package.parent_rate_package_id
                WHERE child_package.rate_package_id = child_run.rate_package_id
            )
            WHERE child_run.parent_model_run_id IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_RATE_PACKAGE AS child_package
                  JOIN pricing.MODEL_RUN AS parent_run
                    ON parent_run.rate_package_id = child_package.parent_rate_package_id
                  WHERE child_package.rate_package_id = child_run.rate_package_id
              )
            """
        )
        connection.commit()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rebuilt_table = False
            for schema, table, column in _OFFLINE_NULLABILITY_UPGRADES:
                rebuilt_table = (
                    _relax_offline_column_nullability(
                        connection,
                        schema=schema,
                        table=table,
                        column=column,
                    )
                    or rebuilt_table
                )
            if rebuilt_table:
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS pricing.UX_MODEL_RUN_RATE_PACKAGE
                    ON MODEL_RUN(rate_package_id)
                    WHERE rate_package_id IS NOT NULL
                    """
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    finally:
        connection.close()


def open_offline_sqlite(
    root: str | Path,
) -> tuple[Engine, dict[str, Path]]:
    """Open a persistent local store and ensure its schema is current."""
    paths = offline_database_paths(root)
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    return engine, paths
