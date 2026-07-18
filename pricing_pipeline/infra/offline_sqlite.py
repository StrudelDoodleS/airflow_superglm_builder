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
    ("pricing", "DATASET_MANIFEST", "offset_column", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "offset_source_column", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "offset_label", "TEXT"),
    ("pricing", "DATASET_MANIFEST", "export_weight_column", "TEXT"),
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
        "candidate_superglm_git_sha",
        (
            "TEXT CHECK ("
            "(candidate_artifact_format IS NULL "
            "OR candidate_artifact_format <> 'superglm-candidate-joblib-v3' "
            "OR candidate_superglm_git_sha IS NOT NULL) "
            "AND (candidate_superglm_git_sha IS NULL "
            "OR (length(candidate_superglm_git_sha) = 40 "
            "AND candidate_superglm_git_sha NOT GLOB '*[^0-9a-f]*'))"
            ")"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "model_source_sha256",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "builder_source_sha256",
        (
            "TEXT CHECK (builder_source_sha256 IS NULL OR "
            "(length(builder_source_sha256) = 64 "
            "AND builder_source_sha256 NOT GLOB '*[^0-9a-f]*'))"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "materialized_split_sha256",
        (
            "TEXT CHECK (materialized_split_sha256 IS NULL OR "
            "(length(materialized_split_sha256) = 64 "
            "AND materialized_split_sha256 NOT GLOB '*[^0-9a-f]*'))"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "runtime_sha256",
        (
            "TEXT CHECK (runtime_sha256 IS NULL OR "
            "(length(runtime_sha256) = 64 "
            "AND runtime_sha256 NOT GLOB '*[^0-9a-f]*'))"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "candidate_superglm_sha256",
        (
            "TEXT CHECK (candidate_superglm_sha256 IS NULL OR "
            "(length(candidate_superglm_sha256) = 64 "
            "AND candidate_superglm_sha256 NOT GLOB '*[^0-9a-f]*'))"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "validation_curve_reason",
        "TEXT",
    ),
    (
        "pricing",
        "MODEL_RUN",
        "validation_curve_status",
        (
            "TEXT CHECK ("
            "(validation_curve_status IS NULL AND validation_curve_reason IS NULL) "
            "OR (validation_curve_status IS NOT NULL "
            "AND validation_curve_status = 'COMPLETE' "
            "AND validation_curve_reason IS NULL) "
            "OR (validation_curve_status IS NOT NULL "
            "AND validation_curve_status = 'UNAVAILABLE' "
            "AND validation_curve_reason IS NOT NULL "
            "AND length(trim(validation_curve_reason)) > 0)"
            ")"
        ),
    ),
    (
        "pricing",
        "MODEL_RUN",
        "validation_source_model_run_id",
        "TEXT REFERENCES MODEL_RUN(model_run_id)",
    ),
    (
        "pricing",
        "PRICING_RATE_PACKAGE",
        "build_fingerprint_sha256",
        (
            "TEXT CHECK (build_fingerprint_sha256 IS NULL OR "
            "(parent_rate_package_id IS NULL "
            "AND length(build_fingerprint_sha256) = 64 "
            "AND build_fingerprint_sha256 NOT GLOB '*[^0-9a-f]*'))"
        ),
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
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
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


def _ensure_fold_metric_foreign_keys(connection) -> None:
    foreign_keys: dict[tuple[int, str], list[tuple[int, str, str]]] = {}
    for row in connection.execute("PRAGMA pricing.foreign_key_list('CV_FOLD_METRIC')").fetchall():
        foreign_keys.setdefault((int(row[0]), str(row[2])), []).append(
            (int(row[1]), str(row[3]), str(row[4]))
        )
    contract = {
        (table, tuple((child, parent) for _, child, parent in sorted(columns)))
        for (_, table), columns in foreign_keys.items()
    }
    expected = {
        ("MODEL_RUN", (("model_run_id", "model_run_id"),)),
        (
            "CV_FOLD",
            (("split_set_id", "split_set_id"), ("fold_no", "fold_no")),
        ),
    }
    if contract == expected:
        return

    orphan = connection.execute(
        """
        SELECT
            metric.model_run_id,
            metric.split_set_id,
            metric.fold_no,
            run.model_run_id IS NULL AS missing_run,
            fold.split_set_id IS NULL AS missing_fold
        FROM pricing.CV_FOLD_METRIC AS metric
        LEFT JOIN pricing.MODEL_RUN AS run
          ON run.model_run_id = metric.model_run_id
        LEFT JOIN pricing.CV_FOLD AS fold
          ON fold.split_set_id = metric.split_set_id
         AND fold.fold_no = metric.fold_no
        WHERE run.model_run_id IS NULL
           OR fold.split_set_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if orphan is not None:
        missing = []
        if bool(orphan[3]):
            missing.append("missing MODEL_RUN")
        if bool(orphan[4]):
            missing.append("missing CV_FOLD")
        raise RuntimeError(
            "cannot add CV_FOLD_METRIC foreign keys: orphan evidence "
            f"({orphan[0]!r}, {orphan[1]!r}, fold {orphan[2]!r}) is " + " and ".join(missing)
        )

    connection.execute("PRAGMA legacy_alter_table=ON")
    try:
        connection.execute(
            "ALTER TABLE pricing.CV_FOLD_METRIC RENAME TO __offline_upgrade_cv_fold_metric"
        )
    finally:
        connection.execute("PRAGMA legacy_alter_table=OFF")
    connection.execute(
        """
        CREATE TABLE pricing.CV_FOLD_METRIC (
            model_run_id TEXT NOT NULL,
            split_set_id TEXT NOT NULL,
            fold_no INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            CONSTRAINT PK_CV_FOLD_METRIC
                PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name),
            CONSTRAINT FK_CV_FOLD_METRIC_MODEL_RUN
                FOREIGN KEY (model_run_id) REFERENCES MODEL_RUN(model_run_id),
            CONSTRAINT FK_CV_FOLD_METRIC_FOLD
                FOREIGN KEY (split_set_id, fold_no)
                REFERENCES CV_FOLD(split_set_id, fold_no)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO pricing.CV_FOLD_METRIC (
            model_run_id, split_set_id, fold_no, metric_name, metric_value
        )
        SELECT model_run_id, split_set_id, fold_no, metric_name, metric_value
        FROM pricing.__offline_upgrade_cv_fold_metric
        """
    )
    connection.execute("DROP TABLE pricing.__offline_upgrade_cv_fold_metric")


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
        connection.execute(
            """
            WITH RECURSIVE package_lineage (
                candidate_model_run_id,
                rate_package_id,
                parent_rate_package_id,
                lineage_depth
            ) AS (
                SELECT
                    candidate_run.model_run_id,
                    candidate_package.rate_package_id,
                    candidate_package.parent_rate_package_id,
                    0
                FROM pricing.MODEL_RUN AS candidate_run
                JOIN pricing.PRICING_RATE_PACKAGE AS candidate_package
                  ON candidate_package.rate_package_id = candidate_run.rate_package_id

                UNION ALL

                SELECT
                    package_lineage.candidate_model_run_id,
                    parent_package.rate_package_id,
                    parent_package.parent_rate_package_id,
                    package_lineage.lineage_depth + 1
                FROM package_lineage
                JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
                  ON parent_package.rate_package_id = package_lineage.parent_rate_package_id
                WHERE package_lineage.lineage_depth < 100
            ),
            provable_validation_source AS (
                SELECT
                    package_lineage.candidate_model_run_id,
                    source_run.model_run_id
                FROM package_lineage
                JOIN pricing.MODEL_RUN AS source_run
                  ON source_run.rate_package_id = package_lineage.rate_package_id
                JOIN pricing.CV_SPLIT_SET AS source_split
                  ON source_split.split_set_id = source_run.split_set_id
                 AND source_split.manifest_id = source_run.manifest_id
                WHERE package_lineage.parent_rate_package_id IS NULL
            )
            UPDATE pricing.MODEL_RUN AS candidate_run
            SET validation_source_model_run_id = (
                SELECT source_run.model_run_id
                FROM provable_validation_source AS source_run
                WHERE source_run.candidate_model_run_id = candidate_run.model_run_id
            )
            WHERE candidate_run.validation_source_model_run_id IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM provable_validation_source AS source_run
                  WHERE source_run.candidate_model_run_id = candidate_run.model_run_id
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
            _ensure_fold_metric_foreign_keys(connection)
            if rebuilt_table:
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS pricing.UX_MODEL_RUN_RATE_PACKAGE
                    ON MODEL_RUN(rate_package_id)
                    WHERE rate_package_id IS NOT NULL
                    """
                )
            connection.commit()
            post_upgrade_path = OFFLINE_DDL_DIR / "pricing_post_upgrade.sql"
            connection.executescript(post_upgrade_path.read_text(encoding="utf-8"))
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
