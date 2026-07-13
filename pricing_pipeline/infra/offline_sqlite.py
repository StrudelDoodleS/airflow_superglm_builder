"""Persistent attached-schema SQLite storage for local pricing workflows."""

from __future__ import annotations

import fcntl
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


OFFLINE_DDL_DIR = Path(__file__).resolve().parents[2] / "db" / "offline_sqlite"
COORDINATOR_DB_FILE = "coordinator.sqlite"
SCHEMA_DB_FILES = {
    "pricing": "pricing.sqlite",
    "pricing_stg": "pricing_stg.sqlite",
    "mlops": "mlops.sqlite",
}


@contextmanager
def local_publish_lock(root: str | Path) -> Iterator[BinaryIO]:
    """Serialize local staging/publication across notebook processes."""
    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    lock_path = resolved / ".publish.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def apply_offline_ddl(engine: Engine) -> None:
    """Create any missing local tables without deleting existing data."""
    connection = engine.raw_connection()
    try:
        for schema in SCHEMA_DB_FILES:
            ddl_path = OFFLINE_DDL_DIR / f"{schema}.sql"
            connection.executescript(ddl_path.read_text(encoding="utf-8"))
        connection.commit()
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
