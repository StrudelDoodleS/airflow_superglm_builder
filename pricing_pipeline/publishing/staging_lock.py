"""Transaction lock shared by staging replacement and package publication."""

from __future__ import annotations

from sqlalchemy import text


_LOCK_TIMEOUT_MS = 10_000
_LOCK_RESOURCE_PREFIX = "pricing_staging_export:"


def acquire_staging_export_lock(connection, export_id: str) -> None:
    """Serialize SQL Server staging mutations and package reads for one export."""
    dialect = getattr(connection, "dialect", None)
    if dialect is None:
        dialect = getattr(getattr(connection, "engine", None), "dialect", None)
    if getattr(dialect, "name", None) != "mssql":
        return

    cleaned_export_id = str(export_id).strip()
    if not cleaned_export_id:
        raise ValueError("export_id is required for the staging transaction lock")
    lock_result = connection.execute(
        text(
            """
            DECLARE @lock_result INT;
            EXEC @lock_result = sys.sp_getapplock
                @Resource = :lock_resource,
                @LockMode = 'Exclusive',
                @LockOwner = 'Transaction',
                @LockTimeout = :lock_timeout_ms;
            SELECT @lock_result;
            """
        ),
        {
            "lock_resource": f"{_LOCK_RESOURCE_PREFIX}{cleaned_export_id}",
            "lock_timeout_ms": _LOCK_TIMEOUT_MS,
        },
    ).scalar_one()
    if int(lock_result) < 0:
        raise RuntimeError(
            f"could not acquire staging publication lock for export_id={cleaned_export_id!r}"
        )
