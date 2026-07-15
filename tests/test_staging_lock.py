from __future__ import annotations

from types import SimpleNamespace

import pytest

from pricing_pipeline.publishing.staging_lock import acquire_staging_export_lock


class _ScalarResult:
    def __init__(self, value: int):
        self.value = value

    def scalar_one(self) -> int:
        return self.value


class _Connection:
    dialect = SimpleNamespace(name="mssql")

    def __init__(self, result: int = 0):
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement, params):
        self.calls.append((str(statement), dict(params)))
        return _ScalarResult(self.result)


def test_staging_export_lock_is_transaction_owned_and_export_scoped():
    connection = _Connection()

    acquire_staging_export_lock(connection, "export-1")

    sql, params = connection.calls[0]
    assert "sys.sp_getapplock" in sql
    assert "@LockMode = 'Exclusive'" in sql
    assert "@LockOwner = 'Transaction'" in sql
    assert params == {
        "lock_resource": "pricing_staging_export:export-1",
        "lock_timeout_ms": 10_000,
    }


def test_staging_export_lock_rejects_sql_server_lock_failure():
    with pytest.raises(RuntimeError, match="export-1"):
        acquire_staging_export_lock(_Connection(result=-1), "export-1")


def test_staging_export_lock_is_noop_for_non_sql_server_connections():
    connection = _Connection()
    connection.dialect = SimpleNamespace(name="sqlite")

    acquire_staging_export_lock(connection, "export-1")

    assert connection.calls == []
