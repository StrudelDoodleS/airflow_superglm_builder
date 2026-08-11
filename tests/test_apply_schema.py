from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import apply_schema
from scripts.apply_schema import parse_args, verify_expected_database


class _ScalarResult:
    def __init__(self, value: str):
        self.value = value

    def scalar_one(self) -> str:
        return self.value


class _Connection:
    def __init__(self, database: str):
        self.database = database
        self.statements: list[str] = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return _ScalarResult(self.database)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _Engine:
    def __init__(self, connection: _Connection):
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


def test_apply_schema_accepts_an_explicit_expected_database():
    args = parse_args(
        [
            "--runtime-module",
            "work_runtime.database",
            "--expected-database",
            "PricingAudit",
        ]
    )

    assert args.runtime_module == "work_runtime.database"
    assert args.expected_database == "PricingAudit"


def test_apply_schema_database_guard_accepts_only_the_exact_target():
    connection = _Connection("PricingAudit")

    assert verify_expected_database(connection, "PricingAudit") == "PricingAudit"
    assert connection.statements == ["SELECT DB_NAME();"]


def test_apply_schema_database_guard_rejects_mismatch_before_migrations():
    connection = _Connection("WrongDatabase")

    with pytest.raises(RuntimeError, match="Refusing to apply migrations"):
        verify_expected_database(connection, "PricingAudit")

    assert connection.statements == ["SELECT DB_NAME();"]


def test_apply_schema_database_guard_rejects_an_empty_expectation():
    connection = _Connection("PricingAudit")

    with pytest.raises(ValueError, match="expected database name is required"):
        verify_expected_database(connection, " ")

    assert connection.statements == []


def test_apply_schema_main_checks_database_before_calling_migrations(monkeypatch, tmp_path):
    connection = _Connection("WrongDatabase")
    engine = _Engine(connection)
    runtime = SimpleNamespace(
        settings=SimpleNamespace(pricing_database="ConfiguredDatabase"),
        get_engine=lambda: engine,
    )
    applied: list[tuple[object, Path]] = []

    monkeypatch.setattr(
        apply_schema,
        "parse_args",
        lambda: Namespace(
            runtime_module="work_runtime.database",
            expected_database="PricingAudit",
        ),
    )
    monkeypatch.setattr(apply_schema, "load_env", lambda: None)
    monkeypatch.setattr(apply_schema, "get_runtime", lambda _module: runtime)
    monkeypatch.setattr(apply_schema, "_schema_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "pricing_pipeline.infra.migrations.migration_files",
        lambda _path: [tmp_path / "V001__test.sql"],
    )
    monkeypatch.setattr(
        "pricing_pipeline.infra.migrations.apply_migrations",
        lambda supplied_engine, supplied_path: applied.append((supplied_engine, supplied_path)),
    )

    with pytest.raises(RuntimeError, match="WrongDatabase"):
        apply_schema.main()

    assert connection.statements == ["SELECT DB_NAME();"]
    assert applied == []
