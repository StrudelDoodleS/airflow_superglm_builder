from __future__ import annotations

from pathlib import Path

import pytest

from pricing_pipeline.infra.reset_schema import (
    CONFIRMATION_FLAG,
    build_drop_batches,
    normalize_schema_names,
    reset_and_reseed_schema,
    verify_expected_database,
)
from scripts import reset_remote_pricing_schema


def test_normalize_schema_names_defaults_to_owned_pricing_schemas():
    assert normalize_schema_names(()) == ("pricing", "pricing_stg", "mlops")


def test_normalize_schema_names_rejects_unsafe_schema_name():
    with pytest.raises(ValueError, match="schema name"):
        normalize_schema_names(("pricing; DROP TABLE dbo.Users",))


def test_drop_batches_remove_owned_objects_before_migration_tracking():
    batches = build_drop_batches(("pricing", "pricing_stg", "mlops"))
    joined = "\n".join(batches)

    expected_order = [
        "ALTER TABLE",
        "DROP TRIGGER",
        "DROP VIEW",
        "DROP PROCEDURE",
        "DROP FUNCTION",
        "DROP TABLE",
        "DROP TABLE IF EXISTS dbo.SCHEMA_MIGRATION",
        "DROP TABLE IF EXISTS dbo.SCHEMA_CONFIGURATION",
    ]
    positions = [joined.index(fragment) for fragment in expected_order]
    assert positions == sorted(positions)
    assert "s.name IN (N'pricing', N'pricing_stg', N'mlops')" in joined


def test_verify_expected_database_rejects_wrong_target_before_dropping():
    class FakeConnection:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((str(sql), params))
            return self

        def scalar_one(self):
            return "WrongDb"

    con = FakeConnection()

    with pytest.raises(RuntimeError, match="Refusing to reset database"):
        verify_expected_database(con, "ExpectedDb")

    assert len(con.executed) == 1
    assert "DB_NAME()" in con.executed[0][0]


def test_cli_requires_confirmation_for_execute():
    parser = reset_remote_pricing_schema.build_parser()

    args = parser.parse_args(
        [
            "--expected-database",
            "MVA",
            "--execute",
        ]
    )

    with pytest.raises(SystemExit, match=CONFIRMATION_FLAG):
        reset_remote_pricing_schema.validate_args(args)


def test_cli_accepts_dry_run_without_confirmation():
    parser = reset_remote_pricing_schema.build_parser()

    args = parser.parse_args(
        [
            "--expected-database",
            "MVA",
        ]
    )

    reset_remote_pricing_schema.validate_args(args)


def test_cli_uses_default_migration_dir():
    parser = reset_remote_pricing_schema.build_parser()

    args = parser.parse_args(["--expected-database", "MVA"])

    assert args.schema_dir == Path("db/migrations")


def test_execute_requires_migration_files_before_any_database_statement(tmp_path):
    class FakeConnection:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((str(sql), params))
            return self

    class FakeBegin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeEngine:
        def __init__(self):
            self.connection = FakeConnection()

        def execution_options(self, **_options):
            return self

        def begin(self):
            return FakeBegin(self.connection)

    engine = FakeEngine()

    with pytest.raises(RuntimeError, match="No schema DDL files"):
        reset_and_reseed_schema(
            engine,
            migrations_dir=tmp_path,
            expected_database="MVA",
            execute=True,
        )

    assert engine.connection.executed == []
