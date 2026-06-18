from __future__ import annotations

from pathlib import Path

import pytest

from pricing_pipeline.infra.reset_schema import (
    CONFIRMATION_FLAG,
    SCHEMA_CONFIG_KEYS,
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


def test_normalize_schema_names_rejects_schemas_outside_runtime_allowlist():
    with pytest.raises(ValueError, match="not configured runtime schema"):
        normalize_schema_names(("pricing", "pricing_stg", "dbo"))


def test_normalize_schema_names_accepts_runtime_configured_names():
    assert normalize_schema_names(("python_pricing", "python_stg", "python_mlops")) == (
        "python_pricing",
        "python_stg",
        "python_mlops",
    )


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


def test_cli_uses_runtime_schema_names_by_default():
    class FakeRuntime:
        def __init__(self):
            self.settings = type(
                "FakeSettings",
                (),
                {
                    "schema_names": type(
                        "FakeSchemaNames",
                        (),
                        {
                            "as_execution_options": lambda self: {
                                "pricing_schema": "python_pricing",
                                "pricing_staging_schema": "python_stg",
                                "mlops_schema": "python_mlops",
                            }
                        },
                    )()
                },
            )()

    assert reset_remote_pricing_schema.schema_names_from_runtime(FakeRuntime(), ()) == (
        "python_pricing",
        "python_stg",
        "python_mlops",
    )


def test_cli_rejects_schema_override_outside_runtime_schema_names():
    class FakeRuntime:
        def __init__(self):
            self.settings = type(
                "FakeSettings",
                (),
                {
                    "schema_names": type(
                        "FakeSchemaNames",
                        (),
                        {
                            "as_execution_options": lambda self: {
                                "pricing_schema": "python_pricing",
                                "pricing_staging_schema": "python_stg",
                                "mlops_schema": "python_mlops",
                            }
                        },
                    )()
                },
            )()

    with pytest.raises(ValueError, match="not configured runtime schema"):
        reset_remote_pricing_schema.schema_names_from_runtime(
            FakeRuntime(),
            ("python_pricing", "python_stg", "dbo"),
        )


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


def test_execute_uses_single_transaction_for_drop_and_migrations(
    tmp_path,
    monkeypatch,
):
    migration = tmp_path / "V001__example.sql"
    migration.write_text("CREATE TABLE pricing.EXAMPLE(id INT);\n", encoding="utf-8")

    class ScalarResult:
        def __init__(self, value):
            self.value = value

        def scalar_one(self):
            return self.value

    class MappingResult:
        def mappings(self):
            return self

        def one_or_none(self):
            return None

    class RowsResult:
        def all(self):
            return []

    class FakeConnection:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            statement = str(sql)
            self.executed.append(statement)
            if "SELECT DB_NAME()" in statement:
                return ScalarResult("MVA")
            if "sp_getapplock" in statement:
                return ScalarResult(0)
            if "FROM dbo.SCHEMA_CONFIGURATION" in statement:
                return RowsResult()
            if "FROM dbo.SCHEMA_MIGRATION" in statement:
                return MappingResult()
            return ScalarResult(None)

    class FakeBegin:
        def __init__(self, engine):
            self.engine = engine

        def __enter__(self):
            self.engine.begin_count += 1
            return self.engine.connection

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeEngine:
        def __init__(self):
            self.connection = FakeConnection()
            self.begin_count = 0

        def execution_options(self, **_options):
            return self

        def begin(self):
            return FakeBegin(self)

    engine = FakeEngine()
    monkeypatch.setattr("pricing_pipeline.infra.migrations.getpass.getuser", lambda: "tester")

    result = reset_and_reseed_schema(
        engine,
        migrations_dir=tmp_path,
        expected_database="MVA",
        execute=True,
    )

    assert engine.begin_count == 1
    assert result.applied_migrations == ("V001__example.sql",)
    statements = "\n".join(engine.connection.executed)
    assert "DROP TABLE" in statements
    assert "CREATE TABLE pricing.EXAMPLE" in statements


def test_schema_config_key_order_is_stable():
    assert SCHEMA_CONFIG_KEYS == (
        "pricing_schema",
        "pricing_staging_schema",
        "mlops_schema",
    )
