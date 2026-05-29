import struct
import sys
from types import SimpleNamespace

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra import db
from pricing_pipeline.infra.db import build_odbc_connect_string
from pricing_pipeline.infra.schema import SchemaNames, render_sql_schemas


def test_settings_defaults_are_local_dev_safe():
    settings = Settings.from_env({})
    assert settings.mssql_server == "mssql,1433"
    assert settings.pricing_database == "PricingLab"
    assert settings.mssql_sqlalchemy_dialect == "mssql+pyodbc"
    assert settings.mssql_auth_mode == "sql_password"
    assert settings.pricing_schema == "pricing"
    assert settings.pricing_staging_schema == "pricing_stg"
    assert settings.mlops_schema == "mlops"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.mlflow_enabled is True


def test_settings_can_disable_optional_mlflow_tracking():
    settings = Settings.from_env({"PRICING_ENABLE_MLFLOW": "false"})

    assert settings.mlflow_enabled is False


def test_odbc_connection_string_targets_database():
    settings = Settings.from_env(
        {
            "MSSQL_SERVER": "localhost,1433",
            "MSSQL_DATABASE": "PricingLab",
            "MSSQL_USER": "sa",
            "MSSQL_PASSWORD": "secret",
            "MSSQL_DRIVER": "ODBC Driver 18 for SQL Server",
            "MSSQL_ENCRYPT": "no",
            "MSSQL_TRUST_SERVER_CERT": "yes",
        }
    )
    odbc = build_odbc_connect_string(settings, database=settings.pricing_database)
    assert "SERVER=localhost,1433" in odbc
    assert "DATABASE=PricingLab" in odbc
    assert "PWD=secret" in odbc


def test_odbc_connection_string_brace_escapes_password_delimiters():
    semicolon_settings = Settings.from_env({"MSSQL_PASSWORD": "sec;ret"})
    semicolon_odbc = build_odbc_connect_string(
        semicolon_settings, database=semicolon_settings.pricing_database
    )
    assert "PWD={sec;ret};" in semicolon_odbc

    brace_settings = Settings.from_env({"MSSQL_PASSWORD": "sec}ret"})
    brace_odbc = build_odbc_connect_string(
        brace_settings, database=brace_settings.pricing_database
    )
    assert "PWD={sec}}ret};" in brace_odbc


def test_odbc_connection_string_prevents_password_attribute_injection():
    settings = Settings.from_env({"MSSQL_PASSWORD": "sec;Encrypt=yes"})

    odbc = build_odbc_connect_string(settings, database=settings.pricing_database)

    assert "PWD={sec;Encrypt=yes};" in odbc
    assert ";Encrypt=yes;" not in odbc


def test_odbc_connection_string_omits_password_for_azure_token_auth():
    settings = Settings.from_env({"MSSQL_AUTH_MODE": "azure_token"})

    odbc = build_odbc_connect_string(settings, database=settings.pricing_database)

    assert "UID=" not in odbc
    assert "PWD=" not in odbc
    assert "DATABASE=PricingLab" in odbc


def test_settings_load_and_validate_custom_schema_names():
    settings = Settings.from_env(
        {
            "PRICING_SCHEMA": "python_pricing",
            "PRICING_STAGING_SCHEMA": "python_pricing_stg",
            "MLOPS_SCHEMA": "python_mlops",
        }
    )

    assert settings.schema_names == SchemaNames(
        pricing="python_pricing",
        pricing_staging="python_pricing_stg",
        mlops="python_mlops",
    )


def test_settings_reject_invalid_schema_names():
    try:
        Settings.from_env({"PRICING_SCHEMA": "pricing-prod"})
    except ValueError as exc:
        assert "PRICING_SCHEMA" in str(exc)
    else:
        raise AssertionError("invalid schema name should fail clearly")


def test_render_sql_schemas_rewrites_schema_tokens_without_touching_table_names():
    rendered = render_sql_schemas(
        """
        IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
            EXEC('CREATE SCHEMA pricing');
        SELECT * FROM pricing.PRICING_MODEL;
        SELECT * FROM pricing_stg.STG_RATE_CELL;
        SELECT * FROM mlops.MODEL_RUN_METRIC;
        """,
        SchemaNames(
            pricing="python_pricing",
            pricing_staging="python_pricing_stg",
            mlops="python_mlops",
        ),
    )

    assert "name = 'python_pricing'" in rendered
    assert "CREATE SCHEMA python_pricing" in rendered
    assert "python_pricing.PRICING_MODEL" in rendered
    assert "python_pricing_stg.STG_RATE_CELL" in rendered
    assert "python_mlops.MODEL_RUN_METRIC" in rendered
    assert "python_pricing.PRICING_MODEL" in rendered


def test_pymssql_sqlalchemy_url_uses_host_port_database_and_escaped_credentials():
    settings = Settings.from_env(
        {
            "MSSQL_SQLALCHEMY_DIALECT": "mssql+pymssql",
            "MSSQL_SERVER": "localhost,1433",
            "MSSQL_DATABASE": "PricingLab",
            "MSSQL_USER": "pricing user",
            "MSSQL_PASSWORD": "sec/ret@word",
        }
    )

    url = db.build_sqlalchemy_url(settings, database=settings.pricing_database)

    assert url == (
        "mssql+pymssql://pricing%20user:sec%2Fret%40word@localhost:1433/PricingLab"
    )


def test_sqlalchemy_url_rejects_unknown_mssql_dialect():
    settings = Settings.from_env({"MSSQL_SQLALCHEMY_DIALECT": "sqlite"})

    try:
        db.build_sqlalchemy_url(settings, database=settings.pricing_database)
    except ValueError as exc:
        assert "MSSQL_SQLALCHEMY_DIALECT" in str(exc)
    else:
        raise AssertionError("unknown SQLAlchemy dialect should fail clearly")


def test_get_engine_only_enables_fast_executemany_for_pyodbc(monkeypatch):
    calls = []

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return object()

    monkeypatch.setattr(db, "create_engine", fake_create_engine)

    pyodbc_settings = Settings.from_env({"MSSQL_SQLALCHEMY_DIALECT": "mssql+pyodbc"})
    pymssql_settings = Settings.from_env({"MSSQL_SQLALCHEMY_DIALECT": "mssql+pymssql"})

    db.get_engine(pyodbc_settings)
    db.get_engine(pymssql_settings)

    assert calls[0][1]["fast_executemany"] is True
    assert "fast_executemany" not in calls[1][1]
    assert calls[0][1]["future"] is True
    assert calls[1][1]["future"] is True


def test_get_engine_can_use_azure_sql_access_token(monkeypatch):
    calls = []

    class FakeEngine:
        def __init__(self):
            self.execution_options_calls = []

        def update_execution_options(self, **kwargs):
            self.execution_options_calls.append(kwargs)

    fake_engine = FakeEngine()

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return fake_engine

    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    monkeypatch.setattr(db, "_azure_sql_access_token_struct", lambda settings: b"packed-token")

    settings = Settings.from_env({"MSSQL_AUTH_MODE": "azure_token"})
    engine = db.get_engine(settings)

    assert engine is fake_engine
    assert calls[0][0].startswith("mssql+pyodbc:///?odbc_connect=")
    assert calls[0][1]["connect_args"]["attrs_before"] == {
        db.SQL_COPT_SS_ACCESS_TOKEN: b"packed-token"
    }
    assert fake_engine.execution_options_calls == [
        {
            "pricing_schema": "pricing",
            "pricing_staging_schema": "pricing_stg",
            "mlops_schema": "mlops",
        }
    ]


def test_azure_sql_access_token_struct_matches_pyodbc_encoding(monkeypatch):
    calls = []

    class FakeCredential:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def get_token(self, scope):
            calls.append(("get_token", scope))
            return SimpleNamespace(token="abc")

    monkeypatch.setitem(
        sys.modules,
        "azure.identity",
        SimpleNamespace(DefaultAzureCredential=FakeCredential),
    )

    settings = Settings.from_env({"MSSQL_TOKEN_SCOPE": "scope://unit-test"})
    packed = db._azure_sql_access_token_struct(settings)
    token_bytes = "abc".encode("utf-16-le")

    assert packed == struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    assert calls == [
        ("init", {"exclude_interactive_browser_credential": False}),
        ("get_token", "scope://unit-test"),
    ]


def test_ensure_database_uses_autocommit_connection_when_creating(monkeypatch):
    class FakeScalarResult:
        def __init__(self, value):
            self.value = value

        def scalar(self):
            return self.value

    class FakeConnection:
        def __init__(self):
            self.execution_options_calls = []
            self.executed = []

        def execution_options(self, **options):
            self.execution_options_calls.append(options)
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            sql = str(statement)
            self.executed.append((sql, params))
            if sql.startswith("SELECT 1 FROM sys.databases"):
                return FakeScalarResult(None)
            return FakeScalarResult(None)

    class FakeEngine:
        def __init__(self):
            self.connection = FakeConnection()

        def begin(self):
            raise AssertionError(
                "ensure_database must not create databases in a transaction"
            )

        def connect(self):
            return self.connection

    engine = FakeEngine()
    monkeypatch.setattr(db, "get_engine", lambda settings, *, database: engine)

    db.ensure_database(Settings.from_env({}), "Pricing]Lab")

    assert engine.connection.execution_options_calls == [
        {"isolation_level": "AUTOCOMMIT"}
    ]
    assert engine.connection.executed == [
        (
            "SELECT 1 FROM sys.databases WHERE name = :database",
            {"database": "Pricing]Lab"},
        ),
        ("CREATE DATABASE [Pricing]]Lab]", None),
    ]
