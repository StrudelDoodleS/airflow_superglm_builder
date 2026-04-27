from pricing_pipeline import db
from pricing_pipeline.config import Settings
from pricing_pipeline.db import build_odbc_connect_string


def test_settings_defaults_are_local_dev_safe():
    settings = Settings.from_env({})
    assert settings.mssql_server == "mssql,1433"
    assert settings.pricing_database == "PricingLab"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


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
