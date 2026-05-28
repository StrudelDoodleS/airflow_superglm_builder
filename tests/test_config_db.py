from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra import db
from pricing_pipeline.infra.db import build_odbc_connect_string


def test_settings_defaults_are_local_dev_safe():
    settings = Settings.from_env({})
    assert settings.mssql_server == "mssql,1433"
    assert settings.pricing_database == "PricingLab"
    assert settings.mssql_sqlalchemy_dialect == "mssql+pyodbc"
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
