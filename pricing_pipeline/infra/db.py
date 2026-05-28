from __future__ import annotations

from urllib.parse import quote, quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.config import Settings


def _format_odbc_value(value: str, *, always_brace: bool = False) -> str:
    needs_braces = always_brace or any(char in value for char in ";{}")
    if not needs_braces:
        return value
    return "{" + value.replace("}", "}}") + "}"


def build_odbc_connect_string(settings: Settings, *, database: str) -> str:
    return (
        f"DRIVER={_format_odbc_value(settings.mssql_driver, always_brace=True)};"
        f"SERVER={_format_odbc_value(settings.mssql_server)};"
        f"DATABASE={_format_odbc_value(database)};"
        f"UID={_format_odbc_value(settings.mssql_user)};"
        f"PWD={_format_odbc_value(settings.mssql_password)};"
        f"Encrypt={_format_odbc_value(settings.mssql_encrypt)};"
        f"TrustServerCertificate={_format_odbc_value(settings.mssql_trust_server_cert)};"
    )


def _format_server_netloc(server: str) -> str:
    host, separator, port = server.strip().rpartition(",")
    if separator and port.isdigit():
        return f"{host}:{port}"
    return server.strip()


def build_pymssql_url(settings: Settings, *, database: str) -> str:
    user = quote(settings.mssql_user, safe="")
    password = quote(settings.mssql_password, safe="")
    database_name = quote(database, safe="")
    return (
        f"mssql+pymssql://{user}:{password}"
        f"@{_format_server_netloc(settings.mssql_server)}/{database_name}"
    )


def build_sqlalchemy_url(settings: Settings, *, database: str) -> str:
    dialect = settings.mssql_sqlalchemy_dialect.strip().lower()
    if dialect == "mssql+pymssql":
        return build_pymssql_url(settings, database=database)
    if dialect != "mssql+pyodbc":
        raise ValueError(
            "MSSQL_SQLALCHEMY_DIALECT must be one of: mssql+pyodbc, mssql+pymssql"
        )
    odbc = build_odbc_connect_string(settings, database=database)
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"


def get_engine(settings: Settings, *, database: str | None = None) -> Engine:
    engine_kwargs = {"future": True}
    if settings.mssql_sqlalchemy_dialect.strip().lower() == "mssql+pyodbc":
        engine_kwargs["fast_executemany"] = True
    return create_engine(
        build_sqlalchemy_url(settings, database=database or settings.pricing_database),
        **engine_kwargs,
    )


def ensure_database(settings: Settings, database: str) -> None:
    master = get_engine(settings, database="master")
    escaped = database.replace("]", "]]")
    with master.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
        exists = con.execute(
            text("SELECT 1 FROM sys.databases WHERE name = :database"),
            {"database": database},
        ).scalar()
        if not exists:
            con.execute(text(f"CREATE DATABASE [{escaped}]"))
