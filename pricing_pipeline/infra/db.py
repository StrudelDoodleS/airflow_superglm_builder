from __future__ import annotations

from urllib.parse import quote_plus

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


def build_sqlalchemy_url(settings: Settings, *, database: str) -> str:
    odbc = build_odbc_connect_string(settings, database=database)
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"


def get_engine(settings: Settings, *, database: str | None = None) -> Engine:
    return create_engine(
        build_sqlalchemy_url(settings, database=database or settings.pricing_database),
        fast_executemany=True,
        future=True,
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
