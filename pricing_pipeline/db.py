from __future__ import annotations

from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pricing_pipeline.config import Settings


def build_odbc_connect_string(settings: Settings, *, database: str) -> str:
    return (
        f"DRIVER={{{settings.mssql_driver}}};"
        f"SERVER={settings.mssql_server};"
        f"DATABASE={database};"
        f"UID={settings.mssql_user};"
        f"PWD={settings.mssql_password};"
        f"Encrypt={settings.mssql_encrypt};"
        f"TrustServerCertificate={settings.mssql_trust_server_cert};"
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
    with master.begin() as con:
        exists = con.execute(
            text("SELECT 1 FROM sys.databases WHERE name = :database"),
            {"database": database},
        ).scalar()
        if not exists:
            con.execute(text(f"CREATE DATABASE [{escaped}]"))
