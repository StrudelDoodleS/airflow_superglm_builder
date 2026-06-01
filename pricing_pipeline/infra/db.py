from __future__ import annotations

import struct
from urllib.parse import quote, quote_plus

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.schema import render_sql_schemas


SQL_COPT_SS_ACCESS_TOKEN = 1256


def _format_odbc_value(value: str, *, always_brace: bool = False) -> str:
    needs_braces = always_brace or any(char in value for char in ";{}")
    if not needs_braces:
        return value
    return "{" + value.replace("}", "}}") + "}"


def build_odbc_connect_string(
    settings: Settings,
    *,
    database: str,
    include_credentials: bool | None = None,
) -> str:
    if include_credentials is None:
        include_credentials = settings.mssql_auth_mode.strip().lower() == "sql_password"

    parts = [
        f"DRIVER={_format_odbc_value(settings.mssql_driver, always_brace=True)};",
        f"SERVER={_format_odbc_value(settings.mssql_server)};",
        f"DATABASE={_format_odbc_value(database)};",
    ]
    if include_credentials:
        parts.extend(
            [
                f"UID={_format_odbc_value(settings.mssql_user)};",
                f"PWD={_format_odbc_value(settings.mssql_password)};",
            ]
        )
    parts.extend(
        [
            f"Encrypt={_format_odbc_value(settings.mssql_encrypt)};",
            f"TrustServerCertificate={_format_odbc_value(settings.mssql_trust_server_cert)};",
        ]
    )
    return "".join(parts)


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
    auth_mode = settings.mssql_auth_mode.strip().lower()
    if auth_mode == "azure_token" and dialect != "mssql+pyodbc":
        raise ValueError(
            "MSSQL_AUTH_MODE=azure_token requires "
            "MSSQL_SQLALCHEMY_DIALECT=mssql+pyodbc"
        )
    if dialect == "mssql+pymssql":
        return build_pymssql_url(settings, database=database)
    if dialect != "mssql+pyodbc":
        raise ValueError(
            "MSSQL_SQLALCHEMY_DIALECT must be one of: mssql+pyodbc, mssql+pymssql"
        )
    odbc = build_odbc_connect_string(settings, database=database)
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"


def _azure_sql_access_token_struct(settings: Settings) -> bytes:
    try:
        from azure.identity import DefaultAzureCredential
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MSSQL_AUTH_MODE=azure_token requires the azure-identity package"
        ) from exc

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    token = credential.get_token(settings.mssql_token_scope).token
    token_bytes = token.encode("utf-16-le")
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


def _attach_schema_renderer(engine: Engine, schemas) -> None:
    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _render_schema_names(_conn, _cursor, statement, parameters, _context, _executemany):
        return render_sql_schemas(statement, schemas), parameters


def configure_engine(engine: Engine, schemas) -> Engine:
    """Attach pricing schema execution options/rendering to an externally built engine."""
    if hasattr(engine, "update_execution_options"):
        engine.update_execution_options(**schemas.as_execution_options())
    attached = getattr(engine, "_pricing_schema_renderer_schemas", None)
    if hasattr(engine, "dispatch") and attached != schemas:
        _attach_schema_renderer(engine, schemas)
        setattr(engine, "_pricing_schema_renderer_schemas", schemas)
    return engine


def get_engine(settings: Settings, *, database: str | None = None) -> Engine:
    dialect = settings.mssql_sqlalchemy_dialect.strip().lower()
    auth_mode = settings.mssql_auth_mode.strip().lower()
    if auth_mode not in {"sql_password", "azure_token"}:
        raise ValueError("MSSQL_AUTH_MODE must be one of: sql_password, azure_token")

    engine_kwargs = {
        "future": True,
        "execution_options": settings.schema_names.as_execution_options(),
    }
    if dialect == "mssql+pyodbc":
        engine_kwargs["fast_executemany"] = True
    if auth_mode == "azure_token":
        engine_kwargs["connect_args"] = {
            "attrs_before": {
                SQL_COPT_SS_ACCESS_TOKEN: _azure_sql_access_token_struct(settings)
            }
        }

    engine = create_engine(
        build_sqlalchemy_url(settings, database=database or settings.pricing_database),
        **engine_kwargs,
    )
    return configure_engine(engine, settings.schema_names)


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
