"""Shared SQL Server connection helpers."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_engine() -> Engine:
    load_env()

    server = os.getenv("MSSQL_SERVER", "localhost,1433")
    database = os.getenv("MSSQL_DATABASE", "PricingLab")
    user = os.getenv("MSSQL_USER", "sa")
    password = os.getenv("MSSQL_PASSWORD", "YourStrong(!)Password123")
    driver = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
    encrypt = os.getenv("MSSQL_ENCRYPT", "no")
    trust_cert = os.getenv("MSSQL_TRUST_SERVER_CERT", "yes")

    odbc = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
    )

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}",
        fast_executemany=True,
        future=True,
    )


def split_sql_server_batches(sql_text: str) -> list[str]:
    """Split a SQL Server script on GO batch separators."""
    batches: list[str] = []
    current: list[str] = []

    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)

    batch = "\n".join(current).strip()
    if batch:
        batches.append(batch)
    return batches


def run_sql_file(engine: Engine, path: Path) -> None:
    sql_text = path.read_text(encoding="utf-8")
    with engine.begin() as con:
        for batch in split_sql_server_batches(sql_text):
            con.execute(text(batch))
