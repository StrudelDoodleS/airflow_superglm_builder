"""Shared SQL Server connection helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra import db as shared_db  # noqa: E402
from pricing_pipeline.infra.config import Settings  # noqa: E402
from pricing_pipeline.infra.runtime import (  # noqa: E402
    PipelineRuntime,
    runtime_from_env_or_module,
)


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _settings_from_env() -> Settings:
    load_env()
    return Settings.from_env(os.environ)


def get_runtime(runtime_module: str | None = None) -> PipelineRuntime:
    load_env()
    return runtime_from_env_or_module(runtime_module, env=os.environ)


def build_sqlalchemy_url(*, database: str | None = None) -> str:
    settings = _settings_from_env()
    return shared_db.build_sqlalchemy_url(
        settings,
        database=database or settings.pricing_database,
    )


def get_engine(
    *,
    database: str | None = None,
    runtime_module: str | None = None,
) -> Engine:
    return get_runtime(runtime_module).get_engine(database=database)


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
