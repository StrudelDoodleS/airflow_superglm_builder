"""Shared SQL Server connection helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def get_runtime(runtime_module: str | None = None) -> PipelineRuntime:
    load_env()
    return runtime_from_env_or_module(runtime_module, env=os.environ)


def get_engine(
    *,
    database: str | None = None,
    runtime_module: str | None = None,
) -> Engine:
    return get_runtime(runtime_module).get_engine(database=database)
