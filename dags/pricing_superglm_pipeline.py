from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

from pricing_pipeline.config import Settings
from pricing_pipeline.db import ensure_database, get_engine
from pricing_pipeline.fremtpl import load_fremtpl_raw
from pricing_pipeline.manifest import create_fremtpl_manifest, new_manifest_id
from pricing_pipeline.migrations import apply_migrations
from pricing_pipeline.pipeline import run_training_export_publish


_REPO_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "db" / "migrations"
_DOCKER_SCHEMA_DIR = Path("/opt/pricing/db/migrations")
SCHEMA_DIR = Path(
    os.environ.get(
        "PRICING_SCHEMA_DIR",
        os.environ.get(
            "PRICING_MIGRATIONS_DIR",
            str(_DOCKER_SCHEMA_DIR if _DOCKER_SCHEMA_DIR.exists() else _REPO_SCHEMA_DIR),
        ),
    )
)
FREMTPL_DATASET_NAME = "freMTPL2freq"


def _settings() -> Settings:
    return Settings.from_env(os.environ)


def _context_date_iso(context: dict) -> str:
    dag_run = context.get("dag_run")
    run_datetime = (
        context.get("logical_date")
        or getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "run_after", None)
        or context.get("data_interval_start")
        or datetime.now(UTC)
    )
    return run_datetime.date().isoformat()


@dag(
    dag_id="pricing_superglm_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pricing", "superglm", "mlflow"],
)
def _pricing_superglm_pipeline():
    @task
    def apply_pricing_schema() -> list[str]:
        settings = _settings()
        if not settings.skip_database_create:
            ensure_database(settings, settings.pricing_database)
        return apply_migrations(get_engine(settings), SCHEMA_DIR)

    @task
    def load_raw() -> int:
        settings = _settings()
        return load_fremtpl_raw(get_engine(settings))

    @task
    def create_manifest() -> str:
        settings = _settings()
        manifest_id = new_manifest_id(FREMTPL_DATASET_NAME)
        return create_fremtpl_manifest(get_engine(settings), manifest_id=manifest_id)

    @task
    def train_and_publish(manifest_id: str) -> dict[str, str]:
        settings = _settings()
        context = get_current_context()
        logical_date = _context_date_iso(context)
        return run_training_export_publish(
            get_engine(settings),
            settings=settings,
            manifest_id=manifest_id,
            dag_id=context["dag"].dag_id,
            airflow_run_id=context["run_id"],
            logical_date=logical_date,
        )

    schema_applied = apply_pricing_schema()
    raw_rows = load_raw()
    manifest_id = create_manifest()
    published = train_and_publish(manifest_id)

    schema_applied >> raw_rows >> manifest_id >> published


pricing_superglm_pipeline = _pricing_superglm_pipeline()
