from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airflow.sdk import dag, get_current_context, task

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.data.manifest import create_dataset_manifest_with_split, new_manifest_id
from pricing_pipeline.infra.migrations import apply_migrations
from pricing_pipeline.infra.runtime import PipelineRuntime, runtime_from_env_or_module
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelSpec
from pricing_pipeline.orchestration.pipeline import publish_model_export, train_and_export_model


_REPO_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"
_DOCKER_SCHEMA_DIR = Path("/opt/pricing/db/migrations")


def schema_dir_from_env() -> Path:
    return Path(
        os.environ.get(
            "PRICING_SCHEMA_DIR",
            os.environ.get(
                "PRICING_MIGRATIONS_DIR",
                str(_DOCKER_SCHEMA_DIR if _DOCKER_SCHEMA_DIR.exists() else _REPO_SCHEMA_DIR),
            ),
        )
    )


SCHEMA_DIR = schema_dir_from_env()


def settings_from_env() -> Settings:
    return Settings.from_env(os.environ)


def runtime_from_dag_config(runtime_module: str | None = None) -> PipelineRuntime:
    return runtime_from_env_or_module(runtime_module, env=os.environ)


def context_date_iso(context: dict[str, Any]) -> str:
    dag_run = context.get("dag_run")
    run_datetime = (
        context.get("logical_date")
        or getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "run_after", None)
        or context.get("data_interval_start")
        or datetime.now(UTC)
    )
    return run_datetime.date().isoformat()


def build_pricing_model_dag(
    *,
    dag_id: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    schedule=None,
    tags: list[str] | None = None,
    runtime_module: str | None = None,
):
    @dag(
        dag_id=dag_id,
        start_date=datetime(2026, 1, 1),
        schedule=schedule,
        catchup=False,
        tags=tags or ["pricing", spec.model_key.lower(), "mlflow"],
    )
    def _pricing_model_dag():
        @task
        def apply_pricing_schema() -> list[str]:
            runtime = runtime_from_dag_config(runtime_module)
            settings = runtime.settings
            if not settings.skip_database_create:
                runtime.ensure_database(settings.pricing_database)
            return apply_migrations(runtime.get_engine(), schema_dir_from_env())

        @task
        def prepare_dataset() -> dict[str, str | None]:
            runtime = runtime_from_dag_config(runtime_module)
            settings = runtime.settings
            engine = runtime.get_engine()
            if spec.dataset.raw_loader is not None:
                spec.dataset.raw_loader(engine)
            manifest_id = new_manifest_id(spec.dataset.dataset_name)
            return create_dataset_manifest_with_split(
                engine,
                dataset=spec.dataset,
                manifest_id=manifest_id,
                validation_split=model_config.validation_split,
                validation_split_artifact_root=settings.validation_split_artifact_root,
            ).to_dict()

        @task
        def train_and_export(manifest_result: dict[str, str | None]) -> dict[str, Any]:
            runtime = runtime_from_dag_config(runtime_module)
            settings = runtime.settings
            context = get_current_context()
            logical_date = context_date_iso(context)
            return train_and_export_model(
                runtime.get_engine(),
                settings=settings,
                manifest_id=str(manifest_result["manifest_id"]),
                split_set_id=manifest_result.get("split_set_id"),
                dag_id=context["dag"].dag_id,
                airflow_run_id=context["run_id"],
                logical_date=logical_date,
                spec=spec,
            ).to_dict()

        @task
        def publish_export(export: dict[str, Any]) -> dict[str, str]:
            runtime = runtime_from_dag_config(runtime_module)
            return publish_model_export(
                runtime.get_engine(),
                export,
                model_config=model_config,
            )

        schema_applied = apply_pricing_schema()
        manifest_id = prepare_dataset()
        export = train_and_export(manifest_id)
        published = publish_export(export)

        schema_applied >> manifest_id >> export >> published

    return _pricing_model_dag()
