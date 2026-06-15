"""Airflow TaskFlow wrappers for the MTPL frequency custom build."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pricing_models.mtpl_frequency.data import DEFAULT_OUTPUT_ROOT, prepare_source_data
from pricing_models.mtpl_frequency.modeling import train_validate_export_model
from pricing_pipeline.orchestration.airflow_run_metadata import (
    merge_prepared_payload_metadata,
    task_run_metadata,
)


def prepare_source_data_task(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    runtime_module: str | None = None,
    task_id: str = "prepare_source_data",
):
    from airflow.sdk import get_current_context, task
    from pricing_pipeline.infra.runtime import runtime_from_env_or_module

    @task(task_id=task_id)
    def _prepare_source_data() -> dict[str, Any]:
        runtime = runtime_from_env_or_module(runtime_module)
        metadata = task_run_metadata(get_current_context(), output_root=output_root)
        payload = prepare_source_data(
            runtime.get_engine(),
            run_key=metadata["run_key"],
            output_dir=Path(metadata["output_dir"]),
        )
        return merge_prepared_payload_metadata(metadata, payload)

    return _prepare_source_data


def train_validate_export_task(
    *,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "train_validate_export",
):
    from airflow.sdk import task
    from pricing_pipeline.infra.runtime import runtime_from_env_or_module

    @task(task_id=task_id)
    def _train_validate_export(prepared: Mapping[str, Any]) -> dict[str, Any]:
        runtime = runtime_from_env_or_module(runtime_module)
        return train_validate_export_model(
            dict(prepared),
            engine=runtime.get_engine(),
            settings=runtime.settings,
            created_by=created_by,
        )

    return _train_validate_export
