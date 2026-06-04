from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pricing_pipeline.data.manifest import (
    DatasetManifestResult,
    create_dataset_manifest_with_split,
    new_manifest_id,
)
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec


def create_model_build_manifest(
    engine,
    *,
    dataset: DatasetSpec,
    model_config: ModelBuildConfig,
    validation_split_artifact_root: Path | None,
    created_by: str,
) -> DatasetManifestResult:
    return create_dataset_manifest_with_split(
        engine,
        dataset=dataset,
        manifest_id=new_manifest_id(dataset.dataset_name),
        validation_split=model_config.validation_split,
        validation_split_artifact_root=validation_split_artifact_root,
        created_by=created_by,
    )


def create_prepared_dataset_manifest_task(
    *,
    model_config: ModelBuildConfig,
    dataset_builder: Callable[[Mapping[str, Any]], DatasetSpec],
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "create_dataset_manifest",
):
    from airflow.sdk import task

    @task(task_id=task_id)
    def _create_dataset_manifest(prepared: Mapping[str, Any]) -> dict[str, Any]:
        runtime = runtime_from_env_or_module(runtime_module)
        payload = dict(prepared)
        dataset = dataset_builder(payload)
        manifest = create_model_build_manifest(
            runtime.get_engine(),
            dataset=dataset,
            model_config=model_config,
            validation_split_artifact_root=runtime.settings.validation_split_artifact_root,
            created_by=created_by,
        )
        payload["manifest_id"] = manifest.manifest_id
        payload["split_set_id"] = manifest.split_set_id
        return payload

    return _create_dataset_manifest
