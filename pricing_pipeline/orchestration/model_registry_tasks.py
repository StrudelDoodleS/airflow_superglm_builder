from __future__ import annotations

from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.model_registry import ensure_pricing_model


def register_pricing_model_task(
    *,
    model_config: ModelBuildConfig,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "register_pricing_model",
):
    from airflow.sdk import task

    @task(task_id=task_id)
    def _register_pricing_model() -> int:
        runtime = runtime_from_env_or_module(runtime_module)
        return ensure_pricing_model(
            runtime.get_engine(),
            model_name=model_config.model_name,
            model_label=model_config.model_label,
            target_name=model_config.target_name,
            model_type=model_config.model_type,
            created_by=created_by,
        )

    return _register_pricing_model
