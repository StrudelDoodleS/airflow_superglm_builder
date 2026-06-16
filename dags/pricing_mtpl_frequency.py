"""Explicit custom TaskFlow DAG for the freMTPL frequency model."""

from __future__ import annotations

from airflow.sdk import dag

from pricing_models.mtpl_frequency.airflow_tasks import (
    prepare_source_data_task,
    train_validate_export_task,
)
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.model_registry_tasks import register_pricing_model_task
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(
    dag_id="pricing_mtpl_frequency",
    schedule=None,
    catchup=False,
    tags=["pricing", "mtpl", "frequency", "custom-tasks"],
)
def pricing_mtpl_frequency():
    registered = register_pricing_model_task(
        model_config=MODEL_CONFIG,
        task_id="register_mtpl_frequency",
    )()
    prepared = prepare_source_data_task()()
    completed = train_validate_export_task()(prepared)
    published = publish_completed_model_build_task(model_config=MODEL_CONFIG)(completed)

    registered >> prepared >> completed >> published


pricing_mtpl_frequency()
