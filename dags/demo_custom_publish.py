from __future__ import annotations

from airflow.sdk import dag

from pricing_models.demo_custom_publish.airflow_tasks import (
    prepare_training_data_task,
    train_validate_export_task,
)
from pricing_models.demo_custom_publish.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.model_registry_tasks import register_pricing_model_task
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(
    dag_id="demo_custom_publish",
    schedule=None,
    catchup=False,
    tags=["pricing", "demo", "custom-tasks"],
)
def demo_custom_publish():
    # Reusable SQL lifecycle task. The TOML-backed MODEL_CONFIG supplies the
    # registry fields, so individual model DAGs do not need custom registry code.
    registered = register_pricing_model_task(
        model_config=MODEL_CONFIG,
        task_id="register_demo_model",
    )()

    # Model-owned ETL task. In a work model this is where you would call your
    # own source-query/transform/materialization code and return a small dict.
    prepared_training = prepare_training_data_task()()

    # Model-owned training/export task. It builds the final pandas model frame,
    # writes the frame-backed manifest, and returns CompletedModelBuild.to_dict().
    completed_build = train_validate_export_task()(prepared_training)

    # Reusable SQL lifecycle task. Deployment stays separate so people can
    # review the package candidate before moving a slot pointer.
    published = publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
    )(completed_build)

    registered >> prepared_training >> completed_build >> published


demo_custom_publish()
