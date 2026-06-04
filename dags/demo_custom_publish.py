from __future__ import annotations

from airflow.sdk import dag

from pricing_models.demo_custom_publish.spec import DATASET_SPEC, MODEL_CONFIG
from pricing_models.demo_custom_publish.tasks import (
    prepare_training_data_task,
    register_demo_model_task,
    train_validate_export_task,
)
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
    registered = register_demo_model_task(model_config=MODEL_CONFIG)()
    prepared_training = prepare_training_data_task(model_config=MODEL_CONFIG)()
    completed_build = train_validate_export_task()(prepared_training)
    published = publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
        dataset=DATASET_SPEC,
    )(completed_build)
    registered >> prepared_training >> completed_build >> published


demo_custom_publish()
