from __future__ import annotations

from airflow.sdk import dag

from pricing_models.demo_custom_publish.spec import DATASET_SPEC, MODEL_CONFIG
from pricing_models.demo_custom_publish.tasks import (
    prepare_training_data_task,
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
    training_frame_path = prepare_training_data_task()()
    completed_build = train_validate_export_task()(training_frame_path)
    publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
        dataset=DATASET_SPEC,
    )(completed_build)


demo_custom_publish()
