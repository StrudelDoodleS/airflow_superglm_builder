from __future__ import annotations

from airflow.sdk import dag

from pricing_models.demo_custom_publish.airflow_tasks import (
    prepare_training_data_task,
    train_validate_export_task,
)
from pricing_models.demo_custom_publish.data import dataset_spec_from_prepared_training
from pricing_models.demo_custom_publish.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.manifest_tasks import (
    create_prepared_dataset_manifest_task,
)
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

    # Reusable SQL lifecycle task. The only model-specific bit is the
    # dataset_builder, which converts the upstream prepared payload into the
    # DatasetSpec used for manifest/split metadata.
    manifested_training = create_prepared_dataset_manifest_task(
        model_config=MODEL_CONFIG,
        dataset_builder=dataset_spec_from_prepared_training,
        task_id="create_training_manifest",
    )(prepared_training)

    # Model-owned training/export task. It receives the manifest IDs from the
    # prior task and returns CompletedModelBuild.to_dict() for the publish task.
    completed_build = train_validate_export_task()(manifested_training)

    # Reusable SQL lifecycle task. Because completed_build already includes
    # manifest_id/split_set_id, this task does not need a DatasetSpec here.
    published = publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
    )(completed_build)

    registered >> prepared_training >> manifested_training >> completed_build >> published


demo_custom_publish()
