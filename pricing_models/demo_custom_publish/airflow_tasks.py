from __future__ import annotations

from pathlib import Path

from pricing_models.demo_custom_publish.data import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    build_demo_training_frame,
    materialize_training_source,
    read_training_frame,
    training_table_for_run,
    write_training_frame,
)
from pricing_models.demo_custom_publish.modeling import (
    MODEL_KEY,
    build_final_model_frame,
    create_demo_model_frame_manifest,
    export_superglm_completed_build,
)
from pricing_models.demo_custom_publish.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.airflow_run_metadata import (
    merge_prepared_payload_metadata,
    task_run_metadata,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export
from pricing_pipeline.publishing.rating_export import build_export_id


def prepare_training_data_task(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
    runtime_module: str | None = None,
    task_id: str = "prepare_training_data",
):
    from airflow.sdk import get_current_context, task
    from pricing_pipeline.infra.runtime import runtime_from_env_or_module

    @task(task_id=task_id)
    def _prepare_training_data() -> dict[str, str]:
        runtime = runtime_from_env_or_module(runtime_module)
        metadata = task_run_metadata(get_current_context(), output_root=output_dir)
        run_key = metadata["run_key"]
        table_name = training_table_for_run(run_key)
        run_output_dir = Path(metadata["output_dir"])
        frame = build_demo_training_frame(row_count=row_count, seed=seed)
        materialize_training_source(runtime.get_engine(), frame, table_name=table_name)
        payload = {
            "training_frame_path": write_training_frame(frame, run_output_dir),
            "training_table": table_name,
        }
        return merge_prepared_payload_metadata(metadata, payload)

    return _prepare_training_data


def train_validate_export_task(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "train_validate_export",
):
    from airflow.sdk import task
    from pricing_pipeline.infra.runtime import runtime_from_env_or_module

    @task(task_id=task_id)
    def _train_validate_export(prepared_training: dict[str, str | None]) -> dict[str, object]:
        runtime = runtime_from_env_or_module(runtime_module)
        effective_from = str(prepared_training["effective_from"])
        data_as_of_date = str(prepared_training.get("data_as_of_date") or effective_from)
        export_run_key = prepared_training.get("run_key") or "manual"
        export_id = build_export_id(
            MODEL_KEY,
            str(export_run_key),
        )
        model_version = resolve_model_version_for_export(
            runtime.get_engine(),
            model_key=MODEL_KEY,
            export_id=export_id,
        )
        frame = read_training_frame(str(prepared_training["training_frame_path"]))
        final_frame = build_final_model_frame(frame)
        manifest = create_demo_model_frame_manifest(
            runtime.get_engine(),
            frame=final_frame,
            data_as_of_date=data_as_of_date,
            validation_split=MODEL_CONFIG.validation_split,
            validation_split_artifact_root=runtime.settings.validation_split_artifact_root,
            created_by=created_by,
        )
        completed = export_superglm_completed_build(
            final_frame,
            output_dir=prepared_training.get("output_dir") or output_dir,
            model_version=model_version,
            effective_from=effective_from,
            created_by=created_by,
            export_id=export_id,
            manifest_id=manifest.manifest_id,
            split_set_id=manifest.split_set_id,
        )
        return completed

    return _train_validate_export
