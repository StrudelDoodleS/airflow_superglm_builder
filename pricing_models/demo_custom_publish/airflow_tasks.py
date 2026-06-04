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
    effective_from_for_run,
    export_superglm_completed_build,
    next_trained_model_version,
)
from pricing_pipeline.orchestration.run_context import run_key_for_value
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
        context = get_current_context()
        run_value = context.get("run_id") or _context_logical_date(context)
        run_key = run_key_for_value(run_value)
        table_name = training_table_for_run(run_value)
        run_output_dir = Path(output_dir) / run_key
        frame = build_demo_training_frame(row_count=row_count, seed=seed)
        materialize_training_source(runtime.get_engine(), frame, table_name=table_name)
        return {
            "training_frame_path": write_training_frame(frame, run_output_dir),
            "training_table": table_name,
            "output_dir": str(run_output_dir),
            "run_key": run_key,
        }

    return _prepare_training_data


def _context_logical_date(context: dict[str, object]) -> object | None:
    value = context.get("logical_date")
    if value is not None:
        return value
    dag_run = context.get("dag_run")
    return (
        getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "run_after", None)
        or getattr(dag_run, "execution_date", None)
    )


def train_validate_export_task(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "train_validate_export",
):
    from airflow.sdk import get_current_context, task
    from pricing_pipeline.infra.runtime import runtime_from_env_or_module

    @task(task_id=task_id)
    def _train_validate_export(prepared_training: dict[str, str | None]) -> dict[str, object]:
        runtime = runtime_from_env_or_module(runtime_module)
        context = get_current_context()
        model_version = next_trained_model_version(runtime.get_engine())
        effective_from = effective_from_for_run(_context_logical_date(context))
        export_run_key = prepared_training.get("run_key") or run_key_for_value(
            context.get("run_id") or model_version
        )
        export_id = build_export_id(
            MODEL_KEY,
            str(export_run_key),
        )
        completed = export_superglm_completed_build(
            read_training_frame(str(prepared_training["training_frame_path"])),
            output_dir=prepared_training.get("output_dir") or output_dir,
            model_version=model_version,
            effective_from=effective_from,
            created_by=created_by,
            export_id=export_id,
        )
        completed["manifest_id"] = prepared_training.get("manifest_id")
        completed["split_set_id"] = prepared_training.get("split_set_id")
        return completed

    return _train_validate_export
