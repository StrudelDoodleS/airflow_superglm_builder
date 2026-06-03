from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import text
from airflow.sdk import get_current_context, task

from pricing_pipeline.data.manifest import (
    create_dataset_manifest_with_split,
    new_manifest_id,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.orchestration.pipeline import publish_model_export
from pricing_pipeline.publishing.publisher import validate_model_on_engine
from pricing_pipeline.publishing.rating_export import build_export_id


_DEFAULT_PYTHON_DAG_ID = "python_publish_completed_model_build"


class CompletedModelBuildError(ValueError):
    """Raised when a completed-build payload cannot be published."""


@dataclass(frozen=True)
class CompletedModelBuild:
    rating_workbook_path: str
    model_version: str
    effective_from: str
    created_by: str | None = None
    export_id: str | None = None
    dag_id: str | None = None
    airflow_run_id: str | None = None
    mlflow_run_id: str | None = None
    manifest_id: str | None = None
    split_set_id: str | None = None
    model_artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: "CompletedModelBuild | Mapping[str, Any]",
    ) -> "CompletedModelBuild":
        if isinstance(value, cls):
            return value
        data = dict(value)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise CompletedModelBuildError(
                "unknown completed build field(s): " + ", ".join(unknown)
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise CompletedModelBuildError(
                f"invalid completed build payload: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletedModelPublishResult:
    model_id: int
    model_key: str
    model_version: str
    manifest_id: str
    split_set_id: str | None
    export_id: str
    rate_package_id: int
    package_version: int
    package_status: str
    rating_workbook_path: str
    mlflow_run_id: str | None = None
    was_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_text(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise CompletedModelBuildError(f"{field_name} is required")
    return str(value).strip()


def _existing_workbook(path_value: str | None) -> str:
    path = Path(_required_text(path_value, "rating_workbook_path"))
    if not path.exists():
        raise CompletedModelBuildError(
            f"rating_workbook_path does not exist: {path.as_posix()}"
        )
    return str(path)


def _resolve_export_id(model_key: str, export_id: str | None, airflow_run_id: str | None) -> str:
    if export_id is not None and str(export_id).strip():
        return str(export_id).strip()
    if airflow_run_id is not None and str(airflow_run_id).strip():
        return build_export_id(model_key, str(airflow_run_id).strip())
    raise CompletedModelBuildError("export_id or airflow_run_id is required")


def _fill_if_blank(payload: dict[str, Any], field_name: str, value: str | None) -> None:
    if not payload.get(field_name) and value is not None and str(value).strip():
        payload[field_name] = str(value).strip()


def validate_existing_manifest(engine, manifest_id: str) -> None:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        exists = con.execute(
            text(
                f"""
                SELECT 1
                FROM {schemas.pricing}.DATASET_MANIFEST
                WHERE manifest_id = :manifest_id
                """
            ),
            {"manifest_id": manifest_id},
        ).scalar_one_or_none()
    if exists is None:
        raise CompletedModelBuildError(f"manifest_id {manifest_id!r} was not found")


def publish_completed_model_build(
    engine,
    *,
    settings: Settings,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec | None,
    completed_build: CompletedModelBuild | Mapping[str, Any],
    package_status: str | None = None,
    created_by: str | None = None,
) -> CompletedModelPublishResult:
    build = CompletedModelBuild.from_mapping(completed_build)
    rating_workbook_path = _existing_workbook(build.rating_workbook_path)
    model_version = _required_text(build.model_version, "model_version")
    effective_from = _required_text(build.effective_from, "effective_from")
    resolved_created_by = _required_text(
        build.created_by or created_by,
        "created_by",
    )
    dag_id = _required_text(build.dag_id or _DEFAULT_PYTHON_DAG_ID, "dag_id")
    airflow_run_id = (
        str(build.airflow_run_id).strip()
        if build.airflow_run_id is not None and str(build.airflow_run_id).strip()
        else None
    )
    export_id = _resolve_export_id(model_config.model_key, build.export_id, airflow_run_id)
    if airflow_run_id is None:
        airflow_run_id = export_id
    resolved_package_status = _required_text(
        package_status or model_config.default_package_status,
        "package_status",
    )

    if build.manifest_id is None and dataset is None:
        raise CompletedModelBuildError(
            "dataset is required when manifest_id is not supplied"
        )

    model_id = validate_model_on_engine(engine, model_config)
    if build.manifest_id is None:
        manifest_result = create_dataset_manifest_with_split(
            engine,
            dataset=dataset,
            manifest_id=new_manifest_id(dataset.dataset_name),
            validation_split=model_config.validation_split,
            validation_split_artifact_root=settings.validation_split_artifact_root,
            created_by=resolved_created_by,
        )
        manifest_id = manifest_result.manifest_id
        split_set_id = manifest_result.split_set_id
    else:
        manifest_id = _required_text(build.manifest_id, "manifest_id")
        validate_existing_manifest(engine, manifest_id)
        split_set_id = build.split_set_id

    export = ModelExportResult(
        model_id=model_id,
        model_key=model_config.model_key,
        model_version=model_version,
        model_type=model_config.model_type,
        target_name=model_config.target_name,
        deployment_slot=model_config.deployment_slot,
        manifest_id=manifest_id,
        dag_id=dag_id,
        airflow_run_id=airflow_run_id,
        mlflow_run_id=build.mlflow_run_id or "",
        split_set_id=split_set_id,
        export_id=export_id,
        rating_workbook_path=rating_workbook_path,
        effective_from=effective_from,
        created_by=resolved_created_by,
        package_status=resolved_package_status,
    )
    publish_config = replace(
        model_config,
        default_package_status=resolved_package_status,
    )
    publish_result = publish_model_export(
        engine,
        export,
        model_config=publish_config,
    )

    return CompletedModelPublishResult(
        model_id=int(model_id),
        model_key=model_config.model_key,
        model_version=model_version,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        export_id=str(publish_result["export_id"]),
        rate_package_id=int(publish_result["rate_package_id"]),
        package_version=int(publish_result["package_version"]),
        package_status=str(publish_result.get("package_status") or resolved_package_status),
        rating_workbook_path=str(
            publish_result.get("rating_workbook_path") or rating_workbook_path
        ),
        mlflow_run_id=str(publish_result.get("mlflow_run_id") or build.mlflow_run_id or "")
        or None,
        was_existing=bool(publish_result.get("was_existing", False)),
    )


def publish_completed_model_build_task(
    *,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "publish_completed_model_build",
):
    @task(task_id=task_id)
    def _publish(completed_build: Mapping[str, Any]) -> dict[str, Any]:
        context = get_current_context()
        dag = context.get("dag")
        payload = CompletedModelBuild.from_mapping(completed_build).to_dict()
        dag_id = getattr(dag, "dag_id", None)
        run_id = context.get("run_id")
        _fill_if_blank(payload, "dag_id", dag_id)
        _fill_if_blank(payload, "airflow_run_id", run_id)
        _fill_if_blank(payload, "created_by", created_by)
        if not payload.get("export_id") and payload.get("airflow_run_id"):
            payload["export_id"] = build_export_id(
                model_config.model_key,
                str(payload["airflow_run_id"]),
            )

        runtime = runtime_from_env_or_module(runtime_module, env=os.environ)
        result = publish_completed_model_build(
            runtime.get_engine(),
            settings=runtime.settings,
            model_config=model_config,
            dataset=dataset,
            completed_build=payload,
            created_by=created_by,
        )
        return result.to_dict()

    return _publish
