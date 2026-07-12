from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy import text

from pricing_pipeline.data.manifest import (
    create_dataset_manifest_with_split,
    new_manifest_id,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.db import configure_engine
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.orchestration.pipeline import publish_model_export
from pricing_pipeline.publishing.publisher import validate_model_on_engine
from pricing_pipeline.publishing.rating_export import build_export_id
from pricing_pipeline.workbench.artifacts import load_candidate_bundle


_DEFAULT_PYTHON_DAG_ID = "python_publish_completed_model_build"


class CompletedModelBuildError(ValueError):
    """Raised when a completed-build payload cannot be published."""


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", ()))
        msg = error.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "invalid completed build payload: " + "; ".join(parts)


def _normalise_effective_from(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError("must be a date, datetime, or ISO date string")
    if not value.strip():
        raise ValueError("is required")

    cleaned = value.strip()
    try:
        return datetime.fromisoformat(cleaned).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(cleaned).isoformat()
        except ValueError as exc:
            raise ValueError("must be a date, datetime, or ISO date string") from exc


class CompletedModelBuild(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

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
    candidate_artifact_path: str | None = None
    candidate_artifact_sha256: str | None = None
    candidate_artifact_format: str | None = None
    candidate_artifact_size_bytes: int | None = None
    candidate_python_version: str | None = None
    candidate_superglm_version: str | None = None
    model_source_sha256: str | None = None
    publication_receipt_path: str | None = None
    publication_receipt_sha256: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_scopes: dict[str, str] = Field(default_factory=dict)
    fold_metrics: tuple[dict[str, int | str | float], ...] = ()

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise CompletedModelBuildError(_format_validation_error(exc)) from exc

    @classmethod
    def from_mapping(
        cls,
        value: "CompletedModelBuild | Mapping[str, Any]",
    ) -> "CompletedModelBuild":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise CompletedModelBuildError("invalid completed build payload: expected a mapping")
        data = dict(value)
        allowed = set(cls.model_fields)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise CompletedModelBuildError(
                "unknown completed build field(s): " + ", ".join(unknown)
            )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if self.candidate_artifact_path is None:
            for field_name in (
                "candidate_artifact_path",
                "candidate_artifact_sha256",
                "candidate_artifact_format",
                "candidate_artifact_size_bytes",
                "candidate_python_version",
                "candidate_superglm_version",
                "model_source_sha256",
            ):
                payload.pop(field_name)
        if self.publication_receipt_path is None:
            payload.pop("publication_receipt_path")
        if self.publication_receipt_sha256 is None:
            payload.pop("publication_receipt_sha256")
        if not self.metric_scopes:
            payload.pop("metric_scopes")
        if not self.fold_metrics:
            payload.pop("fold_metrics")
        return payload

    @field_validator("rating_workbook_path", "model_version", mode="before")
    @classmethod
    def _required_non_empty_text(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("is required")
        return str(value).strip()

    @field_validator("effective_from", mode="before")
    @classmethod
    def _effective_from_date_text(cls, value: Any) -> str:
        return _normalise_effective_from(value)

    @field_validator(
        "created_by",
        "export_id",
        "dag_id",
        "airflow_run_id",
        "mlflow_run_id",
        "manifest_id",
        "split_set_id",
        "model_artifact_path",
        "candidate_artifact_path",
        "candidate_artifact_format",
        "candidate_python_version",
        "candidate_superglm_version",
        "publication_receipt_path",
        mode="before",
    )
    @classmethod
    def _optional_non_empty_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("publication_receipt_sha256", mode="before")
    @classmethod
    def _optional_sha256(cls, value: Any) -> str | None:
        if value is None:
            return None
        digest = str(value).strip()
        if (
            len(digest) != 64
            or digest.lower() != digest
            or not all(char in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                "publication_receipt_sha256 must be a 64-character lowercase hex SHA-256 digest"
            )
        return digest

    @field_validator("candidate_artifact_sha256", "model_source_sha256", mode="before")
    @classmethod
    def _optional_candidate_sha256(cls, value: Any) -> str | None:
        if value is None:
            return None
        digest = str(value).strip()
        if (
            len(digest) != 64
            or digest.lower() != digest
            or not all(char in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("must be a 64-character lowercase hex SHA-256 digest")
        return digest

    @field_validator("candidate_artifact_size_bytes", mode="before")
    @classmethod
    def _optional_positive_size(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("must be a positive integer")
        size = int(value)
        if size <= 0:
            raise ValueError("must be a positive integer")
        return size

    @model_validator(mode="after")
    def _receipt_path_and_hash_are_paired(self) -> "CompletedModelBuild":
        if (self.publication_receipt_path is None) != (self.publication_receipt_sha256 is None):
            raise ValueError(
                "publication_receipt_path and publication_receipt_sha256 must be supplied together"
            )
        return self

    @model_validator(mode="after")
    def _candidate_artifact_fields_are_complete(self) -> "CompletedModelBuild":
        field_names = (
            "candidate_artifact_path",
            "candidate_artifact_sha256",
            "candidate_artifact_format",
            "candidate_artifact_size_bytes",
            "candidate_python_version",
            "candidate_superglm_version",
            "model_source_sha256",
        )
        present = [name for name in field_names if getattr(self, name) is not None]
        if present and len(present) != len(field_names):
            missing = [name for name in field_names if name not in present]
            raise ValueError(
                "candidate artifact fields must be supplied together; missing: "
                + ", ".join(missing)
            )
        return self

    @field_validator("metrics", mode="before")
    @classmethod
    def _finite_numeric_metrics(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("metrics must be a mapping of metric name to finite number")

        metrics: dict[str, float] = {}
        for key, raw_metric in value.items():
            metric_name = str(key).strip()
            if not metric_name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(raw_metric, bool) or not isinstance(raw_metric, Real):
                raise ValueError(f"metric {metric_name!r} must be a finite number")

            metric_value = float(raw_metric)
            if not math.isfinite(metric_value):
                raise ValueError(f"metric {metric_name!r} must be finite")
            metrics[metric_name] = metric_value

        return metrics

    @field_validator("metric_scopes", mode="before")
    @classmethod
    def _metric_scope_values(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("metric_scopes must be a mapping")
        scopes: dict[str, str] = {}
        for key, raw_scope in value.items():
            metric_name = str(key).strip()
            scope = str(raw_scope).strip()
            if not metric_name or not scope:
                raise ValueError("metric scope names and values must be non-empty")
            scopes[metric_name] = scope
        return scopes

    @field_validator("fold_metrics", mode="before")
    @classmethod
    def _fold_metric_values(cls, value: Any) -> tuple[dict[str, int | str | float], ...]:
        if value is None:
            return ()
        records: list[dict[str, int | str | float]] = []
        for raw in value:
            if not isinstance(raw, Mapping):
                raise ValueError("fold_metrics entries must be mappings")
            try:
                fold_no = int(raw["fold_no"])
                metric_name = str(raw["metric_name"]).strip()
                metric_value = float(raw["metric_value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "fold_metrics entries require fold_no, metric_name, and metric_value"
                ) from exc
            if fold_no <= 0 or not metric_name or not math.isfinite(metric_value):
                raise ValueError("fold_metrics entries must contain valid finite values")
            records.append(
                {
                    "fold_no": fold_no,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )
        return tuple(records)

    @model_validator(mode="after")
    def _metric_scopes_reference_metrics(self) -> "CompletedModelBuild":
        unknown = sorted(set(self.metric_scopes) - set(self.metrics))
        if unknown:
            raise ValueError("metric_scopes reference unknown metrics: " + ", ".join(unknown))
        return self


@dataclass(frozen=True)
class CompletedModelPublishResult:
    model_id: int
    model_name: str
    model_version: str
    manifest_id: str
    split_set_id: str | None
    export_id: str
    rate_package_id: int
    package_version: int
    package_status: str
    rating_workbook_path: str
    mlflow_run_id: str | None = None
    publication_receipt_path: str | None = None
    publication_receipt_sha256: str | None = None
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
        raise CompletedModelBuildError(f"rating_workbook_path does not exist: {path.as_posix()}")
    return str(path)


def _resolve_export_id(model_name: str, export_id: str | None, airflow_run_id: str | None) -> str:
    if export_id is not None and str(export_id).strip():
        return str(export_id).strip()
    if airflow_run_id is not None and str(airflow_run_id).strip():
        return build_export_id(model_name, str(airflow_run_id).strip())
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


def _verify_candidate_artifact(
    build: CompletedModelBuild,
    *,
    manifest_id: str,
    split_set_id: str | None,
    allowed_root: str | Path,
) -> None:
    if build.candidate_artifact_path is None:
        return

    try:
        bundle = load_candidate_bundle(
            build.candidate_artifact_path,
            expected_sha256=build.candidate_artifact_sha256,
            expected_size_bytes=build.candidate_artifact_size_bytes,
            expected_format=build.candidate_artifact_format,
            expected_python_version=build.candidate_python_version,
            expected_superglm_version=build.candidate_superglm_version,
            allowed_root=allowed_root,
        )
    except Exception as exc:
        raise CompletedModelBuildError(
            f"candidate artifact verification failed: {exc}"
        ) from exc

    expected_lineage = {
        "manifest_id": manifest_id,
        "split_set_id": split_set_id,
        "model_source_sha256": build.model_source_sha256,
    }
    for field_name, expected_value in expected_lineage.items():
        actual_value = getattr(bundle, field_name)
        if actual_value != expected_value:
            raise CompletedModelBuildError(
                f"candidate artifact {field_name} does not match completed-build "
                f"lineage: expected={expected_value!r}, actual={actual_value!r}"
            )


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
    engine = configure_engine(engine, settings.schema_names)
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
    export_id = _resolve_export_id(model_config.model_name, build.export_id, airflow_run_id)
    if airflow_run_id is None:
        airflow_run_id = export_id
    resolved_package_status = _required_text(
        package_status or model_config.default_package_status,
        "package_status",
    )

    if build.manifest_id is None and dataset is None:
        raise CompletedModelBuildError("dataset is required when manifest_id is not supplied")

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

    _verify_candidate_artifact(
        build,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        allowed_root=settings.workbench_artifact_root,
    )

    export = ModelExportResult(
        model_id=model_id,
        model_name=model_config.model_name,
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
        publication_receipt_path=build.publication_receipt_path,
        publication_receipt_sha256=build.publication_receipt_sha256,
        candidate_artifact_path=build.candidate_artifact_path,
        candidate_artifact_sha256=build.candidate_artifact_sha256,
        candidate_artifact_format=build.candidate_artifact_format,
        candidate_artifact_size_bytes=build.candidate_artifact_size_bytes,
        candidate_python_version=build.candidate_python_version,
        candidate_superglm_version=build.candidate_superglm_version,
        model_source_sha256=build.model_source_sha256,
        metrics=dict(build.metrics),
        metric_scopes=dict(build.metric_scopes),
        fold_metrics=tuple(dict(item) for item in build.fold_metrics),
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
        model_name=model_config.model_name,
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
        mlflow_run_id=str(publish_result.get("mlflow_run_id") or build.mlflow_run_id or "") or None,
        publication_receipt_path=str(
            publish_result.get("publication_receipt_path") or build.publication_receipt_path or ""
        )
        or None,
        publication_receipt_sha256=str(
            publish_result.get("publication_receipt_sha256")
            or build.publication_receipt_sha256
            or ""
        )
        or None,
        was_existing=bool(publish_result.get("was_existing", False)),
    )


def publish_completed_model_build_task(
    *,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec | None = None,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "publish_completed_model_build",
):
    from airflow.sdk import get_current_context, task

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
                model_config.model_name,
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
