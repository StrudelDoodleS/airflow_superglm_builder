from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild


def effective_from_for_run(value: date | datetime | str | None = None) -> str:
    if value is None:
        return date.today().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("effective_from must be a date, datetime, or ISO date string")

    cleaned = value.strip()
    try:
        return datetime.fromisoformat(cleaned).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(cleaned).isoformat()
        except ValueError as exc:
            raise ValueError("effective_from must be a date, datetime, or ISO date string") from exc


def required_payload_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"prepared payload field {field_name!r} is required")
    return str(value).strip()


def completed_model_build_payload(
    *,
    rating_workbook_path: str | Path,
    model_version: str,
    effective_from: str,
    export_id: str,
    created_by: str,
    manifest_id: str,
    split_set_id: str | None,
    mlflow_run_id: str | None = None,
    model_artifact_path: str | Path | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not str(manifest_id).strip():
        raise ValueError("manifest_id is required")

    return CompletedModelBuild(
        rating_workbook_path=str(rating_workbook_path),
        model_version=model_version,
        effective_from=effective_from,
        created_by=created_by,
        export_id=export_id,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        mlflow_run_id=mlflow_run_id,
        model_artifact_path=(str(model_artifact_path) if model_artifact_path is not None else None),
        metrics=metrics or {},
    ).to_dict()
