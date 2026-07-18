from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublishResult:
    mlflow_run_id: str
    export_id: str
    rate_package_id: int
    package_version: int
    rating_workbook_path: str
    package_status: str = "PUBLISHED"
    was_existing: bool = False
    model_run_id: int | None = None


@dataclass(frozen=True)
class CompletedModelPublishResult:
    """Durable package and model-run identity returned to notebook callers."""

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
    model_run_id: int | None = None
    mlflow_run_id: str | None = None
    publication_receipt_path: str | None = None
    publication_receipt_sha256: str | None = None
    candidate_artifact_path: str | None = None
    was_existing: bool = False


@dataclass(frozen=True)
class DeploymentResult:
    model_id: int
    deployment_slot: str
    previous_rate_package_id: int | None
    rate_package_id: int
    package_version: int
    deployed_by: str
    deployment_reason: str
