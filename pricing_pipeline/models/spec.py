from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingFrame:
    X: pd.DataFrame
    y: np.ndarray
    exposure: np.ndarray
    offset: np.ndarray


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    source_system: str
    manifest_sql: str
    pk_columns: tuple[str, ...]
    target_column: str
    weight_column: str | None = None
    raw_loader: Callable[..., int] | None = None
    default_n_splits: int = 5
    default_random_state: int = 42


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    target_name: str
    model_type: str
    experiment_name: str
    deployment_slot: str
    dataset: DatasetSpec
    training_sql: str
    feature_columns: tuple[str, ...]
    build_model: Callable[[], Any]
    build_training_frame: Callable[[pd.DataFrame], TrainingFrame | tuple[Any, ...]]
    model_label: str | None = None
    offset_label: str = "log(Exposure)"
    package_status: str = "DRAFT"


@dataclass(frozen=True)
class ModelExportResult:
    model_id: int
    model_name: str
    model_version: str
    model_type: str
    target_name: str
    deployment_slot: str
    manifest_id: str
    dag_id: str
    airflow_run_id: str
    mlflow_run_id: str
    split_set_id: str | None
    export_id: str
    rating_workbook_path: str
    effective_from: str | None
    created_by: str
    package_status: str = "DRAFT"
    publication_receipt_path: str | None = None
    publication_receipt_sha256: str | None = None
    candidate_artifact_path: str | None = None
    candidate_artifact_sha256: str | None = None
    candidate_artifact_format: str | None = None
    candidate_artifact_size_bytes: int | None = None
    candidate_python_version: str | None = None
    candidate_superglm_version: str | None = None
    model_source_sha256: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metric_scopes: dict[str, str] = field(default_factory=dict)
    fold_metrics: tuple[dict[str, int | str | float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.publication_receipt_path is None:
            payload.pop("publication_receipt_path")
        if self.publication_receipt_sha256 is None:
            payload.pop("publication_receipt_sha256")
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
        if not self.metrics:
            payload.pop("metrics")
        if not self.metric_scopes:
            payload.pop("metric_scopes")
        if not self.fold_metrics:
            payload.pop("fold_metrics")
        return payload

    @classmethod
    def from_mapping(cls, value: "ModelExportResult | Mapping[str, Any]") -> "ModelExportResult":
        if isinstance(value, cls):
            return value
        data = dict(value)
        data["model_id"] = int(data["model_id"])
        if "fold_metrics" in data:
            data["fold_metrics"] = tuple(dict(item) for item in data["fold_metrics"])
        return cls(**data)


def coerce_training_frame(value: TrainingFrame | tuple[Any, ...]) -> TrainingFrame:
    if isinstance(value, TrainingFrame):
        return value
    X, y, exposure, offset = value
    return TrainingFrame(X=X, y=y, exposure=exposure, offset=offset)
