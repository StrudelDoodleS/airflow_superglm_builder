from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
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
    model_key: str
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
    model_key: str
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
    effective_from: str
    created_by: str
    package_status: str = "DRAFT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: "ModelExportResult | Mapping[str, Any]") -> "ModelExportResult":
        if isinstance(value, cls):
            return value
        data = dict(value)
        data["model_id"] = int(data["model_id"])
        return cls(**data)


def coerce_training_frame(value: TrainingFrame | tuple[Any, ...]) -> TrainingFrame:
    if isinstance(value, TrainingFrame):
        return value
    X, y, exposure, offset = value
    return TrainingFrame(X=X, y=y, exposure=exposure, offset=offset)
