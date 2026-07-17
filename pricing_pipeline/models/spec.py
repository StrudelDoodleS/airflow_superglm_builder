from __future__ import annotations

import math
from datetime import date, datetime
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)


class ApprovedModelBuildError(ValueError):
    """Raised when an approved notebook build is incomplete or invalid."""


class ValidationSplitResult(BaseModel):
    """Held-out metrics and authoritative row counts for one validation split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validation_split_no: int
    n_train: int
    n_validation: int
    metrics: Mapping[str, float]

    @field_validator("validation_split_no", "n_train", "n_validation", mode="before")
    @classmethod
    def _positive_integer(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def _finite_metrics(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("must contain at least one metric")
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

    @field_validator("metrics")
    @classmethod
    def _freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(dict(value))

    @field_serializer("metrics")
    def _serialise_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)


class ApprovedModelBuild(BaseModel):
    """Immutable notebook output passed unchanged into local or remote publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: int
    model_name: str
    model_version: str
    model_type: str
    target_name: str
    deployment_slot: str
    manifest_id: str
    split_set_id: str | None = None
    export_id: str
    rating_workbook_path: str
    rating_workbook_sha256: str
    effective_from: str | None = None
    created_by: str
    mlflow_run_id: str | None = None
    publication_receipt_path: str
    publication_receipt_sha256: str
    candidate_artifact_path: str
    candidate_artifact_sha256: str
    candidate_artifact_format: str
    candidate_artifact_size_bytes: int
    candidate_python_version: str
    candidate_superglm_version: str
    candidate_superglm_git_sha: str
    model_source_sha256: str
    model_frame_sha256: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_scopes: dict[str, str] = Field(default_factory=dict)
    fold_metrics: tuple[Mapping[str, int | str | float], ...] = ()
    validation_splits: tuple[ValidationSplitResult, ...] = ()

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            details = []
            for error in exc.errors():
                location = ".".join(str(item) for item in error.get("loc", ()))
                message = error.get("msg", "invalid value")
                details.append(f"{location}: {message}" if location else message)
            raise ApprovedModelBuildError(
                "invalid completed build payload: " + "; ".join(details)
            ) from exc

    @field_validator(
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "deployment_slot",
        "manifest_id",
        "export_id",
        "rating_workbook_path",
        "created_by",
        "publication_receipt_path",
        "candidate_artifact_path",
        "candidate_artifact_format",
        "candidate_python_version",
        "candidate_superglm_version",
        mode="before",
    )
    @classmethod
    def _required_text(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("is required")
        return str(value).strip()

    @field_validator("split_set_id", "mlflow_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value).strip() or None

    @field_validator("model_id", mode="before")
    @classmethod
    def _positive_model_id(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("must be a positive integer")
        model_id = int(value)
        if model_id <= 0:
            raise ValueError("must be a positive integer")
        return model_id

    @field_validator("effective_from", mode="before")
    @classmethod
    def _effective_from_date(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a date, datetime, or ISO date string")
        try:
            return datetime.fromisoformat(value.strip()).date().isoformat()
        except ValueError:
            try:
                return date.fromisoformat(value.strip()).isoformat()
            except ValueError as exc:
                raise ValueError("must be a date, datetime, or ISO date string") from exc

    @field_validator(
        "rating_workbook_sha256",
        "publication_receipt_sha256",
        "candidate_artifact_sha256",
        "model_source_sha256",
        "model_frame_sha256",
        mode="before",
    )
    @classmethod
    def _sha256(cls, value: Any) -> str:
        digest = str(value).strip()
        if (
            len(digest) != 64
            or digest.lower() != digest
            or not all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("must be a 64-character lowercase hex SHA-256 digest")
        return digest

    @field_validator("candidate_artifact_size_bytes", mode="before")
    @classmethod
    def _positive_artifact_size(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("must be a positive integer")
        size = int(value)
        if size <= 0:
            raise ValueError("must be a positive integer")
        return size

    @field_validator("candidate_superglm_git_sha", mode="before")
    @classmethod
    def _superglm_git_sha(cls, value: Any) -> str:
        digest = str(value).strip()
        if (
            len(digest) != 40
            or digest.lower() != digest
            or not all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("must be a 40-character lowercase hex git SHA")
        return digest

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
    def _metric_scopes(cls, value: Any) -> dict[str, str]:
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
    def _fold_metrics(cls, value: Any) -> tuple[Mapping[str, int | str | float], ...]:
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

    @field_validator("fold_metrics")
    @classmethod
    def _freeze_fold_metrics(
        cls,
        value: tuple[Mapping[str, int | str | float], ...],
    ) -> tuple[Mapping[str, int | str | float], ...]:
        return tuple(MappingProxyType(dict(record)) for record in value)

    @field_serializer("fold_metrics")
    def _serialise_fold_metrics(
        self,
        value: tuple[Mapping[str, int | str | float], ...],
    ) -> tuple[dict[str, int | str | float], ...]:
        return tuple(dict(record) for record in value)

    @model_validator(mode="after")
    def _scopes_reference_metrics(self) -> "ApprovedModelBuild":
        unknown = sorted(set(self.metric_scopes) - set(self.metrics))
        if unknown:
            raise ValueError("metric_scopes reference unknown metrics: " + ", ".join(unknown))
        return self

    @model_validator(mode="after")
    def _validation_evidence_is_consistent(self) -> "ApprovedModelBuild":
        split_numbers = tuple(split.validation_split_no for split in self.validation_splits)
        if split_numbers != tuple(range(1, len(self.validation_splits) + 1)):
            raise ValueError("validation_splits must be numbered consecutively from 1")
        if self.validation_splits:
            metric_names = tuple(self.validation_splits[0].metrics)
            if any(tuple(split.metrics) != metric_names for split in self.validation_splits[1:]):
                raise ValueError(
                    "validation_splits must contain the same metrics in requested order"
                )
            expected_fold_metrics = {
                (split.validation_split_no, metric_name): metric_value
                for split in self.validation_splits
                for metric_name, metric_value in split.metrics.items()
            }
            actual_fold_metrics: dict[tuple[int, str], float] = {}
            duplicate_key = False
            for metric in self.fold_metrics:
                key = (int(metric["fold_no"]), str(metric["metric_name"]))
                if key in actual_fold_metrics:
                    duplicate_key = True
                actual_fold_metrics[key] = float(metric["metric_value"])
            if duplicate_key or actual_fold_metrics != expected_fold_metrics:
                raise ValueError("fold_metrics must exactly match validation_splits")
        return self


# Historical public names remain aliases; there is one record and one validator.
CompletedModelBuildError = ApprovedModelBuildError
CompletedModelBuild = ApprovedModelBuild
ModelExportResult = ApprovedModelBuild
