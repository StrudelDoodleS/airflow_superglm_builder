from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from superglm import cross_validate


class StandardSuperGLMError(ValueError):
    """Raised when the shared SuperGLM build contract is violated."""


@dataclass(frozen=True)
class ModelInputs:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: pd.Series | np.ndarray | None = None
    sample_weight_name: str | None = None
    offset: pd.Series | np.ndarray | None = None
    export_weight: pd.Series | np.ndarray | None = None
    export_weight_name: str | None = None


@dataclass(frozen=True)
class FoldMetric:
    fold_no: int
    metric_name: str
    metric_value: float


@dataclass(frozen=True)
class CVEvidence:
    fold_indices: tuple[tuple[np.ndarray, np.ndarray], ...]
    report: dict[str, Any]
    metrics: dict[str, float]
    fold_metrics: tuple[FoldMetric, ...]


class PrecomputedSplitter:
    def __init__(
        self,
        folds: Iterable[tuple[Any, Any]],
        *,
        row_count: int,
    ) -> None:
        if row_count <= 0:
            raise StandardSuperGLMError("row_count must be positive")

        validated: list[tuple[np.ndarray, np.ndarray]] = []
        seen_test_rows: set[int] = set()
        for fold_no, (raw_train, raw_test) in enumerate(folds, start=1):
            train = self._indices(raw_train, fold_no=fold_no, role="train", row_count=row_count)
            test = self._indices(raw_test, fold_no=fold_no, role="test", row_count=row_count)
            if not len(train) or not len(test):
                raise StandardSuperGLMError(
                    f"fold {fold_no} train and test indices must both be non-empty"
                )
            overlap = sorted(set(train.tolist()) & set(test.tolist()))
            if overlap:
                raise StandardSuperGLMError(
                    f"fold {fold_no} train/test rows overlap: {overlap}"
                )
            duplicate_test = sorted(seen_test_rows & set(test.tolist()))
            if duplicate_test:
                raise StandardSuperGLMError(
                    "duplicate test-row membership is not supported by the standard "
                    f"OOF contract: {duplicate_test}"
                )
            seen_test_rows.update(test.tolist())
            validated.append((train, test))

        if not validated:
            raise StandardSuperGLMError("at least one validation fold is required")
        self._folds = tuple(validated)
        self.row_count = int(row_count)
        self.oof_coverage = len(seen_test_rows) / self.row_count

    @staticmethod
    def _indices(
        raw: Any,
        *,
        fold_no: int,
        role: str,
        row_count: int,
    ) -> np.ndarray:
        values = np.asarray(raw)
        if values.ndim != 1:
            raise StandardSuperGLMError(f"fold {fold_no} {role} indices must be one-dimensional")
        if not np.issubdtype(values.dtype, np.integer):
            raise StandardSuperGLMError(f"fold {fold_no} {role} indices must be integers")
        indices = values.astype(np.int64, copy=True)
        if len(np.unique(indices)) != len(indices):
            raise StandardSuperGLMError(f"fold {fold_no} {role} indices contain duplicates")
        if len(indices) and (indices.min() < 0 or indices.max() >= row_count):
            raise StandardSuperGLMError(
                f"fold {fold_no} {role} indices are outside row range 0..{row_count - 1}"
            )
        return indices

    @property
    def folds(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return tuple((train.copy(), test.copy()) for train, test in self._folds)

    def split(self, X, y=None, groups=None):
        del X, y, groups
        yield from self.folds


def _json_primitive(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_primitive(item) for item in value.to_dict("records")]
    if isinstance(value, np.ndarray):
        return [_json_primitive(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_primitive(value.item())
    if isinstance(value, dict):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_score(value: Any, *, label: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise StandardSuperGLMError(f"{label} must be finite")
    return score


def cv_result_to_records(
    result,
    *,
    oof_coverage: float,
) -> tuple[dict[str, Any], dict[str, float], tuple[FoldMetric, ...]]:
    mean_scores = {
        str(name): _finite_score(value, label=f"mean score {name!r}")
        for name, value in result.mean_scores.items()
    }
    pooled_scores = {
        str(name): _finite_score(value, label=f"pooled score {name!r}")
        for name, value in result.pooled_scores.items()
    }
    std_scores = {
        str(name): _finite_score(value, label=f"standard-deviation score {name!r}")
        for name, value in result.std_scores.items()
    }
    metrics = {f"cv_mean_{name}": value for name, value in mean_scores.items()}
    metrics.update({f"cv_pooled_{name}": value for name, value in pooled_scores.items()})
    metrics.update({f"cv_std_{name}": value for name, value in std_scores.items()})
    metrics["cv_oof_coverage"] = float(oof_coverage)

    metric_names = tuple(mean_scores)
    fold_metrics: list[FoldMetric] = []
    for record in result.fold_scores.to_dict("records"):
        fold_no = int(record["fold"]) + 1
        for metric_name in metric_names:
            if metric_name in record:
                fold_metrics.append(
                    FoldMetric(
                        fold_no=fold_no,
                        metric_name=metric_name,
                        metric_value=_finite_score(
                            record[metric_name],
                            label=f"fold {fold_no} score {metric_name!r}",
                        ),
                    )
                )

    fold_indices = [
        {"train": train.tolist(), "test": test.tolist()}
        for train, test in (result.fold_indices or [])
    ]
    report = {
        "schema_version": 1,
        "scope": "cv",
        "fold_scores": _json_primitive(result.fold_scores),
        "mean_scores": mean_scores,
        "pooled_scores": pooled_scores,
        "std_scores": std_scores,
        "fold_indices": fold_indices,
        "oof_coverage": float(oof_coverage),
        "oof_predictions": _json_primitive(result.oof_predictions),
    }
    return report, metrics, tuple(fold_metrics)


def _validate_input_lengths(inputs: ModelInputs) -> None:
    row_count = len(inputs.X)
    values = {
        "y": inputs.y,
        "sample_weight": inputs.sample_weight,
        "offset": inputs.offset,
        "export_weight": inputs.export_weight,
    }
    for name, value in values.items():
        if value is not None and len(value) != row_count:
            raise StandardSuperGLMError(
                f"{name} length {len(value)} does not match X row count {row_count}"
            )


def run_cross_validation(
    model,
    inputs: ModelInputs,
    *,
    split_indices: Iterable[tuple[Any, Any]],
    fit_mode: str,
    scoring: str | Callable | Sequence[str | Callable],
    cross_validate_fn: Callable[..., Any] = cross_validate,
) -> CVEvidence:
    _validate_input_lengths(inputs)
    splitter = PrecomputedSplitter(split_indices, row_count=len(inputs.X))
    result = cross_validate_fn(
        model,
        inputs.X,
        inputs.y,
        cv=splitter,
        sample_weight=inputs.sample_weight,
        offset=inputs.offset,
        fit_mode=fit_mode,
        scoring=scoring,
        return_estimators=False,
        return_oof=True,
        error_score="raise",
    )
    if result.fold_indices is None:
        raise StandardSuperGLMError("SuperGLM CV did not return fold indices")

    non_converged = result.fold_scores.loc[
        ~result.fold_scores["converged"].astype(bool), "fold"
    ].tolist()
    if non_converged:
        fold_numbers = [int(value) + 1 for value in non_converged]
        if len(fold_numbers) == 1:
            raise StandardSuperGLMError(f"fold {fold_numbers[0]} did not converge")
        raise StandardSuperGLMError(f"folds {fold_numbers} did not converge")

    report, metrics, fold_metrics = cv_result_to_records(
        result,
        oof_coverage=splitter.oof_coverage,
    )
    return CVEvidence(
        fold_indices=tuple(
            (np.asarray(train).copy(), np.asarray(test).copy())
            for train, test in result.fold_indices
        ),
        report=report,
        metrics=metrics,
        fold_metrics=fold_metrics,
    )
