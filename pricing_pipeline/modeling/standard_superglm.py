from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pricing_pipeline.build_identity import (
    BuildIdentity,
    BuildIdentityError,
    stable_build_export_id,
    verify_build_identity,
)
from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    create_model_frame_manifest_with_split,
)
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ApprovedModelBuild, ValidationSplitResult
from pricing_pipeline.modeling.validation_curves import (
    ValidationCurveCapture,
    normalize_validation_curves,
    validation_curve_capture_failure,
)
from pricing_pipeline.modeling.superglm_identity import exact_superglm_cross_validate
from pricing_pipeline.publishing.rating_export import export_rating_tables
from pricing_pipeline.publishing.superglm_metadata import build_superglm_publication_receipt
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    write_publication_receipt,
)
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


class StandardSuperGLMError(ValueError):
    """Raised when the shared SuperGLM build contract is violated."""


_SAFE_ATTEMPT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ModelInputs:
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame | np.ndarray
    sample_weight: pd.Series | np.ndarray | None = None
    sample_weight_name: str | None = None
    offset: pd.Series | np.ndarray | None = None
    offset_source: pd.Series | np.ndarray | None = None
    offset_source_name: str | None = None
    export_weight: pd.Series | np.ndarray | None = None
    export_weight_name: str | None = None
    row_ids: pd.DataFrame | None = None


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
    validation_splits: tuple[ValidationSplitResult, ...]
    validation_curve_capture: ValidationCurveCapture


def run_standard_superglm_build(
    engine,
    *,
    frame: pd.DataFrame,
    inputs: ModelInputs,
    superglm_model: Any,
    split_indices: Iterable[tuple[Any, Any]],
    expected_build_identity: BuildIdentity,
    fit_mode: str,
    scoring: str | Callable | Sequence[str | Callable],
    output_dir: str | Path,
    model_id: int,
    model_config: ModelBuildConfig,
    model_version: str,
    export_id: str,
    effective_from: str | None,
    manifest_spec: ModelFrameManifestSpec,
    split_artifact_root: str | Path,
    model_source_root: str | Path,
    created_by: str,
    offset_contract: OffsetExportContract | None = None,
    cross_validate_fn: Callable[..., Any] = exact_superglm_cross_validate,
) -> ApprovedModelBuild:
    _validate_input_lengths(inputs)
    _validate_canonical_row_ids(
        frame,
        inputs,
        pk_columns=manifest_spec.pk_columns,
    )
    resolved_offset_contract = _resolved_offset_contract(inputs, offset_contract)
    offset_source_name = _weight_name(
        inputs.offset_source,
        inputs.offset_source_name,
        role="offset_source",
    )
    _validate_manifest_offset_contract(
        frame,
        manifest_spec,
        resolved_offset_contract,
        offset=inputs.offset,
        offset_source=inputs.offset_source,
        offset_source_name=offset_source_name,
    )
    folds = tuple(split_indices)
    identity_inputs = {
        "frame": frame,
        "model_config": model_config,
        "manifest_spec": manifest_spec,
        "superglm_model": superglm_model,
        "split_indices": folds,
        "fit_mode": fit_mode,
        "scoring": scoring,
        "model_source_root": model_source_root,
    }
    _verify_expected_build_identity(expected_build_identity, identity_inputs)
    if export_id != stable_build_export_id(expected_build_identity):
        raise StandardSuperGLMError(
            "export_id must equal the stable export identity for this build fingerprint"
        )
    try:
        cv_model = deepcopy(superglm_model)
        final_model = deepcopy(superglm_model)
    except Exception as exc:
        raise StandardSuperGLMError(
            "superglm_model must be an unfitted, copyable SuperGLM model"
        ) from exc
    evidence = run_cross_validation(
        cv_model,
        inputs,
        split_indices=folds,
        fit_mode=fit_mode,
        scoring=scoring,
        cross_validate_fn=cross_validate_fn,
    )
    fitted, telemetry = fit_full_model(final_model, inputs, fit_mode=fit_mode)
    fit_weight_name = _weight_name(
        inputs.sample_weight,
        inputs.sample_weight_name,
        role="sample_weight",
    )
    export_weight_name = (
        _weight_name(
            inputs.export_weight,
            inputs.export_weight_name,
            role="export_weight",
        )
        if inputs.export_weight is not None
        else None
    )
    _verify_expected_build_identity(expected_build_identity, identity_inputs)

    manifest = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=manifest_spec,
        validation_split=model_config.validation_split,
        validation_split_artifact_root=Path(split_artifact_root),
        split_indices=list(evidence.fold_indices),
        created_by=created_by,
    )
    if manifest.model_frame_sha256 != expected_build_identity.model_frame_sha256:
        raise StandardSuperGLMError(
            "persisted manifest model-frame hash does not match the approved build identity"
        )
    run_dir = _manifest_attempt_directory(output_dir, manifest.manifest_id)
    try:
        workbook_path = run_dir / "rating_tables.xlsx"
        export_options: dict[str, Any] = {}
        if inputs.offset is not None:
            export_options["offset"] = inputs.offset
        if resolved_offset_contract.handling == "EXPORTED_FACTOR":
            export_options.update(
                offset_source=inputs.offset_source,
                offset_name=resolved_offset_contract.source_factor_name,
                offset_kind="auto",
            )
        export_rating_tables(
            fitted,
            inputs.X,
            inputs.y,
            inputs.export_weight,
            output_path=workbook_path,
            **export_options,
        )
        workbook_sha256 = hash_file_sha256(workbook_path)
        receipt = build_superglm_publication_receipt(
            fitted,
            offset_contract=resolved_offset_contract,
            fit_sample_weight_name=fit_weight_name,
            export_weight_name=export_weight_name,
        )
        receipt_path = run_dir / "publication_receipt.json"
        receipt_sha256 = write_publication_receipt(receipt, receipt_path)

        cv_report = dict(evidence.report)
        cv_report["full_fit_telemetry"] = telemetry
        cv_report["model_name"] = model_config.model_name
        cv_report["fit_mode"] = fit_mode
        cv_report["scoring"] = _scoring_labels(scoring)
        bundle = CandidateBundle(
            fitted_model=fitted,
            X=inputs.X.copy(),
            y=np.asarray(inputs.y).copy(),
            sample_weight=(
                None if inputs.sample_weight is None else np.asarray(inputs.sample_weight).copy()
            ),
            offset=None if inputs.offset is None else np.asarray(inputs.offset).copy(),
            offset_source=(
                None if inputs.offset_source is None else np.asarray(inputs.offset_source).copy()
            ),
            export_weight=(
                None if inputs.export_weight is None else np.asarray(inputs.export_weight).copy()
            ),
            cv_report=cv_report,
            model_name=model_config.model_name,
            model_version=model_version,
            export_id=export_id,
            manifest_id=manifest.manifest_id,
            split_set_id=manifest.split_set_id,
            pk_columns=manifest_spec.pk_columns,
            row_order_sha256=expected_build_identity.row_order_sha256,
            model_source_sha256=expected_build_identity.model_source_sha256,
            build_fingerprint_sha256=expected_build_identity.build_fingerprint_sha256,
            builder_source_sha256=expected_build_identity.builder_source_sha256,
            materialized_split_sha256=expected_build_identity.materialized_split_sha256,
            runtime_sha256=expected_build_identity.runtime_sha256,
            candidate_superglm_sha256=expected_build_identity.candidate_superglm_sha256,
            model_frame_sha256=manifest.model_frame_sha256,
            offset_contract=resolved_offset_contract,
            fit_sample_weight_name=fit_weight_name,
            offset_source_name=offset_source_name,
            export_weight_name=export_weight_name,
        )
        artifact = save_candidate_bundle(bundle, run_dir / "candidate_bundle.joblib")
        if (
            artifact.python_version != expected_build_identity.candidate_python_version
            or artifact.superglm_version != expected_build_identity.candidate_superglm_version
            or artifact.superglm_git_sha != expected_build_identity.candidate_superglm_git_sha
        ):
            raise StandardSuperGLMError(
                "candidate artifact runtime does not match the approved build identity"
            )
        _verify_expected_build_identity(expected_build_identity, identity_inputs)
        fold_metric_records = tuple(
            {
                "fold_no": metric.fold_no,
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value,
            }
            for metric in evidence.fold_metrics
        )
        completed_build = ApprovedModelBuild(
            model_id=model_id,
            model_name=model_config.model_name,
            rating_workbook_path=str(workbook_path),
            rating_workbook_sha256=workbook_sha256,
            model_version=model_version,
            model_type=model_config.model_type,
            target_name=model_config.target_name,
            deployment_slot=model_config.deployment_slot,
            effective_from=effective_from,
            export_id=export_id,
            created_by=created_by,
            manifest_id=manifest.manifest_id,
            split_set_id=manifest.split_set_id,
            candidate_artifact_path=str(artifact.path),
            candidate_artifact_sha256=artifact.sha256,
            candidate_artifact_format=artifact.format,
            candidate_artifact_size_bytes=artifact.size_bytes,
            candidate_python_version=artifact.python_version,
            candidate_superglm_version=artifact.superglm_version,
            candidate_superglm_git_sha=artifact.superglm_git_sha,
            build_fingerprint_sha256=expected_build_identity.build_fingerprint_sha256,
            builder_source_sha256=expected_build_identity.builder_source_sha256,
            materialized_split_sha256=expected_build_identity.materialized_split_sha256,
            runtime_sha256=expected_build_identity.runtime_sha256,
            candidate_superglm_sha256=expected_build_identity.candidate_superglm_sha256,
            row_order_sha256=expected_build_identity.row_order_sha256,
            model_source_sha256=expected_build_identity.model_source_sha256,
            model_frame_sha256=manifest.model_frame_sha256,
            publication_receipt_path=str(receipt_path),
            publication_receipt_sha256=receipt_sha256,
            metrics=evidence.metrics,
            metric_scopes={name: "cv" for name in evidence.metrics},
            fold_metrics=fold_metric_records,
            validation_splits=evidence.validation_splits,
            validation_curve_status=evidence.validation_curve_capture.status,
            validation_curve_reason=evidence.validation_curve_capture.reason,
            validation_curve_points=evidence.validation_curve_capture.points,
        )
        return completed_build
    except BaseException:
        # The manifest/split was committed first and remains durable frame evidence.
        # Only the incomplete, retry-local artifact directory is disposable here.
        shutil.rmtree(run_dir)
        raise


def _verify_expected_build_identity(
    expected: BuildIdentity,
    identity_inputs: dict[str, Any],
) -> None:
    try:
        verify_build_identity(expected, **identity_inputs)
    except BuildIdentityError as exc:
        raise StandardSuperGLMError(str(exc)) from exc


def _validate_input_lengths(inputs: ModelInputs) -> None:
    row_count = len(inputs.X)
    values = {
        "y": inputs.y,
        "sample_weight": inputs.sample_weight,
        "offset": inputs.offset,
        "offset_source": inputs.offset_source,
        "export_weight": inputs.export_weight,
    }
    for name, value in values.items():
        if value is not None and len(value) != row_count:
            raise StandardSuperGLMError(
                f"{name} length {len(value)} does not match X row count {row_count}"
            )


def _validate_canonical_row_ids(
    frame: pd.DataFrame,
    inputs: ModelInputs,
    *,
    pk_columns: tuple[str, ...],
) -> None:
    row_ids = inputs.row_ids
    if row_ids is None:
        raise StandardSuperGLMError("a publishable standard build requires ModelInputs.row_ids")
    if not isinstance(row_ids, pd.DataFrame):
        raise StandardSuperGLMError("ModelInputs.row_ids must be a pandas DataFrame")

    expected_columns = list(pk_columns)
    if list(row_ids.columns) != expected_columns:
        raise StandardSuperGLMError(
            "ModelInputs.row_ids primary-key columns must exactly match manifest "
            f"pk_columns in order: expected={expected_columns!r}, "
            f"actual={list(row_ids.columns)!r}"
        )
    missing_frame_columns = [column for column in expected_columns if column not in frame]
    if missing_frame_columns:
        raise StandardSuperGLMError(
            "canonical frame is missing manifest primary-key columns: "
            + ", ".join(missing_frame_columns)
        )

    frame_row_count = len(frame)
    if len(inputs.X) != frame_row_count:
        raise StandardSuperGLMError(
            "ModelInputs.X row count does not match canonical frame row count: "
            f"X={len(inputs.X)}, frame={frame_row_count}"
        )
    if len(row_ids) != frame_row_count:
        raise StandardSuperGLMError(
            "ModelInputs.row_ids row count does not match canonical frame row count: "
            f"row_ids={len(row_ids)}, frame={frame_row_count}"
        )
    if not row_ids.index.equals(frame.index):
        raise StandardSuperGLMError(
            "ModelInputs.row_ids index/order does not match the canonical frame"
        )

    canonical_row_ids = frame.loc[:, expected_columns]
    if not row_ids.equals(canonical_row_ids):
        raise StandardSuperGLMError(
            "ModelInputs.row_ids primary-key values/order do not match the canonical frame"
        )
    if row_ids.isna().any().any():
        raise StandardSuperGLMError("ModelInputs.row_ids primary-key columns contain null values")
    if row_ids.duplicated(subset=expected_columns).any():
        raise StandardSuperGLMError(
            "ModelInputs.row_ids primary-key columns contain duplicate values"
        )

    identity_index = canonical_row_identity_index(row_ids)
    aligned_values = {
        "X": inputs.X,
        "y": inputs.y,
        "sample_weight": inputs.sample_weight,
        "offset": inputs.offset,
        "offset_source": inputs.offset_source,
        "export_weight": inputs.export_weight,
    }
    for name, value in aligned_values.items():
        if value is None:
            continue
        if not isinstance(value, pd.Series | pd.DataFrame):
            raise StandardSuperGLMError(
                f"ModelInputs.{name} must be a pandas Series/DataFrame carrying "
                "the canonical PK identity index"
            )
        if not value.index.identical(identity_index):
            raise StandardSuperGLMError(
                f"ModelInputs.{name} identity index/order does not exactly match "
                "the canonical PK identity index"
            )


def canonical_row_identity_index(row_ids: pd.DataFrame) -> pd.Index:
    """Build the stable PK identity index carried by publishable model inputs."""
    if len(row_ids.columns) == 1:
        column = row_ids.columns[0]
        return pd.Index(
            row_ids.iloc[:, 0].to_numpy(copy=True),
            name=column,
        )
    return pd.MultiIndex.from_frame(
        row_ids,
        names=list(row_ids.columns),
    )


def hash_model_source(root: str | Path) -> str:
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise StandardSuperGLMError(f"model source root does not exist: {source_root}")
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".ipynb", ".py", ".sql", ".toml"}
        and ".ipynb_checkpoints" not in path.relative_to(source_root).parts
    )
    if not paths:
        raise StandardSuperGLMError(
            f"model source root contains no .ipynb, .py, .sql, or .toml files: {source_root}"
        )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.suffix.lower() == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StandardSuperGLMError(f"invalid model notebook source: {path}") from exc
            cells = notebook.get("cells")
            if not isinstance(cells, list):
                raise StandardSuperGLMError(f"invalid model notebook cells: {path}")
            source_cells = []
            for cell in cells:
                if not isinstance(cell, dict):
                    raise StandardSuperGLMError(f"invalid model notebook cell: {path}")
                raw_source = cell.get("source", "")
                source = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)
                source_cells.append(
                    {
                        "cell_type": str(cell.get("cell_type") or ""),
                        "source": source,
                    }
                )
            source_bytes = json.dumps(
                source_cells,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        else:
            source_bytes = path.read_bytes()
        digest.update(source_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def run_cross_validation(
    model,
    inputs: ModelInputs,
    *,
    split_indices: Iterable[tuple[Any, Any]],
    fit_mode: str,
    scoring: str | Callable | Sequence[str | Callable],
    cross_validate_fn: Callable[..., Any] = exact_superglm_cross_validate,
) -> CVEvidence:
    _validate_input_lengths(inputs)
    splitter = PrecomputedSplitter(split_indices, row_count=len(inputs.X))
    cv_options = {
        "cv": splitter,
        "sample_weight": inputs.sample_weight,
        "offset": inputs.offset,
        "fit_mode": fit_mode,
        "scoring": scoring,
        "return_oof": True,
        "error_score": "raise",
    }
    try:
        result = cross_validate_fn(
            model,
            inputs.X,
            inputs.y,
            return_estimators=True,
            **cv_options,
        )
    except Exception as capture_exc:
        try:
            result = cross_validate_fn(
                model,
                inputs.X,
                inputs.y,
                return_estimators=False,
                **cv_options,
            )
        except Exception as fallback_exc:
            raise fallback_exc from capture_exc
        curve_capture = validation_curve_capture_failure(capture_exc)
    else:
        curve_capture = None
    if result.fold_indices is None:
        raise StandardSuperGLMError("SuperGLM CV did not return fold indices")
    requested_folds = splitter.folds
    try:
        returned_folds = PrecomputedSplitter(
            result.fold_indices,
            row_count=len(inputs.X),
        ).folds
    except StandardSuperGLMError as exc:
        raise StandardSuperGLMError(
            "SuperGLM CV returned fold indices do not exactly match requested validation splits"
        ) from exc
    if len(returned_folds) != len(requested_folds) or any(
        not np.array_equal(returned_train, requested_train)
        or not np.array_equal(returned_validation, requested_validation)
        for (returned_train, returned_validation), (
            requested_train,
            requested_validation,
        ) in zip(returned_folds, requested_folds, strict=True)
    ):
        raise StandardSuperGLMError(
            "SuperGLM CV returned fold indices do not exactly match requested validation splits"
        )

    non_converged = result.fold_scores.loc[
        ~result.fold_scores["converged"].astype(bool), "fold"
    ].tolist()
    if non_converged:
        fold_numbers = [int(value) + 1 for value in non_converged]
        if len(fold_numbers) == 1:
            raise StandardSuperGLMError(f"fold {fold_numbers[0]} did not converge")
        raise StandardSuperGLMError(f"folds {fold_numbers} did not converge")

    report, metrics, fold_metrics, validation_splits = cv_result_to_records(
        result,
        oof_coverage=splitter.oof_coverage,
        scoring=scoring,
        fold_indices=returned_folds,
    )
    if curve_capture is None:
        curve_capture = normalize_validation_curves(
            getattr(result, "curve_similarity", None),
            estimators=getattr(result, "estimators", None),
            fold_count=len(returned_folds),
        )
    return CVEvidence(
        fold_indices=returned_folds,
        report=report,
        metrics=metrics,
        fold_metrics=fold_metrics,
        validation_splits=validation_splits,
        validation_curve_capture=curve_capture,
    )


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
                raise StandardSuperGLMError(f"fold {fold_no} train/test rows overlap: {overlap}")
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


def cv_result_to_records(
    result,
    *,
    oof_coverage: float,
    scoring: str | Callable | Sequence[str | Callable],
    fold_indices: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    tuple[FoldMetric, ...],
    tuple[ValidationSplitResult, ...],
]:
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

    metric_names = _requested_metric_names(
        scoring,
        mean_scores,
        result.fold_scores.columns,
    )
    fold_metrics: list[FoldMetric] = []
    validation_splits: list[ValidationSplitResult] = []
    result_folds = tuple(result.fold_indices or ()) if fold_indices is None else fold_indices
    fold_score_records = result.fold_scores.to_dict("records")
    if len(fold_score_records) != len(result_folds):
        raise StandardSuperGLMError(
            "SuperGLM fold_scores must contain one row per materialized split"
        )
    expected_fold_numbers = list(range(len(result_folds)))
    reported_fold_numbers = [record.get("fold") for record in fold_score_records]
    if reported_fold_numbers != expected_fold_numbers:
        raise StandardSuperGLMError(
            f"SuperGLM fold_scores fold numbering must be exactly 0 through {len(result_folds) - 1}"
        )

    for row_no, record in enumerate(fold_score_records):
        fold_no = int(record["fold"]) + 1
        train_indices, validation_indices = result_folds[row_no]
        reported_counts = (
            ("n_train", "n_train", len(train_indices)),
            ("n_test", "n_validation", len(validation_indices)),
        )
        for reported_name, materialized_name, materialized_count in reported_counts:
            if reported_name not in record:
                continue
            reported_count = record[reported_name]
            if (
                isinstance(reported_count, bool)
                or not isinstance(reported_count, Integral)
                or reported_count <= 0
            ):
                raise StandardSuperGLMError(
                    f"fold {fold_no} reported {reported_name} must be a positive integer"
                )
            if reported_count != materialized_count:
                raise StandardSuperGLMError(
                    f"fold {fold_no} reported {reported_name}={reported_count} but "
                    f"materialized {materialized_name}={materialized_count}"
                )
        missing_metrics = [name for name in metric_names if name not in record]
        if missing_metrics:
            raise StandardSuperGLMError(
                f"fold {fold_no} is missing requested metrics: {', '.join(missing_metrics)}"
            )
        split_metrics: dict[str, float] = {}
        for metric_name in metric_names:
            metric_value = _finite_score(
                record[metric_name],
                label=f"fold {fold_no} score {metric_name!r}",
            )
            split_metrics[metric_name] = metric_value
            fold_metrics.append(
                FoldMetric(
                    fold_no=fold_no,
                    metric_name=metric_name,
                    metric_value=metric_value,
                )
            )
        validation_splits.append(
            ValidationSplitResult(
                validation_split_no=fold_no,
                n_train=len(train_indices),
                n_validation=len(validation_indices),
                metrics=split_metrics,
            )
        )

    fold_index_records = [
        {"train": train.tolist(), "test": test.tolist()} for train, test in result_folds
    ]
    report = {
        "schema_version": 1,
        "scope": "cv",
        "fold_scores": _json_primitive(result.fold_scores),
        "mean_scores": mean_scores,
        "pooled_scores": pooled_scores,
        "std_scores": std_scores,
        "fold_indices": fold_index_records,
        "oof_coverage": float(oof_coverage),
        "oof_predictions": _json_primitive(result.oof_predictions),
    }
    return report, metrics, tuple(fold_metrics), tuple(validation_splits)


def _requested_metric_names(
    scoring: str | Callable | Sequence[str | Callable],
    mean_scores: dict[str, float],
    fold_score_columns: Iterable[Any],
) -> tuple[str, ...]:
    values = (scoring,) if isinstance(scoring, str) or callable(scoring) else tuple(scoring)
    explicit_metric_names = tuple(value for value in values if isinstance(value, str))
    missing_mean_metrics = [name for name in explicit_metric_names if name not in mean_scores]
    if missing_mean_metrics:
        raise StandardSuperGLMError(
            "SuperGLM mean_scores is missing requested metrics: " + ", ".join(missing_mean_metrics)
        )
    if len(explicit_metric_names) == len(values):
        return explicit_metric_names
    metric_names = tuple(str(name) for name in fold_score_columns if str(name) in mean_scores)
    missing_metrics = [name for name in mean_scores if name not in metric_names]
    if missing_metrics:
        raise StandardSuperGLMError(
            "SuperGLM fold_scores is missing aggregate metrics: " + ", ".join(missing_metrics)
        )
    return metric_names


def _finite_score(value: Any, *, label: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise StandardSuperGLMError(f"{label} must be finite")
    return score


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


def fit_full_model(model, inputs: ModelInputs, *, fit_mode: str):
    if fit_mode not in {"fit", "fit_reml"}:
        raise StandardSuperGLMError(f"unsupported fit_mode {fit_mode!r}")
    fit_fn = getattr(model, fit_mode, None)
    if not callable(fit_fn):
        raise StandardSuperGLMError(f"model has no callable {fit_mode} method")
    fitted = fit_fn(
        inputs.X,
        inputs.y,
        sample_weight=inputs.sample_weight,
        offset=inputs.offset,
    )
    telemetry_fn = getattr(fitted, "training_telemetry", None)
    telemetry = telemetry_fn() if callable(telemetry_fn) else {}
    if telemetry.get("converged") is False:
        raise StandardSuperGLMError("full training fit did not converge")
    return fitted, _json_primitive(telemetry)


def _resolved_offset_contract(
    inputs: ModelInputs,
    offset_contract: OffsetExportContract | None,
) -> OffsetExportContract:
    if inputs.offset is None:
        if offset_contract is None:
            return OffsetExportContract(handling="NONE")
        if offset_contract.handling != "NONE":
            raise StandardSuperGLMError(
                "offset contract must use handling='NONE' when fit offset is absent"
            )
        return offset_contract
    if offset_contract is None or offset_contract.handling == "NONE":
        raise StandardSuperGLMError(
            "a fitted offset requires a model-owned non-NONE OffsetExportContract"
        )
    return offset_contract


def _validate_manifest_offset_contract(
    frame: pd.DataFrame,
    manifest_spec: ModelFrameManifestSpec,
    contract: OffsetExportContract,
    *,
    offset: pd.Series | np.ndarray | None,
    offset_source: pd.Series | np.ndarray | None,
    offset_source_name: str | None,
) -> None:
    expects_offset = contract.handling != "NONE"
    if (manifest_spec.offset_column is not None) != expects_offset:
        requirement = "non-null" if expects_offset else "null"
        raise StandardSuperGLMError(
            f"handling {contract.handling} requires manifest offset_column to be {requirement}"
        )
    if manifest_spec.offset_label != contract.label:
        raise StandardSuperGLMError("manifest offset_label must match offset_contract.label")

    expected_source = contract.source_name if contract.handling == "EXPORTED_FACTOR" else None
    if manifest_spec.offset_source_column != expected_source:
        raise StandardSuperGLMError(
            f"handling {contract.handling} requires manifest offset_source_column to match "
            f"offset_contract.source_name {expected_source!r}"
        )
    if offset_source_name != expected_source:
        raise StandardSuperGLMError(
            f"ModelInputs.offset_source must match handling {contract.handling} source "
            f"{expected_source!r}"
        )
    if expected_source is not None:
        if expected_source not in frame:
            raise StandardSuperGLMError(
                f"manifest offset_source_column {expected_source!r} is missing from the model frame"
            )
        if not np.array_equal(
            frame[expected_source].to_numpy(),
            np.asarray(offset_source),
        ):
            raise StandardSuperGLMError(
                "manifest offset_source_column values must match ModelInputs.offset_source"
            )

    if not expects_offset:
        return
    offset_column = manifest_spec.offset_column
    if offset_column not in frame:
        raise StandardSuperGLMError(
            f"manifest offset_column {offset_column!r} is missing from the model frame"
        )
    if not np.array_equal(frame[offset_column].to_numpy(), np.asarray(offset)):
        raise StandardSuperGLMError("manifest offset_column values must match ModelInputs.offset")


def _weight_name(
    value: pd.Series | np.ndarray | None,
    explicit_name: str | None,
    *,
    role: str,
) -> str | None:
    if value is None:
        if explicit_name is not None:
            raise StandardSuperGLMError(f"{role}_name was supplied without {role}")
        return None
    if explicit_name is not None and str(explicit_name).strip():
        return str(explicit_name).strip()
    if isinstance(value, pd.Series) and value.name is not None and str(value.name).strip():
        return str(value.name).strip()
    raise StandardSuperGLMError(
        f"{role} uses an unnamed array; supply {role}_name once in ModelInputs"
    )


def _manifest_attempt_directory(output_dir: str | Path, manifest_id: str) -> Path:
    if not isinstance(manifest_id, str) or not _SAFE_ATTEMPT_COMPONENT.fullmatch(manifest_id):
        raise StandardSuperGLMError(
            "manifest_id must be a safe path component using letters, numbers, '.', '_', or '-'"
        )
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = (output_root / manifest_id).resolve()
    if attempt_dir.parent != output_root:
        raise StandardSuperGLMError(
            f"manifest attempt directory is outside run output directory {output_root}"
        )
    try:
        attempt_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise StandardSuperGLMError(
            f"manifest attempt directory already exists; refusing to overwrite: {attempt_dir}"
        ) from exc
    return attempt_dir


def hash_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scoring_labels(
    scoring: str | Callable | Sequence[str | Callable],
) -> list[str]:
    values = (scoring,) if isinstance(scoring, str) or callable(scoring) else tuple(scoring)
    labels = []
    for value in values:
        if isinstance(value, str):
            labels.append(value)
            continue
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        labels.append(f"{module}.{name}" if module and name else type(value).__name__)
    return labels
