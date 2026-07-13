from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from sqlalchemy import text

from pricing_models.registry import get_model_config
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.file_lock import exclusive_file_lock
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.modeling.standard_superglm import ModelInputs, call_review_hook
from pricing_pipeline.publishing.lineage import record_model_run_on_connection
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.rating_export import export_rating_tables
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.publishing.superglm_metadata import build_superglm_publication_receipt
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    write_publication_receipt,
)
from pricing_pipeline.workbench.artifacts import (
    CandidateArtifactError,
    CandidateArtifactMetadata,
    CandidateBundle,
    load_candidate_bundle,
    save_candidate_bundle,
)
from pricing_pipeline.workbench.submission import (
    EDITED_MODEL_FORMAT,
    EditorSubmission,
    EditorSubmissionError,
    load_verified_submission,
    sha256_file,
)


@dataclass(frozen=True)
class ChampionSnapshot:
    deployment_slot: str
    rate_package_id: int | None
    bundle: CandidateBundle | None
    unavailable_reason: str | None = None

    @property
    def status(self) -> str:
        if self.rate_package_id is None:
            return "NO_CHAMPION"
        if self.bundle is None:
            return "UNAVAILABLE"
        return "COMPARED"

    def revision_metadata(self) -> dict[str, Any]:
        return {
            "available": self.status == "COMPARED",
            "deployment_slot": self.deployment_slot,
            "rate_package_id": self.rate_package_id,
            "reason": self.unavailable_reason,
            "status": self.status,
        }


@dataclass(frozen=True)
class ParentCandidate:
    model_id: int
    model_name: str
    model_version: str
    package_version: int
    rate_package_id: int
    model_run_id: int
    effective_from: str | None
    effective_to: str | None
    config: ModelBuildConfig
    bundle: CandidateBundle
    champion: ChampionSnapshot


@dataclass(frozen=True)
class EditorExport:
    export_id: str
    rating_workbook_path: str
    publication_receipt_path: str
    publication_receipt_sha256: str
    candidate_artifact_path: str
    candidate_artifact_sha256: str
    candidate_artifact_format: str
    candidate_artifact_size_bytes: int
    candidate_python_version: str
    candidate_superglm_version: str
    revision_metadata_json: str
    metrics: dict[str, float]
    metric_scopes: dict[str, str]
    edited_model: Any
    bundle: CandidateBundle


@dataclass(frozen=True)
class EditorPublicationResult:
    submission_id: str
    model_name: str
    parent_rate_package_id: int
    rate_package_id: int
    package_version: int
    model_run_id: int
    package_status: str
    was_existing: bool


@dataclass(frozen=True)
class EditorPublicationAttempt:
    staging_dir: Path
    final_dir: Path


def _editor_export_id(submission: EditorSubmission) -> str:
    return f"editor__{submission.submission_id.replace('-', '_')}"


def _submission_directory(
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
) -> Path:
    root = Path(allowed_root).expanduser().resolve()
    submission_path = Path(submission.path).expanduser().resolve()
    if not submission_path.is_relative_to(root):
        raise EditorSubmissionError(
            f"submission path is outside configured artifact root {root}: {submission_path}"
        )
    return submission_path.parent


@contextmanager
def _editor_publication_lock(submission_dir: Path) -> Iterator[None]:
    lock_path = submission_dir / "publication.lock"
    with exclusive_file_lock(lock_path):
        yield


def _new_editor_publication_attempt(submission_dir: Path) -> EditorPublicationAttempt:
    attempt_id = uuid4().hex
    published_root = submission_dir / "published"
    staging_root = published_root / ".staging"
    attempts_root = published_root / "attempts"
    staging_root.mkdir(parents=True, exist_ok=True)
    attempts_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / attempt_id
    final_dir = attempts_root / attempt_id
    staging_dir.mkdir()
    return EditorPublicationAttempt(staging_dir=staging_dir, final_dir=final_dir)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _remove_unpublished_editor_attempts(submission_dir: Path) -> None:
    published_root = submission_dir / "published"
    for root_name in (".staging", "attempts"):
        root = published_root / root_name
        if not root.is_dir():
            continue
        for child in root.iterdir():
            _remove_path(child)


def _resolve_existing_editor_publication(
    engine,
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
) -> EditorPublicationResult | None:
    schemas = schema_names_from_connectable(engine)
    query = text(
        f"""
        SELECT
            pm.model_name,
            rp.rate_package_id,
            rp.package_version,
            rp.package_status,
            rp.parent_rate_package_id,
            mr.model_run_id,
            mr.run_status,
            mr.candidate_artifact_path,
            mr.candidate_artifact_sha256,
            mr.candidate_artifact_format,
            mr.candidate_artifact_size_bytes,
            mr.candidate_python_version,
            mr.candidate_superglm_version
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr
          ON mr.rate_package_id = rp.rate_package_id
        WHERE pm.model_name = :model_name
          AND rp.parent_rate_package_id = :parent_rate_package_id
          AND rp.source_export_id = :export_id
        """
    )
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                query,
                {
                    "model_name": submission.model_name,
                    "parent_rate_package_id": submission.parent_rate_package_id,
                    "export_id": _editor_export_id(submission),
                },
            )
            .mappings()
            .all()
        )
    if not rows:
        return None
    if len(rows) != 1:
        raise EditorSubmissionError(
            "editor publication requires lineage repair: "
            f"expected one package/run, found {len(rows)}"
        )
    row = dict(rows[0])
    if (
        row.get("model_run_id") is None
        or str(row.get("package_status") or "").upper() != "PUBLISHED"
        or str(row.get("run_status") or "").upper() != "SUCCESS"
    ):
        raise EditorSubmissionError(
            "editor publication requires lineage repair: package/run is incomplete"
        )
    artifact_fields = (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
    )
    if any(row.get(field) is None for field in artifact_fields):
        raise EditorSubmissionError(
            "editor publication requires lineage repair: candidate artifact metadata is incomplete"
        )
    try:
        load_candidate_bundle(
            row["candidate_artifact_path"],
            expected_sha256=row["candidate_artifact_sha256"],
            expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
            expected_format=row["candidate_artifact_format"],
            expected_python_version=row["candidate_python_version"],
            expected_superglm_version=row["candidate_superglm_version"],
            allowed_root=allowed_root,
        )
    except CandidateArtifactError as exc:
        raise EditorSubmissionError(
            f"existing editor publication candidate artifact failed verification: {exc}"
        ) from exc
    return EditorPublicationResult(
        submission_id=submission.submission_id,
        model_name=str(row["model_name"]),
        parent_rate_package_id=int(row["parent_rate_package_id"]),
        rate_package_id=int(row["rate_package_id"]),
        package_version=int(row["package_version"]),
        model_run_id=int(row["model_run_id"]),
        package_status=str(row["package_status"]),
        was_existing=True,
    )


def load_parent_candidate(
    engine,
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
    model_config: ModelBuildConfig | None = None,
) -> ParentCandidate:
    schemas = schema_names_from_connectable(engine)
    query = text(
        f"""
        SELECT
            pm.model_id,
            pm.model_name,
            rp.model_version,
            rp.package_version,
            rp.rate_package_id,
            rp.effective_from_date,
            rp.effective_to_date,
            mr.model_run_id,
            mr.run_status,
            mr.manifest_id,
            split_link.split_set_id,
            mr.candidate_artifact_path,
            mr.candidate_artifact_sha256,
            mr.candidate_artifact_format,
            mr.candidate_artifact_size_bytes,
            mr.candidate_python_version,
            mr.candidate_superglm_version,
            mr.model_source_sha256
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        JOIN {schemas.pricing}.MODEL_RUN AS mr
          ON mr.rate_package_id = rp.rate_package_id
        LEFT JOIN {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link
          ON split_link.model_run_id = mr.model_run_id
         AND split_link.manifest_id = mr.manifest_id
         AND split_link.dataset_role = 'training'
         AND split_link.split_role = 'validation'
        WHERE rp.rate_package_id = :rate_package_id
          AND mr.model_run_id = :model_run_id
        """
    )
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                query,
                {
                    "rate_package_id": submission.parent_rate_package_id,
                    "model_run_id": submission.parent_model_run_id,
                },
            )
            .mappings()
            .all()
        )
    if len(rows) != 1:
        raise EditorSubmissionError(
            "parent package must resolve exactly one successful model run; "
            f"found {len(rows)}"
        )
    row = dict(rows[0])
    expected = {
        "model_name": submission.model_name,
        "package_version": submission.source_package_version,
        "rate_package_id": submission.parent_rate_package_id,
        "model_run_id": submission.parent_model_run_id,
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "candidate_artifact_path": submission.baseline_candidate_path,
        "candidate_artifact_sha256": submission.baseline_candidate_sha256,
        "model_source_sha256": submission.model_source_sha256,
    }
    mismatches = [
        name
        for name, value in expected.items()
        if str(row.get(name)) != str(value)
    ]
    if str(row.get("run_status") or "").upper() != "SUCCESS":
        mismatches.append("run_status")
    if mismatches:
        raise EditorSubmissionError(
            "parent SQL lineage no longer matches the submission: " + ", ".join(mismatches)
        )

    bundle = load_candidate_bundle(
        row["candidate_artifact_path"],
        expected_sha256=row["candidate_artifact_sha256"],
        expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
        expected_format=row["candidate_artifact_format"],
        expected_python_version=row["candidate_python_version"],
        expected_superglm_version=row["candidate_superglm_version"],
        allowed_root=allowed_root,
    )
    if bundle.manifest_id != submission.manifest_id:
        raise EditorSubmissionError("parent bundle manifest does not match the submission")
    if bundle.split_set_id != submission.split_set_id:
        raise EditorSubmissionError("parent bundle split set does not match the submission")
    config = model_config or get_model_config(submission.model_name)
    configured_name = getattr(config, "model_name", submission.model_name)
    if str(configured_name) != submission.model_name:
        raise EditorSubmissionError(
            "explicit model config does not match the editor submission model_name"
        )
    champion = _load_champion_bundle(
        engine,
        model_id=int(row["model_id"]),
        deployment_slot=config.deployment_slot,
        allowed_root=allowed_root,
        parent_bundle=bundle,
    )
    return ParentCandidate(
        model_id=int(row["model_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        package_version=int(row["package_version"]),
        rate_package_id=int(row["rate_package_id"]),
        model_run_id=int(row["model_run_id"]),
        effective_from=(
            None
            if row.get("effective_from_date") is None
            else str(row["effective_from_date"])
        ),
        effective_to=(
            None if row.get("effective_to_date") is None else str(row["effective_to_date"])
        ),
        config=config,
        bundle=bundle,
        champion=champion,
    )


def _load_champion_bundle(
    engine,
    *,
    model_id: int,
    deployment_slot: str,
    allowed_root: Path,
    parent_bundle: CandidateBundle,
) -> ChampionSnapshot:
    schemas = schema_names_from_connectable(engine)
    query = text(
        f"""
        SELECT
            deployment.rate_package_id,
            mr.run_status,
            mr.candidate_artifact_path,
            mr.candidate_artifact_sha256,
            mr.candidate_artifact_format,
            mr.candidate_artifact_size_bytes,
            mr.candidate_python_version,
            mr.candidate_superglm_version
        FROM {schemas.pricing}.PRICING_MODEL_DEPLOYMENT AS deployment
        LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr
          ON mr.rate_package_id = deployment.rate_package_id
        WHERE deployment.model_id = :model_id
          AND deployment.deployment_slot = :deployment_slot
          AND deployment.effective_to_ts IS NULL
        """
    )
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                query,
                {"model_id": model_id, "deployment_slot": deployment_slot},
            )
            .mappings()
            .all()
        )
    if not rows:
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=None,
            bundle=None,
            unavailable_reason=f"no champion is deployed in {deployment_slot}",
        )
    if len(rows) != 1:
        raise EditorSubmissionError(
            f"{len(rows)} current champion runs resolved in {deployment_slot}; "
            "comparison identity is ambiguous"
        )
    row = dict(rows[0])
    rate_package_id = int(row["rate_package_id"])
    if str(row.get("run_status") or "").upper() != "SUCCESS":
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason="the deployed champion has no successful candidate run",
        )
    required = (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
    )
    if any(row.get(name) is None for name in required):
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason="the deployed champion has no candidate artifact",
        )
    try:
        champion = load_candidate_bundle(
            row["candidate_artifact_path"],
            expected_sha256=row["candidate_artifact_sha256"],
            expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
            expected_format=row["candidate_artifact_format"],
            expected_python_version=row["candidate_python_version"],
            expected_superglm_version=row["candidate_superglm_version"],
            allowed_root=allowed_root,
        )
    except CandidateArtifactError as exc:
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason=f"the deployed champion artifact could not be verified: {exc}",
        )
    if list(champion.X.columns) != list(parent_bundle.X.columns):
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason="the deployed champion uses a different prepared feature frame",
        )
    if champion.offset_contract.get("handling") != parent_bundle.offset_contract.get(
        "handling"
    ):
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason="the deployed champion uses a different offset contract",
        )
    return ChampionSnapshot(
        deployment_slot=deployment_slot,
        rate_package_id=rate_package_id,
        bundle=champion,
    )


def _load_edited_model(
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
) -> Any:
    path = Path(submission.edited_model_path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not path.is_relative_to(root):
        raise EditorSubmissionError(f"edited model is outside artifact root {root}: {path}")
    if submission.edited_model_format != EDITED_MODEL_FORMAT:
        raise EditorSubmissionError(
            f"unsupported edited model format {submission.edited_model_format!r}"
        )
    if path.stat().st_size != int(submission.edited_model_size_bytes):
        raise EditorSubmissionError("edited model byte-size verification failed")
    if sha256_file(path) != submission.edited_model_sha256:
        raise EditorSubmissionError("edited model SHA-256 verification failed")
    if submission.edited_model_python_version.split(".")[:2] != platform.python_version().split(
        "."
    )[:2]:
        raise EditorSubmissionError("edited model Python version is incompatible")
    try:
        runtime_superglm_version = version("superglm")
    except PackageNotFoundError:
        runtime_superglm_version = "unknown"
    if submission.edited_model_superglm_version != runtime_superglm_version:
        raise EditorSubmissionError(
            "edited model SuperGLM version is incompatible: "
            f"artifact={submission.edited_model_superglm_version!r}, "
            f"runtime={runtime_superglm_version!r}"
        )
    envelope = joblib.load(path)
    if not isinstance(envelope, dict) or envelope.get("format") != EDITED_MODEL_FORMAT:
        raise EditorSubmissionError("edited model envelope has an invalid format")
    if envelope.get("python_version") != submission.edited_model_python_version:
        raise EditorSubmissionError("edited model Python metadata is inconsistent")
    if envelope.get("superglm_version") != submission.edited_model_superglm_version:
        raise EditorSubmissionError("edited model SuperGLM metadata is inconsistent")
    return envelope["model"]


def _predict(model: Any, bundle: CandidateBundle) -> np.ndarray:
    if bundle.offset is None:
        prediction = model.predict(bundle.X)
    else:
        prediction = model.predict(bundle.X, offset=bundle.offset)
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(values) != len(bundle.X) or not np.isfinite(values).all():
        raise EditorSubmissionError("model returned invalid training predictions")
    return values


def _mean_model_deviance(
    model: Any,
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    distribution = getattr(model, "_distribution", None)
    deviance_unit = getattr(distribution, "deviance_unit", None)
    if callable(deviance_unit):
        unit_values = np.asarray(deviance_unit(y, prediction), dtype=float)
        if unit_values.shape != y.shape or not np.isfinite(unit_values).all():
            raise EditorSubmissionError("model distribution returned invalid unit deviance")
        return float(np.average(unit_values, weights=weights))
    if np.all(y >= 0) and np.all(prediction > 0):
        return float(mean_poisson_deviance(y, prediction, sample_weight=weights))
    return None


def training_comparison_metrics(
    baseline_model: Any,
    edited_model: Any,
    bundle: CandidateBundle,
    *,
    comparison_name: str,
) -> tuple[dict[str, float], dict[str, str]]:
    name = str(comparison_name).strip().lower()
    if not name:
        raise ValueError("comparison_name is required")
    baseline = _predict(baseline_model, bundle)
    edited = _predict(edited_model, bundle)
    weights = (
        np.ones(len(bundle.X), dtype=float)
        if bundle.sample_weight is None
        else np.asarray(bundle.sample_weight, dtype=float)
    )
    if len(weights) != len(bundle.X) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise EditorSubmissionError("training comparison weights are invalid")
    absolute_delta = np.abs(edited - baseline)
    relative_delta = absolute_delta / np.maximum(np.abs(baseline), 1e-12)
    prefix = f"editor_training_{name}"
    metrics = {
        f"{prefix}_mean_absolute_prediction_delta": float(
            np.average(absolute_delta, weights=weights)
        ),
        f"{prefix}_max_absolute_prediction_delta": float(np.max(absolute_delta)),
        f"{prefix}_mean_absolute_relative_change": float(
            np.average(relative_delta, weights=weights)
        ),
    }
    y = np.asarray(bundle.y, dtype=float)
    if len(y) == len(bundle.X) and np.isfinite(y).all():
        baseline_deviance = _mean_model_deviance(
            baseline_model,
            y,
            baseline,
            weights,
        )
        edited_deviance = _mean_model_deviance(
            edited_model,
            y,
            edited,
            weights,
        )
    else:
        baseline_deviance = None
        edited_deviance = None
    if baseline_deviance is not None and edited_deviance is not None:
        metrics[f"{prefix}_baseline_deviance"] = baseline_deviance
        metrics[f"{prefix}_edited_deviance"] = edited_deviance
        delta_name = (
            "editor_training_deviance_delta"
            if name == "parent"
            else f"{prefix}_deviance_delta"
        )
        metrics[delta_name] = edited_deviance - baseline_deviance
    scope = f"editor_training_{name}"
    return metrics, {metric_name: scope for metric_name in metrics}


def inherited_cv_metrics(
    bundle: CandidateBundle,
) -> tuple[dict[str, float], dict[str, str]]:
    report = bundle.cv_report
    metrics: dict[str, float] = {}
    for report_name, metric_prefix in (
        ("mean_scores", "cv_mean"),
        ("pooled_scores", "cv_pooled"),
        ("std_scores", "cv_std"),
    ):
        values = report.get(report_name) or {}
        for metric_name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise EditorSubmissionError(
                    f"inherited CV metric {report_name}.{metric_name} is not finite"
                )
            metrics[f"{metric_prefix}_{metric_name}"] = value
    if report.get("oof_coverage") is not None:
        coverage = float(report["oof_coverage"])
        if not math.isfinite(coverage):
            raise EditorSubmissionError("inherited CV OOF coverage is not finite")
        metrics["cv_oof_coverage"] = coverage
    return metrics, {metric_name: "inherited_cv" for metric_name in metrics}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _revision_with_publisher_identity(value: str, created_by: str) -> str:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise EditorSubmissionError("editor revision metadata must be a JSON object")
    publisher_identity = str(created_by).strip()
    if not publisher_identity:
        raise EditorSubmissionError("publisher identity is required")
    payload["published_by"] = publisher_identity
    return _canonical_json(payload)


def export_edited_model(
    parent: ParentCandidate,
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
    write_dir: str | Path,
    published_dir: str | Path,
) -> EditorExport:
    edited_model = _load_edited_model(submission, allowed_root=allowed_root)
    root = Path(allowed_root).expanduser().resolve()
    output_dir = Path(write_dir).expanduser().resolve()
    final_dir = Path(published_dir).expanduser().resolve()
    if not output_dir.is_relative_to(root) or not final_dir.is_relative_to(root):
        raise EditorSubmissionError("editor publication attempt is outside artifact root")
    if not output_dir.is_dir():
        raise EditorSubmissionError("editor publication staging directory does not exist")
    workbook_write_path = output_dir / "rating_tables.xlsx"
    workbook_path = final_dir / "rating_tables.xlsx"
    export_options = dict(parent.bundle.offset_export_options or {})
    if parent.bundle.offset is not None:
        export_options["offset"] = parent.bundle.offset
    export_rating_tables(
        edited_model,
        parent.bundle.X,
        parent.bundle.y,
        parent.bundle.export_weight,
        output_path=workbook_write_path,
        mlflow_client=None,
        **export_options,
    )
    receipt = build_superglm_publication_receipt(
        edited_model,
        offset_contract=OffsetExportContract.model_validate(parent.bundle.offset_contract),
        fit_sample_weight_name=parent.bundle.fit_sample_weight_name,
        export_weight_name=parent.bundle.export_weight_name,
    )
    receipt_write_path = output_dir / "publication_receipt.json"
    receipt_path = final_dir / "publication_receipt.json"
    receipt_sha256 = write_publication_receipt(receipt, receipt_write_path)

    metrics, metric_scopes = inherited_cv_metrics(parent.bundle)
    parent_metrics, parent_scopes = training_comparison_metrics(
        parent.bundle.fitted_model,
        edited_model,
        parent.bundle,
        comparison_name="parent",
    )
    metrics.update(parent_metrics)
    metric_scopes.update(parent_scopes)
    champion_bundle = parent.champion.bundle
    if champion_bundle is not None:
        champion_metrics, champion_scopes = training_comparison_metrics(
            champion_bundle.fitted_model,
            edited_model,
            parent.bundle,
            comparison_name="champion",
        )
        metrics.update(champion_metrics)
        metric_scopes.update(champion_scopes)
    champion_comparison = parent.champion.revision_metadata()
    review_hook = None
    if parent.bundle.review_hook_module and parent.bundle.review_hook_name:
        review_module = importlib.import_module(parent.bundle.review_hook_module)
        review_hook = getattr(review_module, parent.bundle.review_hook_name, None)
        if not callable(review_hook):
            raise EditorSubmissionError(
                "model-local review hook can no longer be imported: "
                f"{parent.bundle.review_hook_module}.{parent.bundle.review_hook_name}"
            )
    review_artifact = call_review_hook(
        review_hook,
        fitted_model=edited_model,
        inputs=ModelInputs(
            X=parent.bundle.X,
            y=parent.bundle.y,
            sample_weight=parent.bundle.sample_weight,
            sample_weight_name=parent.bundle.fit_sample_weight_name,
            offset=parent.bundle.offset,
            export_weight=parent.bundle.export_weight,
            export_weight_name=parent.bundle.export_weight_name,
        ),
        output_path=output_dir / "rating_tables_review.xlsx",
        allowed_root=output_dir,
    )
    edited_bundle = replace(
        parent.bundle,
        fitted_model=edited_model,
        review_artifact=(
            None
            if review_artifact is None
            else {
                "path": str(final_dir / Path(review_artifact.path).name),
                "sha256": review_artifact.sha256,
                "size_bytes": review_artifact.size_bytes,
            }
        ),
    )
    artifact: CandidateArtifactMetadata = save_candidate_bundle(
        edited_bundle,
        output_dir / "candidate_bundle.joblib",
    )
    artifact = replace(
        artifact,
        path=str(final_dir / "candidate_bundle.joblib"),
    )
    revision_metadata = {
        "kind": "SUPERGLM_EDITOR",
        "schema_version": 1,
        "submission_id": submission.submission_id,
        "reason": submission.reason,
        "claimed_identity": submission.claimed_identity,
        "parent_rate_package_id": submission.parent_rate_package_id,
        "parent_model_run_id": submission.parent_model_run_id,
        "submission_path": submission.path,
        "submission_sha256": submission.sha256,
        "editor_session_path": submission.editor_session_path,
        "editor_session_sha256": submission.editor_session_sha256,
        "editor_session_size_bytes": submission.editor_session_size_bytes,
        "edited_model_path": submission.edited_model_path,
        "edited_model_sha256": submission.edited_model_sha256,
        "edited_model_size_bytes": submission.edited_model_size_bytes,
        "edited_model_format": submission.edited_model_format,
        "baseline_candidate_sha256": submission.baseline_candidate_sha256,
        "baseline_cv_metrics": {
            name: value for name, value in metrics.items() if name.startswith("cv_")
        },
        "comparison_metrics": {
            name: value
            for name, value in metrics.items()
            if name.startswith("editor_training_")
        },
        "champion_comparison": champion_comparison,
    }
    return EditorExport(
        export_id=_editor_export_id(submission),
        rating_workbook_path=str(workbook_path),
        publication_receipt_path=str(receipt_path),
        publication_receipt_sha256=receipt_sha256,
        candidate_artifact_path=artifact.path,
        candidate_artifact_sha256=artifact.sha256,
        candidate_artifact_format=artifact.format,
        candidate_artifact_size_bytes=artifact.size_bytes,
        candidate_python_version=artifact.python_version,
        candidate_superglm_version=artifact.superglm_version,
        revision_metadata_json=_canonical_json(revision_metadata),
        metrics=metrics,
        metric_scopes=metric_scopes,
        edited_model=edited_model,
        bundle=edited_bundle,
    )


def stage_editor_export(
    engine,
    parent: ParentCandidate,
    export: EditorExport,
    created_by: str,
) -> None:
    stage_rating_export(
        engine,
        workbook_path=Path(export.rating_workbook_path),
        export_id=export.export_id,
        model_name=parent.model_name,
        model_version=parent.model_version,
        target_name=parent.config.target_name,
        model_type=parent.config.model_type,
        effective_from=parent.effective_from,
        effective_to=parent.effective_to,
        created_by=created_by,
        replace=True,
        model_id=parent.model_id,
        publication_receipt_path=export.publication_receipt_path,
        publication_receipt_sha256=export.publication_receipt_sha256,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _published_offset_source(bundle: CandidateBundle) -> pd.Series:
    options = bundle.offset_export_options or {}
    if "offset_source" not in options or options["offset_source"] is None:
        raise EditorSubmissionError(
            "EXPORTED_FACTOR SQL parity requires "
            "bundle.offset_export_options['offset_source']"
        )

    raw_source = options["offset_source"]
    if isinstance(raw_source, str):
        if raw_source not in bundle.X.columns:
            raise EditorSubmissionError(
                f"EXPORTED_FACTOR offset_source column {raw_source!r} is missing "
                "from bundle.X"
            )
        raw_source = bundle.X[raw_source]

    if isinstance(raw_source, pd.Series):
        values = raw_source.reset_index(drop=True)
    else:
        try:
            values = pd.Series(raw_source).reset_index(drop=True)
        except (TypeError, ValueError) as exc:
            shape = getattr(raw_source, "shape", None)
            if shape is not None and len(shape) > 1:
                raise EditorSubmissionError(
                    "EXPORTED_FACTOR offset_source must be one-dimensional; "
                    f"received shape {shape}"
                ) from exc
            raise EditorSubmissionError(
                "EXPORTED_FACTOR offset_source must be a one-dimensional array-like"
            ) from exc

    if len(values) != len(bundle.X):
        raise EditorSubmissionError(
            f"EXPORTED_FACTOR offset_source length {len(values)} does not match "
            f"candidate row count {len(bundle.X)}"
        )
    if values.isna().any():
        raise EditorSubmissionError("EXPORTED_FACTOR offset_source contains missing values")
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not pd.api.types.is_number(value):
            continue
        try:
            is_finite = bool(np.isfinite(value))
        except TypeError:
            is_finite = math.isfinite(value)
        if not is_finite:
            raise EditorSubmissionError(
                "EXPORTED_FACTOR offset_source contains non-finite numeric values"
            )
    return values


def verify_package_sql_parity(
    connection,
    *,
    rate_package_id: int,
    edited_model: Any,
    bundle: CandidateBundle,
    sample_size: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-8,
    execute_params_hook: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> None:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    count = min(int(sample_size), len(bundle.X))
    if count == 0:
        raise EditorSubmissionError("cannot verify SQL parity on an empty candidate")
    sample = bundle.X.iloc[:count]
    sample_offset = None if bundle.offset is None else np.asarray(bundle.offset)[:count]
    if sample_offset is None:
        expected = np.asarray(edited_model.predict(sample), dtype=float)
    else:
        expected = np.asarray(edited_model.predict(sample, offset=sample_offset), dtype=float)
    contract = OffsetExportContract.model_validate(bundle.offset_contract)
    sample_published_offset_source = (
        _published_offset_source(bundle).iloc[:count]
        if contract.handling == "EXPORTED_FACTOR"
        else None
    )

    schemas = schema_names_from_connectable(connection)
    statement = text(
        f"""
        EXEC {schemas.pricing}.PREDICT_RATE_PACKAGE
            @rate_package_id = :rate_package_id,
            @features_json = :features_json,
            @exposure = :exposure,
            @include_breakdown = 0
        """
    )
    for position, (_, row) in enumerate(sample.iterrows()):
        features = {str(name): _json_value(value) for name, value in row.items()}
        exposure = 1.0
        if sample_published_offset_source is not None:
            features[str(contract.published_factor_name)] = _json_value(
                sample_published_offset_source.iloc[position]
            )
        elif sample_offset is not None and contract.handling == "ALREADY_APPLIED_SQL_EXPOSURE":
            exposure = float(np.exp(sample_offset[position]))
        params: dict[str, Any] = {
            "rate_package_id": int(rate_package_id),
            "features_json": _canonical_json(features),
            "exposure": exposure,
        }
        if execute_params_hook is not None:
            params = execute_params_hook(params, float(expected[position]))
        actual = float(connection.execute(statement, params).mappings().one()["prediction"])
        if not np.isclose(actual, expected[position], rtol=rtol, atol=atol):
            raise EditorSubmissionError(
                "edited package failed Python/SQL parity at sample row "
                f"{position}: python={expected[position]!r}, sql={actual!r}"
            )


def record_derived_model_run(connection, **kwargs) -> int:
    return record_model_run_on_connection(connection, **kwargs)


def _publish_new_editor_submission(
    engine,
    *,
    submission: EditorSubmission,
    allowed_root: str | Path,
    attempt: EditorPublicationAttempt,
    dag_id: str,
    airflow_run_id: str,
    created_by: str,
    model_config: ModelBuildConfig | None = None,
) -> EditorPublicationResult:
    parent_kwargs: dict[str, Any] = {"allowed_root": allowed_root}
    if model_config is not None:
        parent_kwargs["model_config"] = model_config
    parent = load_parent_candidate(engine, submission, **parent_kwargs)
    exported = export_edited_model(
        parent,
        submission,
        allowed_root=allowed_root,
        write_dir=attempt.staging_dir,
        published_dir=attempt.final_dir,
    )
    os.rename(attempt.staging_dir, attempt.final_dir)
    try:
        stage_editor_export(engine, parent, exported, created_by)
        revision_metadata_json = _revision_with_publisher_identity(
            exported.revision_metadata_json,
            created_by,
        )

        def validate_draft(connection, rate_package_id: int) -> None:
            verify_package_sql_parity(
                connection,
                rate_package_id=rate_package_id,
                edited_model=exported.edited_model,
                bundle=exported.bundle,
            )

        model_run_id: int | None = None

        def write_package_lineage(connection, rate_package_id: int) -> None:
            nonlocal model_run_id
            model_run_id = record_derived_model_run(
                connection,
                dag_id=dag_id,
                airflow_run_id=airflow_run_id,
                mlflow_run_id=f"editor::{submission.submission_id}",
                manifest_id=submission.manifest_id,
                split_set_id=submission.split_set_id,
                export_id=exported.export_id,
                model_id=parent.model_id,
                model_name=parent.model_name,
                model_version=parent.model_version,
                rate_package_id=rate_package_id,
                rating_workbook_path=exported.rating_workbook_path,
                run_status="SUCCESS",
                created_by=created_by,
                publication_receipt_path=exported.publication_receipt_path,
                publication_receipt_sha256=exported.publication_receipt_sha256,
                candidate_artifact_path=exported.candidate_artifact_path,
                candidate_artifact_sha256=exported.candidate_artifact_sha256,
                candidate_artifact_format=exported.candidate_artifact_format,
                candidate_artifact_size_bytes=exported.candidate_artifact_size_bytes,
                candidate_python_version=exported.candidate_python_version,
                candidate_superglm_version=exported.candidate_superglm_version,
                model_source_sha256=submission.model_source_sha256,
                metrics=exported.metrics,
                metric_scopes=exported.metric_scopes,
                parent_model_run_id=submission.parent_model_run_id,
            )

        published = publish_rating_package(
            engine,
            export_id=exported.export_id,
            created_by=created_by,
            package_status=parent.config.default_package_status,
            parent_rate_package_id=submission.parent_rate_package_id,
            revision_metadata_json=revision_metadata_json,
            draft_validator=validate_draft,
            package_lineage_writer=write_package_lineage,
            expected_staged_metadata={
                "export_id": exported.export_id,
                "model_id": parent.model_id,
                "model_name": parent.model_name,
                "model_version": parent.model_version,
                "effective_from_date": parent.effective_from,
                "effective_to_date": parent.effective_to,
                "source_file": str(Path(exported.rating_workbook_path).resolve()),
                "publication_receipt_sha256": exported.publication_receipt_sha256,
            },
        )
    except BaseException:
        _remove_path(attempt.final_dir)
        raise
    if model_run_id is None:
        raise RuntimeError("package publication did not record editor lineage")
    return EditorPublicationResult(
        submission_id=submission.submission_id,
        model_name=parent.model_name,
        parent_rate_package_id=submission.parent_rate_package_id,
        rate_package_id=published.rate_package_id,
        package_version=published.package_version,
        model_run_id=model_run_id,
        package_status=published.package_status,
        was_existing=published.was_existing,
    )


def publish_editor_submission(
    engine,
    *,
    settings: Settings,
    submission_path: str,
    submission_sha256: str,
    dag_id: str,
    airflow_run_id: str,
    created_by: str,
    model_config: ModelBuildConfig | None = None,
) -> EditorPublicationResult:
    submission = load_verified_submission(
        submission_path,
        submission_sha256,
        allowed_root=settings.workbench_artifact_root,
    )
    submission_dir = _submission_directory(
        submission,
        allowed_root=settings.workbench_artifact_root,
    )
    with _editor_publication_lock(submission_dir):
        existing = _resolve_existing_editor_publication(
            engine,
            submission,
            allowed_root=settings.workbench_artifact_root,
        )
        if existing is not None:
            return existing
        _remove_unpublished_editor_attempts(submission_dir)
        attempt = _new_editor_publication_attempt(submission_dir)
        try:
            publish_kwargs: dict[str, Any] = {
                "submission": submission,
                "allowed_root": settings.workbench_artifact_root,
                "attempt": attempt,
                "dag_id": dag_id,
                "airflow_run_id": airflow_run_id,
                "created_by": created_by,
            }
            if model_config is not None:
                publish_kwargs["model_config"] = model_config
            return _publish_new_editor_submission(engine, **publish_kwargs)
        finally:
            _remove_path(attempt.staging_dir)
