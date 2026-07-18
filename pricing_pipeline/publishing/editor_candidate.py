from __future__ import annotations

import json
import math
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.file_lock import exclusive_file_lock
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.lineage import record_model_run
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.rating_export import export_rating_tables
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.publishing.superglm_metadata import build_superglm_publication_receipt
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
    write_publication_receipt,
)
from pricing_pipeline.workbench.artifacts import (
    CandidateArtifactError,
    CandidateArtifactMetadata,
    CandidateBundle,
    load_completed_build_candidate_bundle,
    load_candidate_bundle,
    save_candidate_bundle,
)
from pricing_pipeline.workbench.submission import (
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
    completed_build: ApprovedModelBuild
    publication_receipt: SuperGLMPublicationReceipt
    revision_metadata: dict[str, Any]
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


def publish_editor_submission(
    engine,
    *,
    settings: Settings,
    submission_path: str,
    submission_sha256: str,
    dag_id: str,
    airflow_run_id: str,
    created_by: str,
    model_config: ModelBuildConfig,
) -> EditorPublicationResult:
    publisher_identity = str(created_by).strip()
    if not publisher_identity:
        raise EditorSubmissionError("publisher identity is required")
    submission = load_verified_submission(
        submission_path,
        submission_sha256,
        allowed_root=settings.workbench_artifact_root,
    )
    submitted_slot = str(submission.deployment_slot or "").strip().upper()
    configured_slot = str(model_config.deployment_slot or "").strip().upper()
    if not submitted_slot or submitted_slot != configured_slot:
        raise EditorSubmissionError(
            "explicit model config deployment_slot does not match the editor submission"
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
            result = _publish_new_editor_submission(
                engine,
                submission=submission,
                allowed_root=settings.workbench_artifact_root,
                attempt=attempt,
                dag_id=dag_id,
                airflow_run_id=airflow_run_id,
                created_by=publisher_identity,
                model_config=model_config,
                expected_database=settings.pricing_database,
            )
        except BaseException as exc:
            _remove_path_preserving(
                attempt.staging_dir,
                allowed_root=submission_dir,
                original=exc,
            )
            _remove_empty_attempt_roots(attempt, original=exc)
            raise
        _remove_path(attempt.staging_dir, allowed_root=submission_dir)
        _remove_empty_attempt_roots(attempt)
        return result


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


def _verify_model_frame_sha256(
    bundle: CandidateBundle,
    sql_digest: Any,
    *,
    context: str,
) -> None:
    if (
        not isinstance(sql_digest, str)
        or len(sql_digest) != 64
        or any(character not in "0123456789abcdef" for character in sql_digest)
    ):
        raise EditorSubmissionError(
            f"{context} SQL manifest model_frame_sha256 is missing or invalid"
        )
    if bundle.model_frame_sha256 is None:
        raise EditorSubmissionError(f"{context} bundle model_frame_sha256 is missing")
    if bundle.model_frame_sha256 != sql_digest:
        raise EditorSubmissionError(
            f"{context} bundle model_frame_sha256 does not match the SQL manifest"
        )


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
            mr.parent_model_run_id,
            mr.run_status,
            mr.model_version,
            mr.export_id,
            mr.manifest_id,
            manifest.model_frame_sha256,
            mr.rating_workbook_path,
            mr.rating_workbook_sha256,
            split_link.split_set_id,
            mr.model_source_sha256,
            mr.candidate_artifact_path,
            mr.candidate_artifact_sha256,
            mr.candidate_artifact_format,
            mr.candidate_artifact_size_bytes,
            mr.candidate_python_version,
            mr.candidate_superglm_version,
            mr.candidate_superglm_git_sha
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr
          ON mr.rate_package_id = rp.rate_package_id
        LEFT JOIN {schemas.pricing}.DATASET_MANIFEST AS manifest
          ON manifest.manifest_id = mr.manifest_id
        LEFT JOIN {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link
          ON split_link.model_run_id = mr.model_run_id
         AND split_link.manifest_id = mr.manifest_id
         AND split_link.dataset_role = 'training'
         AND split_link.split_role = 'validation'
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
    workbook_path = Path(str(row.get("rating_workbook_path") or "")).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not workbook_path.is_relative_to(root) or not workbook_path.is_file():
        raise EditorSubmissionError(
            "existing editor publication rating workbook is missing or outside the artifact root"
        )
    expected_workbook_sha256 = str(row.get("rating_workbook_sha256") or "")
    if sha256_file(workbook_path) != expected_workbook_sha256:
        raise EditorSubmissionError(
            "existing editor publication rating workbook SHA-256 verification failed"
        )
    artifact_fields = (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "candidate_superglm_git_sha",
    )
    if any(row.get(field) is None for field in artifact_fields):
        raise EditorSubmissionError(
            "editor publication requires lineage repair: candidate artifact metadata is incomplete"
        )
    expected_lineage = {
        "model_name": submission.model_name,
        "export_id": _editor_export_id(submission),
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "model_source_sha256": submission.model_source_sha256,
    }
    expected_sql_lineage = {
        **expected_lineage,
        "parent_model_run_id": submission.parent_model_run_id,
    }
    sql_mismatches = [
        field for field, expected in expected_sql_lineage.items() if row.get(field) != expected
    ]
    if sql_mismatches:
        raise EditorSubmissionError(
            "existing editor publication SQL lineage does not match the submission: "
            + ", ".join(sql_mismatches)
        )
    try:
        bundle = load_candidate_bundle(
            row["candidate_artifact_path"],
            expected_sha256=row["candidate_artifact_sha256"],
            expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
            expected_format=row["candidate_artifact_format"],
            expected_python_version=row["candidate_python_version"],
            expected_superglm_version=row["candidate_superglm_version"],
            expected_superglm_git_sha=row["candidate_superglm_git_sha"],
            allowed_root=allowed_root,
        )
    except CandidateArtifactError as exc:
        raise EditorSubmissionError(
            f"existing editor publication candidate artifact failed verification: {exc}"
        ) from exc
    expected_bundle = {
        **expected_lineage,
        "model_version": row.get("model_version"),
    }
    bundle_mismatches = [
        field for field, expected in expected_bundle.items() if getattr(bundle, field) != expected
    ]
    if bundle_mismatches:
        raise EditorSubmissionError(
            "existing editor publication bundle lineage does not match the submission: "
            + ", ".join(bundle_mismatches)
        )
    _verify_model_frame_sha256(
        bundle,
        row.get("model_frame_sha256"),
        context="existing editor publication",
    )
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


def _remove_unpublished_editor_attempts(submission_dir: Path) -> None:
    published_root = submission_dir / "published"
    for root_name in (".staging", "attempts"):
        root = published_root / root_name
        if not root.is_dir():
            continue
        for child in root.iterdir():
            _remove_path(child, allowed_root=published_root)


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


def _remove_path(path: Path, *, allowed_root: str | Path) -> None:
    root = Path(allowed_root).expanduser().resolve()
    target = Path(os.path.abspath(Path(path).expanduser()))
    resolved_parent = target.parent.resolve()
    if target == root or not resolved_parent.is_relative_to(root):
        raise EditorSubmissionError(
            f"refusing cleanup outside the editor attempt root {root}: {target}"
        )
    if target.is_symlink():
        target.unlink(missing_ok=True)
        return
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(root):
        raise EditorSubmissionError(
            f"refusing cleanup outside the editor attempt root {root}: {resolved_target}"
        )
    if resolved_target.is_file():
        resolved_target.unlink(missing_ok=True)
    elif resolved_target.is_dir():
        shutil.rmtree(resolved_target)


def _remove_path_preserving(
    path: Path,
    *,
    allowed_root: str | Path,
    original: BaseException,
) -> None:
    try:
        _remove_path(path, allowed_root=allowed_root)
    except BaseException as cleanup_exc:
        original.add_note(f"failed to remove editor attempt path {path}: {cleanup_exc!r}")


def _remove_empty_attempt_roots(
    attempt: EditorPublicationAttempt,
    *,
    original: BaseException | None = None,
) -> None:
    roots = (attempt.staging_dir.parent, attempt.final_dir.parent)
    for root in roots:
        _remove_empty_directory(root, original=original)
    published_root = attempt.staging_dir.parent.parent
    if published_root.name == "published" and attempt.final_dir.parent.parent == published_root:
        _remove_empty_directory(published_root, original=original)


def _remove_empty_directory(
    path: Path,
    *,
    original: BaseException | None,
) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError as cleanup_exc:
        try:
            is_non_empty = any(path.iterdir())
        except BaseException as inspect_exc:
            if original is None:
                raise
            original.add_note(f"failed to inspect editor attempt directory {path}: {inspect_exc!r}")
            return
        if is_non_empty:
            return
        if original is None:
            raise
        original.add_note(
            f"failed to remove empty editor attempt directory {path}: {cleanup_exc!r}"
        )


def load_parent_candidate(
    engine,
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
    model_config: ModelBuildConfig,
) -> ParentCandidate:
    submitted_slot = str(submission.deployment_slot or "").strip().upper()
    configured_slot = str(model_config.deployment_slot or "").strip().upper()
    if not submitted_slot or submitted_slot != configured_slot:
        raise EditorSubmissionError(
            "explicit model config deployment_slot does not match the editor submission"
        )

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
            mr.model_version AS run_model_version,
            mr.export_id,
            mr.manifest_id,
            manifest.model_frame_sha256,
            split_link.split_set_id,
            mr.candidate_artifact_path,
            mr.candidate_artifact_sha256,
            mr.candidate_artifact_format,
            mr.candidate_artifact_size_bytes,
            mr.candidate_python_version,
            mr.candidate_superglm_version,
            mr.candidate_superglm_git_sha,
            mr.model_source_sha256
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        JOIN {schemas.pricing}.MODEL_RUN AS mr
          ON mr.rate_package_id = rp.rate_package_id
        JOIN {schemas.pricing}.DATASET_MANIFEST AS manifest
          ON manifest.manifest_id = mr.manifest_id
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
            f"parent package must resolve exactly one successful model run; found {len(rows)}"
        )
    row = dict(rows[0])
    expected = {
        "model_name": submission.model_name,
        "run_model_version": row["model_version"],
        "package_version": submission.source_package_version,
        "rate_package_id": submission.parent_rate_package_id,
        "model_run_id": submission.parent_model_run_id,
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "candidate_artifact_sha256": submission.baseline_candidate_sha256,
        "model_source_sha256": submission.model_source_sha256,
    }
    mismatches = [name for name, value in expected.items() if str(row.get(name)) != str(value)]
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
        expected_superglm_git_sha=row["candidate_superglm_git_sha"],
        allowed_root=allowed_root,
    )
    for field_name, expected_value in (
        ("model_name", row["model_name"]),
        ("model_version", row["run_model_version"]),
        ("export_id", row["export_id"]),
        ("manifest_id", submission.manifest_id),
        ("split_set_id", submission.split_set_id),
        ("model_source_sha256", submission.model_source_sha256),
    ):
        if getattr(bundle, field_name) != expected_value:
            raise EditorSubmissionError(
                f"parent bundle {field_name} does not match SQL/submission lineage"
            )
    _verify_model_frame_sha256(
        bundle,
        row.get("model_frame_sha256"),
        context="parent candidate",
    )
    config = model_config
    configured_name = config.model_name
    if str(configured_name) != submission.model_name:
        raise EditorSubmissionError(
            "explicit model config does not match the editor submission model_name"
        )
    champion = _load_champion_bundle(
        engine,
        model_id=int(row["model_id"]),
        deployment_slot=submitted_slot,
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
            None if row.get("effective_from_date") is None else str(row["effective_from_date"])
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
            mr.candidate_superglm_version,
            mr.candidate_superglm_git_sha
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
        "candidate_superglm_git_sha",
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
            expected_superglm_git_sha=row["candidate_superglm_git_sha"],
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
    try:
        champion_contract = OffsetExportContract.model_validate(champion.offset_contract)
        parent_contract = OffsetExportContract.model_validate(parent_bundle.offset_contract)
    except ValueError:
        return ChampionSnapshot(
            deployment_slot=deployment_slot,
            rate_package_id=rate_package_id,
            bundle=None,
            unavailable_reason="the deployed champion has an invalid offset contract",
        )
    if champion_contract != parent_contract:
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
    parent: ParentCandidate,
    submission: EditorSubmission,
    *,
    allowed_root: str | Path,
) -> Any:
    path = Path(submission.editor_session_path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not path.is_relative_to(root):
        raise EditorSubmissionError(f"editor session is outside artifact root {root}: {path}")
    if path.stat().st_size != int(submission.editor_session_size_bytes):
        raise EditorSubmissionError("editor session byte-size verification failed")
    if sha256_file(path) != submission.editor_session_sha256:
        raise EditorSubmissionError("editor session SHA-256 verification failed")
    from superglm.editor import EditorSession

    session = EditorSession.load(path, model=parent.bundle.fitted_model)
    return session.to_model(
        X=parent.bundle.X,
        y=parent.bundle.y,
        sample_weight=parent.bundle.sample_weight,
        offset=parent.bundle.offset,
    )


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
            "editor_training_deviance_delta" if name == "parent" else f"{prefix}_deviance_delta"
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


def export_edited_model(
    parent: ParentCandidate,
    submission: EditorSubmission,
    *,
    created_by: str,
    allowed_root: str | Path,
    write_dir: str | Path,
    published_dir: str | Path,
) -> EditorExport:
    edited_model = _load_edited_model(parent, submission, allowed_root=allowed_root)
    root = Path(allowed_root).expanduser().resolve()
    output_dir = Path(write_dir).expanduser().resolve()
    final_dir = Path(published_dir).expanduser().resolve()
    if not output_dir.is_relative_to(root) or not final_dir.is_relative_to(root):
        raise EditorSubmissionError("editor publication attempt is outside artifact root")
    if not output_dir.is_dir():
        raise EditorSubmissionError("editor publication staging directory does not exist")
    workbook_write_path = output_dir / "rating_tables.xlsx"
    workbook_path = final_dir / "rating_tables.xlsx"
    contract = parent.bundle.offset_contract
    export_options: dict[str, Any] = {}
    if parent.bundle.offset is not None:
        export_options["offset"] = parent.bundle.offset
    if contract.handling == "EXPORTED_FACTOR":
        export_options.update(
            offset_source=parent.bundle.offset_source,
            offset_name=contract.source_factor_name,
            offset_kind="auto",
        )
    export_rating_tables(
        edited_model,
        parent.bundle.X,
        parent.bundle.y,
        parent.bundle.export_weight,
        output_path=workbook_write_path,
        **export_options,
    )
    workbook_sha256 = sha256_file(workbook_write_path)
    receipt = build_superglm_publication_receipt(
        edited_model,
        offset_contract=contract,
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
    edited_bundle = replace(
        parent.bundle,
        fitted_model=edited_model,
        model_name=parent.model_name,
        model_version=parent.model_version,
        export_id=_editor_export_id(submission),
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
        "baseline_candidate_sha256": submission.baseline_candidate_sha256,
        "baseline_cv_metrics": {
            name: value for name, value in metrics.items() if name.startswith("cv_")
        },
        "comparison_metrics": {
            name: value for name, value in metrics.items() if name.startswith("editor_training_")
        },
        "champion_comparison": champion_comparison,
    }
    completed_build = ApprovedModelBuild(
        model_id=parent.model_id,
        model_name=parent.model_name,
        model_version=parent.model_version,
        model_type=parent.config.model_type,
        target_name=parent.config.target_name,
        deployment_slot=submission.deployment_slot,
        manifest_id=submission.manifest_id,
        split_set_id=submission.split_set_id,
        export_id=_editor_export_id(submission),
        rating_workbook_path=str(workbook_path),
        rating_workbook_sha256=workbook_sha256,
        effective_from=parent.effective_from,
        created_by=created_by,
        publication_receipt_path=str(receipt_path),
        publication_receipt_sha256=receipt_sha256,
        candidate_artifact_path=artifact.path,
        candidate_artifact_sha256=artifact.sha256,
        candidate_artifact_format=artifact.format,
        candidate_artifact_size_bytes=artifact.size_bytes,
        candidate_python_version=artifact.python_version,
        candidate_superglm_version=artifact.superglm_version,
        candidate_superglm_git_sha=artifact.superglm_git_sha,
        build_fingerprint_sha256=edited_bundle.build_fingerprint_sha256,
        builder_source_sha256=edited_bundle.builder_source_sha256,
        materialized_split_sha256=edited_bundle.materialized_split_sha256,
        runtime_sha256=edited_bundle.runtime_sha256,
        candidate_superglm_sha256=edited_bundle.candidate_superglm_sha256,
        row_order_sha256=edited_bundle.row_order_sha256,
        model_source_sha256=edited_bundle.model_source_sha256,
        model_frame_sha256=edited_bundle.model_frame_sha256,
        metrics=metrics,
        metric_scopes=metric_scopes,
    )
    return EditorExport(
        completed_build=completed_build,
        publication_receipt=receipt,
        revision_metadata=revision_metadata,
        edited_model=edited_model,
        bundle=edited_bundle,
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


def verify_package_sql_parity(
    connection,
    *,
    rate_package_id: int,
    edited_model: Any,
    bundle: CandidateBundle,
    publication_receipt: SuperGLMPublicationReceipt | None = None,
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
    contract = bundle.offset_contract
    sample_published_offset_source = (
        pd.Series(bundle.offset_source).reset_index(drop=True).iloc[:count]
        if contract.handling == "EXPORTED_FACTOR"
        else None
    )
    published_by_source: dict[str, str] = {}
    if publication_receipt is not None:
        for metadata in publication_receipt.term_metadata.values():
            feature_kind = metadata.get("feature_kind")
            if feature_kind == "categorical_interaction":
                source_names = metadata.get("parent_names")
                published_names = metadata.get("input_column_names")
                if isinstance(source_names, list | tuple) and isinstance(
                    published_names, list | tuple
                ):
                    published_by_source.update(
                        (str(source), str(published))
                        for source, published in zip(
                            source_names,
                            published_names,
                            strict=True,
                        )
                    )
                continue
            if feature_kind == "offset":
                continue
            source_name = metadata.get("source_term_name")
            published_name = metadata.get("published_term_name")
            if source_name is not None and published_name is not None:
                published_by_source[str(source_name)] = str(published_name)

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
        features = {
            published_by_source.get(str(name), str(name)): _json_value(value)
            for name, value in row.items()
        }
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


def _publish_new_editor_submission(
    engine,
    *,
    submission: EditorSubmission,
    allowed_root: str | Path,
    attempt: EditorPublicationAttempt,
    dag_id: str,
    airflow_run_id: str,
    created_by: str,
    model_config: ModelBuildConfig,
    expected_database: str,
) -> EditorPublicationResult:
    parent = load_parent_candidate(
        engine,
        submission,
        allowed_root=allowed_root,
        model_config=model_config,
    )
    exported = export_edited_model(
        parent,
        submission,
        created_by=created_by,
        allowed_root=allowed_root,
        write_dir=attempt.staging_dir,
        published_dir=attempt.final_dir,
    )
    build = exported.completed_build
    os.rename(attempt.staging_dir, attempt.final_dir)
    try:
        verified_bundle = _verify_edited_export_artifacts(
            exported,
            parent=parent,
            submission=submission,
            allowed_root=allowed_root,
        )
        if sha256_file(build.rating_workbook_path) != build.rating_workbook_sha256:
            raise EditorSubmissionError("edited rating workbook SHA-256 changed before staging")
        content_sha256 = stage_rating_export(
            engine,
            workbook_path=Path(build.rating_workbook_path),
            export_id=build.export_id,
            expected_database=expected_database,
            model_name=build.model_name,
            model_version=build.model_version,
            target_name=build.target_name,
            model_type=build.model_type,
            effective_from=build.effective_from,
            effective_to=parent.effective_to,
            created_by=build.created_by,
            replace=True,
            model_id=build.model_id,
            publication_receipt_path=build.publication_receipt_path,
            publication_receipt_sha256=build.publication_receipt_sha256,
        )
        if sha256_file(build.rating_workbook_path) != build.rating_workbook_sha256:
            raise EditorSubmissionError("edited rating workbook changed during staging")
        revision_metadata = {
            **exported.revision_metadata,
            "published_by": created_by,
        }

        def validate_draft(connection, rate_package_id: int) -> None:
            verify_package_sql_parity(
                connection,
                rate_package_id=rate_package_id,
                edited_model=exported.edited_model,
                bundle=verified_bundle,
                publication_receipt=exported.publication_receipt,
            )

        def write_package_lineage(connection, rate_package_id: int) -> int:
            return record_model_run(
                None,
                build=build,
                dag_id=dag_id,
                airflow_run_id=airflow_run_id,
                rate_package_id=rate_package_id,
                parent_model_run_id=submission.parent_model_run_id,
                connection=connection,
            )

        published = publish_rating_package(
            engine,
            export_id=build.export_id,
            expected_database=expected_database,
            created_by=build.created_by,
            parent_rate_package_id=submission.parent_rate_package_id,
            revision_metadata=revision_metadata,
            draft_validator=validate_draft,
            package_lineage_writer=write_package_lineage,
            expected_staged_metadata={
                "export_id": build.export_id,
                "model_id": build.model_id,
                "model_name": build.model_name,
                "model_version": build.model_version,
                "effective_from_date": build.effective_from,
                "effective_to_date": parent.effective_to,
                "source_file": str(Path(build.rating_workbook_path).resolve()),
                "publication_receipt_sha256": build.publication_receipt_sha256,
                "staging_content_sha256": content_sha256,
            },
        )
        if published.was_existing:
            existing = _resolve_existing_editor_publication(
                engine,
                submission,
                allowed_root=allowed_root,
            )
            if existing is None or existing.rate_package_id != published.rate_package_id:
                raise EditorSubmissionError(
                    "existing editor package changed before lineage validation"
                )
            _remove_path(attempt.final_dir, allowed_root=allowed_root)
            return existing
    except BaseException as exc:
        _remove_path_preserving(
            attempt.final_dir,
            allowed_root=allowed_root,
            original=exc,
        )
        raise
    if published.model_run_id is None:
        raise RuntimeError("package publication did not record editor lineage")
    return EditorPublicationResult(
        submission_id=submission.submission_id,
        model_name=build.model_name,
        parent_rate_package_id=submission.parent_rate_package_id,
        rate_package_id=published.rate_package_id,
        package_version=published.package_version,
        model_run_id=published.model_run_id,
        package_status=published.package_status,
        was_existing=published.was_existing,
    )


def _verify_edited_export_artifacts(
    exported: EditorExport,
    *,
    parent: ParentCandidate,
    submission: EditorSubmission,
    allowed_root: str | Path,
) -> CandidateBundle:
    build = exported.completed_build
    root = Path(allowed_root).expanduser().resolve()
    workbook_path = _verify_edited_file(
        build.rating_workbook_path,
        expected_sha256=build.rating_workbook_sha256,
        root=root,
        label="edited rating workbook",
    )
    receipt_path = _verify_edited_file(
        build.publication_receipt_path,
        expected_sha256=build.publication_receipt_sha256,
        root=root,
        label="edited publication receipt",
    )
    try:
        bundle = load_completed_build_candidate_bundle(
            build,
            allowed_root=root,
        )
    except CandidateArtifactError as exc:
        raise EditorSubmissionError(
            f"edited candidate artifact failed verification: {exc}"
        ) from exc

    artifact_path = Path(build.candidate_artifact_path).expanduser().resolve()
    if {workbook_path.parent, receipt_path.parent, artifact_path.parent} != {artifact_path.parent}:
        raise EditorSubmissionError(
            "edited candidate artifact files do not share one verified attempt directory"
        )

    build_lineage = {
        "model_id": parent.model_id,
        "model_name": parent.model_name,
        "model_version": parent.model_version,
        "model_type": parent.config.model_type,
        "target_name": parent.config.target_name,
        "deployment_slot": submission.deployment_slot,
        "export_id": _editor_export_id(submission),
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "model_source_sha256": submission.model_source_sha256,
    }
    build_mismatches = [
        field_name
        for field_name, expected in build_lineage.items()
        if getattr(build, field_name) != expected
    ]
    if build_mismatches:
        raise EditorSubmissionError(
            "edited candidate artifact build/SQL lineage mismatch: " + ", ".join(build_mismatches)
        )

    row_count = len(parent.bundle.X)
    row_roles = {
        "X": bundle.X,
        "y": bundle.y,
        "sample_weight": bundle.sample_weight,
        "offset": bundle.offset,
        "offset_source": bundle.offset_source,
        "export_weight": bundle.export_weight,
    }
    bad_lengths = [
        role
        for role, values in row_roles.items()
        if values is not None and len(values) != row_count
    ]
    if bad_lengths:
        raise EditorSubmissionError(
            "edited candidate artifact row count mismatch: " + ", ".join(bad_lengths)
        )
    if list(bundle.X.columns) != list(parent.bundle.X.columns):
        raise EditorSubmissionError(
            "edited candidate artifact feature columns do not match the verified parent"
        )
    if bundle.pk_columns != parent.bundle.pk_columns:
        raise EditorSubmissionError(
            "edited candidate artifact primary-key columns do not match the verified parent"
        )
    if bundle.offset_contract != parent.bundle.offset_contract:
        raise EditorSubmissionError(
            "edited candidate artifact offset contract does not match the verified parent"
        )
    return bundle


def _verify_edited_file(
    path: str | Path,
    *,
    expected_sha256: str,
    root: Path,
    label: str,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise EditorSubmissionError(f"{label} is outside configured artifact root {root}")
    if not resolved.is_file():
        raise EditorSubmissionError(f"{label} does not exist: {resolved}")
    if sha256_file(resolved) != expected_sha256:
        raise EditorSubmissionError(f"{label} SHA-256 verification failed")
    return resolved
