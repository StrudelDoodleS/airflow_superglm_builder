from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from pricing_pipeline.workbench.airflow import AirflowClient

if TYPE_CHECKING:
    from pricing_pipeline.workbench.core import Candidate


SUBMISSION_FORMAT = "superglm-editor-submission-v1"
EDITED_MODEL_FORMAT = "superglm-edited-model-joblib-v1"
EDITOR_DAG_ID = "pricing_publish_editor_candidate"


class EditorSubmissionError(RuntimeError):
    """Raised when an editor submission is incomplete or fails verification."""


@dataclass
class EditorSubmission:
    submission_id: str
    model_name: str
    source_package_version: int
    parent_rate_package_id: int
    parent_model_run_id: int
    manifest_id: str
    split_set_id: str | None
    reason: str
    claimed_identity: str
    created_at: str
    editor_session_path: str
    editor_session_sha256: str
    editor_session_size_bytes: int
    edited_model_path: str
    edited_model_sha256: str
    edited_model_size_bytes: int
    edited_model_format: str
    edited_model_python_version: str
    edited_model_superglm_version: str
    baseline_candidate_path: str
    baseline_candidate_sha256: str
    baseline_candidate_format: str | None
    baseline_candidate_size_bytes: int | None
    baseline_candidate_python_version: str | None
    baseline_candidate_superglm_version: str | None
    model_source_sha256: str
    path: str
    sha256: str
    dag_id: str = EDITOR_DAG_ID
    dag_run_id: str | None = None
    state: str = "saved"
    _airflow_client: AirflowClient | Any | None = field(default=None, repr=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": SUBMISSION_FORMAT,
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "source_package_version": self.source_package_version,
            "parent_rate_package_id": self.parent_rate_package_id,
            "parent_model_run_id": self.parent_model_run_id,
            "manifest_id": self.manifest_id,
            "split_set_id": self.split_set_id,
            "reason": self.reason,
            "claimed_identity": self.claimed_identity,
            "created_at": self.created_at,
            "editor_session_path": self.editor_session_path,
            "editor_session_sha256": self.editor_session_sha256,
            "editor_session_size_bytes": self.editor_session_size_bytes,
            "edited_model_path": self.edited_model_path,
            "edited_model_sha256": self.edited_model_sha256,
            "edited_model_size_bytes": self.edited_model_size_bytes,
            "edited_model_format": self.edited_model_format,
            "edited_model_python_version": self.edited_model_python_version,
            "edited_model_superglm_version": self.edited_model_superglm_version,
            "baseline_candidate_path": self.baseline_candidate_path,
            "baseline_candidate_sha256": self.baseline_candidate_sha256,
            "baseline_candidate_format": self.baseline_candidate_format,
            "baseline_candidate_size_bytes": self.baseline_candidate_size_bytes,
            "baseline_candidate_python_version": self.baseline_candidate_python_version,
            "baseline_candidate_superglm_version": self.baseline_candidate_superglm_version,
            "model_source_sha256": self.model_source_sha256,
        }


def _superglm_version() -> str:
    try:
        return version("superglm")
    except PackageNotFoundError:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_edited_model(model: Any, path: Path) -> tuple[str, int, str, str]:
    python_version = platform.python_version()
    superglm_version = _superglm_version()
    envelope = {
        "format": EDITED_MODEL_FORMAT,
        "python_version": python_version,
        "superglm_version": superglm_version,
        "model": model,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(envelope, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path), path.stat().st_size, python_version, superglm_version


def _submission_id(
    *,
    parent_rate_package_id: int,
    editor_session_sha256: str,
    edited_model_sha256: str,
) -> str:
    identity = json.dumps(
        {
            "parent_rate_package_id": int(parent_rate_package_id),
            "editor_session_sha256": editor_session_sha256,
            "edited_model_sha256": edited_model_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"submission-{hashlib.sha256(identity).hexdigest()[:24]}"


def create_editor_submission(
    candidate: Candidate,
    *,
    editor_session: Any,
    reason: str,
    airflow_client: AirflowClient | Any,
    claimed_identity: str = "prototype-local-not-authenticated",
) -> EditorSubmission:
    cleaned_reason = str(reason).strip()
    if not cleaned_reason:
        raise ValueError("A non-empty reason is required to submit editor changes")

    root = Path(candidate.workbench.settings.workbench_artifact_root).expanduser().resolve()
    submissions_root = root / candidate.model_name / "editor_submissions"
    submissions_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".submission-", dir=submissions_root))
    final_directory: Path | None = None

    try:
        staged_session_path = staging / "editor_session.json"
        editor_session.save(staged_session_path)
        session_sha256 = sha256_file(staged_session_path)
        session_size = staged_session_path.stat().st_size

        edited_model = editor_session.to_model(
            X=candidate.bundle.X,
            y=candidate.bundle.y,
            sample_weight=candidate.bundle.sample_weight,
            offset=candidate.bundle.offset,
        )
        staged_model_path = staging / "edited_model.joblib"
        model_sha256, model_size, python_version, superglm_version = _save_edited_model(
            edited_model,
            staged_model_path,
        )
        submission_id = _submission_id(
            parent_rate_package_id=candidate.rate_package_id,
            editor_session_sha256=session_sha256,
            edited_model_sha256=model_sha256,
        )
        final_directory = submissions_root / submission_id
        if final_directory.exists():
            raise EditorSubmissionError(
                f"submission {submission_id} already exists; use its existing handle"
            )
        final_directory.mkdir()

        final_session_path = final_directory / "editor_session.json"
        editor_session.save(final_session_path)
        if sha256_file(final_session_path) != session_sha256:
            raise EditorSubmissionError("editor session changed while the submission was saved")
        if final_session_path.stat().st_size != session_size:
            raise EditorSubmissionError("editor session byte size changed while saving")

        final_model_path = final_directory / "edited_model.joblib"
        os.replace(staged_model_path, final_model_path)
        submission_path = final_directory / "submission.json"
        technical = candidate.technical
        submission = EditorSubmission(
            submission_id=submission_id,
            model_name=candidate.model_name,
            source_package_version=candidate.package_version,
            parent_rate_package_id=candidate.rate_package_id,
            parent_model_run_id=candidate.model_run_id,
            manifest_id=candidate.bundle.manifest_id,
            split_set_id=candidate.bundle.split_set_id,
            reason=cleaned_reason,
            claimed_identity=claimed_identity,
            created_at=datetime.now(UTC).isoformat(),
            editor_session_path=str(final_session_path),
            editor_session_sha256=session_sha256,
            editor_session_size_bytes=session_size,
            edited_model_path=str(final_model_path),
            edited_model_sha256=model_sha256,
            edited_model_size_bytes=model_size,
            edited_model_format=EDITED_MODEL_FORMAT,
            edited_model_python_version=python_version,
            edited_model_superglm_version=superglm_version,
            baseline_candidate_path=str(technical.get("candidate_artifact_path") or ""),
            baseline_candidate_sha256=str(
                technical.get("candidate_artifact_sha256") or ""
            ),
            baseline_candidate_format=technical.get("candidate_artifact_format"),
            baseline_candidate_size_bytes=technical.get("candidate_artifact_size_bytes"),
            baseline_candidate_python_version=technical.get("candidate_python_version"),
            baseline_candidate_superglm_version=technical.get(
                "candidate_superglm_version"
            ),
            model_source_sha256=candidate.bundle.model_source_sha256,
            path=str(submission_path),
            sha256="",
            _airflow_client=airflow_client,
        )
        _write_json_atomic(submission.to_payload(), submission_path)
        submission.sha256 = sha256_file(submission_path)
    except BaseException:
        if final_directory is not None:
            shutil.rmtree(final_directory, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    run = airflow_client.trigger_dag(
        EDITOR_DAG_ID,
        run_id=f"manual__{submission.submission_id}",
        conf={
            "submission_path": submission.path,
            "submission_sha256": submission.sha256,
        },
    )
    submission.dag_id = run.dag_id
    submission.dag_run_id = run.dag_run_id
    submission.state = run.state
    return submission


def load_verified_submission(
    path: str | Path,
    expected_sha256: str,
    *,
    allowed_root: str | Path | None = None,
) -> EditorSubmission:
    submission_path = Path(path).expanduser().resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if not submission_path.is_relative_to(root):
            raise EditorSubmissionError(
                f"submission is outside configured artifact root {root}: {submission_path}"
            )
    if not submission_path.is_file():
        raise EditorSubmissionError(f"submission does not exist: {submission_path}")
    actual_sha256 = sha256_file(submission_path)
    if actual_sha256 != expected_sha256:
        raise EditorSubmissionError("submission SHA-256 does not match the trigger metadata")
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    if payload.pop("format", None) != SUBMISSION_FORMAT:
        raise EditorSubmissionError("submission has an unsupported format")

    session_path = Path(payload["editor_session_path"]).expanduser().resolve()
    model_path = Path(payload["edited_model_path"]).expanduser().resolve()
    for artifact_path in (session_path, model_path):
        if allowed_root is not None and not artifact_path.is_relative_to(root):
            raise EditorSubmissionError(f"submission artifact is outside {root}: {artifact_path}")
        if not artifact_path.is_file():
            raise EditorSubmissionError(f"submission artifact does not exist: {artifact_path}")
    if sha256_file(session_path) != payload["editor_session_sha256"]:
        raise EditorSubmissionError("editor session SHA-256 verification failed")
    if session_path.stat().st_size != int(payload["editor_session_size_bytes"]):
        raise EditorSubmissionError("editor session byte-size verification failed")
    if sha256_file(model_path) != payload["edited_model_sha256"]:
        raise EditorSubmissionError("edited model SHA-256 verification failed")
    if model_path.stat().st_size != int(payload["edited_model_size_bytes"]):
        raise EditorSubmissionError("edited model byte-size verification failed")

    return EditorSubmission(
        **payload,
        path=str(submission_path),
        sha256=actual_sha256,
    )
