from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import platform
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

import joblib

from pricing_pipeline.workbench.airflow import AirflowClient

if TYPE_CHECKING:
    from pricing_pipeline.workbench.core import Candidate


SUBMISSION_FORMAT = "superglm-editor-submission-v1"
EDITED_MODEL_FORMAT = "superglm-edited-model-joblib-v1"
EDITOR_DAG_ID = "pricing_publish_editor_candidate"


class EditorSubmissionError(RuntimeError):
    """Raised when an editor submission is incomplete or fails verification."""


@dataclass(frozen=True)
class SubmissionStatus:
    state: str
    model_name: str
    source_package_version: int
    package_version: int | None
    rate_package_id: int | None
    model_run_id: int | None
    airflow_url: str
    message: str | None = None


@dataclass
class EditorSubmission:
    submission_id: str
    model_name: str
    deployment_slot: str
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
    _workbench: Any | None = field(default=None, repr=False)
    published_package_version: int | None = field(default=None, repr=False)
    published_rate_package_id: int | None = field(default=None, repr=False)
    published_model_run_id: int | None = field(default=None, repr=False)
    reviewed_champion_status: str | None = field(default=None, repr=False)
    reviewed_champion_rate_package_id: int | None = field(default=None, repr=False)
    reviewed_deployment_slot: str | None = field(default=None, repr=False)
    reviewed_champion_reason: str | None = field(default=None, repr=False)

    @property
    def parent_package_version(self) -> int:
        return self.source_package_version

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": SUBMISSION_FORMAT,
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "deployment_slot": self.deployment_slot,
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

    def status(self) -> SubmissionStatus:
        if self._airflow_client is None or self.dag_run_id is None:
            raise RuntimeError("This submission has no live Airflow run handle")
        run = self._airflow_client.get_dag_run(self.dag_id, self.dag_run_id)
        airflow_url = self._airflow_client.dag_run_ui_url(self.dag_id, self.dag_run_id)
        triggering_user_name = (run.payload or {}).get("triggering_user_name")
        if triggering_user_name and str(triggering_user_name).strip():
            self.claimed_identity = str(triggering_user_name).strip()
        state = str(run.state).lower()
        message = None
        if state in {"queued", "scheduled", "deferred", "up_for_retry"}:
            friendly_state = "queued"
        elif state in {"running", "restarting"}:
            friendly_state = "running"
        elif state == "success":
            if self._workbench is None:
                raise RuntimeError("This submission cannot resolve its published SQL lineage")
            resolved = self._workbench.resolve_editor_publication(self)
            self.published_package_version = int(resolved["package_version"])
            self.published_rate_package_id = int(resolved["rate_package_id"])
            self.published_model_run_id = int(resolved["model_run_id"])
            self.reviewed_champion_status = str(resolved["reviewed_champion_status"])
            reviewed_rate_package_id = resolved["reviewed_champion_rate_package_id"]
            self.reviewed_champion_rate_package_id = (
                None
                if reviewed_rate_package_id is None
                else int(reviewed_rate_package_id)
            )
            self.reviewed_deployment_slot = str(resolved["reviewed_deployment_slot"])
            reviewed_reason = resolved.get("reviewed_champion_reason")
            self.reviewed_champion_reason = (
                None if reviewed_reason is None else str(reviewed_reason)
            )
            friendly_state = "published"
        elif state in {"failed", "upstream_failed"}:
            friendly_state = "failed"
            message = "Airflow could not publish the edited candidate"
        else:
            friendly_state = state
        self.state = friendly_state
        return SubmissionStatus(
            state=friendly_state,
            model_name=self.model_name,
            source_package_version=self.source_package_version,
            package_version=self.published_package_version,
            rate_package_id=self.published_rate_package_id,
            model_run_id=self.published_model_run_id,
            airflow_url=airflow_url,
            message=message,
        )

    def request_deployment(
        self,
        *,
        reason: str,
        deployment_slot: str | None = None,
    ):
        cleaned_reason = str(reason).strip()
        if not cleaned_reason:
            raise ValueError("A non-empty reason is required to request deployment")
        if self.state != "published" or self.published_package_version is None:
            raise RuntimeError("The edited candidate must be published before deployment")
        if self._airflow_client is None:
            raise RuntimeError("This submission has no live Airflow client")
        status = self.reviewed_champion_status
        if status not in {"COMPARED", "NO_CHAMPION", "UNAVAILABLE"}:
            raise RuntimeError("The edited candidate has no valid reviewed champion evidence")
        if status == "UNAVAILABLE":
            detail = self.reviewed_champion_reason or "comparison artifact was unavailable"
            raise RuntimeError(f"Champion comparison was unavailable: {detail}")
        reviewed_slot = str(self.reviewed_deployment_slot or "").strip().upper()
        if not reviewed_slot:
            raise RuntimeError("The edited candidate has no reviewed deployment slot")
        requested_slot = str(deployment_slot or reviewed_slot).strip().upper()
        if requested_slot != reviewed_slot:
            raise ValueError(
                "deployment_slot does not match the slot used for champion comparison"
            )
        expected_current_rate_package_id = self.reviewed_champion_rate_package_id
        if status == "COMPARED" and expected_current_rate_package_id is None:
            raise RuntimeError("Compared champion evidence has no rate package ID")
        if status == "NO_CHAMPION" and expected_current_rate_package_id is not None:
            raise RuntimeError("NO_CHAMPION evidence unexpectedly identifies a rate package")
        slot = reviewed_slot
        deployment_identity = json.dumps(
            {
                "submission_id": self.submission_id,
                "package_version": self.published_package_version,
                "expected_current_rate_package_id": expected_current_rate_package_id,
                "deployment_slot": slot,
                "reason": cleaned_reason,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        suffix = hashlib.sha256(deployment_identity).hexdigest()[:12]
        return self._airflow_client.trigger_dag(
            "pricing_deploy_rate_package",
            run_id=f"manual__deploy__{self.submission_id}__{suffix}",
            conf={
                "model_name": self.model_name,
                "package_version": self.published_package_version,
                "expected_current_rate_package_id": expected_current_rate_package_id,
                "deployment_slot": slot,
                "deployment_reason": cleaned_reason,
            },
        )


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


def _trigger_editor_dag(
    submission: EditorSubmission,
    airflow_client: AirflowClient | Any,
) -> EditorSubmission:
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


def _reuse_existing_submission(
    path: Path,
    *,
    root: Path,
    candidate: Candidate,
    reason: str,
    claimed_identity: str,
    deployment_slot: str,
    editor_session_sha256: str,
    editor_session_size_bytes: int,
    edited_model_sha256: str,
    edited_model_size_bytes: int,
    edited_model_python_version: str,
    edited_model_superglm_version: str,
    airflow_client: AirflowClient | Any,
) -> EditorSubmission:
    if not path.is_file():
        raise EditorSubmissionError(
            f"submission directory already exists without a submission record: {path.parent}"
        )
    existing = load_verified_submission(
        path,
        sha256_file(path),
        allowed_root=root,
    )
    technical = candidate.technical
    expected = {
        "submission_id": path.parent.name,
        "model_name": candidate.model_name,
        "deployment_slot": deployment_slot,
        "source_package_version": candidate.package_version,
        "parent_rate_package_id": candidate.rate_package_id,
        "parent_model_run_id": candidate.model_run_id,
        "manifest_id": candidate.bundle.manifest_id,
        "split_set_id": candidate.bundle.split_set_id,
        "reason": reason,
        "claimed_identity": claimed_identity,
        "editor_session_sha256": editor_session_sha256,
        "editor_session_size_bytes": editor_session_size_bytes,
        "editor_session_path": str(path.parent / "editor_session.json"),
        "edited_model_sha256": edited_model_sha256,
        "edited_model_size_bytes": edited_model_size_bytes,
        "edited_model_path": str(path.parent / "edited_model.joblib"),
        "edited_model_format": EDITED_MODEL_FORMAT,
        "edited_model_python_version": edited_model_python_version,
        "edited_model_superglm_version": edited_model_superglm_version,
        "baseline_candidate_path": str(technical.get("candidate_artifact_path") or ""),
        "baseline_candidate_sha256": str(
            technical.get("candidate_artifact_sha256") or ""
        ),
        "baseline_candidate_format": technical.get("candidate_artifact_format"),
        "baseline_candidate_size_bytes": technical.get("candidate_artifact_size_bytes"),
        "baseline_candidate_python_version": technical.get("candidate_python_version"),
        "baseline_candidate_superglm_version": technical.get(
            "candidate_superglm_version"
        ),
        "model_source_sha256": candidate.bundle.model_source_sha256,
    }
    conflicts = [
        name for name, value in expected.items() if getattr(existing, name) != value
    ]
    if conflicts:
        raise EditorSubmissionError(
            f"submission {existing.submission_id} already exists with incompatible "
            "metadata: " + ", ".join(conflicts)
        )
    existing._airflow_client = airflow_client
    existing._workbench = candidate.workbench
    return existing


def _submission_tree_is_complete(directory: Path) -> bool:
    return directory.is_dir() and all(
        (directory / name).is_file()
        for name in ("editor_session.json", "edited_model.joblib", "submission.json")
    )


def _quarantine_incomplete_submission(
    directory: Path,
    *,
    submissions_root: Path,
    submission_id: str,
) -> bool:
    if directory.parent != submissions_root or directory.name != submission_id:
        raise EditorSubmissionError(
            f"refusing to recover submission path outside reserved directory: {directory}"
        )
    quarantine = submissions_root / f".incomplete-{submission_id}-{uuid4().hex}"
    try:
        os.rename(directory, quarantine)
    except FileNotFoundError:
        return False
    if quarantine.is_symlink() or not quarantine.is_dir():
        quarantine.unlink(missing_ok=True)
    else:
        shutil.rmtree(quarantine)
    return True


@contextmanager
def _submission_root_lock(submissions_root: Path) -> Iterator[None]:
    descriptor = os.open(submissions_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _promote_or_reuse_submission(
    staging: Path,
    final_directory: Path,
    *,
    root: Path,
    candidate: Candidate,
    reason: str,
    claimed_identity: str,
    deployment_slot: str,
    editor_session_sha256: str,
    editor_session_size_bytes: int,
    edited_model_sha256: str,
    edited_model_size_bytes: int,
    edited_model_python_version: str,
    edited_model_superglm_version: str,
    airflow_client: AirflowClient | Any,
) -> EditorSubmission | None:
    with _submission_root_lock(final_directory.parent):
        while True:
            try:
                os.rename(staging, final_directory)
                return None
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise

            if _submission_tree_is_complete(final_directory):
                return _reuse_existing_submission(
                    final_directory / "submission.json",
                    root=root,
                    candidate=candidate,
                    reason=reason,
                    claimed_identity=claimed_identity,
                    deployment_slot=deployment_slot,
                    editor_session_sha256=editor_session_sha256,
                    editor_session_size_bytes=editor_session_size_bytes,
                    edited_model_sha256=edited_model_sha256,
                    edited_model_size_bytes=edited_model_size_bytes,
                    edited_model_python_version=edited_model_python_version,
                    edited_model_superglm_version=edited_model_superglm_version,
                    airflow_client=airflow_client,
                )
            _quarantine_incomplete_submission(
                final_directory,
                submissions_root=final_directory.parent,
                submission_id=final_directory.name,
            )


def _submissions_root(root: Path, model_name: str) -> Path:
    submissions_root = (root / model_name / "editor_submissions").resolve()
    if not submissions_root.is_relative_to(root):
        raise EditorSubmissionError(
            f"submission directory is outside configured artifact root {root}: "
            f"{submissions_root}"
        )
    return submissions_root


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
    submissions_root = _submissions_root(root, candidate.model_name)
    submissions_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".submission-", dir=submissions_root))

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
        deployment_slot = candidate.workbench.model_config(
            candidate.model_name
        ).deployment_slot
        final_session_path = final_directory / "editor_session.json"
        final_model_path = final_directory / "edited_model.joblib"
        submission_path = final_directory / "submission.json"
        technical = candidate.technical
        submission = EditorSubmission(
            submission_id=submission_id,
            model_name=candidate.model_name,
            deployment_slot=deployment_slot,
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
            _workbench=candidate.workbench,
        )
        staged_submission_path = staging / "submission.json"
        _write_json_atomic(submission.to_payload(), staged_submission_path)
        submission.sha256 = sha256_file(staged_submission_path)
        existing = _promote_or_reuse_submission(
            staging,
            final_directory,
            root=root,
            candidate=candidate,
            reason=cleaned_reason,
            claimed_identity=claimed_identity,
            deployment_slot=deployment_slot,
            editor_session_sha256=session_sha256,
            editor_session_size_bytes=session_size,
            edited_model_sha256=model_sha256,
            edited_model_size_bytes=model_size,
            edited_model_python_version=python_version,
            edited_model_superglm_version=superglm_version,
            airflow_client=airflow_client,
        )
        if existing is not None:
            submission = existing
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return _trigger_editor_dag(submission, airflow_client)


def load_verified_submission(
    path: str | Path,
    expected_sha256: str,
    *,
    allowed_root: str | Path,
) -> EditorSubmission:
    submission_path = Path(path).expanduser().resolve()
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
        if not artifact_path.is_relative_to(root):
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
