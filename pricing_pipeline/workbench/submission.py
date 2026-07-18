from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pricing_pipeline.infra.file_lock import exclusive_file_lock

if TYPE_CHECKING:
    from pricing_pipeline.workbench.core import Candidate


SUBMISSION_FORMAT = "superglm-editor-submission-v1"


class EditorSubmissionError(RuntimeError):
    """Raised when an editor submission is incomplete or fails verification."""


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
    baseline_candidate_sha256: str
    model_source_sha256: str
    path: str
    sha256: str

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
            "baseline_candidate_sha256": self.baseline_candidate_sha256,
            "model_source_sha256": self.model_source_sha256,
        }


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


def _submission_tree_is_complete(directory: Path) -> bool:
    return directory.is_dir() and all(
        (directory / name).is_file() for name in ("editor_session.json", "submission.json")
    )


def _quarantine_incomplete_submission(
    directory: Path,
    *,
    submissions_root: Path,
    submission_id: str,
) -> None:
    if directory.parent != submissions_root or directory.name != submission_id:
        raise EditorSubmissionError(
            f"refusing to recover submission path outside reserved directory: {directory}"
        )
    quarantine = submissions_root / f".incomplete-{submission_id}-{uuid4().hex}"
    try:
        os.rename(directory, quarantine)
    except FileNotFoundError:
        return
    if quarantine.is_symlink() or not quarantine.is_dir():
        quarantine.unlink(missing_ok=True)
    else:
        shutil.rmtree(quarantine)


def _promote_or_reuse_submission(
    staging: Path,
    final_directory: Path,
    *,
    root: Path,
    proposed: EditorSubmission,
) -> EditorSubmission:
    with exclusive_file_lock(final_directory.parent / ".submission.lock"):
        while True:
            try:
                os.rename(staging, final_directory)
                return proposed
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise

            if _submission_tree_is_complete(final_directory):
                submission_path = final_directory / "submission.json"
                existing = load_verified_submission(
                    submission_path,
                    sha256_file(submission_path),
                    allowed_root=root,
                )
                conflicts = [
                    field_name
                    for field_name in (
                        "submission_id",
                        "model_name",
                        "deployment_slot",
                        "source_package_version",
                        "parent_rate_package_id",
                        "parent_model_run_id",
                        "manifest_id",
                        "split_set_id",
                        "reason",
                        "claimed_identity",
                        "editor_session_path",
                        "editor_session_sha256",
                        "editor_session_size_bytes",
                        "baseline_candidate_sha256",
                        "model_source_sha256",
                    )
                    if getattr(existing, field_name) != getattr(proposed, field_name)
                ]
                if conflicts:
                    raise EditorSubmissionError(
                        f"submission {existing.submission_id} already exists with "
                        "incompatible metadata: " + ", ".join(conflicts)
                    )
                return existing
            _quarantine_incomplete_submission(
                final_directory,
                submissions_root=final_directory.parent,
                submission_id=final_directory.name,
            )


def save_editor_submission(
    candidate: Candidate,
    *,
    editor_session: Any,
    reason: str,
    claimed_identity: str,
) -> EditorSubmission:
    cleaned_reason = str(reason).strip()
    if not cleaned_reason:
        raise ValueError("A non-empty reason is required to submit editor changes")
    cleaned_identity = str(claimed_identity).strip()
    if not cleaned_identity:
        raise ValueError("A non-empty claimed_identity is required")
    if getattr(editor_session, "reference_model", None) is not candidate.bundle.fitted_model:
        raise EditorSubmissionError(
            "editor_session.reference_model must be the opened candidate fitted model"
        )

    root = Path(candidate.workbench.settings.workbench_artifact_root).expanduser().resolve()
    submissions_root = (root / candidate.model_name / "editor_submissions").resolve()
    if not submissions_root.is_relative_to(root):
        raise EditorSubmissionError(
            f"submission directory is outside configured artifact root {root}: {submissions_root}"
        )
    submissions_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".submission-", dir=submissions_root))

    try:
        staged_session_path = staging / "editor_session.json"
        editor_session.save(staged_session_path)
        session_sha256 = sha256_file(staged_session_path)
        session_size = staged_session_path.stat().st_size

        submission_identity = json.dumps(
            {
                "parent_rate_package_id": int(candidate.rate_package_id),
                "editor_session_sha256": session_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        submission_id = f"submission-{hashlib.sha256(submission_identity).hexdigest()[:24]}"
        final_directory = submissions_root / submission_id
        deployment_slot = candidate.workbench.model_config.deployment_slot
        final_session_path = final_directory / "editor_session.json"
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
            claimed_identity=cleaned_identity,
            created_at=datetime.now(UTC).isoformat(),
            editor_session_path=str(final_session_path),
            editor_session_sha256=session_sha256,
            editor_session_size_bytes=session_size,
            baseline_candidate_sha256=str(technical.get("candidate_artifact_sha256") or ""),
            model_source_sha256=candidate.bundle.model_source_sha256,
            path=str(submission_path),
            sha256="",
        )
        staged_submission_path = staging / "submission.json"
        _write_json_atomic(submission.to_payload(), staged_submission_path)
        submission.sha256 = sha256_file(staged_submission_path)
        submission = _promote_or_reuse_submission(
            staging,
            final_directory,
            root=root,
            proposed=submission,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return submission


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
    if not session_path.is_relative_to(root):
        raise EditorSubmissionError(f"submission artifact is outside {root}: {session_path}")
    if not session_path.is_file():
        raise EditorSubmissionError(f"submission artifact does not exist: {session_path}")
    if sha256_file(session_path) != payload["editor_session_sha256"]:
        raise EditorSubmissionError("editor session SHA-256 verification failed")
    if session_path.stat().st_size != int(payload["editor_session_size_bytes"]):
        raise EditorSubmissionError("editor session byte-size verification failed")
    return EditorSubmission(
        **payload,
        path=str(submission_path),
        sha256=actual_sha256,
    )
