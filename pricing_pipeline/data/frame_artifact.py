"""Verified, file-backed handoff for model frames between analyst notebooks."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from pricing_pipeline.data.manifest import model_frame_evidence

MODEL_FRAME_ARTIFACT_FORMAT = "pricing-model-frame-joblib-v1"


class ModelFrameArtifactError(RuntimeError):
    """Raised when a notebook frame handoff is missing, changed, or invalid."""


@dataclass(frozen=True)
class ModelFrameArtifact:
    path: str
    metadata_path: str
    format: str
    size_bytes: int
    sha256: str
    model_frame_sha256: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    pandas_dtypes: tuple[str, ...]
    created_at: str


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_digest(value: Any, field_name: str) -> str:
    digest = str(value or "")
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ModelFrameArtifactError(
            f"{field_name} must be a 64-character lowercase SHA-256 digest"
        )
    return digest


def _artifact_from_payload(payload: dict[str, Any], *, metadata_path: Path) -> ModelFrameArtifact:
    try:
        artifact = ModelFrameArtifact(
            path=str(payload["path"]),
            metadata_path=str(metadata_path),
            format=str(payload["format"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=_required_digest(payload["sha256"], "sha256"),
            model_frame_sha256=_required_digest(
                payload["model_frame_sha256"],
                "model_frame_sha256",
            ),
            row_count=int(payload["row_count"]),
            column_count=int(payload["column_count"]),
            columns=tuple(str(value) for value in payload["columns"]),
            pandas_dtypes=tuple(str(value) for value in payload["pandas_dtypes"]),
            created_at=str(payload["created_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelFrameArtifactError("model-frame metadata is incomplete or invalid") from exc
    if artifact.format != MODEL_FRAME_ARTIFACT_FORMAT:
        raise ModelFrameArtifactError(
            f"unsupported model-frame artifact format {artifact.format!r}"
        )
    if artifact.size_bytes <= 0 or artifact.row_count <= 0 or artifact.column_count <= 0:
        raise ModelFrameArtifactError("model-frame metadata contains non-positive dimensions")
    if artifact.column_count != len(artifact.columns) or artifact.column_count != len(
        artifact.pandas_dtypes
    ):
        raise ModelFrameArtifactError("model-frame column metadata is inconsistent")
    return artifact


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
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


def inspect_model_frame(path: str | Path) -> ModelFrameArtifact:
    """Read and validate the small JSON receipt without deserializing the frame."""
    artifact_path = Path(path).expanduser().resolve()
    metadata_path = _metadata_path(artifact_path)
    if not metadata_path.is_file():
        raise ModelFrameArtifactError(f"model-frame metadata does not exist: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelFrameArtifactError(
            f"model-frame metadata could not be read: {metadata_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelFrameArtifactError("model-frame metadata must be a JSON object")
    artifact = _artifact_from_payload(payload, metadata_path=metadata_path)
    if Path(artifact.path).expanduser().resolve() != artifact_path:
        raise ModelFrameArtifactError("model-frame metadata path does not match the requested file")
    return artifact


def load_model_frame(path: str | Path) -> pd.DataFrame:
    """Load a frame only after checking its byte-level and frame-level evidence."""
    artifact_path = Path(path).expanduser().resolve()
    artifact = inspect_model_frame(artifact_path)
    if not artifact_path.is_file():
        raise ModelFrameArtifactError(f"model-frame artifact does not exist: {artifact_path}")
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise ModelFrameArtifactError(
            f"model-frame artifact could not be read: {artifact_path}"
        ) from exc
    if len(artifact_bytes) != artifact.size_bytes:
        raise ModelFrameArtifactError(
            "model-frame artifact size does not match its metadata: "
            f"expected={artifact.size_bytes}, actual={len(artifact_bytes)}"
        )
    actual_sha256 = _sha256_bytes(artifact_bytes)
    if actual_sha256 != artifact.sha256:
        raise ModelFrameArtifactError(
            "model-frame artifact SHA-256 does not match its metadata: "
            f"expected={artifact.sha256}, actual={actual_sha256}"
        )
    try:
        envelope = joblib.load(io.BytesIO(artifact_bytes))
    except Exception as exc:
        raise ModelFrameArtifactError(
            f"model-frame artifact could not be deserialized: {artifact_path}"
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("format") != artifact.format:
        raise ModelFrameArtifactError("model-frame artifact envelope is invalid")
    frame = envelope.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise ModelFrameArtifactError("model-frame artifact does not contain a DataFrame")

    actual_frame_sha256, _ = model_frame_evidence(frame)
    if actual_frame_sha256 != artifact.model_frame_sha256:
        raise ModelFrameArtifactError("loaded model-frame evidence does not match its metadata")
    if len(frame) != artifact.row_count:
        raise ModelFrameArtifactError("loaded model-frame row count does not match its metadata")
    if tuple(str(column) for column in frame.columns) != artifact.columns:
        raise ModelFrameArtifactError("loaded model-frame columns do not match their metadata")
    if tuple(str(dtype) for dtype in frame.dtypes) != artifact.pandas_dtypes:
        raise ModelFrameArtifactError("loaded model-frame dtypes do not match their metadata")
    return frame


def save_model_frame(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    replace: bool = False,
) -> ModelFrameArtifact:
    """Atomically save or idempotently reuse the notebook-to-notebook frame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("model frame must not be empty")
    if len(frame.columns) == 0:
        raise ValueError("model frame must contain at least one column")

    artifact_path = Path(path).expanduser().resolve()
    metadata_path = _metadata_path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame_sha256, _ = model_frame_evidence(frame)

    if artifact_path.exists() or metadata_path.exists():
        if artifact_path.is_file() and metadata_path.is_file():
            try:
                existing_frame = load_model_frame(artifact_path)
                existing = inspect_model_frame(artifact_path)
            except ModelFrameArtifactError:
                if not replace:
                    raise
            else:
                existing_sha256, _ = model_frame_evidence(existing_frame)
                if existing_sha256 == model_frame_sha256:
                    return existing
                if not replace:
                    raise FileExistsError(
                        "a different model frame already exists; set replace=True "
                        f"to replace {artifact_path}"
                    )
        elif not replace:
            raise ModelFrameArtifactError(
                "model-frame handoff is incomplete; set replace=True to repair it"
            )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(
            {
                "format": MODEL_FRAME_ARTIFACT_FORMAT,
                "frame": frame.copy(),
            },
            temporary,
            compress=3,
        )
        artifact_bytes = temporary.read_bytes()
        artifact = ModelFrameArtifact(
            path=str(artifact_path),
            metadata_path=str(metadata_path),
            format=MODEL_FRAME_ARTIFACT_FORMAT,
            size_bytes=len(artifact_bytes),
            sha256=_sha256_bytes(artifact_bytes),
            model_frame_sha256=model_frame_sha256,
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=tuple(str(column) for column in frame.columns),
            pandas_dtypes=tuple(str(dtype) for dtype in frame.dtypes),
            created_at=datetime.now(UTC).isoformat(),
        )
        os.replace(temporary, artifact_path)
        _write_json_atomic(asdict(artifact), metadata_path)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "MODEL_FRAME_ARTIFACT_FORMAT",
    "ModelFrameArtifact",
    "ModelFrameArtifactError",
    "inspect_model_frame",
    "load_model_frame",
    "save_model_frame",
]
