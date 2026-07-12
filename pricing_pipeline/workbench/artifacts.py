from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BUNDLE_FORMAT = "superglm-candidate-joblib-v1"


class CandidateArtifactError(RuntimeError):
    """Raised when a candidate artifact cannot be trusted or loaded."""


@dataclass(frozen=True)
class CandidateBundle:
    fitted_model: Any
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: pd.Series | np.ndarray | None
    offset: pd.Series | np.ndarray | None
    export_weight: pd.Series | np.ndarray | None
    cv_report: dict[str, Any]
    manifest_id: str
    split_set_id: str | None
    pk_columns: tuple[str, ...]
    row_order_sha256: str
    model_source_sha256: str
    offset_contract: dict[str, Any]
    review_artifact: dict[str, Any] | None = None
    fit_sample_weight_name: str | None = None
    export_weight_name: str | None = None
    offset_export_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateArtifactMetadata:
    path: str
    sha256: str
    format: str
    size_bytes: int
    python_version: str
    superglm_version: str


def _superglm_version() -> str:
    try:
        return version("superglm")
    except PackageNotFoundError:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _major_minor(value: str) -> tuple[str, str]:
    parts = str(value).split(".")
    if len(parts) < 2:
        return (str(value), "")
    return (parts[0], parts[1])


def _validate_runtime_versions(
    *,
    expected_python_version: str,
    expected_superglm_version: str,
) -> None:
    actual_python = platform.python_version()
    if _major_minor(expected_python_version) != _major_minor(actual_python):
        raise CandidateArtifactError(
            "candidate Python version is incompatible: "
            f"artifact={expected_python_version!r}, runtime={actual_python!r}"
        )

    actual_superglm = _superglm_version()
    if expected_superglm_version != actual_superglm:
        raise CandidateArtifactError(
            "candidate SuperGLM version is incompatible: "
            f"artifact={expected_superglm_version!r}, runtime={actual_superglm!r}"
        )


def save_candidate_bundle(
    bundle: CandidateBundle,
    path: str | Path,
) -> CandidateArtifactMetadata:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    python_version = platform.python_version()
    superglm_version = _superglm_version()
    envelope = {
        "format": BUNDLE_FORMAT,
        "python_version": python_version,
        "superglm_version": superglm_version,
        "bundle": bundle,
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(envelope, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    return CandidateArtifactMetadata(
        path=str(target),
        sha256=_sha256(target),
        format=BUNDLE_FORMAT,
        size_bytes=target.stat().st_size,
        python_version=python_version,
        superglm_version=superglm_version,
    )


def load_candidate_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_format: str,
    expected_python_version: str,
    expected_superglm_version: str,
    allowed_root: str | Path,
) -> CandidateBundle:
    artifact_path = Path(path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not artifact_path.is_relative_to(root):
        raise CandidateArtifactError(
            f"candidate artifact is outside configured artifact root {root}: {artifact_path}"
        )
    if expected_format != BUNDLE_FORMAT:
        raise CandidateArtifactError(
            f"unsupported candidate artifact format {expected_format!r}"
        )

    _validate_runtime_versions(
        expected_python_version=expected_python_version,
        expected_superglm_version=expected_superglm_version,
    )

    if not artifact_path.is_file():
        raise CandidateArtifactError(f"candidate artifact does not exist: {artifact_path}")
    actual_size = artifact_path.stat().st_size
    if actual_size != int(expected_size_bytes):
        raise CandidateArtifactError(
            "candidate artifact byte size does not match SQL metadata: "
            f"expected={expected_size_bytes}, actual={actual_size}"
        )
    actual_sha256 = _sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise CandidateArtifactError(
            "candidate artifact SHA-256 does not match SQL metadata: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    envelope = joblib.load(artifact_path)
    if not isinstance(envelope, dict) or envelope.get("format") != BUNDLE_FORMAT:
        raise CandidateArtifactError("candidate artifact envelope has an invalid format")
    if envelope.get("python_version") != expected_python_version:
        raise CandidateArtifactError("candidate artifact Python metadata is inconsistent")
    if envelope.get("superglm_version") != expected_superglm_version:
        raise CandidateArtifactError("candidate artifact SuperGLM metadata is inconsistent")
    bundle = envelope.get("bundle")
    if not isinstance(bundle, CandidateBundle):
        raise CandidateArtifactError("candidate artifact envelope does not contain a bundle")
    return bundle
