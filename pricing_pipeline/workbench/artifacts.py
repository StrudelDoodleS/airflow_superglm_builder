from __future__ import annotations

import hashlib
import io
import os
import platform
import tempfile
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from packaging.version import InvalidVersion, Version

from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract


BUNDLE_FORMAT = "superglm-candidate-joblib-v2"
EDITED_MODEL_FORMAT = "superglm-edited-model-joblib-v1"


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
    model_name: str
    model_version: str
    export_id: str
    manifest_id: str
    split_set_id: str | None
    pk_columns: tuple[str, ...]
    row_order_sha256: str
    model_source_sha256: str
    offset_contract: OffsetExportContract
    offset_source: pd.Series | np.ndarray | None = None
    fit_sample_weight_name: str | None = None
    offset_source_name: str | None = None
    export_weight_name: str | None = None
    model_frame_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            contract = OffsetExportContract.model_validate(self.offset_contract)
        except ValueError as exc:
            raise CandidateArtifactError(f"invalid offset_contract: {exc}") from exc
        object.__setattr__(self, "offset_contract", contract)

        digest = self.model_frame_sha256
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CandidateArtifactError(
                "model_frame_sha256 must be a 64-character lowercase hex SHA-256 digest"
            )

        if contract.handling == "EXPORTED_FACTOR" and self.offset_source is None:
            raise CandidateArtifactError("EXPORTED_FACTOR requires offset_source")

        for role, values, name in (
            ("offset_source", self.offset_source, self.offset_source_name),
            ("export_weight", self.export_weight, self.export_weight_name),
        ):
            if values is None:
                if name is not None:
                    raise CandidateArtifactError(f"{role}_name was supplied without {role}")
                continue
            if not isinstance(values, pd.Series | np.ndarray):
                raise CandidateArtifactError(
                    f"{role} must be a pandas Series or numpy array"
                )
            if isinstance(values, np.ndarray) and values.ndim != 1:
                raise CandidateArtifactError(f"{role} must be one-dimensional")
            series = pd.Series(values).reset_index(drop=True)
            if len(series) != len(self.X):
                raise CandidateArtifactError(
                    f"{role} length {len(series)} does not match X row count {len(self.X)}"
                )
            if series.isna().any():
                raise CandidateArtifactError(f"{role} contains missing values")
            numeric_values = [
                value
                for value in series
                if not isinstance(value, (bool, np.bool_)) and pd.api.types.is_number(value)
            ]
            if any(not np.isfinite(value) for value in numeric_values):
                raise CandidateArtifactError(f"{role} contains non-finite numeric values")

        if contract.handling != "EXPORTED_FACTOR":
            return
        if not self.offset_source_name:
            raise CandidateArtifactError("EXPORTED_FACTOR requires offset_source_name")
        if self.offset_source_name != contract.source_name:
            raise CandidateArtifactError(
                "offset_source_name must match offset_contract source_name"
            )


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
    try:
        artifact_version = Version(expected_superglm_version)
        runtime_version = Version(actual_superglm)
    except InvalidVersion as exc:
        raise CandidateArtifactError(
            "candidate SuperGLM version is incompatible: "
            f"artifact={expected_superglm_version!r}, runtime={actual_superglm!r}"
        ) from exc
    if (
        artifact_version.release[:2] != runtime_version.release[:2]
        or artifact_version > runtime_version
    ):
        raise CandidateArtifactError(
            "candidate SuperGLM version is incompatible: "
            f"artifact={expected_superglm_version!r}, runtime={actual_superglm!r}"
        )


def save_edited_model(
    editor_session: Any,
    path: str | Path,
) -> CandidateArtifactMetadata:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = Path(editor_session.save_model(target)).expanduser().resolve()
    if saved != target or not target.is_file():
        raise CandidateArtifactError(
            "editor session did not create the requested model artifact"
        )
    return CandidateArtifactMetadata(
        path=str(target),
        sha256=_sha256(target),
        format=EDITED_MODEL_FORMAT,
        size_bytes=target.stat().st_size,
        python_version=platform.python_version(),
        superglm_version=_superglm_version(),
    )


def load_edited_model(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_format: str,
    expected_python_version: str,
    expected_superglm_version: str,
    allowed_root: str | Path,
) -> Any:
    artifact_path = Path(path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not artifact_path.is_relative_to(root):
        raise CandidateArtifactError(
            f"edited model artifact is outside configured artifact root {root}: "
            f"{artifact_path}"
        )
    if expected_format != EDITED_MODEL_FORMAT:
        raise CandidateArtifactError(
            f"unsupported edited model artifact format {expected_format!r}"
        )

    _validate_runtime_versions(
        expected_python_version=expected_python_version,
        expected_superglm_version=expected_superglm_version,
    )

    if not artifact_path.is_file():
        raise CandidateArtifactError(
            f"edited model artifact does not exist: {artifact_path}"
        )
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise CandidateArtifactError(
            f"edited model artifact could not be read: {artifact_path}"
        ) from exc
    actual_size = len(artifact_bytes)
    if actual_size != int(expected_size_bytes):
        raise CandidateArtifactError(
            "edited model artifact byte size does not match submission metadata: "
            f"expected={expected_size_bytes}, actual={actual_size}"
        )
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise CandidateArtifactError(
            "edited model artifact SHA-256 does not match submission metadata: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return joblib.load(io.BytesIO(artifact_bytes))


def save_candidate_bundle(
    bundle: CandidateBundle,
    path: str | Path,
) -> CandidateArtifactMetadata:
    for field_name in ("model_name", "model_version", "export_id"):
        value = getattr(bundle, field_name, None)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CandidateArtifactError(
                f"candidate {field_name} must be a non-empty trimmed string"
            )
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
        raise CandidateArtifactError(f"unsupported candidate artifact format {expected_format!r}")

    _validate_runtime_versions(
        expected_python_version=expected_python_version,
        expected_superglm_version=expected_superglm_version,
    )

    if not artifact_path.is_file():
        raise CandidateArtifactError(f"candidate artifact does not exist: {artifact_path}")
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise CandidateArtifactError(
            f"candidate artifact could not be read: {artifact_path}"
        ) from exc
    actual_size = len(artifact_bytes)
    if actual_size != int(expected_size_bytes):
        raise CandidateArtifactError(
            "candidate artifact byte size does not match SQL metadata: "
            f"expected={expected_size_bytes}, actual={actual_size}"
        )
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise CandidateArtifactError(
            "candidate artifact SHA-256 does not match SQL metadata: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    envelope = joblib.load(io.BytesIO(artifact_bytes))
    if not isinstance(envelope, dict) or envelope.get("format") != BUNDLE_FORMAT:
        raise CandidateArtifactError("candidate artifact envelope has an invalid format")
    if envelope.get("python_version") != expected_python_version:
        raise CandidateArtifactError("candidate artifact Python metadata is inconsistent")
    if envelope.get("superglm_version") != expected_superglm_version:
        raise CandidateArtifactError("candidate artifact SuperGLM metadata is inconsistent")
    bundle = envelope.get("bundle")
    if not isinstance(bundle, CandidateBundle):
        raise CandidateArtifactError("candidate artifact envelope does not contain a bundle")
    bundle = replace(bundle)
    for field_name in ("model_name", "model_version", "export_id"):
        value = getattr(bundle, field_name, None)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CandidateArtifactError(f"candidate artifact has no valid {field_name} identity")
    return bundle
