"""Verified handoff for editor-created SuperGLM level groupings.

SuperGLM 0.26 exposes the defensive ``model.features`` mapping, but currently
stores each fitted categorical ``LevelGrouping`` on the private
``feature_spec._grouping`` attribute. This module is the one allowed
compatibility seam for that final private read/write. Notebook code deals only
in the public helpers below, so a future SuperGLM grouping-export API can
replace this module without changing the notebook workflow.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping
from superglm.features.ordered_categorical import OrderedCategorical

from pricing_pipeline.data.manifest import model_frame_evidence

LEVEL_GROUPING_ARTIFACT_FORMAT = "superglm-level-groupings-joblib-v1"
LEVEL_GROUPING_METADATA_FORMAT = "superglm-level-groupings-metadata-v1"
PRIVATE_SUPERGLM_GROUPING_API = "model.features[*]._grouping"
_SYMBOLIC_CATEGORICAL_BASES = {"first", "most_exposed"}


class LevelGroupingArtifactError(RuntimeError):
    """Raised when an editor grouping handoff cannot be trusted or applied."""


@dataclass(frozen=True)
class LevelGroupingArtifact:
    path: str
    metadata_path: str
    format: str
    size_bytes: int
    sha256: str
    grouping_sha256: str
    python_version: str
    superglm_version: str
    source_model_name: str
    source_package_version: int
    source_manifest_id: str
    source_model_frame_sha256: str
    source_data_as_of_date: str
    feature_names: tuple[str, ...]
    collapsed_group_count: int
    created_at: str


def _superglm_version() -> str:
    try:
        return version("superglm")
    except PackageNotFoundError:
        return "unknown"


def _required_text(value: Any, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise LevelGroupingArtifactError(f"{field_name} is required")
    return cleaned


def _required_sha256(value: Any, field_name: str) -> str:
    digest = _required_text(value, field_name)
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise LevelGroupingArtifactError(
            f"{field_name} must be a 64-character lowercase SHA-256 digest"
        )
    return digest


def _normalise_date(value: Any, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise LevelGroupingArtifactError(
            f"{field_name} must be a date, datetime, or ISO date string"
        )
    cleaned = value.strip()
    try:
        return datetime.fromisoformat(cleaned).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(cleaned).isoformat()
        except ValueError as exc:
            raise LevelGroupingArtifactError(
                f"{field_name} must be a date, datetime, or ISO date string"
            ) from exc


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _grouping_payload(grouping: LevelGrouping) -> dict[str, Any]:
    return {
        "original_to_group": {
            str(original): str(group) for original, group in grouping.original_to_group.items()
        },
        "group_to_originals": {
            str(group): [str(original) for original in originals]
            for group, originals in grouping.group_to_originals.items()
        },
        "all_original_levels": [str(level) for level in grouping.all_original_levels],
        "grouped_levels": [str(level) for level in grouping.grouped_levels],
    }


def _groupings_payload(
    groupings: Mapping[str, LevelGrouping],
) -> dict[str, dict[str, Any]]:
    return {
        feature_name: _grouping_payload(groupings[feature_name])
        for feature_name in sorted(groupings)
    }


def _grouping_sha256(groupings: Mapping[str, LevelGrouping]) -> str:
    canonical = json.dumps(
        _groupings_payload(groupings),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _collapsed_group_count(groupings: Mapping[str, LevelGrouping]) -> int:
    return sum(
        1
        for grouping in groupings.values()
        for originals in grouping.group_to_originals.values()
        if len(originals) > 1
    )


def _validate_grouping_shape(feature_name: str, grouping: Any) -> LevelGrouping:
    if not isinstance(grouping, LevelGrouping):
        raise LevelGroupingArtifactError(
            f"feature {feature_name!r} does not contain a SuperGLM LevelGrouping"
        )

    originals = [str(level) for level in grouping.all_original_levels]
    grouped = [str(level) for level in grouping.grouped_levels]
    if len(originals) != len(set(originals)):
        raise LevelGroupingArtifactError(f"feature {feature_name!r} has duplicate original levels")
    if len(grouped) != len(set(grouped)):
        raise LevelGroupingArtifactError(f"feature {feature_name!r} has duplicate grouped levels")
    if set(grouping.original_to_group) != set(originals):
        raise LevelGroupingArtifactError(
            f"feature {feature_name!r} grouping does not map every original level exactly once"
        )
    if set(grouping.group_to_originals) != set(grouped):
        raise LevelGroupingArtifactError(
            f"feature {feature_name!r} inverse grouping does not match grouped_levels"
        )

    inverse_members: list[str] = []
    for group_label in grouped:
        members = [str(level) for level in grouping.group_to_originals[group_label]]
        if not members:
            raise LevelGroupingArtifactError(
                f"feature {feature_name!r} group {group_label!r} has no levels"
            )
        inverse_members.extend(members)
        for original in members:
            if grouping.original_to_group.get(original) != group_label:
                raise LevelGroupingArtifactError(
                    f"feature {feature_name!r} has inconsistent forward/inverse grouping"
                )
    if len(inverse_members) != len(set(inverse_members)) or set(inverse_members) != set(originals):
        raise LevelGroupingArtifactError(
            f"feature {feature_name!r} grouping is not a partition of its original levels"
        )
    return grouping


def _has_real_collapse(grouping: LevelGrouping) -> bool:
    return any(len(originals) > 1 for originals in grouping.group_to_originals.values())


def extract_editor_level_groupings(editor_session: Any) -> dict[str, LevelGrouping]:
    """Extract all real collapses from an editor session's current model.

    This is the only function that reads SuperGLM's private grouping attribute.
    Identity-only groupings are omitted so an empty mapping means that no
    ``ROUTINE_EDIT`` model needs to be fitted.
    """
    model = getattr(editor_session, "model", None)
    features = getattr(model, "features", None)
    if not isinstance(features, Mapping):
        raise LevelGroupingArtifactError(
            f"SuperGLM editor internals changed: expected {PRIVATE_SUPERGLM_GROUPING_API}"
        )

    extracted: dict[str, LevelGrouping] = {}
    for raw_name, feature_spec in features.items():
        feature_name = _required_text(raw_name, "feature name")
        grouping = getattr(feature_spec, "_grouping", None)
        if grouping is None:
            continue
        validated = _validate_grouping_shape(feature_name, grouping)
        if _has_real_collapse(validated):
            extracted[feature_name] = copy.deepcopy(validated)
    return dict(sorted(extracted.items()))


def _metadata_payload(
    artifact: LevelGroupingArtifact,
    *,
    grouping_payload: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = asdict(artifact)
    payload.pop("path")
    payload.pop("metadata_path")
    payload["metadata_format"] = LEVEL_GROUPING_METADATA_FORMAT
    payload["feature_names"] = list(artifact.feature_names)
    payload["groupings"] = grouping_payload
    return payload


def _artifact_from_metadata(
    payload: dict[str, Any],
    *,
    artifact_path: Path,
    metadata_path: Path,
) -> LevelGroupingArtifact:
    if payload.get("metadata_format") != LEVEL_GROUPING_METADATA_FORMAT:
        raise LevelGroupingArtifactError("unsupported level-grouping metadata format")
    try:
        artifact = LevelGroupingArtifact(
            path=str(artifact_path),
            metadata_path=str(metadata_path),
            format=str(payload["format"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=_required_sha256(payload["sha256"], "sha256"),
            grouping_sha256=_required_sha256(payload["grouping_sha256"], "grouping_sha256"),
            python_version=_required_text(payload["python_version"], "python_version"),
            superglm_version=_required_text(payload["superglm_version"], "superglm_version"),
            source_model_name=_required_text(payload["source_model_name"], "source_model_name"),
            source_package_version=int(payload["source_package_version"]),
            source_manifest_id=_required_text(payload["source_manifest_id"], "source_manifest_id"),
            source_model_frame_sha256=_required_sha256(
                payload["source_model_frame_sha256"],
                "source_model_frame_sha256",
            ),
            source_data_as_of_date=_normalise_date(
                payload["source_data_as_of_date"],
                "source_data_as_of_date",
            ),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            collapsed_group_count=int(payload["collapsed_group_count"]),
            created_at=_required_text(payload["created_at"], "created_at"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LevelGroupingArtifactError(
            "level-grouping metadata is incomplete or invalid"
        ) from exc
    if artifact.format != LEVEL_GROUPING_ARTIFACT_FORMAT:
        raise LevelGroupingArtifactError(
            f"unsupported level-grouping artifact format {artifact.format!r}"
        )
    if artifact.size_bytes <= 0:
        raise LevelGroupingArtifactError("level-grouping artifact size must be positive")
    if artifact.source_package_version <= 0:
        raise LevelGroupingArtifactError("source_package_version must be positive")
    if len(artifact.feature_names) != len(set(artifact.feature_names)):
        raise LevelGroupingArtifactError("feature_names must not contain duplicates")
    if artifact.collapsed_group_count < 0:
        raise LevelGroupingArtifactError("collapsed_group_count must not be negative")
    return artifact


def inspect_level_groupings(path: str | Path) -> LevelGroupingArtifact:
    """Validate and return the readable grouping metadata without unpickling."""
    artifact_path = Path(path).expanduser().resolve()
    metadata_path = _metadata_path(artifact_path)
    if not metadata_path.is_file():
        raise LevelGroupingArtifactError(f"level-grouping metadata does not exist: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LevelGroupingArtifactError(
            f"level-grouping metadata could not be read: {metadata_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise LevelGroupingArtifactError("level-grouping metadata must be a JSON object")
    artifact = _artifact_from_metadata(
        payload,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )
    raw_groupings = payload.get("groupings")
    if not isinstance(raw_groupings, dict):
        raise LevelGroupingArtifactError("level-grouping metadata has no grouping evidence")
    if tuple(sorted(str(name) for name in raw_groupings)) != artifact.feature_names:
        raise LevelGroupingArtifactError("level-grouping feature names do not match their metadata")
    metadata_groupings = _groupings_from_json(raw_groupings)
    if _grouping_sha256(metadata_groupings) != artifact.grouping_sha256:
        raise LevelGroupingArtifactError(
            "level-grouping metadata does not match its semantic SHA-256"
        )
    if _collapsed_group_count(metadata_groupings) != artifact.collapsed_group_count:
        raise LevelGroupingArtifactError(
            "level-grouping collapsed-group count does not match its metadata"
        )
    return artifact


def _verified_artifact_bytes(artifact: LevelGroupingArtifact) -> bytes:
    artifact_path = Path(artifact.path)
    if not artifact_path.is_file():
        raise LevelGroupingArtifactError(f"level-grouping artifact does not exist: {artifact_path}")
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise LevelGroupingArtifactError(
            f"level-grouping artifact could not be read: {artifact_path}"
        ) from exc
    if len(artifact_bytes) != artifact.size_bytes:
        raise LevelGroupingArtifactError("level-grouping artifact size does not match its metadata")
    if _sha256_bytes(artifact_bytes) != artifact.sha256:
        raise LevelGroupingArtifactError(
            "level-grouping artifact SHA-256 does not match its metadata"
        )
    return artifact_bytes


def _groupings_from_json(payload: Mapping[str, Any]) -> dict[str, LevelGrouping]:
    groupings: dict[str, LevelGrouping] = {}
    for raw_name, raw_grouping in payload.items():
        feature_name = _required_text(raw_name, "feature name")
        if not isinstance(raw_grouping, Mapping):
            raise LevelGroupingArtifactError(
                f"feature {feature_name!r} grouping metadata must be an object"
            )
        try:
            grouping = LevelGrouping(
                original_to_group={
                    str(original): str(group)
                    for original, group in raw_grouping["original_to_group"].items()
                },
                group_to_originals={
                    str(group): [str(original) for original in originals]
                    for group, originals in raw_grouping["group_to_originals"].items()
                },
                all_original_levels=[str(level) for level in raw_grouping["all_original_levels"]],
                grouped_levels=[str(level) for level in raw_grouping["grouped_levels"]],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LevelGroupingArtifactError(
                f"feature {feature_name!r} grouping metadata is invalid"
            ) from exc
        groupings[feature_name] = _validate_grouping_shape(feature_name, grouping)
    return dict(sorted(groupings.items()))


def save_editor_level_groupings(
    editor_session: Any,
    path: str | Path,
    *,
    source_model_name: str,
    source_package_version: int,
    source_manifest_id: str,
    source_model_frame_sha256: str,
    source_data_as_of_date: date | datetime | str,
    replace: bool = False,
) -> LevelGroupingArtifact:
    """Atomically save the editor's actual ``LevelGrouping`` objects."""
    model_name = _required_text(source_model_name, "source_model_name")
    try:
        package_version = int(source_package_version)
    except (TypeError, ValueError) as exc:
        raise LevelGroupingArtifactError("source_package_version must be an integer") from exc
    if package_version <= 0:
        raise LevelGroupingArtifactError("source_package_version must be positive")
    manifest_id = _required_text(source_manifest_id, "source_manifest_id")
    frame_sha256 = _required_sha256(
        source_model_frame_sha256,
        "source_model_frame_sha256",
    )
    data_as_of = _normalise_date(source_data_as_of_date, "source_data_as_of_date")
    groupings = extract_editor_level_groupings(editor_session)
    semantic_sha256 = _grouping_sha256(groupings)
    grouping_payload = _groupings_payload(groupings)
    python_version = platform.python_version()
    superglm_version = _superglm_version()

    artifact_path = Path(path).expanduser().resolve()
    metadata_path = _metadata_path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    source_identity = {
        "source_model_name": model_name,
        "source_package_version": package_version,
        "source_manifest_id": manifest_id,
        "source_model_frame_sha256": frame_sha256,
        "source_data_as_of_date": data_as_of,
    }

    if artifact_path.exists() or metadata_path.exists():
        if artifact_path.is_file() and metadata_path.is_file():
            try:
                existing = inspect_level_groupings(artifact_path)
                _verified_artifact_bytes(existing)
            except LevelGroupingArtifactError:
                if not replace:
                    raise
            else:
                unchanged = existing.grouping_sha256 == semantic_sha256 and all(
                    getattr(existing, key) == value for key, value in source_identity.items()
                )
                unchanged = (
                    unchanged
                    and existing.python_version == python_version
                    and existing.superglm_version == superglm_version
                )
                if unchanged:
                    return existing
                if not replace:
                    raise FileExistsError(
                        "a different level-grouping artifact already exists; "
                        f"set replace=True to replace {artifact_path}"
                    )
        elif not replace:
            raise LevelGroupingArtifactError(
                "level-grouping handoff is incomplete; set replace=True to repair it"
            )

    envelope = {
        "format": LEVEL_GROUPING_ARTIFACT_FORMAT,
        "python_version": python_version,
        "superglm_version": superglm_version,
        **source_identity,
        "grouping_sha256": semantic_sha256,
        "groupings": groupings,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(envelope, temporary, compress=3)
        artifact_bytes = temporary.read_bytes()
        artifact = LevelGroupingArtifact(
            path=str(artifact_path),
            metadata_path=str(metadata_path),
            format=LEVEL_GROUPING_ARTIFACT_FORMAT,
            size_bytes=len(artifact_bytes),
            sha256=_sha256_bytes(artifact_bytes),
            grouping_sha256=semantic_sha256,
            python_version=python_version,
            superglm_version=superglm_version,
            source_model_name=model_name,
            source_package_version=package_version,
            source_manifest_id=manifest_id,
            source_model_frame_sha256=frame_sha256,
            source_data_as_of_date=data_as_of,
            feature_names=tuple(sorted(groupings)),
            collapsed_group_count=_collapsed_group_count(groupings),
            created_at=datetime.now(UTC).isoformat(),
        )
        os.replace(temporary, artifact_path)
        _write_json_atomic(
            _metadata_payload(artifact, grouping_payload=grouping_payload),
            metadata_path,
        )
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def _validate_runtime(artifact: LevelGroupingArtifact) -> None:
    artifact_python = artifact.python_version.split(".")[:2]
    runtime_python = platform.python_version().split(".")[:2]
    if artifact_python != runtime_python:
        raise LevelGroupingArtifactError(
            "level-grouping Python version is incompatible: "
            f"artifact={artifact.python_version!r}, runtime={platform.python_version()!r}"
        )
    runtime_superglm = _superglm_version()
    if artifact.superglm_version != runtime_superglm:
        raise LevelGroupingArtifactError(
            "level-grouping SuperGLM version is incompatible: "
            f"artifact={artifact.superglm_version!r}, runtime={runtime_superglm!r}"
        )


def _validate_groupings_against_frame(
    groupings: Mapping[str, LevelGrouping],
    frame: pd.DataFrame,
) -> None:
    for feature_name, grouping in groupings.items():
        if feature_name not in frame.columns:
            raise LevelGroupingArtifactError(
                f"grouped feature {feature_name!r} is missing from the model frame"
            )
        column = frame[feature_name]
        if column.isna().any():
            raise LevelGroupingArtifactError(
                f"grouped feature {feature_name!r} contains missing values"
            )
        actual_levels = {str(value) for value in column.unique()}
        expected_levels = {str(value) for value in grouping.all_original_levels}
        if actual_levels != expected_levels:
            new_levels = sorted(actual_levels - expected_levels)
            missing_levels = sorted(expected_levels - actual_levels)
            raise LevelGroupingArtifactError(
                f"grouped feature {feature_name!r} levels changed since export; "
                f"new={new_levels}, missing={missing_levels}"
            )


def load_level_groupings(
    path: str | Path,
    *,
    frame: pd.DataFrame,
    expected_model_name: str,
    expected_data_as_of_date: date | datetime | str,
    expected_manifest_id: str | None = None,
    allowed_root: str | Path | None = None,
) -> dict[str, LevelGrouping]:
    """Load verified actual groupings and bind them to the current model frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise LevelGroupingArtifactError("frame must be a non-empty pandas DataFrame")
    artifact_path = Path(path).expanduser().resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if not artifact_path.is_relative_to(root):
            raise LevelGroupingArtifactError(
                f"level-grouping artifact is outside allowed root {root}: {artifact_path}"
            )

    artifact = inspect_level_groupings(artifact_path)
    _validate_runtime(artifact)
    if artifact.source_model_name != _required_text(expected_model_name, "expected_model_name"):
        raise LevelGroupingArtifactError(
            "level-grouping artifact belongs to a different model: "
            f"artifact={artifact.source_model_name!r}, expected={expected_model_name!r}"
        )
    expected_data_as_of = _normalise_date(
        expected_data_as_of_date,
        "expected_data_as_of_date",
    )
    if artifact.source_data_as_of_date != expected_data_as_of:
        raise LevelGroupingArtifactError(
            "level-grouping artifact belongs to a different data-as-at version: "
            f"artifact={artifact.source_data_as_of_date!r}, expected={expected_data_as_of!r}"
        )
    if expected_manifest_id is not None and artifact.source_manifest_id != _required_text(
        expected_manifest_id,
        "expected_manifest_id",
    ):
        raise LevelGroupingArtifactError(
            "level-grouping artifact belongs to a different dataset manifest"
        )

    actual_frame_sha256, _ = model_frame_evidence(frame)
    if actual_frame_sha256 != artifact.source_model_frame_sha256:
        raise LevelGroupingArtifactError(
            "level-grouping artifact belongs to a different ordered model frame"
        )
    artifact_bytes = _verified_artifact_bytes(artifact)
    try:
        envelope = joblib.load(io.BytesIO(artifact_bytes))
    except Exception as exc:
        raise LevelGroupingArtifactError(
            f"level-grouping artifact could not be deserialized: {artifact_path}"
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("format") != artifact.format:
        raise LevelGroupingArtifactError("level-grouping artifact envelope is invalid")
    envelope_identity = {
        "python_version": artifact.python_version,
        "superglm_version": artifact.superglm_version,
        "source_model_name": artifact.source_model_name,
        "source_package_version": artifact.source_package_version,
        "source_manifest_id": artifact.source_manifest_id,
        "source_model_frame_sha256": artifact.source_model_frame_sha256,
        "source_data_as_of_date": artifact.source_data_as_of_date,
        "grouping_sha256": artifact.grouping_sha256,
    }
    if any(envelope.get(key) != value for key, value in envelope_identity.items()):
        raise LevelGroupingArtifactError(
            "level-grouping artifact envelope does not match its metadata"
        )
    raw_groupings = envelope.get("groupings")
    if not isinstance(raw_groupings, Mapping):
        raise LevelGroupingArtifactError(
            "level-grouping artifact does not contain a grouping mapping"
        )
    groupings = {
        _required_text(feature_name, "feature name"): _validate_grouping_shape(
            str(feature_name),
            grouping,
        )
        for feature_name, grouping in raw_groupings.items()
    }
    groupings = dict(sorted(groupings.items()))
    if tuple(groupings) != artifact.feature_names:
        raise LevelGroupingArtifactError(
            "level-grouping artifact feature names do not match its metadata"
        )
    if _grouping_sha256(groupings) != artifact.grouping_sha256:
        raise LevelGroupingArtifactError(
            "deserialized level groupings do not match their semantic SHA-256"
        )
    if _collapsed_group_count(groupings) != artifact.collapsed_group_count:
        raise LevelGroupingArtifactError(
            "deserialized collapsed-group count does not match metadata"
        )
    _validate_groupings_against_frame(groupings, frame)
    return {name: copy.deepcopy(grouping) for name, grouping in groupings.items()}


def apply_level_groupings(
    features: Mapping[str, Any],
    groupings: Mapping[str, LevelGrouping],
) -> dict[str, Any]:
    """Return copied feature specs with verified groupings attached.

    Setting ``_grouping`` is the second and final private SuperGLM operation in
    this compatibility module.  Input feature specs are never mutated.
    """
    if not isinstance(features, Mapping) or not features:
        raise LevelGroupingArtifactError("features must be a non-empty mapping")
    copied = copy.deepcopy(dict(features))
    for raw_name, raw_grouping in groupings.items():
        feature_name = _required_text(raw_name, "feature name")
        grouping = _validate_grouping_shape(feature_name, raw_grouping)
        if not _has_real_collapse(grouping):
            raise LevelGroupingArtifactError(
                f"feature {feature_name!r} has no actual collapsed group"
            )
        if feature_name not in copied:
            raise LevelGroupingArtifactError(
                f"grouped feature {feature_name!r} is not in the model feature specification"
            )
        feature_spec = copied[feature_name]
        if not isinstance(feature_spec, Categorical | OrderedCategorical):
            raise LevelGroupingArtifactError(f"grouped feature {feature_name!r} is not categorical")
        base = str(feature_spec.base)
        if base in grouping.original_to_group:
            feature_spec.base = grouping.original_to_group[base]
        elif base not in _SYMBOLIC_CATEGORICAL_BASES and base not in set(grouping.grouped_levels):
            raise LevelGroupingArtifactError(
                f"grouped feature {feature_name!r} has unknown categorical base {base!r}"
            )
        feature_spec._grouping = copy.deepcopy(grouping)
    return copied


__all__ = [
    "LEVEL_GROUPING_ARTIFACT_FORMAT",
    "LEVEL_GROUPING_METADATA_FORMAT",
    "LevelGroupingArtifact",
    "LevelGroupingArtifactError",
    "apply_level_groupings",
    "extract_editor_level_groupings",
    "inspect_level_groupings",
    "load_level_groupings",
    "save_editor_level_groupings",
]
