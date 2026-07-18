"""Stable identity for one publishable SuperGLM root build."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pricing_pipeline.data.manifest import ModelFrameManifestSpec, model_frame_evidence
from pricing_pipeline.data.row_identity import compute_row_order_sha256
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract


class BuildIdentityError(ValueError):
    """Raised when stable build identity cannot be represented or verified."""


@dataclass(frozen=True)
class BuildIdentity:
    """Component hashes and final fingerprint for a publishable root build."""

    build_fingerprint_sha256: str
    model_frame_sha256: str
    row_order_sha256: str
    model_source_sha256: str
    builder_source_sha256: str
    materialized_split_sha256: str
    runtime_sha256: str
    candidate_superglm_sha256: str
    candidate_python_version: str
    candidate_superglm_version: str
    candidate_superglm_git_sha: str


def create_build_identity(
    *,
    frame: pd.DataFrame,
    model_config: ModelBuildConfig,
    manifest_spec: ModelFrameManifestSpec,
    superglm_model: Any,
    split_indices: Sequence[tuple[Any, Any]],
    fit_mode: str,
    scoring: str | Sequence[str],
    offset_contract: OffsetExportContract,
    model_source_root: str | Path,
    builder_source_root: str | Path | None = None,
) -> BuildIdentity:
    """Hash every material input before model-version reservation or fitting."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise BuildIdentityError("model frame must be a non-empty pandas DataFrame")
    if model_config.target_name != manifest_spec.target_column:
        raise BuildIdentityError("model target does not match the manifest target contract")

    fit_mode = _required_text(fit_mode, "fit_mode")
    scoring_names = _scoring_names(scoring)
    if not isinstance(offset_contract, OffsetExportContract):
        raise BuildIdentityError("offset_contract must be a complete OffsetExportContract")
    split_payload = _materialized_split_payload(
        split_indices,
        row_count=len(frame),
        method=model_config.validation_split.method,
        expected_fold_count=model_config.validation_split.n_splits,
    )
    try:
        model_frame_sha256 = model_frame_evidence(frame)[0]
        row_order_sha256 = compute_row_order_sha256(
            frame,
            pk_columns=manifest_spec.pk_columns,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BuildIdentityError(f"model frame identity is invalid: {exc}") from exc

    model_source_sha256 = _source_tree_sha256(
        model_source_root,
        suffixes=frozenset({".ipynb", ".py", ".sql", ".toml"}),
        label="model source",
    )
    builder_root = (
        Path(__file__).resolve().parent
        if builder_source_root is None
        else Path(builder_source_root)
    )
    builder_source_sha256 = _source_tree_sha256(
        builder_root,
        suffixes=frozenset({".py"}),
        label="builder source",
    )

    try:
        from pricing_pipeline.modeling.superglm_identity import canonical_superglm_payload

        superglm_payload = canonical_superglm_payload(superglm_model)
    except Exception as exc:
        raise BuildIdentityError(f"SuperGLM identity is invalid: {exc}") from exc
    runtime = superglm_payload["runtime"]
    candidate_superglm_sha256 = _sha256_plain_json(superglm_payload)
    runtime_payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "superglm_version": runtime["version"],
        "superglm_git_sha": runtime["git_sha"],
    }

    materialized_split_sha256 = _sha256_json(split_payload)
    runtime_sha256 = _sha256_json(runtime_payload)
    build_payload = {
        "schema": "pricing-pipeline-build-identity-v1",
        "model": {
            "name": _required_text(model_config.model_name, "model_name"),
            "type": _required_text(model_config.model_type, "model_type"),
            "target": _required_text(model_config.target_name, "target_name"),
        },
        "data": {
            "model_frame_sha256": model_frame_sha256,
            "row_order_sha256": row_order_sha256,
            "manifest_contract": manifest_spec,
        },
        "superglm": superglm_payload,
        "validation": {
            "definition": model_config.validation_split,
            "materialized_split_sha256": materialized_split_sha256,
        },
        "fit": {
            "mode": fit_mode,
            "scoring": scoring_names,
            "offset_export_contract": offset_contract.model_dump(mode="python"),
        },
        "runtime": runtime_payload,
        "source": {
            "model_source_sha256": model_source_sha256,
            "builder_source_sha256": builder_source_sha256,
        },
    }
    return BuildIdentity(
        build_fingerprint_sha256=_sha256_json(build_payload),
        model_frame_sha256=model_frame_sha256,
        row_order_sha256=row_order_sha256,
        model_source_sha256=model_source_sha256,
        builder_source_sha256=builder_source_sha256,
        materialized_split_sha256=materialized_split_sha256,
        runtime_sha256=runtime_sha256,
        candidate_superglm_sha256=candidate_superglm_sha256,
        candidate_python_version=runtime_payload["python_version"],
        candidate_superglm_version=runtime["version"],
        candidate_superglm_git_sha=runtime["git_sha"],
    )


def verify_build_identity(
    expected: BuildIdentity,
    **build_inputs: Any,
) -> BuildIdentity:
    """Recompute a build identity and name every component that drifted."""
    if not isinstance(expected, BuildIdentity):
        raise BuildIdentityError("expected build identity is required")
    actual = create_build_identity(**build_inputs)
    changed = [
        field.name
        for field in fields(BuildIdentity)
        if getattr(actual, field.name) != getattr(expected, field.name)
    ]
    if changed:
        raise BuildIdentityError("build contract changed during execution: " + ", ".join(changed))
    return expected


def stable_build_export_id(identity: BuildIdentity) -> str:
    """Return the globally safe, retry-stable identity used by staging."""
    if not isinstance(identity, BuildIdentity):
        raise BuildIdentityError("build identity is required")
    return f"build_{identity.build_fingerprint_sha256}"


def _scoring_names(scoring: str | Sequence[str]) -> list[str]:
    values: Sequence[Any]
    if isinstance(scoring, str):
        values = (scoring,)
    elif isinstance(scoring, Sequence) and not isinstance(scoring, bytes | bytearray):
        values = scoring
    else:
        raise BuildIdentityError("publishable builds require named string scorers")
    if not values:
        raise BuildIdentityError("publishable builds require named string scorers")

    names: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise BuildIdentityError("publishable builds require named string scorers")
        names.append(_required_text(value, "scoring"))
    if len(names) != len(set(names)):
        raise BuildIdentityError("scoring names must not contain duplicates")
    return names


def _materialized_split_payload(
    split_indices: Sequence[tuple[Any, Any]],
    *,
    row_count: int,
    method: str,
    expected_fold_count: int | None,
) -> dict[str, Any]:
    try:
        folds = tuple(split_indices)
    except TypeError as exc:
        raise BuildIdentityError("validation split indices must be a finite sequence") from exc
    if not folds:
        raise BuildIdentityError("validation split must contain at least one fold")
    if method in {"train_test_split", "column_holdout"} and len(folds) != 1:
        raise BuildIdentityError("validation split method requires exactly one fold")
    if method == "kfold" and expected_fold_count is not None and len(folds) != expected_fold_count:
        raise BuildIdentityError("validation split fold count does not match its definition")

    payload_folds = []
    validation_rows: list[int] = []
    for fold_no, pair in enumerate(folds, start=1):
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise BuildIdentityError(
                f"validation split {fold_no} must contain train and validation indices"
            )
        train = _split_index_array(pair[0], row_count=row_count, label=f"fold {fold_no} train")
        validation = _split_index_array(
            pair[1],
            row_count=row_count,
            label=f"fold {fold_no} validation",
        )
        if not len(train) or not len(validation):
            raise BuildIdentityError(f"validation split {fold_no} must not be empty")
        if np.intersect1d(train, validation).size:
            raise BuildIdentityError(f"validation split {fold_no} train/validation overlap")
        if len(train) + len(validation) != row_count:
            raise BuildIdentityError(f"validation split {fold_no} must cover every model-frame row")
        validation_rows.extend(int(value) for value in validation)
        payload_folds.append(
            {
                "fold_no": fold_no,
                "train": [int(value) for value in train],
                "validation": [int(value) for value in validation],
            }
        )
    if method in {"kfold", "column_kfold"} and sorted(validation_rows) != list(range(row_count)):
        raise BuildIdentityError(
            "validation split folds must validate every model-frame row exactly once"
        )
    return {"row_count": row_count, "folds": payload_folds}


def _split_index_array(value: Any, *, row_count: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise BuildIdentityError(
            f"validation split {label} indices must be one-dimensional integers"
        )
    normalized = array.astype(np.int64, copy=False)
    if len(normalized) != len(np.unique(normalized)):
        raise BuildIdentityError(f"validation split {label} contains duplicate indices")
    if len(normalized) and (normalized.min() < 0 or normalized.max() >= row_count):
        raise BuildIdentityError(f"validation split {label} contains out-of-range indices")
    return normalized


def _source_tree_sha256(
    root: str | Path,
    *,
    suffixes: frozenset[str],
    label: str,
) -> str:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise BuildIdentityError(f"{label} root does not exist: {source_root}")
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and ".ipynb_checkpoints" not in path.relative_to(source_root).parts
        and "__pycache__" not in path.relative_to(source_root).parts
    )
    if not paths:
        allowed = ", ".join(sorted(suffixes))
        raise BuildIdentityError(f"{label} contains no supported source files ({allowed})")

    digest = hashlib.sha256()
    digest.update(b"pricing-source-tree-v1\0")
    for path in paths:
        if path.is_symlink():
            raise BuildIdentityError(f"{label} contains a symbolic link: {path}")
        relative = path.relative_to(source_root).as_posix()
        source_bytes = _source_file_bytes(path, label=label)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    return digest.hexdigest()


def _source_file_bytes(path: Path, *, label: str) -> bytes:
    if path.suffix.lower() != ".ipynb":
        try:
            return path.read_bytes()
        except OSError as exc:
            raise BuildIdentityError(f"cannot read {label} file: {path}") from exc
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildIdentityError(f"invalid model notebook source: {path}") from exc
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise BuildIdentityError(f"invalid model notebook cells: {path}")
    normalized_cells = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise BuildIdentityError(f"invalid model notebook cell: {path}")
        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            if not all(isinstance(line, str) for line in raw_source):
                raise BuildIdentityError(f"invalid model notebook source lines: {path}")
            source = "".join(raw_source)
        elif isinstance(raw_source, str):
            source = raw_source
        else:
            raise BuildIdentityError(f"invalid model notebook source: {path}")
        normalized_cells.append({"cell_type": str(cell.get("cell_type") or ""), "source": source})
    return _canonical_json_bytes(normalized_cells)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_plain_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        raise BuildIdentityError("canonical build value must not be missing")
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise BuildIdentityError("canonical build value datetime must not be NaT")
        return _canonical_datetime(value)
    if isinstance(value, np.timedelta64):
        if np.isnat(value):
            raise BuildIdentityError("canonical build value timedelta must not be NaT")
        return _canonical_timedelta(value)
    if isinstance(value, pd.Timestamp | datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta | timedelta):
        return _canonical_timedelta(value)
    if isinstance(value, pd.Period):
        if pd.isna(value):
            raise BuildIdentityError("canonical build value period must not be NaT")
        return {
            "type": "period",
            "ordinal": int(value.ordinal),
            "frequency": value.freqstr,
        }
    if isinstance(value, pd.Interval):
        return {
            "type": "interval",
            "closed": value.closed,
            "left": _canonical_value(value.left),
            "right": _canonical_value(value.right),
        }
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise BuildIdentityError("canonical build value floats must be finite")
        return {"type": "float", "value": 0.0 if value == 0.0 else value}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                {
                    "name": field.name,
                    "value": _canonical_value(getattr(value, field.name)),
                }
                for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        normalized = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise BuildIdentityError("canonical build mappings require non-empty string keys")
            normalized.append(
                {
                    "key": _canonical_value(key),
                    "value": _canonical_value(item),
                }
            )
        normalized.sort(key=lambda item: item["key"]["value"])
        return {"type": "mapping", "items": normalized}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {
            "type": "sequence",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": [_canonical_value(item) for item in value],
        }
    raise BuildIdentityError(
        f"unsupported canonical build value {type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_datetime(value: Any) -> dict[str, str]:
    try:
        timestamp = pd.Timestamp(value)
        encoded = timestamp.isoformat()
    except (OverflowError, TypeError, ValueError) as exc:
        raise BuildIdentityError("canonical build value datetime is not representable") from exc
    if pd.isna(timestamp):
        raise BuildIdentityError("canonical build value datetime must not be NaT")
    return {"type": "datetime", "value": encoded}


def _canonical_timedelta(value: Any) -> dict[str, str]:
    try:
        duration = pd.Timedelta(value)
        encoded = duration.isoformat()
    except (OverflowError, TypeError, ValueError) as exc:
        raise BuildIdentityError("canonical build value timedelta is not representable") from exc
    if pd.isna(duration):
        raise BuildIdentityError("canonical build value timedelta must not be NaT")
    return {"type": "timedelta", "value": encoded}


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BuildIdentityError(f"{field_name} must be a non-empty trimmed string")
    return value
