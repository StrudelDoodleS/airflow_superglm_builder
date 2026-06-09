from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pricing_pipeline.data.row_identity import compute_row_order_sha256

FOLD_ASSIGNMENT_FORMAT = "fold_assignment_v1"
HOLDOUT_ASSIGNMENT_FORMAT = "holdout_assignment_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split_artifact_npz(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    validation_split,
    pk_columns: tuple[str, ...],
    row_count: int,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pk_column_array = np.array(_normalize_pk_columns(pk_columns))

    if validation_split.method in {"kfold", "column_kfold"}:
        arrays = {
            "split_format": np.array(FOLD_ASSIGNMENT_FORMAT),
            "pk_columns": pk_column_array,
            "test_fold": _fold_assignment(folds, row_count=row_count),
        }
    elif validation_split.method in {"train_test_split", "column_holdout"}:
        arrays = {
            "split_format": np.array(HOLDOUT_ASSIGNMENT_FORMAT),
            "pk_columns": pk_column_array,
            "is_testing_set": _holdout_assignment(folds, row_count=row_count),
        }
    else:
        arrays = {}
        for fold_no, (train_idx, test_idx) in sorted(folds.items()):
            arrays[f"fold_{fold_no}_train_idx"] = np.asarray(train_idx).astype(
                np.int64,
                copy=False,
            )
            arrays[f"fold_{fold_no}_test_idx"] = np.asarray(test_idx).astype(
                np.int64,
                copy=False,
            )

    np.savez_compressed(output_path, **arrays)
    return file_sha256(output_path)


def load_split_artifact_npz(
    split_set,
    *,
    frame=None,
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not split_set.artifact_uri:
        raise ValueError(f"split_set {split_set.split_set_id} has no artifact_uri")

    path = Path(split_set.artifact_uri)
    if split_set.artifact_sha256:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != split_set.artifact_sha256:
            raise ValueError(
                "artifact_sha256 mismatch for "
                f"{split_set.split_set_id}: expected {split_set.artifact_sha256}, "
                f"got {actual_sha256}"
            )

    with np.load(path, allow_pickle=False) as loaded:
        if "split_format" not in loaded.files:
            return _load_legacy_folds(loaded, split_set=split_set)

        split_format = _string_scalar(_read_array(loaded, "split_format"), key="split_format")
        _validate_compact_context(split_set, frame=frame, pk_columns=pk_columns)
        pk_columns = _normalize_pk_columns(pk_columns)
        artifact_pk_columns = _string_vector(_read_array(loaded, "pk_columns"), key="pk_columns")
        if artifact_pk_columns != pk_columns:
            raise ValueError(
                "pk_columns mismatch for "
                f"{split_set.split_set_id}: expected {pk_columns}, got {artifact_pk_columns}"
            )
        _validate_frame_identity(split_set, frame=frame, pk_columns=pk_columns)

        if split_format == FOLD_ASSIGNMENT_FORMAT:
            test_fold = _validate_test_fold(
                _read_array(loaded, "test_fold"),
                split_set=split_set,
            )
            return _folds_from_test_fold(test_fold, fold_count=split_set.fold_count)
        if split_format == HOLDOUT_ASSIGNMENT_FORMAT:
            is_testing_set = _validate_is_testing_set(
                _read_array(loaded, "is_testing_set"),
                split_set=split_set,
            )
            return {1: _indices_from_test_mask(is_testing_set)}

    raise ValueError(f"Unsupported split_format: {split_format!r}")


def _normalize_pk_columns(pk_columns: tuple[str, ...] | None) -> tuple[str, ...]:
    if not pk_columns:
        raise ValueError("pk_columns is required")
    normalized = tuple(str(column) for column in pk_columns)
    if any(not column.strip() for column in normalized):
        raise ValueError("pk_columns must not contain blank values")
    return normalized


def _small_unsigned_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _validated_indices(indices: Any, *, row_count: int, key: str) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{key} must be an integer array")
    if len(array) and (array.min() < 0 or array.max() >= row_count):
        raise ValueError(f"{key} contains row indices outside artifact row_count")
    return array.astype(np.int64, copy=False)


def _fold_assignment(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_count: int,
) -> np.ndarray:
    if not folds:
        raise ValueError("folds must contain at least one fold")

    dtype = _small_unsigned_dtype(max(folds))
    test_fold = np.zeros(row_count, dtype=dtype)
    for fold_no, (_, test_idx) in sorted(folds.items()):
        if fold_no < 1:
            raise ValueError("fold numbers must be one-based")
        test_idx = _validated_indices(test_idx, row_count=row_count, key=f"fold_{fold_no}_test_idx")
        if np.any(test_fold[test_idx] != 0):
            raise ValueError("test rows must not appear in more than one fold")
        test_fold[test_idx] = fold_no

    if np.any(test_fold == 0):
        raise ValueError("each row must appear in exactly one test fold")
    return test_fold


def _holdout_assignment(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_count: int,
) -> np.ndarray:
    if sorted(folds) != [1]:
        raise ValueError("holdout artifacts require exactly one fold")
    _, test_idx = folds[1]
    test_idx = _validated_indices(test_idx, row_count=row_count, key="fold_1_test_idx")
    is_testing_set = np.zeros(row_count, dtype=np.bool_)
    is_testing_set[test_idx] = True
    return is_testing_set


def _read_array(loaded, key: str) -> np.ndarray:
    try:
        return loaded[key]
    except KeyError as exc:
        raise ValueError(f"compact artifact missing {key}") from exc
    except ValueError as exc:
        raise ValueError(f"{key} must not be an object array") from exc


def _reject_object_array(array: np.ndarray, *, key: str) -> None:
    if array.dtype.kind == "O":
        raise ValueError(f"{key} must not be an object array")


def _string_scalar(array: np.ndarray, *, key: str) -> str:
    _reject_object_array(array, key=key)
    if array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{key} must be a string")
    if array.size != 1:
        raise ValueError(f"{key} must contain exactly one value")
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _string_vector(array: np.ndarray, *, key: str) -> tuple[str, ...]:
    _reject_object_array(array, key=key)
    if array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{key} must be a string array")
    if array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional array")
    values: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
        else:
            values.append(str(value))
    return tuple(values)


def _validate_compact_context(split_set, *, frame, pk_columns: tuple[str, ...] | None) -> None:
    if frame is None:
        raise ValueError("compact artifact requires the model frame")
    if pk_columns is None:
        raise ValueError("compact artifact requires pk_columns")


def _validate_frame_identity(
    split_set,
    *,
    frame,
    pk_columns: tuple[str, ...],
) -> None:
    if len(frame) != split_set.row_count:
        raise ValueError(
            f"row_count mismatch for {split_set.split_set_id}: "
            f"expected {split_set.row_count}, got {len(frame)}"
        )
    actual_hash = compute_row_order_sha256(frame, pk_columns=pk_columns)
    if actual_hash != split_set.row_order_sha256:
        raise ValueError(
            "row_order_sha256 mismatch for "
            f"{split_set.split_set_id}: expected {split_set.row_order_sha256}, got {actual_hash}"
        )


def _load_legacy_folds(loaded, *, split_set) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_no in range(1, split_set.fold_count + 1):
        folds[fold_no] = (
            _read_array(loaded, f"fold_{fold_no}_train_idx"),
            _read_array(loaded, f"fold_{fold_no}_test_idx"),
        )
    return folds


def _validate_assignment_length(
    array: np.ndarray,
    *,
    split_set,
    key: str,
) -> np.ndarray:
    if array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional array")
    if len(array) != split_set.row_count:
        raise ValueError(
            f"{key} length mismatch for {split_set.split_set_id}: "
            f"expected {split_set.row_count}, got {len(array)}"
        )
    return array


def _validate_test_fold(array: np.ndarray, *, split_set) -> np.ndarray:
    _reject_object_array(array, key="test_fold")
    array = _validate_assignment_length(array, split_set=split_set, key="test_fold")
    if array.dtype.kind in {"S", "U", "b"}:
        raise ValueError("test_fold must be a numeric integer array")

    if np.issubdtype(array.dtype, np.floating):
        if np.isnan(array).any() or not np.isfinite(array).all():
            raise ValueError("test_fold must not contain NaN or infinite values")
        if not np.equal(array, np.floor(array)).all():
            raise ValueError("test_fold must not contain fractional values")
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError("test_fold must be a numeric integer array")

    if np.any(array <= 0):
        raise ValueError("test_fold values must be one-based")
    if np.any(array > split_set.fold_count):
        raise ValueError("test_fold values must not exceed split_set.fold_count")
    return array.astype(np.int64, copy=False)


def _folds_from_test_fold(
    test_fold: np.ndarray,
    *,
    fold_count: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_no in range(1, fold_count + 1):
        test_mask = test_fold == fold_no
        if not np.any(test_mask):
            raise ValueError(f"test_fold does not contain fold {fold_no}")
        folds[fold_no] = _indices_from_test_mask(test_mask)
    return folds


def _validate_is_testing_set(array: np.ndarray, *, split_set) -> np.ndarray:
    _reject_object_array(array, key="is_testing_set")
    array = _validate_assignment_length(array, split_set=split_set, key="is_testing_set")
    if array.dtype.kind in {"S", "U"}:
        raise ValueError("is_testing_set must be a boolean or 0/1 integer array")
    if array.dtype.kind == "b":
        return array.astype(np.bool_, copy=False)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("is_testing_set must be a boolean or 0/1 integer array")
    if not np.isin(array, [0, 1]).all():
        raise ValueError("is_testing_set integer values must be 0 or 1")
    return array.astype(np.bool_, copy=False)


def _indices_from_test_mask(test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("split artifact produced an empty train or test fold")
    return train_idx, test_idx
