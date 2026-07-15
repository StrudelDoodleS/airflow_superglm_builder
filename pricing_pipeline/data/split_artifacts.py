from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

FOLD_ASSIGNMENT_FORMAT = "fold_assignment_v1"
HOLDOUT_ASSIGNMENT_FORMAT = "holdout_assignment_v1"
EXPLICIT_INDICES_FORMAT = "explicit_indices_v1"


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
    elif validation_split.method == "custom":
        arrays = {
            "split_format": np.array(EXPLICIT_INDICES_FORMAT),
            "pk_columns": pk_column_array,
        }
        for fold_no, (train_idx, test_idx) in sorted(folds.items()):
            arrays[f"fold_{fold_no}_train_idx"] = np.asarray(train_idx).astype(
                np.int64,
                copy=False,
            )
            arrays[f"fold_{fold_no}_test_idx"] = np.asarray(test_idx).astype(
                np.int64,
                copy=False,
            )
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


def _normalise_index_array(indices: Any, *, row_count: int, key: str) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{key} must be an integer array")
    if len(array) and (array.min() < 0 or array.max() >= row_count):
        raise ValueError(f"{key} contains row indices outside artifact row_count")
    array = array.astype(np.int64, copy=False)
    if len(array) != len(np.unique(array)):
        raise ValueError(f"{key} contains duplicate row indices")
    return array


def _index_mask(indices: np.ndarray, *, row_count: int) -> np.ndarray:
    mask = np.zeros(row_count, dtype=np.bool_)
    mask[indices] = True
    return mask


def _fold_assignment(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_count: int,
) -> np.ndarray:
    if not folds:
        raise ValueError("folds must contain at least one fold")

    dtype = _small_unsigned_dtype(max(folds))
    test_fold = np.zeros(row_count, dtype=dtype)
    for fold_no, (train_idx, test_idx) in sorted(folds.items()):
        if fold_no < 1:
            raise ValueError("fold numbers must be one-based")
        train_idx = _normalise_index_array(
            train_idx,
            row_count=row_count,
            key=f"fold_{fold_no}_train_idx",
        )
        test_idx = _normalise_index_array(
            test_idx,
            row_count=row_count,
            key=f"fold_{fold_no}_test_idx",
        )
        _validate_train_test_complement(
            train_idx,
            test_idx,
            row_count=row_count,
            train_key=f"fold_{fold_no}_train_idx",
            test_key=f"fold_{fold_no}_test_idx",
        )
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
    train_idx, test_idx = folds[1]
    train_idx = _normalise_index_array(
        train_idx,
        row_count=row_count,
        key="fold_1_train_idx",
    )
    test_idx = _normalise_index_array(test_idx, row_count=row_count, key="fold_1_test_idx")
    _validate_train_test_cover(
        train_idx,
        test_idx,
        row_count=row_count,
        train_key="fold_1_train_idx",
        test_key="fold_1_test_idx",
    )
    is_testing_set = np.zeros(row_count, dtype=np.bool_)
    is_testing_set[test_idx] = True
    return is_testing_set


def _validate_train_test_complement(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    row_count: int,
    train_key: str,
    test_key: str,
) -> None:
    if len(train_idx) == 0:
        raise ValueError(f"{train_key} must not be empty")
    if len(test_idx) == 0:
        raise ValueError(f"{test_key} must not be empty")

    train_mask = _index_mask(train_idx, row_count=row_count)
    test_mask = _index_mask(test_idx, row_count=row_count)
    if np.any(train_mask & test_mask):
        raise ValueError(f"{train_key} and {test_key} must be disjoint")
    if not np.array_equal(train_mask, ~test_mask):
        raise ValueError(f"{train_key} must equal the complement of {test_key}")


def _validate_train_test_cover(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    row_count: int,
    train_key: str,
    test_key: str,
) -> None:
    if len(train_idx) == 0:
        raise ValueError(f"{train_key} must not be empty")
    if len(test_idx) == 0:
        raise ValueError(f"{test_key} must not be empty")

    train_mask = _index_mask(train_idx, row_count=row_count)
    test_mask = _index_mask(test_idx, row_count=row_count)
    if np.any(train_mask & test_mask):
        raise ValueError(f"{train_key} and {test_key} must be disjoint")
    covered = train_mask | test_mask
    if not np.all(covered):
        raise ValueError(f"{train_key} and {test_key} must cover every row")
