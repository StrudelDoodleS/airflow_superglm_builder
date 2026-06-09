# Compact CV Split Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eligible materialized CV `.npz` artifacts with compact assignment artifacts while preserving legacy explicit train/test artifact loading.

**Architecture:** Add a shared split-artifact module that knows how to write and load legacy explicit artifacts plus compact `fold_assignment_v1` and `holdout_assignment_v1` artifacts. Manifest creation and later CV materialization both call that shared module, while SQL continues to store only split metadata and artifact pointers.

**Tech Stack:** Python 3.14, NumPy `.npz`, pandas, SQLAlchemy, pytest, existing `ValidationSplitConfig`, existing `DATASET_MANIFEST` / `CV_SPLIT_SET` / `CV_FOLD` tables.

---

## File Structure

- Create `pricing_pipeline/data/row_identity.py`
  - Owns `compute_row_order_sha256(...)`.
  - Keeps row-order PK canonicalization independent from manifest and split artifact modules.
- Create `pricing_pipeline/data/split_artifacts.py`
  - Owns `.npz` artifact writing/loading.
  - Writes compact artifacts for compact-compatible split methods.
  - Loads legacy explicit artifacts and compact artifacts.
  - Verifies artifact SHA, PK columns, row count, and row-order hash.
- Modify `pricing_pipeline/data/manifest.py`
  - Imports row identity helper from `row_identity.py`.
  - Uses `write_split_artifact_npz(...)` for materialized splits.
  - Passes `pk_columns` into artifact writing.
- Modify `pricing_pipeline/data/cv_splits.py`
  - Imports row identity helper from `row_identity.py`.
  - Uses `load_split_artifact_npz(...)` and `write_split_artifact_npz(...)`.
  - Lets materialized compact artifacts load through `load_cv_folds(...)`, where the final frame is available.
- Modify `scripts/export_cv_indices.py`
  - Keep current CLI behaviour.
  - No new flags required.
  - Continue using `load_cv_folds(...)` and `materialize_cv_folds(...)`.
- Modify `tests/test_manifest.py`
  - Update materialized split expectations from explicit arrays to compact arrays.
- Modify `tests/test_cv_splits.py`
  - Add focused artifact writer/loader tests and update materialization expectations.
- Create `tests/test_compact_cv_split_artifacts_sqlite.py`
  - Offline SQLite-style integration coverage using the existing SQL-facing manifest/load path.

No SQL migrations are part of this plan. No re-seed is required.

---

### Task 1: Extract Row Identity Helper

**Files:**
- Create: `pricing_pipeline/data/row_identity.py`
- Modify: `pricing_pipeline/data/manifest.py`
- Modify: `pricing_pipeline/data/cv_splits.py`
- Test: `tests/test_manifest.py::test_compute_row_order_sha256_depends_on_ordered_primary_keys`
- Test: `tests/test_manifest.py::test_compute_row_order_sha256_supports_composite_primary_keys`
- Test: `tests/test_cv_splits.py::test_replay_cv_folds_rejects_changed_row_order`

- [ ] **Step 1: Create the row identity module**

Create `pricing_pipeline/data/row_identity.py`:

```python
from __future__ import annotations

import hashlib
import json

import pandas as pd


def compute_row_order_sha256(
    frame: pd.DataFrame,
    *,
    pk_column: str | None = None,
    pk_columns: tuple[str, ...] | None = None,
) -> str:
    if pk_columns is None:
        if pk_column is None:
            raise ValueError("pk_column or pk_columns is required")
        pk_columns = (pk_column,)

    digest = hashlib.sha256()
    for row in frame.loc[:, list(pk_columns)].itertuples(index=False, name=None):
        digest.update(json.dumps(row, default=str, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
```

- [ ] **Step 2: Update imports in manifest.py**

In `pricing_pipeline/data/manifest.py`, add:

```python
from pricing_pipeline.data.row_identity import compute_row_order_sha256
```

Then delete the local `compute_row_order_sha256(...)` function from `manifest.py`.

- [ ] **Step 3: Update imports in cv_splits.py**

In `pricing_pipeline/data/cv_splits.py`, replace:

```python
from pricing_pipeline.data.manifest import compute_row_order_sha256
```

with:

```python
from pricing_pipeline.data.row_identity import compute_row_order_sha256
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
rtk uv run pytest -q tests/test_manifest.py::test_compute_row_order_sha256_depends_on_ordered_primary_keys tests/test_manifest.py::test_compute_row_order_sha256_supports_composite_primary_keys tests/test_cv_splits.py::test_replay_cv_folds_rejects_changed_row_order
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/data/row_identity.py pricing_pipeline/data/manifest.py pricing_pipeline/data/cv_splits.py
rtk git commit -m "refactor: extract row identity hashing"
```

---

### Task 2: Add Split Artifact Writer and Loader

**Files:**
- Create: `pricing_pipeline/data/split_artifacts.py`
- Modify: `tests/test_cv_splits.py`

- [ ] **Step 1: Add failing writer/loader tests**

Append these imports to `tests/test_cv_splits.py`:

```python
from pricing_pipeline.data.split_artifacts import FOLD_ASSIGNMENT_FORMAT
from pricing_pipeline.data.split_artifacts import HOLDOUT_ASSIGNMENT_FORMAT
from pricing_pipeline.data.split_artifacts import load_split_artifact_npz
from pricing_pipeline.data.split_artifacts import write_split_artifact_npz
from pricing_pipeline.models.config import ValidationSplitConfig
```

Append these tests to `tests/test_cv_splits.py`:

```python
def test_write_split_artifact_npz_writes_compact_kfold_assignment(tmp_path: Path):
    rows = frame()
    folds = {
        1: (np.array([0, 2, 3]), np.array([1, 4])),
        2: (np.array([1, 3, 4]), np.array([0, 2])),
        3: (np.array([0, 1, 2, 4]), np.array([3])),
    }
    artifact = tmp_path / "compact_kfold.npz"

    artifact_sha = write_split_artifact_npz(
        folds,
        validation_split=ValidationSplitConfig.kfold(n_splits=3, materialize=True),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert artifact_sha
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert str(loaded["split_format"].item()) == FOLD_ASSIGNMENT_FORMAT
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["test_fold"].dtype == np.uint8
    assert loaded["test_fold"].tolist() == [2, 1, 2, 3, 1]
    assert "fold_1_train_idx" not in loaded.files


def test_load_split_artifact_npz_reconstructs_compact_kfold(tmp_path: Path):
    rows = frame()
    artifact = tmp_path / "compact_kfold.npz"
    artifact_sha = write_split_artifact_npz(
        {
            1: (np.array([0, 2, 3]), np.array([1, 4])),
            2: (np.array([1, 3, 4]), np.array([0, 2])),
            3: (np.array([0, 1, 2, 4]), np.array([3])),
        },
        validation_split=ValidationSplitConfig.kfold(n_splits=3, materialize=True),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": artifact_sha,
        }
    )

    folds = load_split_artifact_npz(materialized, frame=rows, pk_columns=("IDpol",))

    assert folds[1][0].tolist() == [0, 2, 3]
    assert folds[1][1].tolist() == [1, 4]
    assert folds[3][0].tolist() == [0, 1, 2, 4]
    assert folds[3][1].tolist() == [3]


def test_write_split_artifact_npz_writes_compact_holdout_assignment(tmp_path: Path):
    rows = frame()
    artifact = tmp_path / "compact_holdout.npz"

    write_split_artifact_npz(
        {1: (np.array([0, 2, 3]), np.array([1, 4]))},
        validation_split=ValidationSplitConfig.train_test_split(test_size=0.4, materialize=True),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert str(loaded["split_format"].item()) == HOLDOUT_ASSIGNMENT_FORMAT
    assert loaded["is_testing_set"].dtype == np.bool_
    assert loaded["is_testing_set"].tolist() == [False, True, False, False, True]


def test_load_split_artifact_npz_rejects_compact_without_frame(tmp_path: Path):
    artifact = tmp_path / "compact_kfold.npz"
    artifact_sha = write_split_artifact_npz(
        {
            1: (np.array([0, 2, 3]), np.array([1, 4])),
            2: (np.array([1, 3, 4]), np.array([0, 2])),
            3: (np.array([0, 1, 2, 4]), np.array([3])),
        },
        validation_split=ValidationSplitConfig.kfold(n_splits=3, materialize=True),
        pk_columns=("IDpol",),
        row_count=5,
        output_path=artifact,
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": artifact_sha,
        }
    )

    with pytest.raises(ValueError, match="compact artifact requires the model frame"):
        load_split_artifact_npz(materialized)


def test_load_split_artifact_npz_keeps_legacy_explicit_support(tmp_path: Path):
    artifact = tmp_path / "legacy.npz"
    np.savez_compressed(
        artifact,
        fold_1_train_idx=np.array([0, 2, 3]),
        fold_1_test_idx=np.array([1, 4]),
        fold_2_train_idx=np.array([1, 3, 4]),
        fold_2_test_idx=np.array([0, 2]),
        fold_3_train_idx=np.array([0, 1, 2, 4]),
        fold_3_test_idx=np.array([3]),
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    folds = load_split_artifact_npz(materialized)

    assert folds[1][0].tolist() == [0, 2, 3]
    assert folds[1][1].tolist() == [1, 4]
    assert folds[3][1].tolist() == [3]


@pytest.mark.parametrize(
    "bad_test_fold",
    [
        np.array([1, 2, 0, 3, 1], dtype=np.int64),
        np.array([1, 2, -1, 3, 1], dtype=np.int64),
        np.array([1, 2, 4, 3, 1], dtype=np.int64),
        np.array([1.0, 2.0, 1.5, 3.0, 1.0], dtype=np.float64),
        np.array(["1", "2", "1", "3", "1"]),
    ],
)
def test_load_split_artifact_npz_rejects_bad_test_fold_values(
    tmp_path: Path,
    bad_test_fold: np.ndarray,
):
    artifact = tmp_path / "bad_fold.npz"
    np.savez_compressed(
        artifact,
        split_format=np.array(FOLD_ASSIGNMENT_FORMAT),
        pk_columns=np.array(["IDpol"]),
        test_fold=bad_test_fold,
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    with pytest.raises(ValueError, match="test_fold"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))


@pytest.mark.parametrize(
    "bad_is_testing_set",
    [
        np.array([0, 1, 2, 0, 1], dtype=np.int64),
        np.array(["False", "True", "False", "False", "True"]),
    ],
)
def test_load_split_artifact_npz_rejects_bad_is_testing_set_values(
    tmp_path: Path,
    bad_is_testing_set: np.ndarray,
):
    artifact = tmp_path / "bad_holdout.npz"
    np.savez_compressed(
        artifact,
        split_format=np.array(HOLDOUT_ASSIGNMENT_FORMAT),
        pk_columns=np.array(["IDpol"]),
        is_testing_set=bad_is_testing_set,
    )
    materialized = CVSplitSet(
        **{
            **train_test_split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    with pytest.raises(ValueError, match="is_testing_set"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run pytest -q tests/test_cv_splits.py::test_write_split_artifact_npz_writes_compact_kfold_assignment tests/test_cv_splits.py::test_load_split_artifact_npz_reconstructs_compact_kfold tests/test_cv_splits.py::test_write_split_artifact_npz_writes_compact_holdout_assignment tests/test_cv_splits.py::test_load_split_artifact_npz_rejects_compact_without_frame tests/test_cv_splits.py::test_load_split_artifact_npz_keeps_legacy_explicit_support
```

Expected: fail with `ModuleNotFoundError: No module named 'pricing_pipeline.data.split_artifacts'`.

- [ ] **Step 3: Create split_artifacts.py**

Create `pricing_pipeline/data/split_artifacts.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pricing_pipeline.data.row_identity import compute_row_order_sha256
from pricing_pipeline.models.config import ValidationSplitConfig


FOLD_ASSIGNMENT_FORMAT = "fold_assignment_v1"
HOLDOUT_ASSIGNMENT_FORMAT = "holdout_assignment_v1"

_FOLD_ASSIGNMENT_METHODS = {"kfold", "column_kfold"}
_HOLDOUT_ASSIGNMENT_METHODS = {"train_test_split", "column_holdout"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_dtype(fold_count: int):
    if fold_count <= np.iinfo(np.uint8).max:
        return np.uint8
    if fold_count <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _normalise_index_array(values: np.ndarray, *, row_count: int, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{field_name} must contain integer row positions")
    index = array.astype(np.int64, copy=False)
    if len(index) and (int(index.min()) < 0 or int(index.max()) >= row_count):
        raise ValueError(f"{field_name} contains row positions outside 0..{row_count - 1}")
    if len(np.unique(index)) != len(index):
        raise ValueError(f"{field_name} must not contain duplicate row positions")
    return index


def _normalise_folds(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_count: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    normalised: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for raw_fold_no, (train_idx, test_idx) in sorted(folds.items()):
        fold_no = int(raw_fold_no)
        normalised[fold_no] = (
            _normalise_index_array(train_idx, row_count=row_count, field_name=f"fold_{fold_no}_train_idx"),
            _normalise_index_array(test_idx, row_count=row_count, field_name=f"fold_{fold_no}_test_idx"),
        )
    return normalised


def _legacy_arrays(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for fold_no, (train_idx, test_idx) in _normalise_folds(folds, row_count=row_count).items():
        arrays[f"fold_{fold_no}_train_idx"] = train_idx.astype(np.int64, copy=False)
        arrays[f"fold_{fold_no}_test_idx"] = test_idx.astype(np.int64, copy=False)
    return arrays


def _fold_assignment_arrays(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    pk_columns: tuple[str, ...],
    row_count: int,
) -> dict[str, np.ndarray]:
    normalised = _normalise_folds(folds, row_count=row_count)
    fold_count = len(normalised)
    seen_test = np.zeros(row_count, dtype=bool)
    test_fold = np.zeros(row_count, dtype=_fold_dtype(fold_count))
    for fold_no, (_, test_idx) in normalised.items():
        if fold_no < 1 or fold_no > fold_count:
            raise ValueError(f"fold numbers must be contiguous from 1 to {fold_count}")
        if np.any(seen_test[test_idx]):
            raise ValueError("compact fold assignment requires each row to be test exactly once")
        seen_test[test_idx] = True
        test_fold[test_idx] = fold_no
    if not bool(np.all(seen_test)):
        raise ValueError("compact fold assignment requires every row to appear in one test fold")
    return {
        "split_format": np.array(FOLD_ASSIGNMENT_FORMAT),
        "pk_columns": np.asarray(pk_columns, dtype=np.str_),
        "test_fold": test_fold,
    }


def _holdout_assignment_arrays(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    pk_columns: tuple[str, ...],
    row_count: int,
) -> dict[str, np.ndarray]:
    normalised = _normalise_folds(folds, row_count=row_count)
    if sorted(normalised) != [1]:
        raise ValueError("compact holdout assignment requires exactly one fold")
    train_idx, test_idx = normalised[1]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("compact holdout assignment requires train and test rows")
    assigned = np.zeros(row_count, dtype=bool)
    assigned[train_idx] = True
    if np.any(assigned[test_idx]):
        raise ValueError("compact holdout assignment requires disjoint train and test rows")
    assigned[test_idx] = True
    if not bool(np.all(assigned)):
        raise ValueError("compact holdout assignment requires every row to be train or test")
    is_testing_set = np.zeros(row_count, dtype=bool)
    is_testing_set[test_idx] = True
    return {
        "split_format": np.array(HOLDOUT_ASSIGNMENT_FORMAT),
        "pk_columns": np.asarray(pk_columns, dtype=np.str_),
        "is_testing_set": is_testing_set,
    }


def write_split_artifact_npz(
    folds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    validation_split: ValidationSplitConfig,
    pk_columns: tuple[str, ...],
    row_count: int,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_split.method in _FOLD_ASSIGNMENT_METHODS:
        arrays = _fold_assignment_arrays(folds, pk_columns=pk_columns, row_count=row_count)
    elif validation_split.method in _HOLDOUT_ASSIGNMENT_METHODS:
        arrays = _holdout_assignment_arrays(folds, pk_columns=pk_columns, row_count=row_count)
    else:
        arrays = _legacy_arrays(folds, row_count=row_count)
    np.savez_compressed(output_path, **arrays)
    return file_sha256(output_path)


def _artifact_format(loaded: np.lib.npyio.NpzFile) -> str | None:
    if "split_format" not in loaded.files:
        return None
    value = loaded["split_format"]
    if value.dtype.kind == "O":
        raise ValueError("split_format must not be an object array")
    return str(value.reshape(()).item())


def _artifact_pk_columns(loaded: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if "pk_columns" not in loaded.files:
        raise ValueError("compact artifact missing pk_columns")
    values = loaded["pk_columns"]
    if values.dtype.kind == "O":
        raise ValueError("pk_columns must not be an object array")
    return tuple(str(item) for item in values.tolist())


def _verify_compact_context(
    split_set: Any,
    loaded: np.lib.npyio.NpzFile,
    *,
    frame: pd.DataFrame | None,
    pk_columns: tuple[str, ...] | None,
    assignment_length: int,
) -> None:
    if frame is None:
        raise ValueError("compact artifact requires the model frame for row identity validation")
    if pk_columns is None:
        raise ValueError("compact artifact requires pk_columns")
    artifact_pk_columns = _artifact_pk_columns(loaded)
    if artifact_pk_columns != tuple(pk_columns):
        raise ValueError(
            f"artifact pk_columns mismatch: expected {tuple(pk_columns)!r}, got {artifact_pk_columns!r}"
        )
    if int(split_set.row_count) != assignment_length:
        raise ValueError(
            f"row_count mismatch for {split_set.split_set_id}: "
            f"expected {split_set.row_count}, got {assignment_length}"
        )
    if len(frame) != int(split_set.row_count):
        raise ValueError(
            f"row_count mismatch for {split_set.split_set_id}: "
            f"expected {split_set.row_count}, got {len(frame)}"
        )
    actual_hash = compute_row_order_sha256(frame, pk_columns=tuple(pk_columns))
    if actual_hash != split_set.row_order_sha256:
        raise ValueError(
            "row_order_sha256 mismatch for "
            f"{split_set.split_set_id}: expected {split_set.row_order_sha256}, got {actual_hash}"
        )


def _load_legacy_explicit(
    split_set: Any,
    loaded: np.lib.npyio.NpzFile,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_no in range(1, int(split_set.fold_count) + 1):
        folds[fold_no] = (
            loaded[f"fold_{fold_no}_train_idx"],
            loaded[f"fold_{fold_no}_test_idx"],
        )
    return folds


def _coerce_test_fold(raw: np.ndarray, *, fold_count: int) -> np.ndarray:
    if raw.ndim != 1:
        raise ValueError("test_fold must be one-dimensional")
    if raw.dtype.kind in {"i", "u"}:
        test_fold = raw.astype(np.int64, copy=False)
    elif raw.dtype.kind == "f":
        if not np.isfinite(raw).all():
            raise ValueError("test_fold must not contain null or NaN values")
        if not np.equal(raw, np.floor(raw)).all():
            raise ValueError("test_fold must contain integer-like values")
        test_fold = raw.astype(np.int64)
    else:
        raise ValueError("test_fold must contain integer-like values")
    if len(test_fold) and (int(test_fold.min()) < 1 or int(test_fold.max()) > fold_count):
        raise ValueError(f"test_fold values must be in 1..{fold_count}")
    return test_fold


def _load_fold_assignment(
    split_set: Any,
    loaded: np.lib.npyio.NpzFile,
    *,
    frame: pd.DataFrame | None,
    pk_columns: tuple[str, ...] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if "test_fold" not in loaded.files:
        raise ValueError("fold_assignment_v1 artifact missing test_fold")
    test_fold = _coerce_test_fold(loaded["test_fold"], fold_count=int(split_set.fold_count))
    _verify_compact_context(
        split_set,
        loaded,
        frame=frame,
        pk_columns=pk_columns,
        assignment_length=len(test_fold),
    )
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_no in range(1, int(split_set.fold_count) + 1):
        test_mask = test_fold == fold_no
        train_mask = ~test_mask
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("compact fold assignment produced an empty train or test fold")
        folds[fold_no] = (train_idx, test_idx)
    return folds


def _coerce_is_testing_set(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 1:
        raise ValueError("is_testing_set must be one-dimensional")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=False)
    if raw.dtype.kind in {"i", "u"}:
        unique_values = set(np.unique(raw).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError("is_testing_set integer values must be 0 or 1")
        return raw.astype(bool, copy=False)
    raise ValueError("is_testing_set must be boolean or integer 0/1")


def _load_holdout_assignment(
    split_set: Any,
    loaded: np.lib.npyio.NpzFile,
    *,
    frame: pd.DataFrame | None,
    pk_columns: tuple[str, ...] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if "is_testing_set" not in loaded.files:
        raise ValueError("holdout_assignment_v1 artifact missing is_testing_set")
    is_testing_set = _coerce_is_testing_set(loaded["is_testing_set"])
    _verify_compact_context(
        split_set,
        loaded,
        frame=frame,
        pk_columns=pk_columns,
        assignment_length=len(is_testing_set),
    )
    train_idx = np.flatnonzero(~is_testing_set)
    test_idx = np.flatnonzero(is_testing_set)
    if len(train_idx) == 0:
        raise ValueError("compact holdout assignment produced no train rows")
    if len(test_idx) == 0:
        raise ValueError("compact holdout assignment produced no test rows")
    return {1: (train_idx, test_idx)}


def load_split_artifact_npz(
    split_set: Any,
    *,
    frame: pd.DataFrame | None = None,
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
        split_format = _artifact_format(loaded)
        if split_format is None:
            return _load_legacy_explicit(split_set, loaded)
        if split_format == FOLD_ASSIGNMENT_FORMAT:
            return _load_fold_assignment(
                split_set,
                loaded,
                frame=frame,
                pk_columns=pk_columns,
            )
        if split_format == HOLDOUT_ASSIGNMENT_FORMAT:
            return _load_holdout_assignment(
                split_set,
                loaded,
                frame=frame,
                pk_columns=pk_columns,
            )
        raise ValueError(f"Unsupported split_format: {split_format}")
```

- [ ] **Step 4: Run split artifact tests**

Run:

```bash
rtk uv run pytest -q tests/test_cv_splits.py::test_write_split_artifact_npz_writes_compact_kfold_assignment tests/test_cv_splits.py::test_load_split_artifact_npz_reconstructs_compact_kfold tests/test_cv_splits.py::test_write_split_artifact_npz_writes_compact_holdout_assignment tests/test_cv_splits.py::test_load_split_artifact_npz_rejects_compact_without_frame tests/test_cv_splits.py::test_load_split_artifact_npz_keeps_legacy_explicit_support tests/test_cv_splits.py::test_load_split_artifact_npz_rejects_bad_test_fold_values tests/test_cv_splits.py::test_load_split_artifact_npz_rejects_bad_is_testing_set_values
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/data/split_artifacts.py tests/test_cv_splits.py
rtk git commit -m "feat: add compact cv split artifact helpers"
```

---

### Task 3: Wire Manifest Materialization to Compact Artifacts

**Files:**
- Modify: `pricing_pipeline/data/manifest.py`
- Modify: `tests/test_manifest.py`

- [ ] **Step 1: Update manifest artifact tests to expect compact arrays**

In `tests/test_manifest.py`, update the train/test materialization assertion that currently expects `["fold_1_test_idx", "fold_1_train_idx"]`.

Replace that assertion block with:

```python
    loaded = np.load(artifact_path, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert str(loaded["split_format"].item()) == "holdout_assignment_v1"
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["is_testing_set"].dtype == np.bool_
    assert int(loaded["is_testing_set"].sum()) == 2
    assert "fold_1_train_idx" not in loaded.files
```

Add this new test to `tests/test_manifest.py` near the existing materialization tests:

```python
def test_create_model_frame_manifest_materializes_compact_column_kfold_artifact(
    monkeypatch,
    tmp_path: Path,
):
    engine = FakeEngine()
    raw_frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "ClaimNb": [0, 1, 0, 2],
            "Exposure": [1.0, 0.5, 1.5, 0.25],
            "fold_number": [1, 2, 1, 2],
        }
    )
    to_sql_calls = []

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append({"name": name, "frame": self.copy()})

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    result = create_model_frame_manifest_with_split(
        engine,
        frame=raw_frame,
        spec=ModelFrameManifestSpec(
            dataset_name="unit_model_frame",
            source_system="unit",
            data_as_of_date="2026-06-09",
            pk_columns=("PolicyID",),
            target_column="ClaimNb",
            weight_column="Exposure",
        ),
        manifest_id="manifest_column_kfold",
        validation_split=ValidationSplitConfig.column_kfold(
            column="fold_number",
            materialize=True,
        ),
        validation_split_artifact_root=tmp_path,
        created_by="unit-test",
    )

    artifact_path = tmp_path / "manifest_column_kfold__column_kfold_fold_number.npz"
    loaded = np.load(artifact_path, allow_pickle=False)
    assert result.split_artifact_uri == str(artifact_path)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert str(loaded["split_format"].item()) == "fold_assignment_v1"
    assert loaded["pk_columns"].tolist() == ["PolicyID"]
    assert loaded["test_fold"].tolist() == [1, 2, 1, 2]

    split_set = next(call for call in to_sql_calls if call["name"] == "CV_SPLIT_SET")[
        "frame"
    ].iloc[0]
    assert split_set["artifact_uri"] == str(artifact_path)
    assert len(split_set["artifact_sha256"]) == 64
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run pytest -q tests/test_manifest.py::test_create_dataset_manifest_with_train_test_split_materializes_artifact tests/test_manifest.py::test_create_model_frame_manifest_materializes_compact_column_kfold_artifact
```

Expected: fail because `write_validation_split_npz(...)` still writes legacy explicit arrays and does not accept `pk_columns`.

- [ ] **Step 3: Update manifest.py to use the shared writer**

In `pricing_pipeline/data/manifest.py`, replace:

```python
from pricing_pipeline.models.config import ValidationSplitConfig
```

with:

```python
from pricing_pipeline.data.row_identity import compute_row_order_sha256
from pricing_pipeline.data.split_artifacts import write_split_artifact_npz
from pricing_pipeline.models.config import ValidationSplitConfig
```

Then replace `write_validation_split_npz(...)` with:

```python
def write_validation_split_npz(
    frame: pd.DataFrame,
    *,
    validation_split: ValidationSplitConfig,
    output_path: Path,
    pk_columns: tuple[str, ...] = ("IDpol",),
) -> str:
    folds = {
        fold_no: (train_idx, test_idx)
        for fold_no, (train_idx, test_idx) in enumerate(
            validation_split_indices(frame, validation_split),
            start=1,
        )
    }
    return write_split_artifact_npz(
        folds,
        validation_split=validation_split,
        pk_columns=pk_columns,
        row_count=len(frame),
        output_path=output_path,
    )
```

Then update the call inside `create_model_frame_manifest_with_split(...)`:

```python
        split_artifact_sha256 = write_validation_split_npz(
            frame,
            validation_split=validation_split,
            output_path=artifact_path,
            pk_columns=spec.pk_columns,
        )
```

- [ ] **Step 4: Run focused manifest tests**

Run:

```bash
rtk uv run pytest -q tests/test_manifest.py::test_create_dataset_manifest_with_train_test_split_materializes_artifact tests/test_manifest.py::test_create_model_frame_manifest_materializes_compact_column_kfold_artifact
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/data/manifest.py tests/test_manifest.py
rtk git commit -m "feat: write compact cv artifacts from manifests"
```

---

### Task 4: Wire CV Split Loading and Later Materialization

**Files:**
- Modify: `pricing_pipeline/data/cv_splits.py`
- Modify: `tests/test_cv_splits.py`

- [ ] **Step 1: Add failing public-path tests**

Append these tests to `tests/test_cv_splits.py`:

```python
def test_load_cv_folds_reconstructs_materialized_compact_artifact_from_dataset_loader(
    tmp_path: Path,
):
    rows = frame()
    artifact = tmp_path / "compact_kfold.npz"
    artifact_sha = write_split_artifact_npz(
        {
            1: (np.array([0, 2, 3]), np.array([1, 4])),
            2: (np.array([1, 3, 4]), np.array([0, 2])),
            3: (np.array([0, 1, 2, 4]), np.array([3])),
        },
        validation_split=ValidationSplitConfig.kfold(n_splits=3, materialize=True),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": artifact_sha,
        }
    )

    class ScalarResult:
        def mappings(self):
            return self

        def one(self):
            return materialized.__dict__

    class Connection:
        def execute(self, statement, params=None):
            return ScalarResult()

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    folds = load_cv_folds(
        Engine(),
        materialized.split_set_id,
        dataset_loader=lambda manifest_id: rows,
        pk_columns=("IDpol",),
    )

    assert folds[1][1].tolist() == [1, 4]
    assert folds[2][1].tolist() == [0, 2]
    assert folds[3][1].tolist() == [3]


def test_materialize_cv_folds_writes_compact_artifact_for_replayable_kfold(
    tmp_path: Path,
):
    executed = []

    class Connection:
        def execute(self, statement, params=None):
            executed.append((str(statement), params))

    class Engine:
        def begin(self):
            return self

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    output_path = tmp_path / "manifest_1__kfold_3_seed_42.npz"

    result = materialize_cv_folds(
        Engine(),
        split_set(),
        frame(),
        output_path=output_path,
        pk_columns=("IDpol",),
    )

    assert result == output_path
    loaded = np.load(output_path, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert loaded["test_fold"].tolist() == [2, 1, 2, 3, 1]
    assert executed[0][1]["artifact_uri"] == str(output_path)
    assert len(executed[0][1]["artifact_sha256"]) == 64
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run pytest -q tests/test_cv_splits.py::test_load_cv_folds_reconstructs_materialized_compact_artifact_from_dataset_loader tests/test_cv_splits.py::test_materialize_cv_folds_writes_compact_artifact_for_replayable_kfold
```

Expected: fail because `load_cv_folds(...)` does not pass a frame into materialized loading and `materialize_cv_folds(...)` still writes explicit arrays.

- [ ] **Step 3: Update cv_splits.py imports**

In `pricing_pipeline/data/cv_splits.py`, replace split artifact related imports with:

```python
from pricing_pipeline.data.row_identity import compute_row_order_sha256
from pricing_pipeline.data.split_artifacts import load_split_artifact_npz
from pricing_pipeline.data.split_artifacts import write_split_artifact_npz
```

Remove the local `file_sha256(...)` function from `cv_splits.py`.

- [ ] **Step 4: Update replay and materialized APIs**

In `pricing_pipeline/data/cv_splits.py`, update `replay_cv_folds(...)` signature and row-order hash call:

```python
def replay_cv_folds(
    split_set: CVSplitSet,
    frame: pd.DataFrame,
    *,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if pk_columns is None:
        if pk_column is None:
            raise ValueError("pk_column or pk_columns is required")
        pk_columns = (pk_column,)
    actual_hash = compute_row_order_sha256(frame, pk_columns=pk_columns)
```

Leave the existing row-count validation, train-test replay, source-column replay,
and sklearn replay branches in place after the `actual_hash` comparison.

Replace `load_materialized_cv_folds(...)` with:

```python
def load_materialized_cv_folds(
    split_set: CVSplitSet,
    *,
    frame: pd.DataFrame | None = None,
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return load_split_artifact_npz(split_set, frame=frame, pk_columns=pk_columns)
```

Before `materialize_cv_folds(...)`, add:

```python
def _validation_split_from_split_set(split_set: CVSplitSet):
    from pricing_pipeline.models.config import ValidationSplitConfig

    params = json.loads(split_set.splitter_params_json or "{}")
    if split_set.splitter_class == "sklearn.model_selection.KFold":
        return ValidationSplitConfig.kfold(
            n_splits=int(params.get("n_splits", split_set.fold_count)),
            random_state=params.get("random_state"),
            shuffle=bool(params.get("shuffle", True)),
            materialize=True,
        )
    if split_set.splitter_class == "sklearn.model_selection.train_test_split":
        return ValidationSplitConfig.train_test_split(
            test_size=float(params.get("test_size", 0.2)),
            random_state=params.get("random_state"),
            shuffle=bool(params.get("shuffle", True)),
            stratify_column=params.get("stratify_column"),
            materialize=True,
        )
    if split_set.splitter_class == "source_column" and params.get("method") == "column_kfold":
        return ValidationSplitConfig.column_kfold(
            column=str(params["column"]),
            materialize=True,
        )
    if split_set.splitter_class == "source_column" and params.get("method") == "column_holdout":
        return ValidationSplitConfig.column_holdout(
            column=str(params["column"]),
            train_values=tuple(params.get("train_values", ())),
            test_values=tuple(params.get("test_values", ())),
            materialize=True,
        )
    return ValidationSplitConfig.none()
```

Update `materialize_cv_folds(...)` signature:

```python
def materialize_cv_folds(
    engine: Engine,
    split_set: CVSplitSet,
    frame: pd.DataFrame,
    *,
    output_path: Path,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> Path:
```

Then replace the full function body with:

```python
    resolved_pk_columns = pk_columns or ((pk_column,) if pk_column is not None else ("IDpol",))
    folds = replay_cv_folds(
        split_set,
        frame,
        pk_column=pk_column,
        pk_columns=resolved_pk_columns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_sha256 = write_split_artifact_npz(
        folds,
        validation_split=_validation_split_from_split_set(split_set),
        pk_columns=resolved_pk_columns,
        row_count=len(frame),
        output_path=output_path,
    )

    with engine.begin() as con:
        con.execute(
            text(
                """
                UPDATE pricing.CV_SPLIT_SET
                SET
                    split_mode = 'MATERIALIZED',
                    artifact_uri = :artifact_uri,
                    artifact_sha256 = :artifact_sha256,
                    runtime_metadata_json = :runtime_metadata_json
                WHERE split_set_id = :split_set_id
                """
            ),
            {
                "split_set_id": split_set.split_set_id,
                "artifact_uri": str(output_path),
                "artifact_sha256": artifact_sha256,
                "runtime_metadata_json": runtime_dependency_metadata(),
            },
        )

    return output_path
```

Update `load_cv_folds(...)`:

```python
def load_cv_folds(
    engine: Engine,
    split_set_id: str,
    *,
    dataset_loader,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    split_set = fetch_split_set(engine, split_set_id)
    if pk_columns is None:
        if pk_column is None:
            raise ValueError("pk_column or pk_columns is required")
        pk_columns = (pk_column,)
    frame = dataset_loader(split_set.manifest_id)
    if split_set.split_mode == "MATERIALIZED":
        return load_materialized_cv_folds(split_set, frame=frame, pk_columns=pk_columns)
    return replay_cv_folds(split_set, frame, pk_columns=pk_columns)
```

- [ ] **Step 5: Update old expectations in tests**

In `tests/test_cv_splits.py`, update `test_materialize_cv_folds_writes_all_folds_updates_split_set_and_hashes_artifact` so it expects compact arrays:

```python
    loaded = np.load(output_path, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert loaded["test_fold"].tolist() == [2, 1, 2, 3, 1]
```

In `test_materialize_cv_folds_supports_replayable_source_column_split_set`, update the artifact assertion:

```python
    loaded = np.load(output_path, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert loaded["test_fold"].tolist() == [2, 1, 2, 1]
```

Keep `test_load_materialized_cv_folds_checks_artifact_hash(...)` as a legacy explicit artifact hash test.

- [ ] **Step 6: Run focused CV split tests**

Run:

```bash
rtk uv run pytest -q tests/test_cv_splits.py
```

Expected: all `test_cv_splits.py` tests pass.

- [ ] **Step 7: Commit**

```bash
rtk git add pricing_pipeline/data/cv_splits.py tests/test_cv_splits.py
rtk git commit -m "feat: load and materialize compact cv artifacts"
```

---

### Task 5: Add Offline SQLite-Style Integration Coverage

**Files:**
- Create: `tests/test_compact_cv_split_artifacts_sqlite.py`

- [ ] **Step 1: Add the offline integration test**

Create `tests/test_compact_cv_split_artifacts_sqlite.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from pricing_pipeline.data.cv_splits import load_cv_folds
from pricing_pipeline.data.cv_splits import fetch_split_set
from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.data.manifest import create_model_frame_manifest_with_split
from pricing_pipeline.models.config import ValidationSplitConfig


def sqlite_engine_with_pricing_schema(tmp_path: Path):
    pricing_db = tmp_path / "pricing.sqlite"
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _attach_pricing_schema(dbapi_connection, connection_record):
        dbapi_connection.execute(f"ATTACH DATABASE '{pricing_db.as_posix()}' AS pricing")

    return engine


def test_compact_materialized_manifest_loads_offline_with_sqlite(tmp_path: Path):
    engine = sqlite_engine_with_pricing_schema(tmp_path)
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "SnapshotMonth": ["2026-05", "2026-05", "2026-05", "2026-05"],
            "ClaimNb": [0, 1, 0, 2],
            "Exposure": [1.0, 0.5, 1.5, 0.25],
            "train_holdout": ["train", "holdout", "train", "train"],
        }
    )

    result = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=ModelFrameManifestSpec(
            dataset_name="sqlite_model_frame",
            source_system="sqlite_unit",
            data_as_of_date="2026-06-09",
            pk_columns=("PolicyID", "SnapshotMonth"),
            target_column="ClaimNb",
            weight_column="Exposure",
        ),
        manifest_id="sqlite_manifest_1",
        validation_split=ValidationSplitConfig.column_holdout(
            column="train_holdout",
            train_values=("train",),
            test_values=("holdout",),
            materialize=True,
        ),
        validation_split_artifact_root=tmp_path,
        created_by="unit-test",
    )

    split_set = fetch_split_set(engine, result.split_set_id)
    folds = load_cv_folds(
        engine,
        result.split_set_id,
        dataset_loader=lambda manifest_id: frame,
        pk_columns=("PolicyID", "SnapshotMonth"),
    )

    artifact_path = Path(split_set.artifact_uri)
    loaded = np.load(artifact_path, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert "fold_1_train_idx" not in loaded.files
    assert str(loaded["split_format"].item()) == "holdout_assignment_v1"
    assert loaded["pk_columns"].tolist() == ["PolicyID", "SnapshotMonth"]
    assert loaded["is_testing_set"].tolist() == [False, True, False, False]
    assert folds[1][0].tolist() == [0, 2, 3]
    assert folds[1][1].tolist() == [1]

    cv_folds = pd.read_sql_query(
        text("SELECT fold_no, n_train, n_test FROM pricing.CV_FOLD ORDER BY fold_no"),
        engine,
    )
    assert cv_folds.to_dict("records") == [{"fold_no": 1, "n_train": 3, "n_test": 1}]

    manifest_rows = pd.read_sql_query(
        text("SELECT pk_columns_json FROM pricing.DATASET_MANIFEST WHERE manifest_id = :id"),
        engine,
        params={"id": "sqlite_manifest_1"},
    )
    assert json.loads(manifest_rows.iloc[0]["pk_columns_json"]) == [
        "PolicyID",
        "SnapshotMonth",
    ]
```

- [ ] **Step 2: Run the offline test**

Run:

```bash
rtk uv run pytest -q tests/test_compact_cv_split_artifacts_sqlite.py
```

Expected: test passes without SQL Server.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/test_compact_cv_split_artifacts_sqlite.py
rtk git commit -m "test: add offline compact cv artifact coverage"
```

---

### Task 6: Full Verification and Cleanup

**Files:**
- Review all touched files.

- [ ] **Step 1: Run focused suites**

Run:

```bash
rtk uv run pytest -q tests/test_cv_splits.py tests/test_manifest.py tests/test_compact_cv_split_artifacts_sqlite.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
rtk uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run lint and formatting checks**

Run:

```bash
rtk uv run ruff check .
rtk uv run ruff format --check pricing_pipeline/data/row_identity.py pricing_pipeline/data/split_artifacts.py pricing_pipeline/data/manifest.py pricing_pipeline/data/cv_splits.py tests/test_cv_splits.py tests/test_manifest.py tests/test_compact_cv_split_artifacts_sqlite.py
```

Expected: both commands pass.

- [ ] **Step 4: Run no-Docker and SQL syntax checks**

Run:

```bash
rtk uv run python scripts/no_docker_services.py menu --dry-run
rtk uv run pytest -q tests/test_sql_server_syntax.py
rtk uv run sqlfluff parse --dialect tsql db/migrations
```

Expected: all commands pass. No SQL migrations were added, so these validate no regressions.

- [ ] **Step 5: Check diff hygiene**

Run:

```bash
rtk git diff --check
rtk git status --short --branch
```

Expected: `git diff --check` prints nothing. `git status` shows only intended branch state.

- [ ] **Step 6: Commit verification-only cleanup if needed**

If formatting or lint changes were required, commit them:

```bash
rtk git add pricing_pipeline/data tests scripts
rtk git commit -m "chore: tidy compact cv artifact implementation"
```

If no changes were required after verification, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage:
  - Compact `fold_assignment_v1` and `holdout_assignment_v1` artifacts: Task 2.
  - `is_testing_set` naming: Task 2.
  - Lower snake case artifact keys: Task 2.
  - Lean v1 with no `row_key_hash`: Task 2 and Task 3.
  - Artifact interpreted with `CV_SPLIT_SET`: Task 2 and Task 4.
  - SHA verification before trusting arrays: Task 2.
  - `np.load(..., allow_pickle=False)`: Task 2.
  - Legacy explicit artifact compatibility: Task 2 and Task 4.
  - Manifest and later materialization use same writer: Task 3 and Task 4.
  - Offline SQLite-style validation: Task 5.
  - No SQL DDL / no re-seed: File structure and Task 6.
- Placeholder scan:
  - No `TBD`.
  - No `TODO`.
  - No undefined "add appropriate" instructions.
- Type consistency:
  - Public fold mapping remains `dict[int, tuple[np.ndarray, np.ndarray]]`.
  - `pk_columns` is consistently `tuple[str, ...]`.
  - Holdout assignment key is consistently `is_testing_set`.
  - Fold assignment key is consistently `test_fold`.
