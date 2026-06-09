from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.data.cv_splits import (
    CVSplitSet,
    load_cv_folds,
    load_materialized_cv_folds,
    materialize_cv_folds,
    replay_cv_folds,
    resolve_splitter,
)
from pricing_pipeline.data.manifest import compute_row_order_sha256
from pricing_pipeline.data.split_artifacts import FOLD_ASSIGNMENT_FORMAT
from pricing_pipeline.data.split_artifacts import HOLDOUT_ASSIGNMENT_FORMAT
from pricing_pipeline.data.split_artifacts import load_split_artifact_npz
from pricing_pipeline.data.split_artifacts import write_split_artifact_npz
from pricing_pipeline.models.config import ValidationSplitConfig
from scripts import export_cv_indices


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [10, 20, 30, 40, 50],
            "ClaimNb": [0, 1, 0, 2, 0],
        }
    )


def split_set() -> CVSplitSet:
    rows = frame()
    return CVSplitSet(
        split_set_id="manifest_1__kfold_3_seed_42",
        manifest_id="manifest_1",
        split_mode="REPLAYABLE",
        splitter_class="sklearn.model_selection.KFold",
        splitter_params_json=json.dumps(
            {"n_splits": 3, "shuffle": True, "random_state": 42},
            sort_keys=True,
        ),
        row_order_sha256=compute_row_order_sha256(rows, pk_column="IDpol"),
        row_count=5,
        fold_count=3,
        artifact_uri=None,
        artifact_sha256=None,
    )


def train_test_split_set() -> CVSplitSet:
    rows = frame()
    return CVSplitSet(
        split_set_id="manifest_1__train_test_split_test_0_4_seed_42",
        manifest_id="manifest_1",
        split_mode="REPLAYABLE",
        splitter_class="sklearn.model_selection.train_test_split",
        splitter_params_json=json.dumps(
            {
                "test_size": 0.4,
                "random_state": 42,
                "shuffle": True,
            },
            sort_keys=True,
        ),
        row_order_sha256=compute_row_order_sha256(rows, pk_column="IDpol"),
        row_count=5,
        fold_count=1,
        artifact_uri=None,
        artifact_sha256=None,
    )


def source_column_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [10, 20, 30, 40],
            "ClaimNb": [0, 1, 0, 2],
            "fold_number": [2, 1, 2, 1],
            "holdout_flag": [0, 0, 1, 0],
        },
        index=[100, 200, 300, 400],
    )


def source_column_kfold_split_set() -> CVSplitSet:
    rows = source_column_frame()
    return CVSplitSet(
        split_set_id="manifest_1__column_kfold_fold_number",
        manifest_id="manifest_1",
        split_mode="REPLAYABLE",
        splitter_class="source_column",
        splitter_params_json=json.dumps(
            {
                "method": "column_kfold",
                "column": "fold_number",
                "fold_values": [1, 2],
            },
            sort_keys=True,
        ),
        row_order_sha256=compute_row_order_sha256(rows, pk_column="IDpol"),
        row_count=4,
        fold_count=2,
        artifact_uri=None,
        artifact_sha256=None,
    )


def source_column_holdout_split_set() -> CVSplitSet:
    rows = source_column_frame()
    return CVSplitSet(
        split_set_id="manifest_1__column_holdout_holdout_flag",
        manifest_id="manifest_1",
        split_mode="REPLAYABLE",
        splitter_class="source_column",
        splitter_params_json=json.dumps(
            {
                "method": "column_holdout",
                "column": "holdout_flag",
                "train_values": [0],
                "test_values": [1],
                "unexpected_values": "error",
            },
            sort_keys=True,
        ),
        row_order_sha256=compute_row_order_sha256(rows, pk_column="IDpol"),
        row_count=4,
        fold_count=1,
        artifact_uri=None,
        artifact_sha256=None,
    )


def test_resolve_splitter_recreates_supported_splitter():
    cv = resolve_splitter(split_set())

    folds = list(cv.split(frame()))

    assert len(folds) == 3
    assert [test_idx.tolist() for _, test_idx in folds] == [[1, 4], [0, 2], [3]]


def test_replay_cv_folds_returns_one_based_fold_mapping():
    folds = replay_cv_folds(split_set(), frame())

    assert sorted(folds) == [1, 2, 3]
    assert folds[1][0].tolist() == [0, 2, 3]
    assert folds[1][1].tolist() == [1, 4]
    assert folds[3][0].tolist() == [0, 1, 2, 4]
    assert folds[3][1].tolist() == [3]


def test_replay_cv_folds_supports_train_test_split_metadata():
    folds = replay_cv_folds(train_test_split_set(), frame())

    assert sorted(folds) == [1]
    assert len(folds[1][0]) == 3
    assert len(folds[1][1]) == 2
    assert sorted(folds[1][0].tolist() + folds[1][1].tolist()) == [0, 1, 2, 3, 4]


def test_replay_cv_folds_supports_source_column_kfold_metadata():
    folds = replay_cv_folds(
        source_column_kfold_split_set(),
        source_column_frame(),
    )

    assert sorted(folds) == [1, 2]
    assert folds[1][0].tolist() == [0, 2]
    assert folds[1][1].tolist() == [1, 3]
    assert folds[2][0].tolist() == [1, 3]
    assert folds[2][1].tolist() == [0, 2]


def test_replay_cv_folds_supports_source_column_holdout_metadata():
    folds = replay_cv_folds(
        source_column_holdout_split_set(),
        source_column_frame(),
    )

    assert sorted(folds) == [1]
    assert folds[1][0].tolist() == [0, 1, 3]
    assert folds[1][1].tolist() == [2]


def test_replay_cv_folds_rejects_missing_source_column():
    changed = source_column_frame().drop(columns=["fold_number"])

    with pytest.raises(ValueError, match="fold_number"):
        replay_cv_folds(source_column_kfold_split_set(), changed)


def test_replay_cv_folds_rejects_changed_row_order():
    changed = frame().sort_values("IDpol", ascending=False)

    with pytest.raises(ValueError, match="row_order_sha256"):
        replay_cv_folds(split_set(), changed)


def test_replay_cv_folds_rejects_disagreeing_pk_column_and_pk_columns():
    with pytest.raises(ValueError, match="pk_column and pk_columns disagree"):
        replay_cv_folds(
            split_set(),
            frame(),
            pk_column="ClaimNb",
            pk_columns=("IDpol",),
        )


def test_replay_cv_folds_accepts_composite_pk_columns_with_legacy_pk_column_default():
    rows = frame()
    composite_split_set = CVSplitSet(
        **{
            **split_set().__dict__,
            "row_order_sha256": compute_row_order_sha256(
                rows,
                pk_columns=("IDpol", "ClaimNb"),
            ),
        }
    )

    folds = replay_cv_folds(
        composite_split_set,
        rows,
        pk_columns=("IDpol", "ClaimNb"),
    )

    assert folds[1][1].tolist() == [1, 4]


def test_replay_cv_folds_accepts_duplicate_single_pk_column_specification():
    folds = replay_cv_folds(
        split_set(),
        frame(),
        pk_column="IDpol",
        pk_columns=("IDpol",),
    )

    assert folds[1][1].tolist() == [1, 4]


def test_load_cv_folds_hides_sql_and_dataset_loader():
    class ScalarResult:
        def __init__(self, mapping):
            self.mapping = mapping

        def mappings(self):
            return self

        def one(self):
            return self.mapping

    class Connection:
        def execute(self, statement, params=None):
            assert "FROM pricing.CV_SPLIT_SET" in str(statement)
            assert params == {"split_set_id": "manifest_1__kfold_3_seed_42"}
            return ScalarResult(split_set().__dict__)

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    folds = load_cv_folds(
        Engine(),
        "manifest_1__kfold_3_seed_42",
        dataset_loader=lambda manifest_id: frame(),
    )

    assert folds[2][1].tolist() == [0, 2]


def test_load_cv_folds_loads_legacy_materialized_artifact_without_dataset_loader(
    tmp_path: Path,
):
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

    class ScalarResult:
        def mappings(self):
            return self

        def one(self):
            return materialized.__dict__

    class Connection:
        def execute(self, statement, params=None):
            assert "FROM pricing.CV_SPLIT_SET" in str(statement)
            assert params == {"split_set_id": materialized.split_set_id}
            return ScalarResult()

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    def dataset_loader(manifest_id: str) -> pd.DataFrame:
        raise AssertionError(f"dataset_loader should not be called for {manifest_id}")

    folds = load_cv_folds(
        Engine(),
        materialized.split_set_id,
        dataset_loader=dataset_loader,
    )

    assert folds[1][1].tolist() == [1, 4]
    assert folds[2][1].tolist() == [0, 2]
    assert folds[3][1].tolist() == [3]


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
    loaded_manifest_ids = []

    class ScalarResult:
        def mappings(self):
            return self

        def one(self):
            return materialized.__dict__

    class Connection:
        def execute(self, statement, params=None):
            assert "FROM pricing.CV_SPLIT_SET" in str(statement)
            assert params == {"split_set_id": materialized.split_set_id}
            return ScalarResult()

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    def dataset_loader(manifest_id: str) -> pd.DataFrame:
        loaded_manifest_ids.append(manifest_id)
        return rows

    folds = load_cv_folds(
        Engine(),
        materialized.split_set_id,
        dataset_loader=dataset_loader,
        pk_columns=("IDpol",),
    )

    assert loaded_manifest_ids == ["manifest_1"]
    assert folds[1][1].tolist() == [1, 4]
    assert folds[2][1].tolist() == [0, 2]
    assert folds[3][1].tolist() == [3]


def test_export_cv_indices_writes_requested_fold(tmp_path: Path):
    folds = {
        1: (np.array([0, 2, 3]), np.array([1, 4])),
        2: (np.array([1, 3, 4]), np.array([0, 2])),
    }
    output_path = tmp_path / "fold_2.npz"

    export_cv_indices.write_fold_npz(folds, fold_no=2, output_path=output_path)

    loaded = np.load(output_path)
    assert loaded["train_idx"].tolist() == [1, 3, 4]
    assert loaded["test_idx"].tolist() == [0, 2]


def test_export_cv_indices_writes_all_folds_artifact(tmp_path: Path):
    folds = {
        1: (np.array([0, 2, 3]), np.array([1, 4])),
        2: (np.array([1, 3, 4]), np.array([0, 2])),
    }
    output_path = tmp_path / "all_folds.npz"

    export_cv_indices.write_all_folds_npz(folds, output_path=output_path)

    loaded = np.load(output_path)
    assert loaded["fold_1_train_idx"].tolist() == [0, 2, 3]
    assert loaded["fold_1_test_idx"].tolist() == [1, 4]
    assert loaded["fold_2_train_idx"].tolist() == [1, 3, 4]
    assert loaded["fold_2_test_idx"].tolist() == [0, 2]


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


def test_materialize_cv_folds_writes_all_folds_updates_split_set_and_hashes_artifact(
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
    )

    assert result == output_path
    loaded = np.load(output_path, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert loaded["test_fold"].tolist() == [2, 1, 2, 3, 1]
    assert executed
    statement, params = executed[0]
    assert "UPDATE pricing.CV_SPLIT_SET" in statement
    assert "split_mode = 'MATERIALIZED'" in statement
    assert params["split_set_id"] == "manifest_1__kfold_3_seed_42"
    assert params["artifact_uri"] == str(output_path)
    assert len(params["artifact_sha256"]) == 64
    assert "runtime_metadata_json" in params


def test_materialize_cv_folds_supports_replayable_source_column_split_set(
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

    output_path = tmp_path / "manifest_1__column_kfold_fold_number.npz"

    result = materialize_cv_folds(
        Engine(),
        source_column_kfold_split_set(),
        source_column_frame(),
        output_path=output_path,
    )

    assert result == output_path
    loaded = np.load(output_path, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert loaded["test_fold"].tolist() == [2, 1, 2, 1]
    assert executed[0][1]["split_set_id"] == "manifest_1__column_kfold_fold_number"


def test_load_materialized_cv_folds_checks_artifact_hash(tmp_path: Path):
    artifact = tmp_path / "folds.npz"
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
            "artifact_sha256": "bad-hash",
        }
    )

    with pytest.raises(ValueError, match="artifact_sha256"):
        load_materialized_cv_folds(materialized)


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
    assert "fold_1_test_idx" not in loaded.files


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
        validation_split=ValidationSplitConfig.train_test_split(
            test_size=0.4,
            materialize=True,
        ),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert str(loaded["split_format"].item()) == HOLDOUT_ASSIGNMENT_FORMAT
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["is_testing_set"].dtype == np.bool_
    assert loaded["is_testing_set"].tolist() == [False, True, False, False, True]


def test_write_split_artifact_npz_writes_compact_column_kfold_assignment(
    tmp_path: Path,
):
    rows = source_column_frame()
    artifact = tmp_path / "compact_column_kfold.npz"

    write_split_artifact_npz(
        {
            1: (np.array([0, 2]), np.array([1, 3])),
            2: (np.array([1, 3]), np.array([0, 2])),
        },
        validation_split=ValidationSplitConfig.column_kfold(
            column="fold_number",
            materialize=True,
        ),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert str(loaded["split_format"].item()) == FOLD_ASSIGNMENT_FORMAT
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["test_fold"].dtype == np.uint8
    assert loaded["test_fold"].tolist() == [2, 1, 2, 1]


def test_write_split_artifact_npz_writes_compact_column_holdout_assignment(
    tmp_path: Path,
):
    rows = source_column_frame()
    artifact = tmp_path / "compact_column_holdout.npz"

    write_split_artifact_npz(
        {1: (np.array([0, 1, 3]), np.array([2]))},
        validation_split=ValidationSplitConfig.column_holdout(
            column="holdout_flag",
            train_values=(0,),
            test_values=(1,),
            materialize=True,
        ),
        pk_columns=("IDpol",),
        row_count=len(rows),
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert str(loaded["split_format"].item()) == HOLDOUT_ASSIGNMENT_FORMAT
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["is_testing_set"].dtype == np.bool_
    assert loaded["is_testing_set"].tolist() == [False, False, True, False]


def test_write_split_artifact_npz_falls_back_to_legacy_explicit_arrays(
    tmp_path: Path,
):
    artifact = tmp_path / "legacy_fallback.npz"

    write_split_artifact_npz(
        {
            1: (np.array([0, 2, 3]), np.array([1, 4])),
            2: (np.array([1, 3, 4]), np.array([0, 2])),
        },
        validation_split=ValidationSplitConfig.none(),
        pk_columns=("IDpol",),
        row_count=5,
        output_path=artifact,
    )

    loaded = np.load(artifact, allow_pickle=False)
    assert sorted(loaded.files) == [
        "fold_1_test_idx",
        "fold_1_train_idx",
        "fold_2_test_idx",
        "fold_2_train_idx",
    ]
    assert loaded["fold_1_train_idx"].tolist() == [0, 2, 3]
    assert loaded["fold_1_test_idx"].tolist() == [1, 4]
    assert "split_format" not in loaded.files
    assert "pk_columns" not in loaded.files


def test_write_split_artifact_npz_rejects_kfold_train_not_complement(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="fold_1_train_idx"):
        write_split_artifact_npz(
            {
                1: (np.array([0, 2]), np.array([1, 4])),
                2: (np.array([1, 3, 4]), np.array([0, 2])),
                3: (np.array([0, 1, 2, 4]), np.array([3])),
            },
            validation_split=ValidationSplitConfig.kfold(n_splits=3, materialize=True),
            pk_columns=("IDpol",),
            row_count=5,
            output_path=tmp_path / "bad_train_complement.npz",
        )


def test_write_split_artifact_npz_rejects_kfold_row_missing_from_test_assignment(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="each row must appear in exactly one test fold"):
        write_split_artifact_npz(
            {
                1: (np.array([0, 2, 3]), np.array([1, 4])),
                2: (np.array([1, 3, 4]), np.array([0, 2])),
            },
            validation_split=ValidationSplitConfig.kfold(n_splits=2, materialize=True),
            pk_columns=("IDpol",),
            row_count=5,
            output_path=tmp_path / "missing_test_assignment.npz",
        )


def test_write_split_artifact_npz_rejects_holdout_train_test_not_covering_all_rows(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="cover every row"):
        write_split_artifact_npz(
            {1: (np.array([0, 2]), np.array([1, 4]))},
            validation_split=ValidationSplitConfig.train_test_split(
                test_size=0.4,
                materialize=True,
            ),
            pk_columns=("IDpol",),
            row_count=5,
            output_path=tmp_path / "bad_holdout_cover.npz",
        )


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


def test_load_split_artifact_npz_checks_artifact_hash(tmp_path: Path):
    artifact = tmp_path / "compact_kfold.npz"
    write_split_artifact_npz(
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
            "artifact_sha256": "bad-hash",
        }
    )

    with pytest.raises(ValueError, match="artifact_sha256"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))


def test_load_split_artifact_npz_rejects_pk_columns_mismatch(tmp_path: Path):
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

    with pytest.raises(ValueError, match="pk_columns"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("ClaimNb",))


def test_load_split_artifact_npz_rejects_changed_row_order(tmp_path: Path):
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
    changed = frame().sort_values("IDpol", ascending=False)

    with pytest.raises(ValueError, match="row_order_sha256"):
        load_split_artifact_npz(materialized, frame=changed, pk_columns=("IDpol",))


@pytest.mark.parametrize(
    ("base_split_set", "arrays", "match"),
    [
        (
            split_set(),
            {
                "split_format": np.array(FOLD_ASSIGNMENT_FORMAT),
                "pk_columns": np.array(["IDpol"]),
                "test_fold": np.array([1, 2, 1, 3], dtype=np.uint8),
            },
            "test_fold length mismatch",
        ),
        (
            train_test_split_set(),
            {
                "split_format": np.array(HOLDOUT_ASSIGNMENT_FORMAT),
                "pk_columns": np.array(["IDpol"]),
                "is_testing_set": np.array([False, True, False, True]),
            },
            "is_testing_set length mismatch",
        ),
    ],
)
def test_load_split_artifact_npz_rejects_assignment_length_mismatch(
    tmp_path: Path,
    base_split_set: CVSplitSet,
    arrays: dict[str, np.ndarray],
    match: str,
):
    artifact = tmp_path / "bad_length.npz"
    np.savez_compressed(artifact, **arrays)
    materialized = CVSplitSet(
        **{
            **base_split_set.__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    with pytest.raises(ValueError, match=match):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))


def test_load_split_artifact_npz_rejects_object_split_format(tmp_path: Path):
    artifact = tmp_path / "object_split_format.npz"
    np.savez_compressed(
        artifact,
        split_format=np.array([FOLD_ASSIGNMENT_FORMAT], dtype=object),
        pk_columns=np.array(["IDpol"]),
        test_fold=np.array([2, 1, 2, 3, 1], dtype=np.uint8),
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    with pytest.raises(ValueError, match="split_format"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))


def test_load_split_artifact_npz_rejects_object_pk_columns(tmp_path: Path):
    artifact = tmp_path / "object_pk_columns.npz"
    np.savez_compressed(
        artifact,
        split_format=np.array(FOLD_ASSIGNMENT_FORMAT),
        pk_columns=np.array(["IDpol"], dtype=object),
        test_fold=np.array([2, 1, 2, 3, 1], dtype=np.uint8),
    )
    materialized = CVSplitSet(
        **{
            **split_set().__dict__,
            "split_mode": "MATERIALIZED",
            "artifact_uri": str(artifact),
            "artifact_sha256": None,
        }
    )

    with pytest.raises(ValueError, match="pk_columns"):
        load_split_artifact_npz(materialized, frame=frame(), pk_columns=("IDpol",))


def test_load_split_artifact_npz_rejects_integral_float_test_fold(tmp_path: Path):
    artifact = tmp_path / "integral_float_fold.npz"
    np.savez_compressed(
        artifact,
        split_format=np.array(FOLD_ASSIGNMENT_FORMAT),
        pk_columns=np.array(["IDpol"]),
        test_fold=np.array([2.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float64),
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
    assert folds[3][0].tolist() == [0, 1, 2, 4]
    assert folds[3][1].tolist() == [3]


@pytest.mark.parametrize(
    "bad_test_fold",
    [
        np.array([1, 2, 0, 3, 1], dtype=np.int64),
        np.array([1, 2, -1, 3, 1], dtype=np.int64),
        np.array([1, 2, 4, 3, 1], dtype=np.int64),
        np.array([1.0, 2.0, 1.5, 3.0, 1.0], dtype=np.float64),
        np.array([1.0, 2.0, np.nan, 3.0, 1.0], dtype=np.float64),
        np.array(["1", "2", "1", "3", "1"]),
        np.array([1, 2, None, 3, 1], dtype=object),
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
        np.array([False, True, None, False, True], dtype=object),
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
