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


def test_replay_cv_folds_rejects_changed_row_order():
    changed = frame().sort_values("IDpol", ascending=False)

    with pytest.raises(ValueError, match="row_order_sha256"):
        replay_cv_folds(split_set(), changed)


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
    loaded = np.load(output_path)
    assert loaded["fold_1_train_idx"].tolist() == [0, 2, 3]
    assert loaded["fold_1_test_idx"].tolist() == [1, 4]
    assert loaded["fold_3_train_idx"].tolist() == [0, 1, 2, 4]
    assert loaded["fold_3_test_idx"].tolist() == [3]
    assert executed
    statement, params = executed[0]
    assert "UPDATE pricing.CV_SPLIT_SET" in statement
    assert "split_mode = 'MATERIALIZED'" in statement
    assert params["split_set_id"] == "manifest_1__kfold_3_seed_42"
    assert params["artifact_uri"] == str(output_path)
    assert len(params["artifact_sha256"]) == 64
    assert "runtime_metadata_json" in params


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
