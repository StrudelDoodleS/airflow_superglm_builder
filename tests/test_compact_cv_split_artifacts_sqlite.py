from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from pricing_pipeline.data.cv_splits import fetch_split_set
from pricing_pipeline.data.cv_splits import load_cv_folds
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
    assert "fold_1_test_idx" not in loaded.files
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
