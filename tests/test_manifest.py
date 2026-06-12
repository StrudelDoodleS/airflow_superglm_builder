import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.data import manifest
from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    create_dataset_manifest_with_split,
    build_column_metadata,
    build_cv_split_set,
    build_validation_split_set,
    compute_row_order_sha256,
    create_dataset_manifest,
    create_model_frame_manifest_with_split,
    create_fremtpl_manifest,
    new_manifest_id,
    runtime_dependency_metadata,
    validation_split_indices,
)
from pricing_pipeline.models.config import ValidationSplitConfig
from pricing_pipeline.models.spec import DatasetSpec


def manifest_frame(**overrides):
    data = {
        "IDpol": [10, 20, 30, 40, 50],
        "ClaimNb": [0, 1, 0, 2, 0],
        "Exposure": [0.5, 1.0, None, 0.25, 0.75],
        "Area": ["A", "B", "A", None, "C"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_new_manifest_id_includes_dataset_date_prefix_and_unique_suffix():
    first = new_manifest_id("freMTPL2freq")
    second = new_manifest_id("freMTPL2freq")

    expected_prefix = f"freMTPL2freq_{date.today():%Y%m%d}_"
    assert first.startswith(expected_prefix)
    assert second.startswith(expected_prefix)
    assert first != second


def test_build_column_metadata_marks_roles_and_counts_columns():
    frame = manifest_frame()

    columns = build_column_metadata(frame, manifest_id="manifest_1")

    assert list(columns.columns) == [
        "manifest_id",
        "ordinal_no",
        "column_name",
        "column_role",
        "pandas_dtype",
        "null_count",
        "distinct_count",
    ]
    assert columns["manifest_id"].tolist() == ["manifest_1"] * len(frame.columns)
    assert columns["ordinal_no"].tolist() == [1, 2, 3, 4]
    assert dict(zip(columns["column_name"], columns["column_role"], strict=True)) == {
        "IDpol": "KEY",
        "ClaimNb": "TARGET",
        "Exposure": "WEIGHT",
        "Area": "FEATURE",
    }

    exposure = columns.loc[columns["column_name"].eq("Exposure")].iloc[0]
    assert exposure["pandas_dtype"] == str(frame["Exposure"].dtype)
    assert exposure["null_count"] == 1
    assert exposure["distinct_count"] == 4

    area = columns.loc[columns["column_name"].eq("Area")].iloc[0]
    assert area["null_count"] == 1
    assert area["distinct_count"] == 3


def test_build_column_metadata_uses_dataset_declared_roles():
    frame = pd.DataFrame(
        {
            "PolicyID": [1, 2],
            "Snapshot": ["2026-01", "2026-02"],
            "LossCost": [10.0, 20.0],
            "Postcode": ["A", "B"],
        }
    )

    columns = build_column_metadata(
        frame,
        manifest_id="manifest_custom",
        pk_columns=("PolicyID", "Snapshot"),
        target_column="LossCost",
        weight_column=None,
    )

    assert dict(zip(columns["column_name"], columns["column_role"], strict=True)) == {
        "PolicyID": "KEY",
        "Snapshot": "KEY",
        "LossCost": "TARGET",
        "Postcode": "FEATURE",
    }


def test_compute_row_order_sha256_depends_on_ordered_primary_keys():
    first = pd.DataFrame({"IDpol": [10, 20, 30]})
    same = pd.DataFrame({"IDpol": [10, 20, 30]})
    reordered = pd.DataFrame({"IDpol": [20, 10, 30]})

    assert compute_row_order_sha256(first, pk_column="IDpol") == compute_row_order_sha256(
        same,
        pk_column="IDpol",
    )
    assert compute_row_order_sha256(first, pk_column="IDpol") != compute_row_order_sha256(
        reordered,
        pk_column="IDpol",
    )


def test_compute_row_order_sha256_supports_composite_primary_keys():
    first = pd.DataFrame({"PolicyID": [10, 10, 20], "Snapshot": ["2026-01", "2026-02", "2026-01"]})
    same = first.copy()
    changed_key_part = pd.DataFrame(
        {"PolicyID": [10, 10, 20], "Snapshot": ["2026-01", "2026-03", "2026-01"]}
    )

    assert compute_row_order_sha256(first, pk_columns=("PolicyID", "Snapshot")) == (
        compute_row_order_sha256(same, pk_columns=("PolicyID", "Snapshot"))
    )
    assert compute_row_order_sha256(first, pk_columns=("PolicyID", "Snapshot")) != (
        compute_row_order_sha256(changed_key_part, pk_columns=("PolicyID", "Snapshot"))
    )


def test_build_cv_split_set_records_replayable_splitter_metadata():
    frame = manifest_frame()

    split_set = build_cv_split_set(
        frame,
        manifest_id="manifest_1",
        n_splits=5,
        random_state=42,
    )

    assert split_set.to_dict("records") == [
        {
            "split_set_id": "manifest_1__kfold_5_seed_42",
            "manifest_id": "manifest_1",
            "split_mode": "REPLAYABLE",
            "splitter_class": "sklearn.model_selection.KFold",
            "splitter_params_json": json.dumps(
                {"n_splits": 5, "shuffle": True, "random_state": 42},
                sort_keys=True,
            ),
            "row_order_sha256": compute_row_order_sha256(frame, pk_column="IDpol"),
            "row_count": 5,
            "fold_count": 5,
            "groups_column": None,
            "stratify_column": None,
            "artifact_uri": None,
            "artifact_sha256": None,
            "runtime_metadata_json": runtime_dependency_metadata(),
            "created_by": "airflow",
        }
    ]


def test_runtime_dependency_metadata_records_python_platform_and_core_versions():
    metadata_json = runtime_dependency_metadata()
    metadata = json.loads(metadata_json)

    assert metadata["python_version"].startswith(f"{sys.version_info.major}.")
    assert metadata["platform"]
    assert metadata["packages"]["numpy"]
    assert metadata["packages"]["pandas"]
    assert metadata["packages"]["sklearn"]
    assert "superglm" in metadata["packages"]


class FakeBeginConnection:
    def __init__(self, events):
        self.events = events
        self.executed = []

    def execute(self, statement, params=None):
        sql = str(statement).strip()
        self.executed.append((sql, params))
        self.events.append(f"execute:{sql.split()[0]}")


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self, execution_options=None):
        self.events = []
        self._execution_options = execution_options or {}
        self.connection = FakeBeginConnection(self.events)

    def begin(self):
        return FakeBegin(self.connection)


def test_create_model_frame_manifest_writes_final_frame_metadata(monkeypatch):
    engine = FakeEngine()
    frame = pd.DataFrame(
        {
            "PolicyID": [2, 1, 3],
            "LossCost": [3.4, 1.2, 0.0],
            "ExposureYears": [1.0, 0.5, 0.25],
            "BandedDriverAge": ["30-39", "18-29", "40-49"],
        }
    )
    to_sql_calls = []

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append(
            {
                "name": name,
                "schema": kwargs.get("schema"),
                "frame": self.copy(),
            }
        )

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    result = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=ModelFrameManifestSpec(
            dataset_name="work_loss_cost_frame",
            source_system="pricing_mart",
            data_as_of_date="2026-06-04",
            pk_columns=("PolicyID",),
            target_column="LossCost",
            weight_column="ExposureYears",
        ),
        manifest_id="manifest_frame_1",
        validation_split=ValidationSplitConfig.train_test_split(
            test_size=0.34,
            random_state=42,
        ),
        created_by="unit-test",
    )

    assert result.manifest_id == "manifest_frame_1"
    assert result.split_set_id == "manifest_frame_1__train_test_split_test_0_34_seed_42"
    assert [(call["name"], call["schema"]) for call in to_sql_calls] == [
        ("DATASET_MANIFEST", "pricing"),
        ("DATASET_COLUMN", "pricing"),
        ("CV_SPLIT_SET", "pricing"),
        ("CV_FOLD", "pricing"),
    ]

    manifest_row = to_sql_calls[0]["frame"].iloc[0]
    assert manifest_row["manifest_id"] == "manifest_frame_1"
    assert manifest_row["dataset_name"] == "work_loss_cost_frame"
    assert manifest_row["source_system"] == "pricing_mart"
    assert manifest_row["data_as_of_date"] == date(2026, 6, 4)
    assert manifest_row["row_count"] == 3
    assert json.loads(manifest_row["pk_columns_json"]) == ["PolicyID"]
    assert manifest_row["target_column"] == "LossCost"
    assert manifest_row["weight_column"] == "ExposureYears"
    assert manifest_row["created_by"] == "unit-test"

    column_roles = dict(
        zip(
            to_sql_calls[1]["frame"]["column_name"],
            to_sql_calls[1]["frame"]["column_role"],
            strict=True,
        )
    )
    assert column_roles == {
        "PolicyID": "KEY",
        "LossCost": "TARGET",
        "ExposureYears": "WEIGHT",
        "BandedDriverAge": "FEATURE",
    }

    split_row = to_sql_calls[2]["frame"].iloc[0]
    assert split_row["row_order_sha256"] == compute_row_order_sha256(
        frame,
        pk_columns=("PolicyID",),
    )


def test_create_model_frame_manifest_records_supplied_custom_split_indices(
    monkeypatch,
    tmp_path,
):
    engine = FakeEngine()
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "LossCost": [1.0, 0.0, 2.0, 0.5],
            "ExposureYears": [1.0, 1.0, 0.5, 0.25],
            "BandedDriverAge": ["18-29", "30-39", "40-49", "50-59"],
        }
    )
    split_indices = [
        (np.asarray([0, 2, 3]), np.asarray([1])),
        (np.asarray([1, 2]), np.asarray([0, 3])),
    ]
    to_sql_calls = []

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append(
            {
                "name": name,
                "schema": kwargs.get("schema"),
                "frame": self.copy(),
            }
        )

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    result = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=ModelFrameManifestSpec(
            dataset_name="work_loss_cost_frame",
            source_system="pricing_mart",
            data_as_of_date="2026-06-04",
            pk_columns=("PolicyID",),
            target_column="LossCost",
            weight_column="ExposureYears",
        ),
        manifest_id="manifest_custom_split",
        validation_split=ValidationSplitConfig.custom(materialize=True),
        validation_split_artifact_root=tmp_path,
        split_indices=split_indices,
        created_by="unit-test",
    )

    assert result.split_set_id == "manifest_custom_split__custom"

    split_set = next(call for call in to_sql_calls if call["name"] == "CV_SPLIT_SET")["frame"].iloc[
        0
    ]
    assert split_set["split_mode"] == "MATERIALIZED"
    assert split_set["splitter_class"] == "custom"
    assert json.loads(split_set["splitter_params_json"]) == {"method": "custom"}
    assert split_set["fold_count"] == 2

    cv_folds = next(call for call in to_sql_calls if call["name"] == "CV_FOLD")["frame"]
    assert cv_folds[["fold_no", "n_train", "n_test"]].to_dict("records") == [
        {"fold_no": 1, "n_train": 3, "n_test": 1},
        {"fold_no": 2, "n_train": 2, "n_test": 2},
    ]

    artifact_path = tmp_path / "manifest_custom_split__custom.npz"
    loaded = np.load(artifact_path, allow_pickle=False)
    assert sorted(loaded.files) == [
        "fold_1_test_idx",
        "fold_1_train_idx",
        "fold_2_test_idx",
        "fold_2_train_idx",
        "pk_columns",
        "split_format",
    ]
    assert loaded["pk_columns"].tolist() == ["PolicyID"]
    assert loaded["fold_1_train_idx"].tolist() == [0, 2, 3]
    assert loaded["fold_1_test_idx"].tolist() == [1]
    assert loaded["fold_2_train_idx"].tolist() == [1, 2]
    assert loaded["fold_2_test_idx"].tolist() == [0, 3]


def test_create_model_frame_manifest_rejects_supplied_split_indices_that_do_not_match_config(
    monkeypatch,
):
    engine = FakeEngine()
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "LossCost": [1.0, 0.0, 2.0, 0.5],
            "ExposureYears": [1.0, 1.0, 0.5, 0.25],
        }
    )
    monkeypatch.setattr(pd.DataFrame, "to_sql", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="method='custom'"):
        create_model_frame_manifest_with_split(
            engine,
            frame=frame,
            spec=ModelFrameManifestSpec(
                dataset_name="work_loss_cost_frame",
                source_system="pricing_mart",
                data_as_of_date="2026-06-04",
                pk_columns=("PolicyID",),
                target_column="LossCost",
                weight_column="ExposureYears",
            ),
            manifest_id="manifest_split_mismatch",
            validation_split=ValidationSplitConfig.train_test_split(
                test_size=0.5,
                random_state=42,
            ),
            split_indices=[
                (np.asarray([0, 1]), np.asarray([2, 3])),
            ],
            created_by="unit-test",
        )


def test_create_model_frame_manifest_accepts_empty_supplied_split_indices_for_none(
    monkeypatch,
):
    engine = FakeEngine()
    frame = manifest_frame()
    to_sql_calls = []

    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        lambda self, name, con, **kwargs: to_sql_calls.append(name),
    )

    result = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=ModelFrameManifestSpec(
            dataset_name="no_validation_frame",
            source_system="pricing_mart",
            data_as_of_date="2026-06-04",
            pk_columns=("IDpol",),
            target_column="ClaimNb",
            weight_column="Exposure",
        ),
        manifest_id="manifest_no_validation_from_recipe",
        validation_split=ValidationSplitConfig.none(),
        split_indices=[],
        created_by="unit-test",
    )

    assert result.split_set_id is None
    assert "CV_SPLIT_SET" not in to_sql_calls
    assert "CV_FOLD" not in to_sql_calls


def test_column_kfold_validation_split_uses_positional_indices_and_stable_fold_order():
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "LossCost": [1.0, 2.0, 0.0, 3.0],
            "fold_number": [2, 1, 2, 1],
        },
        index=[10, 20, 30, 40],
    )

    folds = validation_split_indices(
        frame,
        ValidationSplitConfig.column_kfold(column="fold_number"),
    )

    assert [(train.tolist(), test.tolist()) for train, test in folds] == [
        ([0, 2], [1, 3]),
        ([1, 3], [0, 2]),
    ]


def test_column_holdout_validation_split_supports_numeric_values():
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "LossCost": [1.0, 2.0, 0.0, 3.0],
            "holdout_flag": [0, 0, 1, 0],
        },
        index=[10, 20, 30, 40],
    )

    folds = validation_split_indices(
        frame,
        ValidationSplitConfig.column_holdout(
            column="holdout_flag",
            train_values=(0,),
            test_values=(1,),
        ),
    )

    assert [(train.tolist(), test.tolist()) for train, test in folds] == [([0, 1, 3], [2])]


def test_validation_split_indices_rejects_custom_without_model_supplied_folds():
    with pytest.raises(ValueError, match="custom validation split"):
        validation_split_indices(manifest_frame(), ValidationSplitConfig.custom(materialize=True))


@pytest.mark.parametrize(
    ("frame", "config_factory", "match"),
    [
        (
            pd.DataFrame({"PolicyID": [1, 2], "fold": [1, None]}),
            lambda: ValidationSplitConfig.column_kfold(column="fold"),
            "null",
        ),
        (
            pd.DataFrame({"PolicyID": [1, 2], "fold": [1, 1]}),
            lambda: ValidationSplitConfig.column_kfold(column="fold"),
            "at least two",
        ),
        (
            pd.DataFrame({"PolicyID": [1, 2], "split": ["train", "unknown"]}),
            lambda: ValidationSplitConfig.column_holdout(
                column="split",
                train_values=("train",),
                test_values=("holdout",),
            ),
            "unexpected",
        ),
        (
            pd.DataFrame({"PolicyID": [1, 2], "split": ["train", "train"]}),
            lambda: ValidationSplitConfig.column_holdout(
                column="split",
                train_values=("train",),
                test_values=("holdout",),
            ),
            "test",
        ),
    ],
)
def test_source_column_validation_split_rejects_invalid_frame_values(
    frame,
    config_factory,
    match,
):
    with pytest.raises(ValueError, match=match):
        validation_split_indices(frame, config_factory())


def test_create_model_frame_manifest_materializes_compact_column_kfold_artifact(
    monkeypatch,
    tmp_path,
):
    engine = FakeEngine()
    frame = pd.DataFrame(
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
        frame=frame,
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

    assert result.split_set_id == "manifest_column_kfold__column_kfold_fold_number"
    assert result.split_artifact_uri == str(
        tmp_path / "manifest_column_kfold__column_kfold_fold_number.npz"
    )

    column_roles = dict(
        zip(
            to_sql_calls[1]["frame"]["column_name"],
            to_sql_calls[1]["frame"]["column_role"],
            strict=True,
        )
    )
    assert column_roles["fold_number"] == "SPLIT"

    split_set = to_sql_calls[2]["frame"].iloc[0]
    assert split_set["splitter_class"] == "source_column"
    assert split_set["fold_count"] == 2
    assert split_set["artifact_uri"] == result.split_artifact_uri
    assert len(split_set["artifact_sha256"]) == 64
    assert json.loads(split_set["splitter_params_json"]) == {
        "column": "fold_number",
        "fold_values": [1, 2],
        "method": "column_kfold",
    }

    folds = to_sql_calls[3]["frame"]
    assert folds["fold_no"].tolist() == [1, 2]
    assert folds["n_train"].tolist() == [2, 2]
    assert folds["n_test"].tolist() == [2, 2]

    loaded = np.load(
        tmp_path / "manifest_column_kfold__column_kfold_fold_number.npz",
        allow_pickle=False,
    )
    assert sorted(loaded.files) == ["pk_columns", "split_format", "test_fold"]
    assert str(loaded["split_format"].item()) == "fold_assignment_v1"
    assert loaded["pk_columns"].tolist() == ["PolicyID"]
    assert loaded["test_fold"].tolist() == [1, 2, 1, 2]


def test_build_validation_split_set_records_column_holdout_params():
    frame = pd.DataFrame(
        {
            "PolicyID": [101, 102, 103, 104],
            "LossCost": [1.0, 2.0, 0.0, 3.0],
            "holdout_flag": [0, 0, 1, 0],
        }
    )

    split_set = build_validation_split_set(
        frame,
        manifest_id="manifest_holdout",
        validation_split=ValidationSplitConfig.column_holdout(
            column="holdout_flag",
            train_values=(0,),
            test_values=(1,),
        ),
        pk_columns=("PolicyID",),
        created_by="unit-test",
    )

    row = split_set.iloc[0]
    assert row["split_set_id"] == "manifest_holdout__column_holdout_holdout_flag"
    assert row["splitter_class"] == "source_column"
    assert row["fold_count"] == 1
    assert json.loads(row["splitter_params_json"]) == {
        "column": "holdout_flag",
        "method": "column_holdout",
        "test_values": [1],
        "train_values": [0],
        "unexpected_values": "error",
    }


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            ModelFrameManifestSpec(
                dataset_name="unit",
                source_system="unit",
                data_as_of_date="2026-06-04",
                pk_columns=("split_column",),
                target_column="LossCost",
            ),
            "primary key",
        ),
        (
            ModelFrameManifestSpec(
                dataset_name="unit",
                source_system="unit",
                data_as_of_date="2026-06-04",
                pk_columns=("PolicyID",),
                target_column="split_column",
            ),
            "target",
        ),
        (
            ModelFrameManifestSpec(
                dataset_name="unit",
                source_system="unit",
                data_as_of_date="2026-06-04",
                pk_columns=("PolicyID",),
                target_column="LossCost",
                weight_column="split_column",
            ),
            "weight",
        ),
    ],
)
def test_create_model_frame_manifest_rejects_split_column_role_overlap(spec, match):
    frame = pd.DataFrame(
        {
            "PolicyID": [1, 2, 3, 4],
            "LossCost": [1.0, 0.0, 2.0, 1.5],
            "split_column": [1, 2, 1, 2],
        }
    )

    with pytest.raises(ValueError, match=match):
        create_model_frame_manifest_with_split(
            FakeEngine(),
            frame=frame,
            spec=spec,
            manifest_id="manifest_split_overlap",
            validation_split=ValidationSplitConfig.column_kfold(column="split_column"),
        )


@pytest.mark.parametrize(
    ("bad_frame", "match"),
    [
        (
            pd.DataFrame([[1, 10.0], [2, 20.0]], columns=["PolicyID", "PolicyID"]),
            "duplicate column",
        ),
        (
            pd.DataFrame({"PolicyID": [1, 2], " ": [10.0, 20.0]}),
            "blank column",
        ),
        (
            pd.DataFrame({"PolicyID": [1, None], "LossCost": [10.0, 20.0]}),
            "null",
        ),
        (
            pd.DataFrame({"PolicyID": [1, 1], "LossCost": [10.0, 20.0]}),
            "duplicate",
        ),
    ],
)
def test_create_model_frame_manifest_rejects_invalid_frame(bad_frame, match):
    with pytest.raises(ValueError, match=match):
        create_model_frame_manifest_with_split(
            FakeEngine(),
            frame=bad_frame,
            spec=ModelFrameManifestSpec(
                dataset_name="unit",
                source_system="unit",
                data_as_of_date="2026-06-04",
                pk_columns=("PolicyID",),
                target_column="LossCost",
            ),
            manifest_id="manifest_bad",
            validation_split=ValidationSplitConfig.none(),
        )


def test_create_model_frame_manifest_rejects_bad_split_config():
    with pytest.raises(ValueError, match="n_splits"):
        create_model_frame_manifest_with_split(
            FakeEngine(),
            frame=pd.DataFrame({"PolicyID": [1, 2], "LossCost": [10.0, 20.0]}),
            spec=ModelFrameManifestSpec(
                dataset_name="unit",
                source_system="unit",
                data_as_of_date="2026-06-04",
                pk_columns=("PolicyID",),
                target_column="LossCost",
            ),
            manifest_id="manifest_bad_split",
            validation_split=ValidationSplitConfig.kfold(n_splits=3),
        )


def test_create_fremtpl_manifest_reads_raw_table_and_persists_manifest_sequence(monkeypatch):
    engine = FakeEngine()
    raw_frame = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [0, 1, 0],
            "Exposure": [0.5, 1.0, 0.25],
            "Area": ["A", "B", "C"],
        }
    )
    read_calls = []
    to_sql_calls = []

    def fake_read_sql_query(sql, con):
        read_calls.append((str(sql), con))
        return raw_frame

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append(
            {
                "name": name,
                "con": con,
                "schema": kwargs.get("schema"),
                "if_exists": kwargs.get("if_exists"),
                "index": kwargs.get("index"),
                "chunksize": kwargs.get("chunksize"),
                "frame": self.copy(),
            }
        )
        engine.events.append(f"to_sql:{name}")

    monkeypatch.setattr(manifest.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    created = create_fremtpl_manifest(
        engine,
        manifest_id="manifest_1",
        n_splits=3,
        random_state=123,
        created_by="unit-test",
    )

    assert created == "manifest_1"
    assert read_calls == [("SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine)]
    assert engine.events == [
        "to_sql:DATASET_MANIFEST",
        "to_sql:DATASET_COLUMN",
        "to_sql:CV_SPLIT_SET",
        "to_sql:CV_FOLD",
    ]

    assert [
        (call["name"], call["schema"], call["if_exists"], call["index"]) for call in to_sql_calls
    ] == [
        ("DATASET_MANIFEST", "pricing", "append", False),
        ("DATASET_COLUMN", "pricing", "append", False),
        ("CV_SPLIT_SET", "pricing", "append", False),
        ("CV_FOLD", "pricing", "append", False),
    ]
    assert all(call["con"] is engine.connection for call in to_sql_calls)

    manifest_row = to_sql_calls[0]["frame"].iloc[0]
    assert manifest_row["manifest_id"] == "manifest_1"
    assert manifest_row["dataset_name"] == "freMTPL2freq"
    assert manifest_row["source_system"] == "openml_41214"
    assert manifest_row["data_as_of_date"] == date.today()
    assert manifest_row["row_count"] == 3
    assert json.loads(manifest_row["pk_columns_json"]) == ["IDpol"]
    assert manifest_row["target_column"] == "ClaimNb"
    assert manifest_row["weight_column"] == "Exposure"
    assert manifest_row["created_by"] == "unit-test"

    split_set = to_sql_calls[2]["frame"].iloc[0]
    assert split_set["split_set_id"] == "manifest_1__kfold_3_seed_123"
    folds = to_sql_calls[3]["frame"]
    assert folds["fold_no"].tolist() == [1, 2, 3]
    assert folds["n_train"].tolist() == [2, 2, 2]
    assert folds["n_test"].tolist() == [1, 1, 1]
    assert engine.connection.executed == []


def test_create_fremtpl_manifest_uses_configured_pricing_schema_for_to_sql(monkeypatch):
    engine = FakeEngine({"pricing_schema": "python_pricing"})
    raw_frame = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [0, 1, 0],
            "Exposure": [0.5, 1.0, 0.25],
            "Area": ["A", "B", "C"],
        }
    )
    to_sql_calls = []

    monkeypatch.setattr(
        manifest.pd,
        "read_sql_query",
        lambda sql, con: raw_frame,
    )

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append({"name": name, "schema": kwargs.get("schema")})

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    create_fremtpl_manifest(
        engine,
        manifest_id="manifest_custom_schema",
        n_splits=3,
        random_state=123,
        created_by="unit-test",
    )

    assert [(call["name"], call["schema"]) for call in to_sql_calls] == [
        ("DATASET_MANIFEST", "python_pricing"),
        ("DATASET_COLUMN", "python_pricing"),
        ("CV_SPLIT_SET", "python_pricing"),
        ("CV_FOLD", "python_pricing"),
    ]


def test_create_dataset_manifest_with_train_test_split_materializes_artifact(
    monkeypatch,
    tmp_path,
):
    engine = FakeEngine()
    raw_frame = manifest_frame()
    to_sql_calls = []

    monkeypatch.setattr(
        manifest.pd,
        "read_sql_query",
        lambda sql, con: raw_frame,
    )

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append(
            {
                "name": name,
                "schema": kwargs.get("schema"),
                "frame": self.copy(),
            }
        )

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    result = create_dataset_manifest_with_split(
        engine,
        dataset=DatasetSpec(
            dataset_name="unit",
            source_system="unit",
            manifest_sql="SELECT * FROM unit",
            pk_columns=("IDpol",),
            target_column="ClaimNb",
            weight_column="Exposure",
        ),
        manifest_id="manifest_train_test",
        validation_split=ValidationSplitConfig.train_test_split(
            test_size=0.4,
            random_state=42,
            materialize=True,
        ),
        validation_split_artifact_root=tmp_path,
        created_by="unit-test",
    )

    assert result.manifest_id == "manifest_train_test"
    assert result.split_set_id == "manifest_train_test__train_test_split_test_0_4_seed_42"
    assert result.split_artifact_uri is not None

    artifact_path = tmp_path / "manifest_train_test__train_test_split_test_0_4_seed_42.npz"
    loaded = np.load(artifact_path, allow_pickle=False)
    assert sorted(loaded.files) == ["is_testing_set", "pk_columns", "split_format"]
    assert str(loaded["split_format"].item()) == "holdout_assignment_v1"
    assert loaded["pk_columns"].tolist() == ["IDpol"]
    assert loaded["is_testing_set"].dtype == np.bool_
    assert int(loaded["is_testing_set"].sum()) == 2
    assert "fold_1_train_idx" not in loaded.files

    split_set_call = next(call for call in to_sql_calls if call["name"] == "CV_SPLIT_SET")
    split_set = split_set_call["frame"].iloc[0]
    assert split_set["splitter_class"] == "sklearn.model_selection.train_test_split"
    assert split_set["fold_count"] == 1
    assert split_set["artifact_uri"] == str(artifact_path)
    assert len(split_set["artifact_sha256"]) == 64


def test_create_dataset_manifest_with_none_validation_split_writes_no_cv_tables(
    monkeypatch,
):
    engine = FakeEngine()
    raw_frame = manifest_frame()
    to_sql_calls = []

    monkeypatch.setattr(
        manifest.pd,
        "read_sql_query",
        lambda sql, con: raw_frame,
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        lambda self, name, con, **kwargs: to_sql_calls.append(name),
    )

    result = create_dataset_manifest_with_split(
        engine,
        dataset=DatasetSpec(
            dataset_name="unit",
            source_system="unit",
            manifest_sql="SELECT * FROM unit",
            pk_columns=("IDpol",),
            target_column="ClaimNb",
            weight_column="Exposure",
        ),
        manifest_id="manifest_no_validation",
        validation_split=ValidationSplitConfig.none(),
        created_by="unit-test",
    )

    assert result.manifest_id == "manifest_no_validation"
    assert result.split_set_id is None
    assert "CV_SPLIT_SET" not in to_sql_calls
    assert "CV_FOLD" not in to_sql_calls


def test_create_dataset_manifest_uses_dataset_spec_without_fremtpl_assumptions(monkeypatch):
    engine = FakeEngine()
    raw_frame = pd.DataFrame(
        {
            "PolicyID": [1, 2, 3],
            "Snapshot": ["2026-01", "2026-01", "2026-01"],
            "LossCost": [1.2, 3.4, 0.0],
            "ExposureYears": [0.5, 1.0, 0.25],
            "Segment": ["A", "B", "A"],
        }
    )
    dataset = DatasetSpec(
        dataset_name="work_loss_cost",
        source_system="work_sql",
        manifest_sql="SELECT * FROM actuarial.loss_cost ORDER BY PolicyID, Snapshot",
        pk_columns=("PolicyID", "Snapshot"),
        target_column="LossCost",
        weight_column="ExposureYears",
    )
    read_calls = []
    to_sql_calls = []

    def fake_read_sql_query(sql, con):
        read_calls.append((str(sql), con))
        return raw_frame

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append({"name": name, "frame": self.copy()})

    monkeypatch.setattr(manifest.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    created = create_dataset_manifest(
        engine,
        dataset=dataset,
        manifest_id="work_manifest_1",
        n_splits=3,
        random_state=123,
        created_by="unit-test",
    )

    assert created == "work_manifest_1"
    assert read_calls == [("SELECT * FROM actuarial.loss_cost ORDER BY PolicyID, Snapshot", engine)]
    manifest_row = to_sql_calls[0]["frame"].iloc[0]
    assert manifest_row["dataset_name"] == "work_loss_cost"
    assert manifest_row["source_system"] == "work_sql"
    assert json.loads(manifest_row["pk_columns_json"]) == ["PolicyID", "Snapshot"]
    assert manifest_row["target_column"] == "LossCost"
    assert manifest_row["weight_column"] == "ExposureYears"

    column_roles = dict(
        zip(
            to_sql_calls[1]["frame"]["column_name"],
            to_sql_calls[1]["frame"]["column_role"],
            strict=True,
        )
    )
    assert column_roles == {
        "PolicyID": "KEY",
        "Snapshot": "KEY",
        "LossCost": "TARGET",
        "ExposureYears": "WEIGHT",
        "Segment": "FEATURE",
    }
    split_set = to_sql_calls[2]["frame"].iloc[0]
    assert split_set["row_order_sha256"] == compute_row_order_sha256(
        raw_frame,
        pk_columns=("PolicyID", "Snapshot"),
    )


def test_create_fremtpl_manifest_defaults_to_metadata_only_cv_split_set(monkeypatch):
    engine = FakeEngine()
    raw_frame = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [0, 1, 0],
            "Exposure": [0.5, 1.0, 0.25],
            "Area": ["A", "B", "C"],
        }
    )
    to_sql_calls = []

    monkeypatch.setattr(
        manifest.pd,
        "read_sql_query",
        lambda sql, con: raw_frame,
    )

    def fake_to_sql(self, name, con, **kwargs):
        to_sql_calls.append({"name": name, "frame": self.copy()})
        engine.events.append(f"to_sql:{name}")

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    created = create_fremtpl_manifest(
        engine,
        manifest_id="manifest_2",
        n_splits=3,
        random_state=123,
        created_by="unit-test",
    )

    assert created == "manifest_2"
    assert engine.events == [
        "to_sql:DATASET_MANIFEST",
        "to_sql:DATASET_COLUMN",
        "to_sql:CV_SPLIT_SET",
        "to_sql:CV_FOLD",
    ]
    assert [call["name"] for call in to_sql_calls] == [
        "DATASET_MANIFEST",
        "DATASET_COLUMN",
        "CV_SPLIT_SET",
        "CV_FOLD",
    ]
    assert engine.connection.executed == []

    split_set = to_sql_calls[2]["frame"].iloc[0]
    assert split_set["split_set_id"] == "manifest_2__kfold_3_seed_123"
    assert split_set["row_count"] == 3
    assert split_set["fold_count"] == 3
    assert split_set["split_mode"] == "REPLAYABLE"
    assert json.loads(split_set["runtime_metadata_json"])["packages"]["sklearn"]


def test_load_fremtpl_manifest_script_help_runs_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/load_fremtpl_manifest.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--manifest-id" in result.stdout
    assert "--n-splits" in result.stdout
    assert "--random-state" in result.stdout
    assert "--created-by" in result.stdout
    assert "--data-id" not in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
