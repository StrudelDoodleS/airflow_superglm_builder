import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.data import manifest as manifest_api
from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    build_column_metadata,
    build_validation_split_set,
    compute_row_order_sha256,
    create_model_frame_manifest_with_split,
    new_manifest_id,
    runtime_dependency_metadata,
    validation_split_indices,
)
from pricing_pipeline.models.config import ValidationSplitConfig


NO_VALIDATION = ValidationSplitConfig(
    method="none",
    n_splits=None,
    random_state=None,
    shuffle=False,
)
CUSTOM_VALIDATION = ValidationSplitConfig(
    method="custom",
    n_splits=None,
    random_state=None,
    shuffle=False,
    materialize=True,
)


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


@pytest.mark.parametrize(
    "offset_fields",
    [
        {},
        {
            "offset_column": "LogExposure",
            "offset_label": "log(Exposure)",
        },
        {
            "offset_column": "TermOffset",
            "offset_source_column": "TermMonths",
            "offset_label": "log(TermMonths / 12)",
        },
    ],
    ids=["none", "already-applied-sql-exposure", "exported-factor"],
)
def test_model_frame_manifest_spec_accepts_valid_offset_shapes(offset_fields):
    spec = ModelFrameManifestSpec(
        dataset_name="frequency",
        source_system="test",
        data_as_of_date="2026-06-30",
        pk_columns=("PolicyID",),
        target_column="ClaimNb",
        **offset_fields,
    )

    assert spec.offset_column == offset_fields.get("offset_column")
    assert spec.offset_source_column == offset_fields.get("offset_source_column")
    assert spec.offset_label == offset_fields.get("offset_label")


@pytest.mark.parametrize(
    "offset_fields",
    [
        {"offset_source_column": "TermMonths"},
        {"offset_label": "log(TermMonths / 12)"},
        {"offset_column": "TermOffset"},
        {
            "offset_column": "TermOffset",
            "offset_source_column": "TermMonths",
        },
    ],
    ids=["source-only", "label-only", "offset-only", "offset-and-source-without-label"],
)
def test_model_frame_manifest_spec_rejects_invalid_offset_shapes(offset_fields):
    with pytest.raises(ValueError, match="offset"):
        ModelFrameManifestSpec(
            dataset_name="frequency",
            source_system="test",
            data_as_of_date="2026-06-30",
            pk_columns=("PolicyID",),
            target_column="ClaimNb",
            **offset_fields,
        )


def test_build_column_metadata_marks_roles_and_counts_columns():
    frame = manifest_frame()

    columns = build_column_metadata(
        frame,
        manifest_id="manifest_1",
        spec=ModelFrameManifestSpec(
            dataset_name="freMTPL",
            source_system="test",
            data_as_of_date="2026-06-30",
            pk_columns=("IDpol",),
            target_column="ClaimNb",
            weight_column="Exposure",
            feature_columns=("Area",),
        ),
    )

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
        spec=ModelFrameManifestSpec(
            dataset_name="loss_cost",
            source_system="test",
            data_as_of_date="2026-06-30",
            pk_columns=("PolicyID", "Snapshot"),
            target_column="LossCost",
            feature_columns=("Postcode",),
        ),
    )

    assert dict(zip(columns["column_name"], columns["column_role"], strict=True)) == {
        "PolicyID": "KEY",
        "Snapshot": "KEY",
        "LossCost": "TARGET",
        "Postcode": "FEATURE",
    }


def test_build_column_metadata_records_only_explicit_feature_roles():
    frame = pd.DataFrame(
        {
            "PolicyID": [1, 2],
            "ClaimNb": [0, 1],
            "FitWeight": [1.0, 0.5],
            "TermOffset": [0.0, np.log(3.0)],
            "TermMonths": [12, 36],
            "ExportWeight": [10.0, 20.0],
            "SnapshotDate": ["2026-06-30", "2026-06-30"],
            "Fold": [1, 2],
            "DriverAge": [25, 40],
            "SourceNote": ["a", "b"],
        }
    )

    columns = build_column_metadata(
        frame,
        manifest_id="manifest_roles",
        spec=ModelFrameManifestSpec(
            dataset_name="frequency",
            source_system="test",
            data_as_of_date="2026-06-30",
            pk_columns=("PolicyID",),
            target_column="ClaimNb",
            weight_column="FitWeight",
            offset_column="TermOffset",
            offset_source_column="TermMonths",
            offset_label="log(TermMonths / 12)",
            export_weight_column="ExportWeight",
            data_as_of_column="SnapshotDate",
            feature_columns=("DriverAge",),
        ),
        split_column="Fold",
    )

    assert dict(zip(columns["column_name"], columns["column_role"], strict=True)) == {
        "PolicyID": "KEY",
        "ClaimNb": "TARGET",
        "FitWeight": "WEIGHT",
        "TermOffset": "OFFSET",
        "TermMonths": "OFFSET_SOURCE",
        "ExportWeight": "EXPORT_WEIGHT",
        "SnapshotDate": "DATA_AS_OF",
        "Fold": "SPLIT",
        "DriverAge": "FEATURE",
        "SourceNote": "OTHER",
    }


def test_build_column_metadata_preserves_intentional_operational_role_overlap():
    frame = pd.DataFrame(
        {
            "PolicyID": [1, 2],
            "ClaimNb": [0, 1],
            "TermMonths": [12, 36],
        }
    )

    columns = build_column_metadata(
        frame,
        manifest_id="manifest_shared_operational_column",
        spec=ModelFrameManifestSpec(
            dataset_name="frequency",
            source_system="test",
            data_as_of_date="2026-06-30",
            pk_columns=("PolicyID",),
            target_column="ClaimNb",
            offset_column="TermMonths",
            offset_source_column="TermMonths",
            offset_label="identity(TermMonths)",
            export_weight_column="TermMonths",
        ),
    )

    assert dict(zip(columns["column_name"], columns["column_role"], strict=True)) == {
        "PolicyID": "KEY",
        "ClaimNb": "TARGET",
        "TermMonths": "OFFSET+OFFSET_SOURCE+EXPORT_WEIGHT",
    }


def test_model_frame_evidence_binds_schema_values_and_row_order():
    frame = pd.DataFrame(
        {
            "policy_id": pd.Series([1, 2], dtype="int64"),
            "region": pd.Series(["N", "S"], dtype="string"),
        }
    )

    digest, metadata_json = manifest_api.model_frame_evidence(frame)
    same_digest, _ = manifest_api.model_frame_evidence(frame.copy())
    changed_value = frame.copy()
    changed_value.loc[1, "region"] = "E"
    changed_order = frame.iloc[::-1].reset_index(drop=True)
    changed_columns = frame.loc[:, ["region", "policy_id"]]
    changed_dtype = frame.astype({"policy_id": "float64"})

    assert len(digest) == 64
    assert digest == same_digest
    assert manifest_api.model_frame_evidence(changed_value)[0] != digest
    assert manifest_api.model_frame_evidence(changed_order)[0] != digest
    assert manifest_api.model_frame_evidence(changed_columns)[0] != digest
    assert manifest_api.model_frame_evidence(changed_dtype)[0] != digest
    metadata = json.loads(metadata_json)
    assert metadata["frame_hash"]["algorithm"] == "sha256"
    assert metadata["frame_hash"]["format_version"] == 1
    assert metadata["frame_hash"]["dataframe_index_included"] is False
    assert metadata["packages"]["pandas"] == pd.__version__
    assert metadata["python_version"].startswith(f"{sys.version_info.major}.")


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


def test_runtime_dependency_metadata_records_python_platform_and_core_versions():
    metadata_json = runtime_dependency_metadata()
    metadata = json.loads(metadata_json)

    assert metadata["python_version"].startswith(f"{sys.version_info.major}.")
    assert metadata["platform"]
    assert metadata["packages"]["numpy"]
    assert metadata["packages"]["pandas"]
    assert metadata["packages"]["sklearn"]
    assert "superglm" in metadata["packages"]
    assert metadata["superglm_git_sha"] == "b91fbef5f1ef15aadfa0372963fed3864607d816"


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
            "TermOffset": [0.0, np.log(3.0), np.log(2.0)],
            "PolicyTerm": [12, 36, 24],
            "ExportWeight": [4.0, 5.0, 6.0],
            "SnapshotDate": ["2026-06-30"] * 3,
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
            feature_columns=("BandedDriverAge",),
            offset_column="TermOffset",
            offset_source_column="PolicyTerm",
            offset_label="log(PolicyTerm / 12)",
            export_weight_column="ExportWeight",
            data_as_of_column="SnapshotDate",
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
    assert manifest_row["offset_column"] == "TermOffset"
    assert manifest_row["offset_source_column"] == "PolicyTerm"
    assert manifest_row["offset_label"] == "log(PolicyTerm / 12)"
    assert manifest_row["export_weight_column"] == "ExportWeight"
    assert "exposure_column" not in manifest_row
    assert manifest_row["data_as_of_column"] == "SnapshotDate"
    assert len(manifest_row["model_frame_sha256"]) == 64
    assert result.model_frame_sha256 == manifest_row["model_frame_sha256"]
    assert json.loads(manifest_row["frame_hash_metadata_json"])["packages"]["pandas"]
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
        "TermOffset": "OFFSET",
        "PolicyTerm": "OFFSET_SOURCE",
        "ExportWeight": "EXPORT_WEIGHT",
        "SnapshotDate": "DATA_AS_OF",
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
        validation_split=CUSTOM_VALIDATION,
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
        validation_split=NO_VALIDATION,
        split_indices=[],
        created_by="unit-test",
    )

    assert result.split_set_id is None
    assert "CV_SPLIT_SET" not in to_sql_calls
    assert "CV_FOLD" not in to_sql_calls


def test_kfold_without_shuffle_does_not_pass_a_random_seed_to_sklearn():
    frame = pd.DataFrame({"row_id": range(6)})

    folds = validation_split_indices(
        frame,
        ValidationSplitConfig.kfold(n_splits=3, shuffle=False),
    )

    assert [test.tolist() for _, test in folds] == [[0, 1], [2, 3], [4, 5]]


def test_train_test_split_can_stratify_by_repeated_composite_pk_component():
    frame = pd.DataFrame(
        {
            "policy_id": [101, 101, 102, 102, 103, 103, 104, 104],
            "risk_id": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )

    [(train, test)] = validation_split_indices(
        frame,
        ValidationSplitConfig.train_test_split(
            test_size=0.5,
            random_state=7,
            stratify_column="policy_id",
        ),
    )

    expected_policies = [101, 102, 103, 104]
    assert sorted(frame.iloc[train]["policy_id"]) == expected_policies
    assert sorted(frame.iloc[test]["policy_id"]) == expected_policies


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
        validation_split_indices(manifest_frame(), CUSTOM_VALIDATION)


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
            validation_split=NO_VALIDATION,
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
