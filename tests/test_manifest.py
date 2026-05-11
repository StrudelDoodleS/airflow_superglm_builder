import json
import os
import subprocess
import sys
from datetime import date

import pandas as pd

from pricing_pipeline.data import manifest
from pricing_pipeline.data.manifest import (
    build_column_metadata,
    build_cv_split_set,
    compute_row_order_sha256,
    create_dataset_manifest,
    create_fremtpl_manifest,
    new_manifest_id,
    runtime_dependency_metadata,
)
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
    first = pd.DataFrame(
        {"PolicyID": [10, 10, 20], "Snapshot": ["2026-01", "2026-02", "2026-01"]}
    )
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
    def __init__(self):
        self.events = []
        self.connection = FakeBeginConnection(self.events)

    def begin(self):
        return FakeBegin(self.connection)


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
    assert read_calls == [
        ("SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine)
    ]
    assert engine.events == [
        "to_sql:DATASET_MANIFEST",
        "to_sql:DATASET_COLUMN",
        "to_sql:CV_SPLIT_SET",
        "to_sql:CV_FOLD",
    ]

    assert [(call["name"], call["schema"], call["if_exists"], call["index"]) for call in to_sql_calls] == [
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
    assert read_calls == [
        ("SELECT * FROM actuarial.loss_cost ORDER BY PolicyID, Snapshot", engine)
    ]
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
