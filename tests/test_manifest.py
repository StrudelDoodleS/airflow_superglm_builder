import json
import os
import subprocess
import sys
from datetime import date

import pandas as pd

from pricing_pipeline import manifest
from pricing_pipeline.manifest import (
    build_column_metadata,
    build_cv_splits,
    build_row_keys,
    create_fremtpl_manifest,
    new_manifest_id,
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


def test_build_row_keys_uses_idpol_current_order_and_deterministic_folds():
    frame = manifest_frame()

    row_keys = build_row_keys(
        frame,
        manifest_id="manifest_1",
        n_splits=3,
        random_state=42,
    )

    assert list(row_keys.columns) == [
        "manifest_id",
        "source_pk_text",
        "row_ordinal",
        "cv_fold_no",
    ]
    assert row_keys["manifest_id"].tolist() == ["manifest_1"] * len(frame)
    assert row_keys["source_pk_text"].tolist() == [
        "IDpol=10",
        "IDpol=20",
        "IDpol=30",
        "IDpol=40",
        "IDpol=50",
    ]
    assert row_keys["row_ordinal"].tolist() == [1, 2, 3, 4, 5]
    assert row_keys["cv_fold_no"].tolist() == [2, 1, 2, 3, 1]


def test_build_cv_splits_records_train_folds_json_and_test_fold():
    splits = build_cv_splits("manifest_1", n_splits=4)

    assert list(splits.columns) == [
        "manifest_id",
        "split_no",
        "train_folds_json",
        "test_fold_no",
    ]
    assert splits["split_no"].tolist() == [1, 2, 3, 4]
    assert splits["test_fold_no"].tolist() == [1, 2, 3, 4]
    assert [json.loads(value) for value in splits["train_folds_json"]] == [
        [2, 3, 4],
        [1, 3, 4],
        [1, 2, 4],
        [1, 2, 3],
    ]


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


def test_create_fremtpl_manifest_reads_raw_table_and_persists_manifest_sequence(
    monkeypatch,
):
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
        "execute:TRUNCATE",
        "to_sql:STG_DATASET_ROW_KEY",
        "execute:INSERT",
        "to_sql:CV_SPLIT",
    ]

    assert [(call["name"], call["schema"], call["if_exists"], call["index"]) for call in to_sql_calls] == [
        ("DATASET_MANIFEST", "pricing", "append", False),
        ("DATASET_COLUMN", "pricing", "append", False),
        ("STG_DATASET_ROW_KEY", "pricing", "append", False),
        ("CV_SPLIT", "pricing", "append", False),
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

    staged_keys = to_sql_calls[2]["frame"]
    assert staged_keys["source_pk_text"].tolist() == [
        "IDpol=1",
        "IDpol=2",
        "IDpol=3",
    ]
    assert staged_keys["row_ordinal"].tolist() == [1, 2, 3]
    assert staged_keys["cv_fold_no"].tolist() == [1, 2, 3]

    truncate_sql, truncate_params = engine.connection.executed[0]
    assert truncate_sql == "TRUNCATE TABLE pricing.STG_DATASET_ROW_KEY"
    assert truncate_params is None

    insert_sql, insert_params = engine.connection.executed[1]
    assert "HASHBYTES('SHA2_256', source_pk_text)" in insert_sql
    assert "WHERE manifest_id = :manifest_id" in insert_sql
    assert insert_params == {"manifest_id": "manifest_1"}

    cv_splits = to_sql_calls[3]["frame"]
    assert cv_splits["test_fold_no"].tolist() == [1, 2, 3]
    assert [json.loads(value) for value in cv_splits["train_folds_json"]] == [
        [2, 3],
        [1, 3],
        [1, 2],
    ]


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
