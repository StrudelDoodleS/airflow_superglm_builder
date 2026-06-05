from __future__ import annotations

import builtins
import importlib
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

from pricing_pipeline.orchestration.airflow_run_metadata import (
    context_logical_date,
    merge_prepared_payload_metadata,
    task_run_metadata,
)
from pricing_pipeline.orchestration.run_context import run_key_for_value


class FakeDagRun:
    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def test_airflow_run_metadata_module_does_not_import_airflow(monkeypatch):
    sys.modules.pop("pricing_pipeline.orchestration.airflow_run_metadata", None)
    original_import = builtins.__import__

    def import_without_airflow(name, *args, **kwargs):
        if name == "airflow" or name.startswith("airflow."):
            raise AssertionError("airflow must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_airflow)

    module = importlib.import_module("pricing_pipeline.orchestration.airflow_run_metadata")

    assert module.task_run_metadata


def test_context_logical_date_prefers_direct_context_value():
    assert context_logical_date(
        {
            "logical_date": date(2026, 6, 5),
            "dag_run": FakeDagRun(logical_date=date(2020, 1, 1)),
        }
    ) == date(2026, 6, 5)


@pytest.mark.parametrize("attr_name", ["logical_date", "run_after", "execution_date"])
def test_context_logical_date_reads_dag_run_fallbacks(attr_name):
    expected = datetime(2026, 6, 5, 14, 30)

    assert context_logical_date({"dag_run": FakeDagRun(**{attr_name: expected})}) == expected


def test_task_run_metadata_uses_run_id_for_key_and_logical_date_for_dates(tmp_path):
    metadata = task_run_metadata(
        {
            "run_id": "scheduled__2026-06-05T00:00:00+00:00",
            "logical_date": datetime(2026, 6, 4, 23, 0),
        },
        output_root=tmp_path,
    )

    expected_run_key = run_key_for_value("scheduled__2026-06-05T00:00:00+00:00")
    assert metadata == {
        "run_key": expected_run_key,
        "output_dir": str(tmp_path / expected_run_key),
        "effective_from": "2026-06-04",
        "data_as_of_date": "2026-06-04",
    }


def test_task_run_metadata_is_stable_for_airflow_retry(tmp_path):
    context = {
        "run_id": "scheduled__2026-06-05T00:00:00+00:00",
        "logical_date": datetime(2026, 6, 4, 23, 0),
    }

    first = task_run_metadata(context, output_root=tmp_path)
    second = task_run_metadata(context, output_root=tmp_path)

    assert first == second


def test_task_run_metadata_falls_back_to_manual_when_context_is_empty(tmp_path):
    metadata = task_run_metadata({}, output_root=tmp_path)

    assert metadata["run_key"] == run_key_for_value("manual")
    assert metadata["output_dir"] == str(tmp_path / metadata["run_key"])
    assert len(metadata["effective_from"]) == 10
    assert metadata["data_as_of_date"] == metadata["effective_from"]


def test_merge_prepared_payload_metadata_rejects_run_key_mismatch():
    with pytest.raises(ValueError, match="run_key"):
        merge_prepared_payload_metadata(
            {
                "run_key": "run-1",
                "output_dir": "/tmp/run-1",
                "effective_from": "2026-06-05",
                "data_as_of_date": "2026-06-05",
            },
            {"run_key": "run-2"},
        )


def test_merge_prepared_payload_metadata_allows_selected_payload_overrides():
    result = merge_prepared_payload_metadata(
        {
            "run_key": "run-1",
            "output_dir": "/tmp/run-1",
            "effective_from": "2026-06-05",
            "data_as_of_date": "2026-06-05",
        },
        {
            "run_key": "run-1",
            "output_dir": "/custom/output",
            "effective_from": "2026-06-06",
            "data_as_of_date": "2026-05-31",
            "training_frame_path": "/custom/output/frame.parquet",
        },
    )

    assert result == {
        "run_key": "run-1",
        "output_dir": "/custom/output",
        "effective_from": "2026-06-06",
        "data_as_of_date": "2026-05-31",
        "training_frame_path": "/custom/output/frame.parquet",
    }


def test_prepare_source_payload_path_can_be_formed_from_metadata(tmp_path):
    metadata = task_run_metadata(
        {"run_id": "manual__1", "logical_date": date(2026, 6, 5)},
        output_root=tmp_path,
    )

    assert Path(metadata["output_dir"]).parent == tmp_path
