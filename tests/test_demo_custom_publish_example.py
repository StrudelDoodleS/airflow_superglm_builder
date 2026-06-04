from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from openpyxl import load_workbook


def test_demo_version_helper_increments_trained_model_versions_only():
    from pricing_models.demo_custom_publish.tasks import next_model_version_from_existing

    assert next_model_version_from_existing([]) == "v1"
    assert next_model_version_from_existing(["v1", "v2"]) == "v3"
    assert next_model_version_from_existing(["manual-v99", "v4", "20260604"]) == "v5"


def test_demo_effective_from_uses_run_date_not_hardcoded_string():
    from pricing_models.demo_custom_publish.tasks import (
        effective_from_for_run,
        run_key_for_value,
        training_table_for_run,
    )

    assert effective_from_for_run(date(2026, 6, 4)) == "2026-06-04"
    assert (
        effective_from_for_run(datetime(2026, 6, 4, 23, 15, tzinfo=UTC))
        == "2026-06-04"
    )
    assert (
        run_key_for_value("manual__2026-06-04T23:15:00+00:00").startswith(
            "manual__20260604t2315000000_"
        )
    )
    assert training_table_for_run("manual__2026-06-04T23:15:00+00:00").startswith(
        "DEMO_CUSTOM_PUBLISH_TRAINING_manual__20260604t2315000000_"
    )
    assert len(training_table_for_run("x" * 200)) <= 128


def test_demo_run_scoped_keys_are_collision_safe():
    from pricing_models.demo_custom_publish.tasks import (
        run_key_for_value,
        training_table_for_run,
    )

    punctuation_keys = {
        run_key_for_value("scheduled__2026-06-04T12:00:00+00:00"),
        run_key_for_value("scheduled__20260604T1200000000"),
    }
    long_prefix_keys = {
        run_key_for_value(("x" * 200) + "a"),
        run_key_for_value(("x" * 200) + "b"),
    }
    punctuation_tables = {
        training_table_for_run("scheduled__2026-06-04T12:00:00+00:00"),
        training_table_for_run("scheduled__20260604T1200000000"),
    }
    long_prefix_tables = {
        training_table_for_run(("x" * 200) + "a"),
        training_table_for_run(("x" * 200) + "b"),
    }

    assert len(punctuation_keys) == 2
    assert len(long_prefix_keys) == 2
    assert len(punctuation_tables) == 2
    assert len(long_prefix_tables) == 2
    assert all(len(key) <= 99 for key in punctuation_keys | long_prefix_keys)
    assert all(len(table) <= 128 for table in punctuation_tables | long_prefix_tables)


def test_demo_dataset_spec_can_point_at_run_specific_training_table():
    from pricing_models.demo_custom_publish.tasks import dataset_spec_for_training_table

    dataset = dataset_spec_for_training_table("DEMO_CUSTOM_PUBLISH_TRAINING_run_1")

    assert dataset.dataset_name == "demo_custom_frequency_training"
    assert "pricing_stg.DEMO_CUSTOM_PUBLISH_TRAINING_run_1" in dataset.manifest_sql
    assert dataset.pk_columns == ("policy_id",)
    assert dataset.target_column == "claim_count"
    assert dataset.weight_column == "exposure"


def test_demo_training_export_returns_completed_build_payload(tmp_path: Path):
    from pricing_models.demo_custom_publish.tasks import (
        build_demo_training_frame,
        export_superglm_completed_build,
    )

    frame = build_demo_training_frame(row_count=180, seed=20260604)

    completed = export_superglm_completed_build(
        frame,
        output_dir=tmp_path,
        model_version="v1",
        effective_from="2026-06-04",
        created_by="pytest",
        export_id="demo-custom-publish-test",
    )

    workbook_path = Path(completed["rating_workbook_path"])
    model_path = Path(completed["model_artifact_path"])
    assert workbook_path.parent == tmp_path / "demo-custom-publish-test"
    summary_path = workbook_path.parent / "model_summary.txt"

    assert completed["model_version"] == "v1"
    assert completed["effective_from"] == "2026-06-04"
    assert completed["created_by"] == "pytest"
    assert completed["export_id"] == "demo-custom-publish-test"
    assert completed["mlflow_run_id"] is None
    assert completed["metrics"]["row_count"] == 180
    assert completed["metrics"]["deviance"] > 0
    assert workbook_path.exists()
    assert model_path.exists()
    assert summary_path.exists()

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    assert workbook.sheetnames == [
        "Rating Tables",
        "Discretization Impact",
        "Model Summary",
    ]
    assert workbook["Rating Tables"]["A2"].value == "Base"
    assert "SuperGLM Results" in summary_path.read_text(encoding="utf-8")
