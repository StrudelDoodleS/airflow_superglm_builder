from __future__ import annotations

from datetime import date, datetime

import pytest

from pricing_pipeline.orchestration.completed_build_helpers import (
    completed_model_build_payload,
    effective_from_for_run,
    required_payload_text,
)


def test_effective_from_for_run_normalizes_date_values():
    assert effective_from_for_run(date(2026, 6, 5)) == "2026-06-05"
    assert effective_from_for_run(datetime(2026, 6, 5, 14, 30)) == "2026-06-05"
    assert effective_from_for_run("2026-06-05") == "2026-06-05"
    assert effective_from_for_run("2026-06-05T14:30:00") == "2026-06-05"


@pytest.mark.parametrize("bad", ["", "   ", "not-a-date", 20260605])
def test_effective_from_for_run_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="date"):
        effective_from_for_run(bad)


def test_required_payload_text_returns_stripped_text():
    assert required_payload_text({"field": "  value  "}, "field") == "value"


@pytest.mark.parametrize("payload", [{}, {"field": None}, {"field": "   "}])
def test_required_payload_text_rejects_missing_or_blank_values(payload):
    with pytest.raises(ValueError, match="prepared payload field 'field' is required"):
        required_payload_text(payload, "field")


def test_completed_model_build_payload_includes_optional_mlflow_run_id():
    payload = completed_model_build_payload(
        rating_workbook_path="/tmp/rating.xlsx",
        model_version="v4",
        effective_from="2026-06-05",
        export_id="model__run_1",
        created_by="airflow",
        manifest_id="manifest-1",
        split_set_id="split-1",
        mlflow_run_id="mlflow-1",
        model_artifact_path="/tmp/model.pkl",
        metrics={"deviance": 12.3},
    )

    assert payload == {
        "rating_workbook_path": "/tmp/rating.xlsx",
        "model_version": "v4",
        "effective_from": "2026-06-05",
        "created_by": "airflow",
        "export_id": "model__run_1",
        "dag_id": None,
        "airflow_run_id": None,
        "mlflow_run_id": "mlflow-1",
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
        "model_artifact_path": "/tmp/model.pkl",
        "metrics": {"deviance": 12.3},
    }


def test_completed_model_build_payload_carries_candidate_artifact_and_metrics():
    payload = completed_model_build_payload(
        rating_workbook_path="/tmp/rating.xlsx",
        model_version="v4",
        effective_from="2026-06-05",
        export_id="model__run_1",
        created_by="airflow",
        manifest_id="manifest-1",
        split_set_id="split-1",
        candidate_artifact_path="/tmp/candidate.joblib",
        candidate_artifact_sha256="a" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=123,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        model_source_sha256="b" * 64,
        metrics={"cv_pooled_deviance": 0.42},
        metric_scopes={"cv_pooled_deviance": "cv"},
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
        ),
    )

    assert payload["candidate_artifact_path"] == "/tmp/candidate.joblib"
    assert payload["candidate_artifact_size_bytes"] == 123
    assert payload["metric_scopes"] == {"cv_pooled_deviance": "cv"}
    assert payload["fold_metrics"][0]["fold_no"] == 1


@pytest.mark.parametrize("manifest_id", [None, "", "   "])
def test_completed_model_build_payload_requires_manifest_id(manifest_id):
    with pytest.raises(ValueError, match="manifest_id"):
        completed_model_build_payload(
            rating_workbook_path="/tmp/rating.xlsx",
            model_version="v1",
            effective_from="2026-06-05",
            export_id="model__run_1",
            created_by="airflow",
            manifest_id=manifest_id,
            split_set_id=None,
        )
