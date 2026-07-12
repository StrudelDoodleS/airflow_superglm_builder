from __future__ import annotations

from types import SimpleNamespace

import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.publishing.lifecycle import PublishResult


def test_editor_publisher_creates_child_and_derived_run(monkeypatch, tmp_path):
    from pricing_pipeline.publishing import editor_candidate

    submission = SimpleNamespace(
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
        submission_id="submission-1",
        model_name="HOME_FREQ",
        source_package_version=7,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        reason="Market calibration",
        claimed_identity="prototype-local-not-authenticated",
        model_source_sha256="b" * 64,
    )
    parent = SimpleNamespace(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v4",
        config=SimpleNamespace(
            model_name="HOME_FREQ",
            target_name="claim_count",
            model_type="superglm_poisson",
            default_package_status="PUBLISHED",
        ),
    )
    exported = SimpleNamespace(
        export_id="editor__submission_1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        publication_receipt_path=str(tmp_path / "publication_receipt.json"),
        publication_receipt_sha256="c" * 64,
        candidate_artifact_path=str(tmp_path / "candidate_bundle.joblib"),
        candidate_artifact_sha256="d" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=321,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        revision_metadata_json='{"kind":"SUPERGLM_EDITOR"}',
        metrics={"editor_training_deviance_delta": 0.009},
        metric_scopes={"editor_training_deviance_delta": "editor_training_parent"},
    )
    calls = []
    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda path, digest, **kwargs: submission,
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_parent_candidate",
        lambda engine, loaded_submission: parent,
    )
    monkeypatch.setattr(
        editor_candidate,
        "export_edited_model",
        lambda loaded_parent, loaded_submission: exported,
    )
    monkeypatch.setattr(
        editor_candidate,
        "stage_editor_export",
        lambda engine, loaded_parent, export, created_by: calls.append(
            ("stage", created_by)
        ),
    )
    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        lambda engine, **kwargs: calls.append(("publish", kwargs))
        or PublishResult(
            mlflow_run_id="",
            export_id=exported.export_id,
            rate_package_id=108,
            package_version=8,
            rating_workbook_path=exported.rating_workbook_path,
        ),
    )
    monkeypatch.setattr(
        editor_candidate,
        "record_derived_model_run",
        lambda engine, **kwargs: calls.append(("lineage", kwargs)) or 908,
    )

    result = editor_candidate.publish_editor_submission(
        object(),
        settings=Settings(workbench_artifact_root=tmp_path),
        submission_path=submission.path,
        submission_sha256=submission.sha256,
        dag_id="pricing_publish_editor_candidate",
        airflow_run_id="manual__submission-1",
        created_by="analyst@example.test",
    )

    assert result.parent_rate_package_id == submission.parent_rate_package_id
    assert result.rate_package_id == 108
    assert result.package_version == 8
    assert result.model_run_id == 908
    assert [name for name, _value in calls] == ["stage", "publish", "lineage"]
    publish_kwargs = calls[1][1]
    assert publish_kwargs["parent_rate_package_id"] == submission.parent_rate_package_id
    assert publish_kwargs["revision_metadata_json"] == exported.revision_metadata_json
    lineage_kwargs = calls[2][1]
    assert lineage_kwargs["rate_package_id"] == result.rate_package_id
    assert lineage_kwargs["manifest_id"] == submission.manifest_id
    assert lineage_kwargs["split_set_id"] == submission.split_set_id
    assert lineage_kwargs["candidate_artifact_sha256"] == "d" * 64


def test_training_comparison_metrics_are_stable_and_scoped():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import training_comparison_metrics
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Model:
        def __init__(self, prediction):
            self.prediction = np.asarray(prediction, dtype=float)

        def predict(self, X, offset=None):
            assert len(X) == 3
            return self.prediction

    bundle = CandidateBundle(
        fitted_model=Model([1.0, 2.0, 3.0]),
        X=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        y=np.array([1.0, 1.0, 4.0]),
        sample_weight=np.array([1.0, 2.0, 1.0]),
        offset=None,
        export_weight=None,
        cv_report={},
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    metrics, scopes = training_comparison_metrics(
        bundle.fitted_model,
        Model([1.1, 1.8, 3.2]),
        bundle,
        comparison_name="parent",
    )

    assert metrics["editor_training_parent_mean_absolute_prediction_delta"] == pytest.approx(
        0.175
    )
    assert metrics["editor_training_parent_max_absolute_prediction_delta"] == pytest.approx(0.2)
    assert "editor_training_deviance_delta" in metrics
    assert set(scopes.values()) == {"editor_training_parent"}


def test_package_specific_parity_uses_bounded_rows_and_explicit_package_id():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Model:
        def predict(self, X, offset=None):
            return np.asarray(X["x"], dtype=float) * 2.0

    class Result:
        def __init__(self, prediction):
            self.prediction = prediction

        def mappings(self):
            return self

        def one(self):
            return {"prediction": self.prediction}

    class Connection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params):
            self.calls.append((str(statement), params))
            return Result(params["x_prediction"])

    bundle = CandidateBundle(
        fitted_model=Model(),
        X=pd.DataFrame({"x": np.arange(100, dtype=float)}),
        y=np.ones(100),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )
    connection = Connection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=5,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert len(connection.calls) == 5
    assert all(params["rate_package_id"] == 108 for _sql, params in connection.calls)
    assert all("PREDICT_RATE_PACKAGE" in sql for sql, _params in connection.calls)
