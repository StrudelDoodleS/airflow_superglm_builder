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
    assert publish_kwargs["revision_metadata_json"] == (
        '{"kind":"SUPERGLM_EDITOR","published_by":"analyst@example.test"}'
    )
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
            self._distribution = SimpleNamespace(
                deviance_unit=lambda y, mu: (np.asarray(y) - np.asarray(mu)) ** 2
            )

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
    assert metrics["editor_training_deviance_delta"] == pytest.approx(-0.2675)
    assert set(scopes.values()) == {"editor_training_parent"}


def test_champion_comparison_scores_parent_rows_even_when_training_rows_differ(tmp_path):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import _load_champion_bundle
    from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle

    parent = CandidateBundle(
        fitted_model={"model": "parent"},
        X=pd.DataFrame({"x": [1.0, 2.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        manifest_id="parent-manifest",
        split_set_id="parent-split",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )
    champion = CandidateBundle(
        fitted_model={"model": "champion"},
        X=pd.DataFrame({"x": [8.0, 9.0, 10.0]}),
        y=np.array([1.0, 0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        manifest_id="champion-manifest",
        split_set_id="champion-split",
        pk_columns=("id",),
        row_order_sha256="c" * 64,
        model_source_sha256="d" * 64,
        offset_contract={"handling": "NONE"},
    )
    artifact = save_candidate_bundle(champion, tmp_path / "champion.joblib")

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "run_status": "SUCCESS",
                    "candidate_artifact_path": artifact.path,
                    "candidate_artifact_sha256": artifact.sha256,
                    "candidate_artifact_format": artifact.format,
                    "candidate_artifact_size_bytes": artifact.size_bytes,
                    "candidate_python_version": artifact.python_version,
                    "candidate_superglm_version": artifact.superglm_version,
                }
            ]

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    loaded, reason = _load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert reason is None
    assert loaded is not None
    assert loaded.manifest_id == "champion-manifest"


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


def test_editor_child_inherits_original_cv_baseline_with_explicit_scope():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import inherited_cv_metrics
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    bundle = CandidateBundle(
        fitted_model=object(),
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={
            "mean_scores": {"deviance": 0.48},
            "pooled_scores": {"deviance": 0.47},
            "std_scores": {"deviance": 0.03},
            "oof_coverage": 1.0,
        },
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    metrics, scopes = inherited_cv_metrics(bundle)

    assert metrics == {
        "cv_mean_deviance": 0.48,
        "cv_pooled_deviance": 0.47,
        "cv_std_deviance": 0.03,
        "cv_oof_coverage": 1.0,
    }
    assert set(scopes.values()) == {"inherited_cv"}
