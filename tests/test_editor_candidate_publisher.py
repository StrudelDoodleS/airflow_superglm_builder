from __future__ import annotations

import json
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
    allowed_roots = []
    publish_connection = object()
    publication_is_active = False
    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda path, digest, **kwargs: submission,
    )
    def fake_load_parent_candidate(engine, loaded_submission, *, allowed_root):
        allowed_roots.append(("parent", allowed_root))
        return parent

    def fake_export_edited_model(loaded_parent, loaded_submission, *, allowed_root):
        allowed_roots.append(("edited", allowed_root))
        return exported

    monkeypatch.setattr(editor_candidate, "load_parent_candidate", fake_load_parent_candidate)
    monkeypatch.setattr(editor_candidate, "export_edited_model", fake_export_edited_model)
    monkeypatch.setattr(
        editor_candidate,
        "stage_editor_export",
        lambda engine, loaded_parent, export, created_by: calls.append(
            ("stage", created_by)
        ),
    )
    def fake_publish_rating_package(engine, **kwargs):
        nonlocal publication_is_active
        calls.append(("publish", kwargs))
        publication_is_active = True
        try:
            lineage_writer = kwargs.get("package_lineage_writer")
            if lineage_writer is not None:
                lineage_writer(publish_connection, 108)
        finally:
            publication_is_active = False
        return PublishResult(
            mlflow_run_id="",
            export_id=exported.export_id,
            rate_package_id=108,
            package_version=8,
            rating_workbook_path=exported.rating_workbook_path,
        )

    def fake_record_derived_model_run(connection, **kwargs):
        assert publication_is_active, "lineage must execute inside package publication"
        assert connection is publish_connection
        calls.append(("lineage", kwargs))
        return 908

    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        fake_publish_rating_package,
    )
    monkeypatch.setattr(
        editor_candidate,
        "record_derived_model_run",
        fake_record_derived_model_run,
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
    assert callable(publish_kwargs["package_lineage_writer"])
    lineage_kwargs = calls[2][1]
    assert lineage_kwargs["rate_package_id"] == result.rate_package_id
    assert lineage_kwargs["parent_model_run_id"] == submission.parent_model_run_id
    assert lineage_kwargs["manifest_id"] == submission.manifest_id
    assert lineage_kwargs["split_set_id"] == submission.split_set_id
    assert lineage_kwargs["candidate_artifact_sha256"] == "d" * 64
    assert allowed_roots == [("parent", tmp_path), ("edited", tmp_path)]


@pytest.mark.parametrize(
    "submission_relative_path",
    [
        "submission.json",
        "models/HOME_FREQ/editor/submissions/deep/submission.json",
    ],
)
def test_parent_candidate_uses_exact_configured_root_and_unambiguous_split_link(
    monkeypatch,
    tmp_path,
    submission_relative_path,
):
    from pricing_pipeline.publishing import editor_candidate

    configured_root = tmp_path / "configured-workbench"
    candidate_path = configured_root / "models/HOME_FREQ/runs/deep/candidate.joblib"
    submission = SimpleNamespace(
        path=str(configured_root / submission_relative_path),
        model_name="HOME_FREQ",
        source_package_version=7,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        baseline_candidate_path=str(candidate_path),
        baseline_candidate_sha256="a" * 64,
        model_source_sha256="b" * 64,
    )
    row = {
        "model_id": 17,
        "model_name": submission.model_name,
        "model_version": "v4",
        "package_version": submission.source_package_version,
        "rate_package_id": submission.parent_rate_package_id,
        "effective_from_date": "2026-01-01",
        "effective_to_date": None,
        "model_run_id": submission.parent_model_run_id,
        "run_status": "SUCCESS",
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "candidate_artifact_path": submission.baseline_candidate_path,
        "candidate_artifact_sha256": submission.baseline_candidate_sha256,
        "candidate_artifact_format": "superglm-candidate-joblib-v1",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
        "model_source_sha256": submission.model_source_sha256,
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def __init__(self):
            self.statements = []

        def execute(self, statement, params):
            self.statements.append((str(statement), params))
            return Rows()

    class Begin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def __init__(self):
            self.connection = Connection()

        def begin(self):
            return Begin(self.connection)

    bundle = SimpleNamespace(manifest_id="manifest-1", split_set_id="split-1")
    load_calls = []
    champion_calls = []

    def fake_load_candidate_bundle(path, **kwargs):
        load_calls.append((path, kwargs))
        return bundle

    def fake_load_champion_bundle(engine, **kwargs):
        champion_calls.append(kwargs)
        return None, "no champion"

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    monkeypatch.setattr(editor_candidate, "load_candidate_bundle", fake_load_candidate_bundle)
    monkeypatch.setattr(editor_candidate, "_load_champion_bundle", fake_load_champion_bundle)
    monkeypatch.setattr(
        editor_candidate,
        "get_model_config",
        lambda model_name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    engine = Engine()

    parent = editor_candidate.load_parent_candidate(
        engine,
        submission,
        allowed_root=configured_root,
    )

    assert parent.bundle is bundle
    assert load_calls[0][0] == str(candidate_path)
    assert load_calls[0][1]["allowed_root"] == configured_root
    assert champion_calls[0]["allowed_root"] == configured_root
    statement = engine.connection.statements[0][0]
    assert "split_link.manifest_id = mr.manifest_id" in statement
    assert "split_link.dataset_role = 'training'" in statement
    assert "split_link.split_role = 'validation'" in statement


@pytest.mark.parametrize(
    "submission_relative_path",
    ["submission.json", "crafted/deep/layout/submission.json"],
)
def test_edited_model_root_cannot_be_widened_by_submission_path(
    tmp_path,
    submission_relative_path,
):
    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    configured_root = tmp_path / "configured-workbench"
    outside_model = tmp_path / "outside" / "edited.joblib"
    submission = SimpleNamespace(
        path=str(tmp_path / submission_relative_path),
        edited_model_path=str(outside_model),
    )

    with pytest.raises(EditorSubmissionError, match="outside artifact root"):
        editor_candidate._load_edited_model(
            submission,
            allowed_root=configured_root,
        )


def test_edited_model_loader_supports_nested_path_within_configured_root(
    monkeypatch,
    tmp_path,
):
    import platform

    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.submission import EDITED_MODEL_FORMAT, sha256_file

    configured_root = tmp_path / "configured-workbench"
    model_path = configured_root / "models/HOME_FREQ/editor/deep/edited.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"nested-model")
    python_version = platform.python_version()
    superglm_version = "test-superglm"
    edited_model = object()
    submission = SimpleNamespace(
        path=str(configured_root / "submission.json"),
        edited_model_path=str(model_path),
        edited_model_format=EDITED_MODEL_FORMAT,
        edited_model_size_bytes=model_path.stat().st_size,
        edited_model_sha256=sha256_file(model_path),
        edited_model_python_version=python_version,
        edited_model_superglm_version=superglm_version,
    )
    monkeypatch.setattr(editor_candidate, "version", lambda name: superglm_version)
    monkeypatch.setattr(
        editor_candidate.joblib,
        "load",
        lambda path: {
            "format": EDITED_MODEL_FORMAT,
            "python_version": python_version,
            "superglm_version": superglm_version,
            "model": edited_model,
        },
    )

    loaded = editor_candidate._load_edited_model(
        submission,
        allowed_root=configured_root,
    )

    assert loaded is edited_model


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
                    "rate_package_id": 107,
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

    snapshot = _load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert snapshot.status == "COMPARED"
    assert snapshot.rate_package_id == 107
    assert snapshot.unavailable_reason is None
    assert snapshot.bundle is not None
    assert snapshot.bundle.manifest_id == "champion-manifest"


@pytest.mark.parametrize(
    ("rows", "expected_status", "expected_rate_package_id"),
    [
        ([], "NO_CHAMPION", None),
        (
            [{"rate_package_id": 107, "run_status": "FAILED"}],
            "UNAVAILABLE",
            107,
        ),
    ],
)
def test_champion_snapshot_distinguishes_absent_and_unavailable_champion(
    rows,
    expected_status,
    expected_rate_package_id,
    tmp_path,
):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import _load_champion_bundle
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return rows

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

    parent = CandidateBundle(
        fitted_model=object(),
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        manifest_id="parent-manifest",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    snapshot = _load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert snapshot.status == expected_status
    assert snapshot.rate_package_id == expected_rate_package_id
    assert snapshot.revision_metadata()["deployment_slot"] == "HOME_FREQ_UAT"
    assert snapshot.revision_metadata()["available"] is (expected_status == "COMPARED")


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


class _ParityResult:
    def __init__(self, prediction):
        self.prediction = prediction

    def mappings(self):
        return self

    def one(self):
        return {"prediction": self.prediction}


class _ParityConnection:
    def __init__(self, *, allow_execute=True):
        self.allow_execute = allow_execute
        self.calls = []

    def execute(self, statement, params):
        if not self.allow_execute:
            raise AssertionError("SQL must not execute for an invalid offset source")
        self.calls.append(params)
        return _ParityResult(params["x_prediction"])


def _offset_parity_bundle(*, handling, offset_source=None, published_values=None):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.workbench.artifacts import CandidateBundle

    raw_exposure = np.array([2.0, 4.0])
    fitted_offset = np.log(raw_exposure)
    if published_values is None:
        published_values = raw_exposure

    class Model:
        def predict(self, X, offset=None):
            np.testing.assert_allclose(offset, fitted_offset)
            return np.asarray(X["x"], dtype=float) + np.exp(offset)

    if handling == "EXPORTED_FACTOR":
        contract = {
            "handling": handling,
            "source_factor_name": "Exposure",
            "published_factor_name": "Exposure",
            "source_name": "Exposure",
            "label": "log(Exposure)",
        }
        export_options = {"offset_source": offset_source}
    else:
        contract = {
            "handling": handling,
            "source_name": "Exposure",
            "label": "log(Exposure)",
        }
        export_options = None

    return CandidateBundle(
        fitted_model=Model(),
        X=pd.DataFrame({"x": [1.0, 3.0], "Exposure": published_values}),
        y=np.ones(2),
        sample_weight=None,
        offset=fitted_offset,
        export_weight=None,
        cv_report={},
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract=contract,
        offset_export_options=export_options,
    ), raw_exposure


@pytest.mark.parametrize("source_kind", ["column", "series", "array"])
def test_package_sql_parity_uses_published_offset_source_for_exported_factor(source_kind):
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    raw_exposure = pd.Series([2.0, 4.0], name="Exposure")
    offset_sources = {
        "column": "Exposure",
        "series": raw_exposure,
        "array": raw_exposure.to_numpy(),
    }
    bundle, expected_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=offset_sources[source_kind],
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [
        json.loads(params["features_json"])["Exposure"] for params in connection.calls
    ] == expected_exposure.tolist()


@pytest.mark.parametrize("source_kind", ["column", "series"])
def test_package_sql_parity_preserves_categorical_offset_levels_positionally(source_kind):
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    published_levels = pd.Series(
        ["basic", "premium"],
        index=pd.Index([101, 303]),
        name="Exposure",
        dtype="category",
    )
    offset_source = "Exposure" if source_kind == "column" else published_levels
    bundle, _raw_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=offset_source,
        published_values=published_levels.to_numpy(),
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [
        json.loads(params["features_json"])["Exposure"] for params in connection.calls
    ] == published_levels.tolist()


def test_package_sql_parity_normalizes_mapping_offset_source_positionally():
    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    offset_source = {101: "basic", 303: "premium"}
    bundle, _raw_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=offset_source,
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [
        json.loads(params["features_json"])["Exposure"] for params in connection.calls
    ] == list(offset_source.values())


@pytest.mark.parametrize(
    ("offset_source", "message"),
    [
        ([2.0], "offset_source length 1 does not match candidate row count 2"),
        ([2.0, float("nan")], "offset_source contains missing values"),
        ([2.0, float("inf")], "offset_source contains non-finite numeric values"),
    ],
)
def test_package_sql_parity_rejects_invalid_exported_offset_source_before_sql(
    offset_source,
    message,
):
    import numpy as np

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    bundle, _raw_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=np.asarray(offset_source),
    )

    with pytest.raises(EditorSubmissionError, match=message):
        verify_package_sql_parity(
            _ParityConnection(allow_execute=False),
            rate_package_id=108,
            edited_model=bundle.fitted_model,
            bundle=bundle,
        )


@pytest.mark.parametrize(
    ("dtype", "infinity"),
    [("object", float("inf")), ("category", float("-inf"))],
)
def test_package_sql_parity_rejects_nonfinite_numeric_levels_in_mixed_series(
    dtype,
    infinity,
):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    offset_source = pd.Series(["basic", np.float64(infinity)], dtype=dtype)
    bundle, _raw_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=offset_source,
    )

    with pytest.raises(
        EditorSubmissionError,
        match="offset_source contains non-finite numeric values",
    ):
        verify_package_sql_parity(
            _ParityConnection(allow_execute=False),
            rate_package_id=108,
            edited_model=bundle.fitted_model,
            bundle=bundle,
        )


def test_package_sql_parity_applies_fitted_offset_as_sql_exposure():
    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    bundle, raw_exposure = _offset_parity_bundle(
        handling="ALREADY_APPLIED_SQL_EXPOSURE"
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [params["exposure"] for params in connection.calls] == raw_exposure.tolist()


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
