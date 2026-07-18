from __future__ import annotations

import importlib
import inspect
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from superglm.links import LogLink

from pricing_pipeline.build_identity import BuildIdentity, BuildIdentityError
from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.modeling.superglm_identity import SuperGLMIdentityError
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract


class _FakeModel:
    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        self.fit_offset = None

    def fit_reml(self, X, y, sample_weight=None, offset=None):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_sample_weight = sample_weight
        self.fit_offset = offset
        return self

    def training_telemetry(self):
        return {"converged": True, "n_iter": 4}


def _build_identity(**overrides) -> BuildIdentity:
    values = {
        "build_fingerprint_sha256": "1" * 64,
        "model_frame_sha256": "2" * 64,
        "row_order_sha256": "3" * 64,
        "model_source_sha256": "4" * 64,
        "builder_source_sha256": "5" * 64,
        "materialized_split_sha256": "6" * 64,
        "runtime_sha256": "7" * 64,
        "candidate_superglm_sha256": "8" * 64,
        "candidate_python_version": platform.python_version(),
        "candidate_superglm_version": "0.12.0",
        "candidate_superglm_git_sha": "e21bbdca98b6b511e189ae6c30f4af60ec09d95b",
    }
    values.update(overrides)
    return BuildIdentity(**values)


def _stable_export_id() -> str:
    return "build_" + "1" * 64


def _api():
    try:
        module = importlib.import_module("pricing_pipeline.modeling.standard_superglm")
        return module
    except ModuleNotFoundError as exc:
        pytest.fail(f"standard SuperGLM API is not implemented: {exc}")


@pytest.fixture(autouse=True)
def _accept_fixture_build_identity(monkeypatch):
    monkeypatch.setattr(
        _api(),
        "verify_build_identity",
        lambda expected, **kwargs: expected,
    )


def _folds():
    return [
        (np.array([0, 1]), np.array([2])),
        (np.array([1, 2]), np.array([0])),
    ]


def _model_config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="HOME_FREQ",
        model_label="Home frequency",
        target_name="target",
        model_type="superglm_poisson",
        deployment_slot="HOME_FREQ_CURRENT",
        validation_split=ValidationSplitConfig(
            method="custom",
            n_splits=None,
            random_state=None,
            shuffle=False,
            materialize=True,
        ),
    )


def _cv_result(
    *,
    converged=(True, True),
    oof_predictions=None,
    curve_similarity=None,
    estimators=None,
):
    return SimpleNamespace(
        fold_scores=pd.DataFrame(
            {
                "fold": [0, 1],
                "n_train": [2, 2],
                "n_test": [1, 1],
                "fit_time_s": [0.1, 0.2],
                "score_time_s": [0.01, 0.02],
                "converged": list(converged),
                "n_iter": [3, 4],
                "effective_df": [1.5, 1.7],
                "deviance": [0.4, 0.5],
            }
        ),
        mean_scores={"deviance": np.float64(0.45)},
        pooled_scores={"deviance": np.float64(0.42)},
        std_scores={"deviance": np.float64(0.05)},
        fold_indices=_folds(),
        curve_similarity=curve_similarity,
        oof_predictions=(
            np.array([0.25, np.nan, 0.75]) if oof_predictions is None else oof_predictions
        ),
        estimators=estimators,
    )


def _curve_similarity():
    x = np.array([20.0, 30.0, 40.0])
    return {
        "age": {
            "family": "continuous",
            "domain": {"x": x.copy()},
            "support": {"x": x.copy(), "density": np.array([1.0, 3.0, 2.0])},
            "curves": {
                "response": object(),
                "link": {
                    "fold_0": np.array([-0.2, 0.1, 0.4]),
                    "fold_1": np.array([0.3, 0.2, -0.1]),
                },
            },
        }
    }


def _fold_estimators():
    return [SimpleNamespace(_link=LogLink()), SimpleNamespace(_link=LogLink())]


def _real_curve_data():
    row_count = 60
    rng = np.random.default_rng(20260717)
    age = np.linspace(18.0, 78.0, row_count)
    region = np.resize(np.array(["north", "central", "south"]), row_count)
    region_eta = pd.Series(region).map({"north": -0.2, "central": 0.0, "south": 0.3}).to_numpy()
    y = rng.poisson(np.exp(-0.6 + 0.012 * (age - 40.0) + region_eta))
    return pd.DataFrame({"age": age, "region": region}), y


def _real_curve_model():
    from superglm import Categorical, Numeric, SuperGLM

    return SuperGLM(
        features={
            "age": Numeric(),
            "region": Categorical(base="first"),
        },
        selection_penalty=0.0,
    )


def test_cv_evidence_has_one_per_split_metric_representation():
    api = _api()

    assert not hasattr(api, "Fold" + "Metric")
    assert "fold_" + "metrics" not in api.CVEvidence.__dataclass_fields__


def test_precomputed_splitter_replays_exact_folds():
    api = _api()
    splitter = api.PrecomputedSplitter(_folds(), row_count=3)

    replayed = list(splitter.split(pd.DataFrame(index=range(3))))

    assert [pair[0].tolist() for pair in replayed] == [[0, 1], [1, 2]]
    assert [pair[1].tolist() for pair in replayed] == [[2], [0]]
    assert splitter.oof_coverage == pytest.approx(2 / 3)


def test_precomputed_splitter_rejects_duplicate_test_membership():
    api = _api()
    folds = [
        (np.array([0]), np.array([1])),
        (np.array([2]), np.array([1])),
    ]

    with pytest.raises(api.StandardSuperGLMError, match="duplicate test-row"):
        api.PrecomputedSplitter(folds, row_count=3)


def test_precomputed_splitter_rejects_out_of_range_indices():
    api = _api()
    folds = [(np.array([0, 1]), np.array([3]))]

    with pytest.raises(api.StandardSuperGLMError, match="outside row range"):
        api.PrecomputedSplitter(folds, row_count=3)


def test_cv_report_adapter_returns_json_primitives_and_stable_metrics():
    api = _api()

    report, metrics, validation_splits = api.cv_result_to_records(
        _cv_result(),
        oof_coverage=2 / 3,
        scoring=("deviance",),
    )

    json.dumps(report, allow_nan=False)
    assert report["scope"] == "cv"
    assert report["oof_coverage"] == pytest.approx(2 / 3)
    assert report["oof_predictions"][0] == pytest.approx(0.25)
    assert report["oof_predictions"][1] is None
    assert report["oof_predictions"][2] == pytest.approx(0.75)
    assert metrics == {
        "cv_mean_deviance": pytest.approx(0.45),
        "cv_pooled_deviance": pytest.approx(0.42),
        "cv_std_deviance": pytest.approx(0.05),
        "cv_oof_coverage": pytest.approx(2 / 3),
    }
    assert [split.metrics["deviance"] for split in validation_splits] == pytest.approx([0.4, 0.5])


def test_cv_report_adapter_returns_wide_validation_splits_in_requested_order():
    api = _api()
    result = _cv_result()
    result.fold_scores["nll"] = [0.6, 0.7]
    result.fold_scores["gini"] = [0.8, 0.9]
    result.mean_scores.update(nll=0.65, gini=0.85)
    result.pooled_scores.update(nll=0.64)
    result.std_scores.update(nll=0.05, gini=0.05)

    _, _, validation_splits = api.cv_result_to_records(
        result,
        oof_coverage=2 / 3,
        scoring=("deviance", "nll", "gini"),
    )

    assert [
        {
            "validation_split_no": split.validation_split_no,
            "n_train": split.n_train,
            "n_validation": split.n_validation,
            "metrics": split.metrics,
        }
        for split in validation_splits
    ] == [
        {
            "validation_split_no": 1,
            "n_train": 2,
            "n_validation": 1,
            "metrics": {"deviance": 0.4, "nll": 0.6, "gini": 0.8},
        },
        {
            "validation_split_no": 2,
            "n_train": 2,
            "n_validation": 1,
            "metrics": {"deviance": 0.5, "nll": 0.7, "gini": 0.9},
        },
    ]


def test_cv_report_adapter_does_not_fabricate_omitted_custom_metrics():
    api = _api()
    result = _cv_result()
    result.fold_scores["gini"] = [0.8, 0.9]
    result.mean_scores["gini"] = 0.85
    result.std_scores["gini"] = 0.05

    _, _, validation_splits = api.cv_result_to_records(
        result,
        oof_coverage=2 / 3,
        scoring=("deviance", "gini"),
    )

    assert list(validation_splits[0].metrics) == ["deviance", "gini"]
    assert all("nll" not in split.metrics for split in validation_splits)


def test_cv_report_adapter_uses_actual_keys_from_dict_returning_callable():
    api = _api()

    def pricing_scores(*args, **kwargs):
        del args, kwargs
        return {"calibration": 0.0, "ranking": 0.0}

    result = _cv_result()
    result.fold_scores = result.fold_scores.drop(columns="deviance")
    result.fold_scores["calibration"] = [0.2, 0.3]
    result.fold_scores["ranking"] = [0.7, 0.8]
    result.mean_scores = {"calibration": 0.25, "ranking": 0.75}
    result.pooled_scores = {}
    result.std_scores = {"calibration": 0.05, "ranking": 0.05}

    _, _, validation_splits = api.cv_result_to_records(
        result,
        oof_coverage=2 / 3,
        scoring=(pricing_scores,),
    )

    assert list(validation_splits[0].metrics) == ["calibration", "ranking"]
    assert all(pricing_scores.__name__ not in split.metrics for split in validation_splits)


def test_cv_report_adapter_preserves_mixed_scorer_fold_column_order():
    api = _api()

    def pricing_scores(*args, **kwargs):
        del args, kwargs
        return {"calibration": 0.0, "ranking": 0.0}

    result = _cv_result()
    non_metric_columns = [name for name in result.fold_scores if name != "deviance"]
    result.fold_scores = result.fold_scores[non_metric_columns].assign(
        calibration=[0.2, 0.3],
        ranking=[0.7, 0.8],
        deviance=[0.4, 0.5],
    )
    result.mean_scores = {
        "deviance": 0.45,
        "calibration": 0.25,
        "ranking": 0.75,
    }
    result.std_scores = {
        "deviance": 0.05,
        "calibration": 0.05,
        "ranking": 0.05,
    }

    _, _, validation_splits = api.cv_result_to_records(
        result,
        oof_coverage=2 / 3,
        scoring=(pricing_scores, "deviance"),
    )

    assert list(validation_splits[0].metrics) == [
        "calibration",
        "ranking",
        "deviance",
    ]


def test_cv_report_adapter_rejects_aggregate_metric_missing_from_fold_columns():
    api = _api()

    def pricing_scores(*args, **kwargs):
        del args, kwargs
        return {"custom": 0.0}

    result = _cv_result()
    result.mean_scores = {"deviance": 0.45, "custom": 0.25}
    result.std_scores = {"deviance": 0.05, "custom": 0.05}

    with pytest.raises(
        api.StandardSuperGLMError,
        match="fold_scores is missing aggregate metrics: custom",
    ):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=(pricing_scores, "deviance"),
        )


def test_cv_report_adapter_rejects_requested_metric_missing_from_mean_scores():
    api = _api()
    result = _cv_result()
    result.fold_scores["gini"] = [0.7, 0.8]

    with pytest.raises(
        api.StandardSuperGLMError,
        match="mean_scores is missing requested metrics: gini",
    ):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=("deviance", "gini"),
        )


def test_cv_report_adapter_rejects_missing_requested_fold_metric():
    api = _api()
    result = _cv_result()
    result.fold_scores = result.fold_scores.drop(columns="deviance")

    with pytest.raises(
        api.StandardSuperGLMError,
        match="fold 1 is missing requested metrics: deviance",
    ):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=("deviance",),
        )


def test_cv_report_adapter_rejects_non_finite_requested_fold_metric():
    api = _api()
    result = _cv_result()
    result.fold_scores.loc[1, "deviance"] = float("nan")

    with pytest.raises(
        api.StandardSuperGLMError,
        match="fold 2 score 'deviance' must be finite",
    ):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=("deviance",),
        )


@pytest.mark.parametrize(
    ("fold_values", "match"),
    [
        ([0], "one row per materialized split"),
        ([0, 0], "fold numbering must be exactly 0 through 1"),
        ([1, 2], "fold numbering must be exactly 0 through 1"),
    ],
)
def test_cv_report_adapter_rejects_missing_duplicate_or_misnumbered_fold_rows(
    fold_values,
    match,
):
    api = _api()
    result = _cv_result()
    result.fold_scores = result.fold_scores.iloc[: len(fold_values)].copy()
    result.fold_scores["fold"] = fold_values

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=("deviance",),
        )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("n_test", 2, "fold 2 reported n_test=2 but materialized n_validation=1"),
        ("n_train", True, "fold 2 reported n_train must be a positive integer"),
    ],
)
def test_cv_report_adapter_rejects_invalid_or_mismatched_reported_split_counts(
    column,
    value,
    match,
):
    api = _api()
    result = _cv_result()
    result.fold_scores[column] = result.fold_scores[column].astype(object)
    result.fold_scores.loc[1, column] = value

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.cv_result_to_records(
            result,
            oof_coverage=2 / 3,
            scoring=("deviance",),
        )


def test_run_cross_validation_passes_strict_superglm_options():
    api = _api()
    captured = {}

    def fake_cross_validate(model, X, y, **kwargs):
        captured.update({"model": model, "X": X, "y": y, **kwargs})
        return _cv_result(
            curve_similarity=_curve_similarity(),
            estimators=_fold_estimators(),
        )

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )
    evidence = api.run_cross_validation(
        object(),
        inputs,
        split_indices=_folds(),
        fit_mode="fit_reml",
        scoring=("deviance",),
        cross_validate_fn=fake_cross_validate,
    )

    assert captured["error_score"] == "raise"
    assert captured["return_oof"] is True
    assert captured["return_estimators"] is True
    assert captured["fit_mode"] == "fit_reml"
    assert evidence.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert evidence.fold_indices[0][1].tolist() == [2]
    assert evidence.validation_curve_capture.status == "COMPLETE"
    assert len(evidence.validation_curve_capture.points) == 6
    assert "estimators" not in evidence.report
    assert "curve_similarity" not in evidence.report


def test_curve_capture_exception_retries_once_without_estimators_and_keeps_fallback_metrics(
    monkeypatch,
):
    api = _api()
    calls = []

    def failing_curve_builder(**kwargs):
        del kwargs
        raise RuntimeError("curve comparison\n  exploded " + "x" * 700)

    monkeypatch.setattr(
        api.superglm_curve_similarity,
        "build_cv_curve_similarity",
        failing_curve_builder,
    )

    def fake_cross_validate(model, X, y, **kwargs):
        calls.append((model, X, y, dict(kwargs)))
        if kwargs["return_estimators"]:
            api.superglm_curve_similarity.build_cv_curve_similarity()
        return _cv_result()

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    evidence = api.run_cross_validation(
        object(),
        inputs,
        split_indices=_folds(),
        fit_mode="fit_reml",
        scoring=("deviance",),
        cross_validate_fn=fake_cross_validate,
    )

    assert len(calls) == 2
    assert [call[3]["return_estimators"] for call in calls] == [True, False]
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] is calls[1][1]
    assert calls[0][2] is calls[1][2]
    assert calls[0][3]["cv"] is calls[1][3]["cv"]
    assert calls[0][3]["return_oof"] is calls[1][3]["return_oof"] is True
    assert calls[0][3]["error_score"] == calls[1][3]["error_score"] == "raise"
    assert evidence.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert evidence.validation_curve_capture.status == "UNAVAILABLE"
    assert evidence.validation_curve_capture.points == ()
    assert evidence.validation_curve_capture.reason.startswith(
        "validation curve capture failed: RuntimeError: curve comparison exploded"
    )
    assert "\n" not in evidence.validation_curve_capture.reason
    assert len(evidence.validation_curve_capture.reason) == 500


def test_cv_failure_before_curve_capture_is_not_retried():
    api = _api()
    calls = []

    def fake_cross_validate(*args, **kwargs):
        del args
        calls.append(kwargs["return_estimators"])
        raise SuperGLMIdentityError("audited CV identity failed")

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(SuperGLMIdentityError, match="audited CV identity failed"):
        api.run_cross_validation(
            object(),
            inputs,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            cross_validate_fn=fake_cross_validate,
        )

    assert calls == [True]


def test_curve_capture_fallback_failure_surfaces_with_first_failure_as_cause(monkeypatch):
    api = _api()
    calls = []

    def failing_curve_builder(**kwargs):
        del kwargs
        raise ValueError("curve comparison failed")

    monkeypatch.setattr(
        api.superglm_curve_similarity,
        "build_cv_curve_similarity",
        failing_curve_builder,
    )

    def fake_cross_validate(*args, **kwargs):
        del args
        calls.append(kwargs["return_estimators"])
        if kwargs["return_estimators"]:
            api.superglm_curve_similarity.build_cv_curve_similarity()
        raise RuntimeError("fallback scoring failed")

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(RuntimeError, match="fallback scoring failed") as caught:
        api.run_cross_validation(
            object(),
            inputs,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            cross_validate_fn=fake_cross_validate,
        )

    assert calls == [True, False]
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "curve comparison failed"


def test_successful_cv_with_malformed_curves_keeps_metrics_without_retry_or_partial_points():
    api = _api()
    calls = []
    malformed = _curve_similarity()
    malformed["age"]["domain"] = None

    def fake_cross_validate(*args, **kwargs):
        del args
        calls.append(kwargs["return_estimators"])
        return _cv_result(
            curve_similarity=malformed,
            estimators=_fold_estimators(),
        )

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    evidence = api.run_cross_validation(
        object(),
        inputs,
        split_indices=_folds(),
        fit_mode="fit_reml",
        scoring=("deviance",),
        cross_validate_fn=fake_cross_validate,
    )

    assert calls == [True]
    assert evidence.metrics["cv_mean_deviance"] == pytest.approx(0.45)
    assert evidence.validation_curve_capture.status == "UNAVAILABLE"
    assert evidence.validation_curve_capture.points == ()


def test_successful_cv_with_duplicate_continuous_domain_keeps_valid_metrics():
    api = _api()
    calls = []
    duplicate = _curve_similarity()
    duplicate["age"]["domain"]["x"] = [20, 20.0, 40]
    duplicate["age"]["support"]["x"] = [20, 20.0, 40]

    def fake_cross_validate(*args, **kwargs):
        del args
        calls.append(kwargs["return_estimators"])
        return _cv_result(
            curve_similarity=duplicate,
            estimators=_fold_estimators(),
        )

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    evidence = api.run_cross_validation(
        object(),
        inputs,
        split_indices=_folds(),
        fit_mode="fit_reml",
        scoring=("deviance",),
        cross_validate_fn=fake_cross_validate,
    )

    assert calls == [True]
    assert evidence.metrics["cv_mean_deviance"] == pytest.approx(0.45)
    assert evidence.validation_curve_capture.status == "UNAVAILABLE"
    assert "unique" in evidence.validation_curve_capture.reason
    assert evidence.validation_curve_capture.points == ()


def test_real_pinned_superglm_kfold_captures_numeric_and_categorical_split_points():
    from importlib.metadata import distribution, version

    from sklearn.model_selection import KFold

    api = _api()
    assert version("superglm") == "0.12.0"
    direct_url = json.loads(distribution("superglm").read_text("direct_url.json"))
    assert direct_url["vcs_info"]["commit_id"] == ("e21bbdca98b6b511e189ae6c30f4af60ec09d95b")

    X, y = _real_curve_data()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=20260717).split(X, y))

    evidence = api.run_cross_validation(
        _real_curve_model(),
        api.ModelInputs(X=X, y=y),
        split_indices=folds,
        fit_mode="fit",
        scoring=("deviance",),
    )

    capture = evidence.validation_curve_capture
    assert capture.status == "COMPLETE"
    assert capture.reason is None
    assert [split.validation_split_no for split in evidence.validation_splits] == [1, 2, 3, 4, 5]
    assert all(list(split.metrics) == ["deviance"] for split in evidence.validation_splits)
    assert {point.validation_split_no for point in capture.points} == {1, 2, 3, 4, 5}
    assert {point.term_name for point in capture.points} == {"age", "region"}
    assert len([point for point in capture.points if point.term_name == "age"]) == 5 * 200
    assert len([point for point in capture.points if point.term_name == "region"]) == 5 * 3
    for split_no in (1, 2, 3, 4, 5):
        for term_name in ("age", "region"):
            term_points = [
                point
                for point in capture.points
                if point.validation_split_no == split_no and point.term_name == term_name
            ]
            reference_points = [
                point
                for point in term_points
                if (
                    point.x_numeric == point.reference_value
                    if point.point_kind == "NUMERIC"
                    else point.level_text == point.reference_level
                )
            ]
            assert len(reference_points) == 1
            assert reference_points[0].eta_contribution == 0.0
            assert reference_points[0].relativity == 1.0
    assert "estimators" not in evidence.report
    assert "curve_similarity" not in evidence.report


def test_real_pinned_superglm_one_shot_captures_one_metric_row_and_curve_split():
    from superglm import cross_validate as real_cross_validate

    api = _api()
    X, y = _real_curve_data()
    calls = []

    def recording_cross_validate(*args, **kwargs):
        calls.append(kwargs["return_estimators"])
        return real_cross_validate(*args, **kwargs)

    evidence = api.run_cross_validation(
        _real_curve_model(),
        api.ModelInputs(X=X, y=y),
        split_indices=[(np.arange(45), np.arange(45, 60))],
        fit_mode="fit",
        scoring=("deviance",),
        cross_validate_fn=recording_cross_validate,
    )

    assert calls == [True]
    assert len(evidence.validation_splits) == 1
    split = evidence.validation_splits[0]
    assert (split.validation_split_no, split.n_train, split.n_validation) == (
        1,
        45,
        15,
    )
    assert list(split.metrics) == ["deviance"]
    capture = evidence.validation_curve_capture
    assert capture.status == "COMPLETE"
    assert {point.validation_split_no for point in capture.points} == {1}
    assert {point.term_name for point in capture.points} == {"age", "region"}


@pytest.mark.parametrize(
    "returned_folds",
    [
        _folds()[:1],
        [
            (np.array([0, 2]), np.array([1])),
            (np.array([1, 2]), np.array([0])),
        ],
        list(reversed(_folds())),
    ],
)
def test_run_cross_validation_rejects_returned_folds_that_differ_from_request(
    returned_folds,
):
    api = _api()
    result = _cv_result()
    result.fold_indices = returned_folds
    result.fold_scores = result.fold_scores.iloc[: len(returned_folds)].copy()
    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(
        api.StandardSuperGLMError,
        match="returned fold indices do not exactly match requested validation splits",
    ):
        api.run_cross_validation(
            object(),
            inputs,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            cross_validate_fn=lambda *args, **kwargs: result,
        )


def test_run_cross_validation_rejects_non_converged_fold():
    api = _api()
    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(api.StandardSuperGLMError, match="fold 2 did not converge"):
        api.run_cross_validation(
            object(),
            inputs,
            split_indices=_folds(),
            fit_mode="fit",
            scoring=("deviance",),
            cross_validate_fn=lambda *args, **kwargs: _cv_result(converged=(True, False)),
        )


def test_standard_runner_requires_explicit_canonical_row_ids(tmp_path):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )

    with pytest.raises(api.StandardSuperGLMError, match="requires ModelInputs.row_ids"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=api.ModelInputs(
                X=frame[["age"]],
                y=frame["target"].to_numpy(),
            ),
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )


def test_standard_runner_rejects_uncopyable_model_before_training_or_persistence(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )

    class UncopyableModel:
        def __deepcopy__(self, memo):
            del memo
            raise RuntimeError("copy blocked")

    def must_not_run(*args, **kwargs):
        del args, kwargs
        pytest.fail("training and persistence must not run after model copy failure")

    monkeypatch.setattr(api, "run_cross_validation", must_not_run)
    monkeypatch.setattr(api, "fit_full_model", must_not_run)
    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", must_not_run)
    monkeypatch.setattr(api, "export_rating_tables", must_not_run)
    monkeypatch.setattr(api, "save_candidate_bundle", must_not_run)

    with pytest.raises(
        api.StandardSuperGLMError,
        match="superglm_model must be an unfitted, copyable SuperGLM model",
    ) as exc_info:
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            superglm_model=UncopyableModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from=None,
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "splits").exists()


def test_standard_runner_rejects_fitted_model_before_copy_or_persistence(
    tmp_path,
    monkeypatch,
):
    from superglm import Numeric, SuperGLM

    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": np.arange(20),
            "target": np.resize([0.0, 1.0], 20),
            "age": np.linspace(20.0, 60.0, 20),
        }
    )
    superglm_model = SuperGLM(
        features={"age": Numeric()},
        selection_penalty=0.0,
    ).fit(frame[["age"]], frame["target"])
    assert superglm_model._result is not None

    monkeypatch.setattr(
        api,
        "deepcopy",
        lambda model: pytest.fail(f"fitted model was copied: {model!r}"),
    )
    monkeypatch.setattr(
        api,
        "verify_build_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BuildIdentityError("SuperGLM identity is invalid: model is not pristine")
        ),
    )

    with pytest.raises(
        api.StandardSuperGLMError,
        match="SuperGLM identity is invalid: model is not pristine",
    ):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            superglm_model=superglm_model,
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from=None,
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )

    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "splits").exists()


def test_standard_runner_rejects_model_source_drift_during_training(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    verification_calls = 0

    def verify_identity(expected, **kwargs):
        nonlocal verification_calls
        del kwargs
        verification_calls += 1
        if verification_calls == 2:
            raise BuildIdentityError("build contract changed during execution: model_source_sha256")
        return expected

    monkeypatch.setattr(api, "verify_build_identity", verify_identity)
    monkeypatch.setattr(
        api,
        "create_model_frame_manifest_with_split",
        lambda *args, **kwargs: pytest.fail(
            "source drift must fail before audit evidence is persisted"
        ),
    )
    monkeypatch.setattr(
        api,
        "exact_superglm_cross_validate",
        lambda *args, **kwargs: _cv_result(),
    )

    with pytest.raises(api.StandardSuperGLMError, match="model_source_sha256"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from=None,
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )


@pytest.mark.parametrize(
    ("case", "input_builder", "match"),
    [
        (
            "filtered",
            lambda frame: (
                frame.iloc[:2][["age"]].copy(),
                frame.iloc[:2][["policy_id"]].copy(),
            ),
            "row count",
        ),
        (
            "reordered",
            lambda frame: (
                frame.iloc[::-1][["age"]].copy(),
                frame.iloc[::-1][["policy_id"]].copy(),
            ),
            "index/order",
        ),
        (
            "reset-index",
            lambda frame: (
                frame[["age"]].reset_index(drop=True),
                frame[["policy_id"]].reset_index(drop=True),
            ),
            "index/order",
        ),
        (
            "wrong-pk",
            lambda frame: (
                frame[["age"]].copy(),
                frame[["policy_id"]].rename(columns={"policy_id": "account_id"}),
            ),
            "primary-key columns",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_standard_runner_rejects_inputs_not_aligned_to_canonical_frame(
    tmp_path,
    case,
    input_builder,
    match,
):
    del case
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        },
        index=[10, 11, 12],
    )
    X, row_ids = input_builder(frame)

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=api.ModelInputs(
                X=X,
                y=np.zeros(len(X)),
                row_ids=row_ids,
            ),
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )


@pytest.mark.parametrize(
    ("pk_values", "match"),
    [
        ([1, None, 3], "null"),
        ([1, 1, 3], "duplicate"),
    ],
)
def test_standard_runner_rejects_missing_or_duplicate_row_identity_before_cv(
    tmp_path,
    pk_values,
    match,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": pk_values,
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
        )


def _identity_bound_inputs(api, frame, **overrides):
    row_ids = frame[["policy_id"]].copy()
    identity = pd.Index(row_ids["policy_id"].to_numpy(copy=True), name="policy_id")
    X = frame[["age"]].copy()
    X.index = identity
    values = {
        "X": X,
        "y": pd.Series(
            frame["target"].to_numpy(copy=True),
            index=identity,
            name="target",
        ),
        "row_ids": row_ids,
    }
    values.update(overrides)
    return api.ModelInputs(**values)


def _minimal_standard_build(api, tmp_path):
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.py").write_text("MODEL = 'HOME_FREQ'\n", encoding="utf-8")
    return {
        "frame": frame,
        "inputs": _identity_bound_inputs(api, frame),
        "superglm_model": _FakeModel(),
        "split_indices": _folds(),
        "expected_build_identity": _build_identity(),
        "fit_mode": "fit_reml",
        "scoring": ("deviance",),
        "output_dir": tmp_path / "run",
        "model_id": 17,
        "model_config": _model_config(),
        "model_version": "v1",
        "export_id": _stable_export_id(),
        "effective_from": None,
        "manifest_spec": ModelFrameManifestSpec(
            dataset_name="home_freq_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="target",
            feature_columns=("age",),
        ),
        "split_artifact_root": tmp_path / "splits",
        "model_source_root": source_root,
        "created_by": "pytest",
    }


def _complete_role_build(api, tmp_path):
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0, 1, 0],
            "age": [20.0, 30.0, 40.0],
            "age_squared": [400.0, 900.0, 1600.0],
            "fit_weight": [1.0, 1.5, 2.0],
            "fitted_offset": [0.0, np.log(2.0), np.log(3.0)],
            "raw_term": [12.0, 24.0, 36.0],
            "export_weight": [2.0, 3.0, 4.0],
        }
    )
    row_ids = frame[["policy_id"]].copy()
    identity = pd.Index(row_ids["policy_id"].to_numpy(copy=True), name="policy_id")

    def series(column, *, dtype=None):
        values = frame[column].to_numpy(copy=True)
        if dtype is not None:
            values = values.astype(dtype)
        return pd.Series(values, index=identity, name=column)

    X = frame[["age", "age_squared"]].copy()
    X.index = identity
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.py").write_text("MODEL = 'HOME_FREQ'\n", encoding="utf-8")
    contract = OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="raw_term",
        published_factor_name="RawTerm",
        source_name="raw_term",
        label="log(raw_term / 12)",
    )
    return {
        "frame": frame,
        "inputs": api.ModelInputs(
            X=X,
            y=series("target", dtype=float),
            sample_weight=series("fit_weight"),
            sample_weight_name="fit_weight",
            offset=series("fitted_offset"),
            offset_source=series("raw_term"),
            offset_source_name="raw_term",
            export_weight=series("export_weight"),
            export_weight_name="export_weight",
            row_ids=row_ids,
        ),
        "superglm_model": _FakeModel(),
        "split_indices": _folds(),
        "expected_build_identity": _build_identity(),
        "fit_mode": "fit_reml",
        "scoring": ("deviance",),
        "output_dir": tmp_path / "run",
        "model_id": 17,
        "model_config": _model_config(),
        "model_version": "v1",
        "export_id": _stable_export_id(),
        "effective_from": None,
        "manifest_spec": ModelFrameManifestSpec(
            dataset_name="home_freq_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="target",
            weight_column="fit_weight",
            feature_columns=("age", "age_squared"),
            offset_column="fitted_offset",
            offset_source_column="raw_term",
            offset_label="log(raw_term / 12)",
            export_weight_column="export_weight",
        ),
        "split_artifact_root": tmp_path / "splits",
        "model_source_root": source_root,
        "created_by": "pytest",
        "offset_contract": contract,
    }


def test_publishable_standard_runner_does_not_expose_cv_implementation_choice():
    parameters = inspect.signature(_api().run_standard_superglm_build).parameters

    assert "cross_validate_fn" not in parameters


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("X_columns", "ModelInputs.X.*feature columns"),
        ("X_values", "ModelInputs.X.*values"),
        ("y_name", "ModelInputs.y.*target"),
        ("y_values", "ModelInputs.y.*target"),
        ("sample_weight_name", "sample_weight.*name"),
        ("sample_weight_values", "sample_weight.*values"),
        ("offset_values", "offset.*values"),
        ("offset_source_name", "offset_source.*name"),
        ("offset_source_values", "offset_source.*values"),
        ("export_weight_name", "export_weight.*name"),
        ("export_weight_values", "export_weight.*values"),
        ("row_ids", "row_ids.*values"),
    ],
)
def test_publishable_standard_runner_rejects_each_final_frame_role_drift_before_cv(
    tmp_path,
    monkeypatch,
    mutation,
    match,
):
    api = _api()
    build = _complete_role_build(api, tmp_path)
    inputs = build["inputs"]
    values = {name: getattr(inputs, name) for name in api.ModelInputs.__dataclass_fields__}
    if mutation.endswith("_values"):
        role = mutation.removesuffix("_values")
        changed = values[role].copy()
        changed.iloc[0] = changed.iloc[0] + 1
        values[role] = changed
    elif mutation == "y_name":
        values["y"] = values["y"].rename("wrong_target")
    elif mutation.endswith("_name"):
        values[mutation] = "wrong_column"
    elif mutation == "X_columns":
        values["X"] = values["X"].rename(columns={"age_squared": "hidden_transform"})
    elif mutation == "row_ids":
        changed = values["row_ids"].copy()
        changed.iloc[0, 0] = 999
        values["row_ids"] = changed
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(mutation)
    build["inputs"] = api.ModelInputs(**values)
    monkeypatch.setattr(
        api,
        "run_cross_validation",
        lambda *args, **kwargs: pytest.fail("CV ran after final-frame role drift"),
    )

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(object(), **build)


def test_publishable_runner_uses_deep_snapshots_and_copied_materialized_splits(
    tmp_path,
    monkeypatch,
):
    api = _api()
    build = _complete_role_build(api, tmp_path)
    caller_frame = build["frame"]
    caller_inputs = build["inputs"]
    caller_splits = build["split_indices"]
    expected_frame = caller_frame.copy(deep=True)
    expected_X = caller_inputs.X.copy(deep=True)
    expected_y = caller_inputs.y.copy(deep=True)
    expected_folds = [(train.copy(), validation.copy()) for train, validation in caller_splits]

    def mutating_exact_cv(model, X, y, **kwargs):
        del model, X, y, kwargs
        caller_frame.loc[:, "age_squared"] = -1.0
        caller_inputs.X.loc[:, "age_squared"] = -2.0
        caller_inputs.y.iloc[:] = 99.0
        caller_inputs.sample_weight.iloc[:] = 88.0
        caller_inputs.offset.iloc[:] = 77.0
        caller_inputs.offset_source.iloc[:] = 66.0
        caller_inputs.export_weight.iloc[:] = 55.0
        caller_inputs.row_ids.iloc[0, 0] = 999
        caller_splits[0][0][0] = 2
        caller_splits[0][1][0] = 1
        return _cv_result()

    def capture_full_fit(model, inputs, *, fit_mode):
        del fit_mode
        pd.testing.assert_frame_equal(inputs.X, expected_X)
        pd.testing.assert_series_equal(inputs.y, expected_y)
        return model, {"converged": True}

    def capture_manifest(engine, **kwargs):
        del engine
        pd.testing.assert_frame_equal(kwargs["frame"], expected_frame)
        for actual, expected in zip(kwargs["split_indices"], expected_folds, strict=True):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        raise RuntimeError("trusted snapshots captured")

    monkeypatch.setattr(api, "exact_superglm_cross_validate", mutating_exact_cv)
    monkeypatch.setattr(api, "fit_full_model", capture_full_fit)
    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", capture_manifest)

    with pytest.raises(RuntimeError, match="trusted snapshots captured"):
        api.run_standard_superglm_build(object(), **build)


@pytest.mark.parametrize(
    (
        "handling",
        "manifest_offset",
        "manifest_offset_source",
        "manifest_label",
        "offset_source_mode",
        "match",
    ),
    [
        (
            "NONE",
            "TermOffset",
            "Term",
            "log(Term / 12)",
            None,
            "handling NONE",
        ),
        (
            "EXPORTED_FACTOR",
            "TermOffset",
            None,
            "log(Term / 12)",
            "frame",
            "EXPORTED_FACTOR.*offset_source_column",
        ),
        (
            "EXPORTED_FACTOR",
            "TermOffset",
            "OtherTerm",
            "log(Term / 12)",
            "frame",
            "offset_source_column.*source_name",
        ),
        (
            "EXPORTED_FACTOR",
            "TermOffset",
            "Term",
            "wrong label",
            "frame",
            "offset_label.*label",
        ),
        (
            "ALREADY_APPLIED_SQL_EXPOSURE",
            "TermOffset",
            "Term",
            "log(Term / 12)",
            None,
            "ALREADY_APPLIED_SQL_EXPOSURE.*offset_source_column",
        ),
        (
            "EXPORTED_FACTOR",
            "OtherOffset",
            "Term",
            "log(Term / 12)",
            "frame",
            "ModelInputs.offset values",
        ),
        (
            "EXPORTED_FACTOR",
            "TermOffset",
            "Term",
            "log(Term / 12)",
            "wrong",
            "ModelInputs.offset_source values",
        ),
        (
            "NONE",
            None,
            None,
            None,
            "frame",
            "ModelInputs.offset_source.*manifest has no offset_source column",
        ),
        (
            "ALREADY_APPLIED_SQL_EXPOSURE",
            "TermOffset",
            None,
            "log(Term / 12)",
            "frame",
            "ModelInputs.offset_source.*manifest has no offset_source column",
        ),
    ],
)
def test_standard_runner_rejects_offset_contract_or_input_role_mismatch_before_cv(
    tmp_path,
    handling,
    manifest_offset,
    manifest_offset_source,
    manifest_label,
    offset_source_mode,
    match,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
            "Term": [12.0, 24.0, 36.0],
            "OtherTerm": [12.0, 24.0, 36.0],
            "TermOffset": [0.0, np.log(2.0), np.log(3.0)],
            "OtherOffset": [1.0, 1.0, 1.0],
        }
    )
    identity = pd.Index(frame["policy_id"], name="policy_id")
    offset = pd.Series(frame["TermOffset"].to_numpy(), index=identity, name="TermOffset")
    term = pd.Series(frame["Term"].to_numpy(), index=identity, name="Term")
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "model.py").write_text("MODEL = 'HOME_FREQ'\n")
    input_overrides = {}
    if handling != "NONE":
        input_overrides["offset"] = offset
    if offset_source_mode:
        source = term
        if offset_source_mode == "wrong":
            source = pd.Series([1.0, 2.0, 3.0], index=identity, name="Term")
        input_overrides.update(offset_source=source, offset_source_name="Term")
    inputs = _identity_bound_inputs(api, frame, **input_overrides)
    contract = OffsetExportContract(handling="NONE")
    if handling == "EXPORTED_FACTOR":
        contract = OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Term",
            published_factor_name="Term",
            source_name="Term",
            label="log(Term / 12)",
        )
    elif handling == "ALREADY_APPLIED_SQL_EXPOSURE":
        contract = OffsetExportContract(
            handling="ALREADY_APPLIED_SQL_EXPOSURE",
            source_name="Term",
            label="log(Term / 12)",
        )

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=inputs,
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from=None,
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
                offset_column=manifest_offset,
                offset_source_column=manifest_offset_source,
                offset_label=manifest_label,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
            offset_contract=contract,
        )


def test_manifest_offset_contract_accepts_already_applied_sql_exposure():
    api = _api()
    manifest_spec = ModelFrameManifestSpec(
        dataset_name="home_freq_frame",
        source_system="pytest",
        data_as_of_date="2026-06-30",
        pk_columns=("policy_id",),
        target_column="target",
        offset_column="LogExposure",
        offset_label="log(Exposure)",
    )
    contract = OffsetExportContract(
        handling="ALREADY_APPLIED_SQL_EXPOSURE",
        source_name="Exposure",
        label="log(Exposure)",
    )

    api._validate_manifest_offset_contract(manifest_spec, contract)


def test_manifest_offset_contract_accepts_identity_offset_source():
    api = _api()
    manifest_spec = ModelFrameManifestSpec(
        dataset_name="home_freq_frame",
        source_system="pytest",
        data_as_of_date="2026-06-30",
        pk_columns=("policy_id",),
        target_column="target",
        offset_column="Term",
        offset_source_column="Term",
        offset_label="identity(Term)",
    )
    contract = OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="Term",
        published_factor_name="Term",
        source_name="Term",
        label="identity(Term)",
    )

    api._validate_manifest_offset_contract(manifest_spec, contract)


def test_canonical_validation_rejects_reversed_then_reset_feature_frame():
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    inputs = _identity_bound_inputs(
        api,
        frame,
        X=frame.iloc[::-1][["age"]].reset_index(drop=True),
    )

    with pytest.raises(api.StandardSuperGLMError, match="ModelInputs.X.*identity index"):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


def test_canonical_validation_rejects_reordered_target_series():
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    inputs = _identity_bound_inputs(api, frame)
    reordered_y = inputs.y.iloc[::-1]
    inputs = _identity_bound_inputs(api, frame, y=reordered_y)

    with pytest.raises(api.StandardSuperGLMError, match="ModelInputs.y.*identity index"):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


@pytest.mark.parametrize("field_name", ["sample_weight", "offset", "export_weight"])
def test_canonical_validation_rejects_reordered_optional_row_inputs(field_name):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    identity = pd.Index([2, 1, 0], name="policy_id")
    reordered = pd.Series([1.0, 1.0, 1.0], index=identity, name=field_name)
    inputs = _identity_bound_inputs(api, frame, **{field_name: reordered})

    with pytest.raises(
        api.StandardSuperGLMError,
        match=rf"ModelInputs.{field_name}.*identity index",
    ):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


def test_standard_runner_binds_export_id_to_build_fingerprint_before_copy(
    tmp_path,
    monkeypatch,
):
    api = _api()
    build = _minimal_standard_build(api, tmp_path)
    build["export_id"] = "incidental-attempt-id"
    monkeypatch.setattr(
        api,
        "deepcopy",
        lambda model: pytest.fail(f"model copied before export identity check: {model!r}"),
    )

    with pytest.raises(api.StandardSuperGLMError, match="stable export identity"):
        api.run_standard_superglm_build(object(), **build)


def test_standard_runner_rejects_manifest_frame_mismatch_before_artifacts(
    tmp_path,
    monkeypatch,
):
    api = _api()
    build = _minimal_standard_build(api, tmp_path)
    monkeypatch.setattr(
        api,
        "exact_superglm_cross_validate",
        lambda *args, **kwargs: _cv_result(),
    )
    monkeypatch.setattr(
        api,
        "create_model_frame_manifest_with_split",
        lambda *args, **kwargs: SimpleNamespace(
            manifest_id="manifest-mismatch",
            split_set_id="split-mismatch",
            model_frame_sha256="f" * 64,
        ),
    )
    monkeypatch.setattr(
        api,
        "export_rating_tables",
        lambda *args, **kwargs: pytest.fail("artifacts were created after manifest mismatch"),
    )

    with pytest.raises(api.StandardSuperGLMError, match="manifest model-frame hash"):
        api.run_standard_superglm_build(object(), **build)

    assert not (tmp_path / "run").exists()


def test_standard_runner_final_drift_removes_attempt_artifacts(
    tmp_path,
    monkeypatch,
):
    api = _api()
    build = _minimal_standard_build(api, tmp_path)
    split_evidence = tmp_path / "splits" / "split-final-drift.npz"
    verification_calls = 0

    def verify_identity(expected, **kwargs):
        nonlocal verification_calls
        del kwargs
        verification_calls += 1
        if verification_calls == 3:
            raise BuildIdentityError(
                "build contract changed during execution: builder_source_sha256"
            )
        return expected

    def persist_manifest(*args, **kwargs):
        del args, kwargs
        split_evidence.parent.mkdir(parents=True)
        split_evidence.write_bytes(b"durable split evidence")
        return SimpleNamespace(
            manifest_id="manifest-final-drift",
            split_set_id="split-final-drift",
            model_frame_sha256="2" * 64,
        )

    def export_workbook(model, X, y, weight, output_path, **kwargs):
        del model, X, y, weight, kwargs
        Path(output_path).write_bytes(b"workbook")

    def write_receipt(receipt, path):
        del receipt
        Path(path).write_bytes(b"receipt")
        return "a" * 64

    monkeypatch.setattr(api, "verify_build_identity", verify_identity)
    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", persist_manifest)
    monkeypatch.setattr(api, "export_rating_tables", export_workbook)
    monkeypatch.setattr(api, "build_superglm_publication_receipt", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "write_publication_receipt", write_receipt)
    monkeypatch.setattr(
        api,
        "exact_superglm_cross_validate",
        lambda *args, **kwargs: _cv_result(),
    )

    with pytest.raises(api.StandardSuperGLMError, match="builder_source_sha256"):
        api.run_standard_superglm_build(object(), **build)

    assert verification_calls == 3
    assert not (tmp_path / "run" / "manifest-final-drift").exists()
    assert split_evidence.read_bytes() == b"durable split evidence"


@pytest.mark.parametrize(
    "manifest_id",
    ("../escape", "nested/manifest", "manifest id", ".", ""),
)
def test_manifest_attempt_directory_rejects_unsafe_path_components(
    tmp_path,
    manifest_id,
):
    api = _api()

    with pytest.raises(api.StandardSuperGLMError, match="safe path component"):
        api._manifest_attempt_directory(tmp_path / "run", manifest_id)


def test_standard_runner_removes_partial_attempt_but_keeps_manifest_evidence(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    split_evidence = tmp_path / "splits" / "manifest-failure-split.npz"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.py").write_text("MODEL = 'HOME_FREQ'\n", encoding="utf-8")

    def fake_manifest(engine, **kwargs):
        del engine, kwargs
        split_evidence.parent.mkdir(parents=True)
        split_evidence.write_bytes(b"durable split evidence")
        return SimpleNamespace(
            manifest_id="manifest-failure",
            split_set_id="manifest-failure-split",
            split_artifact_uri=str(split_evidence),
            model_frame_sha256="2" * 64,
        )

    def failing_export(model, X, y, exposure, output_path, **kwargs):
        del model, X, y, exposure, kwargs
        Path(output_path).write_bytes(b"partial workbook")
        raise RuntimeError("artifact export failed")

    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", fake_manifest)
    monkeypatch.setattr(api, "export_rating_tables", failing_export)
    monkeypatch.setattr(
        api,
        "exact_superglm_cross_validate",
        lambda *args, **kwargs: _cv_result(),
    )

    with pytest.raises(RuntimeError, match="artifact export failed"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            superglm_model=_FakeModel(),
            split_indices=_folds(),
            expected_build_identity=_build_identity(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_config=_model_config(),
            model_version="v1",
            export_id=_stable_export_id(),
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
                feature_columns=("age",),
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=source_root,
            created_by="pytest",
        )

    assert not (tmp_path / "run").exists()
    assert split_evidence.read_bytes() == b"durable split evidence"


def test_standard_runner_cleanup_failure_preserves_original_base_exception(
    tmp_path,
    monkeypatch,
):
    api = _api()
    build = _minimal_standard_build(api, tmp_path)
    monkeypatch.setattr(
        api,
        "exact_superglm_cross_validate",
        lambda *args, **kwargs: _cv_result(),
    )
    monkeypatch.setattr(
        api,
        "create_model_frame_manifest_with_split",
        lambda *args, **kwargs: SimpleNamespace(
            manifest_id="manifest-interrupted",
            split_set_id="split-interrupted",
            model_frame_sha256="2" * 64,
        ),
    )
    monkeypatch.setattr(
        api,
        "export_rating_tables",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("original interruption")),
    )
    monkeypatch.setattr(
        api.shutil,
        "rmtree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    with pytest.raises(KeyboardInterrupt, match="original interruption") as exc_info:
        api.run_standard_superglm_build(object(), **build)

    assert any("cleanup failed" in note for note in exc_info.value.__notes__)


def test_standard_runner_uses_model_config_and_returns_approved_build(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
            "Term": [12.0, 36.0, 24.0],
            "TermOffset": [0.0, np.log(3.0), np.log(2.0)],
            "RatingWeight": [2.0, 4.0, 3.0],
        }
    )
    source_root = tmp_path / "pricing_models" / "home_freq"
    (source_root / "sql").mkdir(parents=True)
    (source_root / "modeling.py").write_text("FIT_MODE = 'fit_reml'\n", encoding="utf-8")
    (source_root / "model.toml").write_text('model_name = "HOME_FREQ"\n', encoding="utf-8")
    (source_root / "sql" / "source.sql").write_text("SELECT 1;\n", encoding="utf-8")
    captured = {}
    superglm_model = _FakeModel()
    cv_models = []
    final_models = []
    curve_payload = _curve_similarity()

    def fake_cross_validate(model, *args, **kwargs):
        del args
        cv_models.append(model)
        assert kwargs["return_estimators"] is True
        return _cv_result(
            curve_similarity=curve_payload,
            estimators=_fold_estimators(),
        )

    real_fit_full_model = api.fit_full_model

    def capture_fit_full_model(model, inputs, *, fit_mode):
        final_models.append(model)
        return real_fit_full_model(model, inputs, fit_mode=fit_mode)

    def fake_export(model, X, y, exposure, output_path, **kwargs):
        captured["export_weight"] = exposure
        captured["export_options"] = kwargs
        Path(output_path).write_bytes(b"canonical workbook")
        return Path(output_path)

    manifest_ids = iter(("manifest-1", "manifest-2", "manifest-3"))
    manifest_digests = iter(("a" * 64, "a" * 64, "a" * 64))

    def fake_manifest(engine, **kwargs):
        captured["manifest"] = kwargs
        manifest_id = next(manifest_ids)
        return SimpleNamespace(
            manifest_id=manifest_id,
            split_set_id=f"{manifest_id}-split",
            split_artifact_uri=str(tmp_path / "splits" / f"{manifest_id}-split.npz"),
            model_frame_sha256=next(manifest_digests),
        )

    def fake_receipt_writer(receipt, path):
        Path(path).write_bytes(b"canonical receipt")
        return "c" * 64

    monkeypatch.setattr(api, "export_rating_tables", fake_export)
    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", fake_manifest)
    monkeypatch.setattr(api, "build_superglm_publication_receipt", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "write_publication_receipt", fake_receipt_writer)
    monkeypatch.setattr(api, "fit_full_model", capture_fit_full_model)
    monkeypatch.setattr(api, "exact_superglm_cross_validate", fake_cross_validate)

    base_inputs = _identity_bound_inputs(api, frame)
    term = pd.Series(
        frame["Term"].to_numpy(copy=True),
        index=base_inputs.X.index,
        name="Term",
    )
    rating_weight = pd.Series(
        frame["RatingWeight"].to_numpy(copy=True),
        index=base_inputs.X.index,
        name="RatingWeight",
    )
    inputs = _identity_bound_inputs(
        api,
        frame,
        offset=pd.Series(
            frame["TermOffset"].to_numpy(copy=True),
            index=base_inputs.X.index,
            name="TermOffset",
        ),
        offset_source=term,
        offset_source_name="Term",
        export_weight=rating_weight,
        export_weight_name="RatingWeight",
    )
    validation_split = ValidationSplitConfig(
        method="custom",
        n_splits=None,
        random_state=None,
        shuffle=False,
        materialize=True,
    )
    model_config = ModelBuildConfig(
        model_name="HOME_FREQ",
        model_label="Home frequency",
        target_name="target",
        model_type="superglm_poisson",
        deployment_slot="HOME_FREQ_CURRENT",
        validation_split=validation_split,
    )
    build_kwargs = {
        "frame": frame,
        "inputs": inputs,
        "superglm_model": superglm_model,
        "split_indices": _folds(),
        "expected_build_identity": _build_identity(model_frame_sha256="a" * 64),
        "fit_mode": "fit_reml",
        "scoring": ("deviance",),
        "output_dir": tmp_path / "run",
        "model_id": 17,
        "model_config": model_config,
        "model_version": "v1",
        "export_id": _stable_export_id(),
        "effective_from": "2026-07-12",
        "manifest_spec": ModelFrameManifestSpec(
            dataset_name="home_freq_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="target",
            feature_columns=("age",),
            offset_column="TermOffset",
            offset_source_column="Term",
            offset_label="log(Term / 12)",
            export_weight_column="RatingWeight",
        ),
        "split_artifact_root": tmp_path / "splits",
        "model_source_root": source_root,
        "created_by": "pytest",
        "offset_contract": OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Term",
            published_factor_name="Term",
            source_name="Term",
            label="log(Term / 12)",
        ),
    }
    result = api.run_standard_superglm_build(object(), **build_kwargs)

    from pricing_pipeline.workbench.artifacts import load_candidate_bundle

    assert isinstance(result, ApprovedModelBuild)
    assert result.validation_curve_status == "COMPLETE"
    assert result.validation_curve_reason is None
    assert len(result.validation_curve_points) == 6
    bundle = load_candidate_bundle(
        result.candidate_artifact_path,
        expected_sha256=result.candidate_artifact_sha256,
        expected_size_bytes=result.candidate_artifact_size_bytes,
        expected_format=result.candidate_artifact_format,
        expected_python_version=result.candidate_python_version,
        expected_superglm_version=result.candidate_superglm_version,
        expected_superglm_git_sha=result.candidate_superglm_git_sha,
        allowed_root=tmp_path / "run",
    )
    assert bundle.model_name == "HOME_FREQ"
    assert bundle.model_version == "v1"
    assert bundle.export_id == _stable_export_id()
    assert bundle.model_frame_sha256 == "a" * 64
    assert "estimators" not in bundle.cv_report
    assert "curve_similarity" not in bundle.cv_report
    first_paths = {
        "workbook": Path(result.rating_workbook_path),
        "receipt": Path(result.publication_receipt_path),
        "candidate": Path(result.candidate_artifact_path),
    }
    first_bytes = {name: path.read_bytes() for name, path in first_paths.items()}
    second_result = api.run_standard_superglm_build(object(), **build_kwargs)
    curve_payload = _curve_similarity()
    curve_payload["age"]["domain"]["x"] = [20, 20.0, 40]
    curve_payload["age"]["support"]["x"] = [20, 20.0, 40]
    malformed_result = api.run_standard_superglm_build(object(), **build_kwargs)

    assert [test.tolist() for _, test in captured["manifest"]["split_indices"]] == [
        [2],
        [0],
    ]
    assert captured["manifest"]["validation_split"] == validation_split
    assert len(cv_models) == 3
    assert len(final_models) == 3
    assert all(model is not superglm_model for model in cv_models)
    assert all(model is not superglm_model for model in final_models)
    assert all(
        cv_model is not final_model for cv_model, final_model in zip(cv_models, final_models)
    )
    assert cv_models[0] is not cv_models[1]
    assert final_models[0] is not final_models[1]
    assert bundle.fitted_model is not superglm_model
    assert final_models[0].fit_X.equals(inputs.X)
    assert superglm_model.fit_X is None
    assert superglm_model.fit_y is None
    np.testing.assert_allclose(captured["export_weight"], rating_weight)
    np.testing.assert_allclose(captured["export_options"]["offset"], np.log(term / 12.0))
    np.testing.assert_allclose(captured["export_options"]["offset_source"], term)
    assert captured["export_options"]["offset_name"] == "Term"
    assert captured["export_options"]["offset_kind"] == "auto"
    np.testing.assert_allclose(bundle.offset_source, term)
    np.testing.assert_allclose(bundle.export_weight, rating_weight)
    assert bundle.offset_source_name == "Term"
    assert bundle.export_weight_name == "RatingWeight"
    assert result.manifest_id == "manifest-1"
    assert result.model_frame_sha256 == "a" * 64
    assert result.split_set_id == "manifest-1-split"
    assert second_result.manifest_id == "manifest-2"
    assert second_result.model_frame_sha256 == "a" * 64
    assert isinstance(malformed_result, ApprovedModelBuild)
    assert malformed_result.manifest_id == "manifest-3"
    assert malformed_result.model_frame_sha256 == "a" * 64
    assert malformed_result.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert malformed_result.validation_curve_status == "UNAVAILABLE"
    assert "unique" in malformed_result.validation_curve_reason
    assert malformed_result.validation_curve_points == ()
    second_paths = {
        "workbook": Path(second_result.rating_workbook_path),
        "receipt": Path(second_result.publication_receipt_path),
        "candidate": Path(second_result.candidate_artifact_path),
    }
    assert {path.parent for path in first_paths.values()} == {
        (tmp_path / "run" / "manifest-1").resolve()
    }
    assert {path.parent for path in second_paths.values()} == {
        (tmp_path / "run" / "manifest-2").resolve()
    }
    assert set(first_paths.values()).isdisjoint(second_paths.values())
    assert {name: path.read_bytes() for name, path in first_paths.items()} == first_bytes
    assert Path(result.candidate_artifact_path).exists()
    assert result.candidate_artifact_sha256
    assert result.model_source_sha256
    assert result.rating_workbook_sha256 == api.hash_file_sha256(first_paths["workbook"])
    assert result.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert [split.model_dump() for split in result.validation_splits] == [
        {
            "validation_split_no": 1,
            "n_train": 2,
            "n_validation": 1,
            "metrics": {"deviance": pytest.approx(0.4)},
        },
        {
            "validation_split_no": 2,
            "n_train": 2,
            "n_validation": 1,
            "metrics": {"deviance": pytest.approx(0.5)},
        },
    ]
    assert bundle.cv_report["model_name"] == "HOME_FREQ"
    assert bundle.cv_report["fit_mode"] == "fit_reml"
    assert bundle.cv_report["scoring"] == ["deviance"]
