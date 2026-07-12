from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _api():
    try:
        module = importlib.import_module("pricing_pipeline.modeling.standard_superglm")
        return module
    except ModuleNotFoundError as exc:
        pytest.fail(f"standard SuperGLM API is not implemented: {exc}")


def _folds():
    return [
        (np.array([0, 1]), np.array([2])),
        (np.array([1, 2]), np.array([0])),
    ]


def _cv_result(*, converged=(True, True), oof_predictions=None):
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
        curve_similarity=None,
        oof_predictions=(
            np.array([0.25, np.nan, 0.75])
            if oof_predictions is None
            else oof_predictions
        ),
        estimators=None,
    )


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

    report, metrics, fold_metrics = api.cv_result_to_records(
        _cv_result(),
        oof_coverage=2 / 3,
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
    assert [(item.fold_no, item.metric_name, item.metric_value) for item in fold_metrics] == [
        (1, "deviance", pytest.approx(0.4)),
        (2, "deviance", pytest.approx(0.5)),
    ]


def test_run_cross_validation_passes_strict_superglm_options():
    api = _api()
    captured = {}

    def fake_cross_validate(model, X, y, **kwargs):
        captured.update({"model": model, "X": X, "y": y, **kwargs})
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

    assert captured["error_score"] == "raise"
    assert captured["return_oof"] is True
    assert captured["return_estimators"] is False
    assert captured["fit_mode"] == "fit_reml"
    assert evidence.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert evidence.fold_indices[0][1].tolist() == [2]


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
            cross_validate_fn=lambda *args, **kwargs: _cv_result(
                converged=(True, False)
            ),
        )
