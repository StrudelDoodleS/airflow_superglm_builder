from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM, Tweedie
from superglm.features.spline import _SplineBase

from pricing_pipeline.modeling.scratch_benchmark import (
    ScratchBenchmarkError,
    _required_scratch_estimators,
    fit_boosted_blend,
    superglm_edf_table,
    unconstrained_superglm_features,
)


class _WeightedMeanEstimator:
    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        self.fitted_rate: float | None = None

    def fit(self, X, y, *, sample_weight=None, **_kwargs):
        assert len(X) == len(y)
        self.fitted_rate = max(
            float(np.average(y, weights=sample_weight)) * self.multiplier,
            1e-6,
        )
        return self

    def predict(self, X):
        assert self.fitted_rate is not None
        return np.full(len(X), self.fitted_rate)


def _fake_factories():
    return {
        "catboost": lambda _seed: _WeightedMeanEstimator(0.75),
        "lightgbm": lambda _seed: _WeightedMeanEstimator(1.0),
        "xgboost": lambda _seed: _WeightedMeanEstimator(1.25),
    }


def test_unconstrained_feature_map_has_no_groupings_or_shape_constraints():
    frame = pd.DataFrame(
        {
            "continuous": np.linspace(0.0, 1.0, 30),
            "category": np.resize(["A", "B", "C"], 30),
            "ordered": np.resize(["low", "mid", "high"], 30),
            "linear": np.linspace(-1.0, 1.0, 30),
        }
    )

    features = unconstrained_superglm_features(
        frame,
        categorical_columns=("category",),
        ordered_columns={"ordered": ("low", "mid", "high")},
        linear_columns=("linear",),
        k=6,
        knot_strategy="quantile_tempered",
        knot_alpha=0.35,
    )

    assert list(features) == list(frame)
    assert isinstance(features["continuous"], _SplineBase)
    assert features["continuous"].constraint_kind is None
    assert features["continuous"]._explicit_knots is None
    assert features["continuous"]._lambda_policy is None
    assert features["continuous"].knot_strategy == "quantile_tempered"
    assert features["continuous"].knot_alpha == pytest.approx(0.35)
    assert isinstance(features["category"], Categorical)
    assert features["category"]._grouping is None
    assert isinstance(features["ordered"], OrderedCategorical)
    assert features["ordered"]._grouping is None
    assert isinstance(features["linear"], Numeric)


def test_edf_table_keeps_ordered_specials_separate_from_smooth_curve():
    rng = np.random.default_rng(1729)
    row_count = 160
    ordered = np.resize(["low", "mid", "high", "MISSING"], row_count)
    x = np.linspace(0.0, 1.0, row_count)
    frame = pd.DataFrame({"ordered": ordered, "x": x})
    target = rng.poisson(np.exp(-1.0 + 0.5 * x + 0.2 * (ordered == "high")))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "ordered": OrderedCategorical(
                order=["low", "mid", "high", "MISSING"],
                specials=["MISSING"],
                basis=Spline(kind="ps", k=5),
            ),
            "x": Spline(kind="ps", k=5),
        },
    ).fit_reml(
        frame,
        target,
        max_reml_iter=4,
        runtime_validation="skip",
    )

    edf = superglm_edf_table(model)

    assert {"ordered_smooth", "special", "smooth"} <= set(edf["component_kind"])
    ordered_rows = edf.loc[edf["feature_name"].eq("ordered")]
    assert set(ordered_rows["component_kind"]) == {"ordered_smooth", "special"}
    assert np.isfinite(edf["effective_df"]).all()
    assert (edf["available_dimension"] > 0).all()


def test_boosted_blend_learns_weights_only_from_complete_oof_predictions():
    rng = np.random.default_rng(1729)
    row_count = 120
    exposure = rng.uniform(0.2, 1.0, row_count)
    category = np.resize(["A", "B", "C"], row_count)
    x = rng.normal(size=row_count)
    rate = np.exp(-1.2 + 0.3 * x + 0.25 * (category == "C"))
    target = rng.poisson(exposure * rate)
    frame = pd.DataFrame({"x": x, "category": category})

    blend = fit_boosted_blend(
        frame,
        target,
        categorical_columns=("category",),
        exposure=exposure,
        n_splits=4,
        random_state=12,
        estimator_factories=_fake_factories(),
    )

    assert blend.uses_exposure is True
    assert blend.distribution == "poisson"
    assert blend.tweedie_power is None
    assert set(blend.weights) == {"catboost", "lightgbm", "xgboost"}
    assert sum(blend.weights.values()) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in blend.weights.values())
    assert len(blend.oof_predictions) == row_count
    assert blend.oof_predictions["row_position"].is_unique
    np.testing.assert_allclose(blend.oof_predictions["fit_weight"], exposure)
    assert np.isfinite(blend.oof_predictions.filter(like="_rate")).all().all()
    assert set(blend.metrics["model"]) == {
        "catboost",
        "lightgbm",
        "xgboost",
        "blend",
    }
    base_deviance = blend.metrics.loc[
        blend.metrics["model"].ne("blend"), "mean_unit_deviance"
    ].min()
    blend_deviance = blend.metrics.loc[
        blend.metrics["model"].eq("blend"), "mean_unit_deviance"
    ].item()
    assert blend_deviance <= base_deviance + 1e-10

    expected = blend.predict_expected(frame.iloc[:5], exposure=exposure[:5])
    assert expected.shape == (5,)
    assert np.isfinite(expected).all()
    assert (expected > 0.0).all()


def test_boosted_blend_uses_one_fixed_tweedie_power_for_fit_and_oof_loss():
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 60),
            "category": np.resize(["A", "B", "C"], 60),
        }
    )
    target = np.linspace(0.0, 3.0, 60)
    exposure = np.linspace(0.25, 1.0, 60)
    credibility_weight = np.linspace(1.0, 2.0, 60)

    blend = fit_boosted_blend(
        frame,
        target,
        categorical_columns=("category",),
        sample_weight=credibility_weight,
        exposure=exposure,
        tweedie_power=1.63,
        n_splits=3,
        estimator_factories=_fake_factories(),
    )

    assert blend.distribution == "tweedie"
    assert blend.tweedie_power == pytest.approx(1.63)
    assert blend.power_source == "explicit"
    assert blend.metrics["distribution"].eq("tweedie").all()
    assert blend.metrics["tweedie_power"].eq(1.63).all()
    assert np.isfinite(blend.metrics["mean_unit_deviance"]).all()
    positions = blend.oof_predictions["row_position"].to_numpy(dtype=int)
    np.testing.assert_allclose(
        blend.oof_predictions["fit_weight"],
        credibility_weight[positions] * exposure[positions] ** (2.0 - 1.63),
    )


def test_fitted_superglm_is_the_distribution_source_of_truth():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 60)})
    target = np.linspace(0.1, 3.0, 60)
    reference = SuperGLM(
        family=Tweedie(p=1.63),
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(frame, target)

    blend = fit_boosted_blend(
        frame,
        target,
        reference_superglm=reference,
        n_splits=3,
        estimator_factories=_fake_factories(),
    )

    assert blend.distribution == "tweedie"
    assert blend.tweedie_power == pytest.approx(1.63)
    assert blend.power_source == "reference_superglm"
    assert blend.metrics["power_source"].eq("reference_superglm").all()
    with pytest.raises(ValueError, match="conflicts with the fitted reference"):
        fit_boosted_blend(
            frame,
            target,
            tweedie_power=1.5,
            reference_superglm=reference,
            estimator_factories=_fake_factories(),
        )


def test_fitted_string_poisson_superglm_is_a_valid_reference():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 60)})
    target = np.resize([0.0, 1.0, 0.0, 2.0], 60)
    reference = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(frame, target)

    blend = fit_boosted_blend(
        frame,
        target,
        reference_superglm=reference,
        n_splits=3,
        estimator_factories=_fake_factories(),
    )

    assert blend.distribution == "poisson"
    assert blend.tweedie_power is None
    assert blend.power_source == "reference_superglm"


def test_boosted_blend_protects_shared_distribution_contract():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    target = np.linspace(0.0, 2.0, 30)

    with pytest.raises(ValueError, match="cannot override the shared distribution"):
        fit_boosted_blend(
            frame,
            target,
            tweedie_power=1.5,
            model_parameters={"xgboost": {"tweedie_variance_power": 1.7}},
            estimator_factories=_fake_factories(),
        )
    for invalid in (1.0, 2.0, np.nan):
        with pytest.raises(ValueError, match="strictly between 1 and 2"):
            fit_boosted_blend(
                frame,
                target,
                tweedie_power=invalid,
                estimator_factories=_fake_factories(),
            )


def test_optional_estimators_receive_the_exact_shared_tweedie_power():
    pytest.importorskip("catboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    factories = _required_scratch_estimators(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        thread_count=1,
        tweedie_power=1.63,
        model_parameters={},
    )

    catboost = factories["catboost"](42).get_params()
    lightgbm = factories["lightgbm"](42).get_params()
    xgboost = factories["xgboost"](42).get_params()
    assert catboost["loss_function"] == "Tweedie:variance_power=1.63"
    assert lightgbm["objective"] == "tweedie"
    assert lightgbm["tweedie_variance_power"] == pytest.approx(1.63)
    assert xgboost["objective"] == "reg:tweedie"
    assert xgboost["tweedie_variance_power"] == pytest.approx(1.63)


def test_boosted_blend_rejects_overlapping_cv_positions():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    target = np.resize([0.0, 1.0, 2.0], 30)

    with pytest.raises(ValueError, match="must not overlap"):
        fit_boosted_blend(
            frame,
            target,
            cv=[(np.arange(20), np.arange(10, 30))],
            estimator_factories=_fake_factories(),
        )


def test_boosted_blend_requires_optional_estimators_when_no_factories(monkeypatch):
    def missing_estimators(**_kwargs):
        raise ScratchBenchmarkError("uv sync --extra scratch")

    monkeypatch.setattr(
        "pricing_pipeline.modeling.scratch_benchmark._required_scratch_estimators",
        missing_estimators,
    )

    with pytest.raises(ScratchBenchmarkError, match="uv sync --extra scratch"):
        fit_boosted_blend(
            pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)}),
            np.resize([0.0, 1.0], 30),
        )
