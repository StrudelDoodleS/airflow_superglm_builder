from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_tweedie_deviance
from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM, Tweedie
from superglm.features.spline import _SplineBase

from pricing_pipeline.modeling.scratch_benchmark import (
    ScratchBenchmarkError,
    ScratchBoostedBlend,
    _catboost_frame,
    _category_levels,
    _required_scratch_estimators,
    _tree_frame,
    fit_boosted_blend,
    superglm_edf_table,
    unconstrained_superglm_features,
)

_EXPLORATORY_BLEND_EVIDENCE = (
    "exploratory_cross_fitted_meta_over_global_base_oof_"
    "non_nested_not_unbiased_generalization_estimate"
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


class _ColumnEstimator:
    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y, *, sample_weight=None, **_kwargs):
        del y, sample_weight
        assert self.column in X
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


def _column_factories():
    return {
        "catboost": lambda _seed: _ColumnEstimator("candidate_a"),
        "lightgbm": lambda _seed: _ColumnEstimator("candidate_b"),
        "xgboost": lambda _seed: _ColumnEstimator("candidate_c"),
    }


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
def test_booster_preprocessing_preserves_typed_and_missing_category_identities(backend):
    values = np.empty(40, dtype=object)
    values[0::4] = 1
    values[1::4] = "1"
    values[2::4] = None
    values[3::4] = "__SCRATCH_MISSING__"
    frame = pd.DataFrame({"category": values})

    levels = _category_levels(frame, ("category",))
    tree = _tree_frame(
        frame,
        feature_columns=("category",),
        categorical_levels=levels,
    )
    prepared = _catboost_frame(frame, ("category",), levels) if backend == "catboost" else tree

    assert len(levels["category"]) == 4
    assert prepared["category"].nunique(dropna=False) == 4
    assert prepared["category"].iloc[0] != prepared["category"].iloc[1]
    assert prepared["category"].iloc[2] != prepared["category"].iloc[3]


def test_tree_prediction_rejects_text_match_with_wrong_category_type():
    training = pd.DataFrame({"category": np.resize([1, 2], 30)})
    levels = _category_levels(training, ("category",))
    prediction = pd.DataFrame({"category": ["1", 2]})

    with pytest.raises(
        ScratchBenchmarkError,
        match="categorical benchmark column 'category'.*absent from the training contract",
    ):
        _tree_frame(
            prediction,
            feature_columns=("category",),
            categorical_levels=levels,
        )


def test_unsupported_category_fails_safely_before_any_estimator_fit():
    secret = "private-category-value"

    class UnsupportedCategory:
        def __str__(self):
            return secret

    fit_calls = 0

    class TrackingEstimator(_WeightedMeanEstimator):
        def fit(self, X, y, *, sample_weight=None, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    factories = {
        name: lambda _seed, multiplier=multiplier: TrackingEstimator(multiplier)
        for name, multiplier in {"catboost": 0.75, "lightgbm": 1.0, "xgboost": 1.25}.items()
    }
    values = np.empty(30, dtype=object)
    values[:] = UnsupportedCategory()

    with pytest.raises(ScratchBenchmarkError) as exc_info:
        fit_boosted_blend(
            pd.DataFrame({"category": values}),
            np.resize([0.0, 1.0, 2.0], 30),
            categorical_columns=("category",),
            estimator_factories=factories,
        )

    assert fit_calls == 0
    assert secret not in str(exc_info.value)


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
    blend_deviance = blend.metrics.loc[
        blend.metrics["model"].eq("blend"), "mean_unit_deviance"
    ].item()
    assert blend_deviance == pytest.approx(
        mean_tweedie_deviance(
            blend.oof_predictions["actual_rate"],
            blend.oof_predictions["blend_rate"],
            sample_weight=blend.oof_predictions["fit_weight"],
            power=1.0,
        )
    )
    assert blend.metrics.set_index("model").loc["blend", "evaluation"] == (
        _EXPLORATORY_BLEND_EVIDENCE
    )
    assert set(blend.oof_predictions["blend_evidence"]) == {_EXPLORATORY_BLEND_EVIDENCE}

    expected = blend.predict_expected(frame.iloc[:5], exposure=exposure[:5])
    assert expected.shape == (5,)
    assert np.isfinite(expected).all()
    assert (expected > 0.0).all()


def test_blend_evidence_cross_fits_meta_weights_by_assessment_fold():
    row_count = 40
    first_fold = np.arange(0, 20)
    second_fold = np.arange(20, 40)
    frame = pd.DataFrame(
        {
            "candidate_a": np.r_[np.ones(20), np.full(20, 4.0)],
            "candidate_b": np.r_[np.full(20, 4.0), np.ones(20)],
            "candidate_c": np.full(row_count, 3.0),
        }
    )
    target = np.ones(row_count)
    cv = [
        (second_fold, first_fold),
        (first_fold, second_fold),
    ]

    blend = fit_boosted_blend(
        frame,
        target,
        cv=cv,
        estimator_factories=_column_factories(),
    )

    deployable_weights = np.array([blend.weights[name] for name in _column_factories()])
    assert deployable_weights == pytest.approx([0.5, 0.5, 0.0], abs=1e-8)
    deployable_oof_rate = (
        blend.oof_predictions[["catboost_rate", "lightgbm_rate", "xgboost_rate"]].to_numpy()
        @ deployable_weights
    )
    assert blend.oof_predictions["assessment_fold"].tolist() == [0] * 20 + [1] * 20
    assert blend.oof_predictions["blend_rate"].to_numpy() == pytest.approx(np.full(row_count, 4.0))
    assert not np.allclose(blend.oof_predictions["blend_rate"], deployable_oof_rate)
    blend_metric = blend.metrics.set_index("model").loc["blend"]
    assert blend_metric["evaluation"] == _EXPLORATORY_BLEND_EVIDENCE
    assert blend_metric["blend_weight_scope"] == "final_all_complete_oof"
    assert blend_metric["mean_unit_deviance"] == pytest.approx(2.0 * (4.0 - 1.0 - np.log(4.0)))


def test_blend_contract_disclaims_nested_or_unbiased_generalization_evidence():
    raw_documentation = ScratchBoostedBlend.__doc__

    assert raw_documentation is not None
    documentation = " ".join(raw_documentation.split())
    assert "honest OOF" not in documentation
    assert "global base-OOF matrix" in documentation
    assert "non-nested" in documentation
    assert "not an unbiased generalization estimate" in documentation


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
