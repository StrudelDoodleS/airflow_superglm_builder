from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_tweedie_deviance

from pricing_pipeline.modeling.scratch_diagnostics import (
    blend_evaluation_table,
    blend_weight_label,
    double_lift_table,
    feature_calibration_tables,
    interaction_failure_tables,
    lorenz_curve_table,
    risk_calibration_table,
    unit_tweedie_deviance,
    weighted_quantile_bins,
)


def _diagnostic_case(row_count: int = 240):
    rng = np.random.default_rng(17)
    x = np.linspace(0.0, 1.0, row_count)
    category = np.resize(["A", "B", "C"], row_count)
    noise = rng.normal(size=row_count)
    exposure = np.linspace(0.2, 1.0, row_count)
    credibility = np.linspace(1.0, 3.0, row_count)
    gam_rate = np.exp(-0.2 + 0.35 * x + 0.1 * (category == "C"))
    interaction = np.where((x > 0.65) & (category == "B"), 0.65, 0.0)
    gbm_rate = gam_rate * np.exp(interaction)
    # The interaction is deliberately unsupported by the response. This makes
    # the held-out B/high-x cells an observable GBM failure rather than merely
    # a disagreement with the GAM.
    actual_rate = gam_rate * np.exp(0.02 * noise)
    response = exposure * actual_rate
    features = pd.DataFrame({"x": x, "category": category, "noise": noise})
    return features, response, exposure, credibility, gam_rate, gbm_rate


@pytest.mark.parametrize("power", [1.0, 1.5, 1.9])
def test_row_tweedie_deviance_matches_sklearn_weighted_mean(power):
    actual = np.array([0.0, 0.2, 1.1, 3.0])
    predicted = np.array([0.1, 0.4, 0.9, 2.5])
    weight = np.array([1.0, 2.0, 3.0, 4.0])

    rows = unit_tweedie_deviance(actual, predicted, power=power)

    assert np.average(rows, weights=weight) == pytest.approx(
        mean_tweedie_deviance(actual, predicted, sample_weight=weight, power=power)
    )


def test_weighted_quantile_bins_follow_score_and_balance_business_weight():
    values = np.arange(100, dtype=float)
    weights = np.linspace(1.0, 2.0, 100)

    bins = weighted_quantile_bins(values, weights, n_bins=5)

    assert np.all(np.diff(bins) >= 0)
    assert set(bins) == set(range(5))
    totals = np.bincount(bins, weights=weights)
    assert totals.max() - totals.min() < weights.max() * 2


def test_weighted_quantile_bins_keep_equal_scores_together_independent_of_row_order():
    values = np.repeat([0.0, 1.0, 2.0, 3.0], 4)
    weights = np.array([1.0, 3.0, 2.0, 4.0] * 4)

    bins = weighted_quantile_bins(values, weights, n_bins=5)
    permutation = np.array([15, 0, 8, 3, 12, 5, 10, 1, 14, 7, 4, 9, 2, 13, 6, 11])
    permuted_bins = weighted_quantile_bins(
        values[permutation],
        weights[permutation],
        n_bins=5,
    )
    restored = np.empty_like(permuted_bins)
    restored[permutation] = permuted_bins

    for score in np.unique(values):
        assert np.unique(bins[values == score]).size == 1
    assert restored.tolist() == bins.tolist()


def test_double_lift_is_weight_balanced_and_ordered_by_gbm_over_gam():
    features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    table = double_lift_table(
        features,
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        n_bins=6,
    )

    # This fixture has only two unique GBM/GAM ratios. Tie-safe binning must not
    # manufacture six bins by splitting either score on row order.
    assert table["double_lift_bin"].tolist() == [1, 2]
    assert table["geometric_mean_gbm_to_gam"].is_monotonic_increasing
    assert table["sample_weight"].sum() == pytest.approx(weight.sum())
    for column in (
        "actual_response_index",
        "gam_response_index",
        "gbm_response_index",
        "blend_040_response_index",
        "blend_050_response_index",
    ):
        assert np.average(table[column], weights=table["sample_weight"]) == pytest.approx(1.0)
    assert table.iloc[-1]["gbm_minus_gam_mean_deviance"] > 0
    assert table.iloc[-1]["gbm_observed_to_predicted"] < 1


def test_distinct_blend_weights_have_collision_free_diagnostic_columns():
    features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    table = double_lift_table(
        features,
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        n_bins=4,
        governed_gam_weights=(0.4, 0.403),
    )

    assert "blend_040_response" in table
    assert "blend_w_0p403_response" in table
    assert not np.allclose(table["blend_040_response"], table["blend_w_0p403_response"])


def test_adjacent_float_does_not_share_governed_percentage_column_identity():
    governed_weight = 0.4
    adjacent_weight = float(np.nextafter(governed_weight, 0.0))

    governed_label = blend_weight_label(governed_weight)
    adjacent_label = blend_weight_label(adjacent_weight)

    assert governed_label == "blend_040"
    assert adjacent_label != governed_label
    assert blend_weight_label(float.fromhex(adjacent_weight.hex())) == adjacent_label


def test_fixed_blend_table_keeps_offset_and_credibility_weight_distinct():
    _features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    table = blend_evaluation_table(
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        gam_weights=(0.0, 0.4, 0.5, 1.0),
    )

    assert table["gam_weight"].tolist() == [0.0, 0.4, 0.5, 1.0]
    assert table["actual_response"].iloc[0] == pytest.approx(np.average(response, weights=weight))
    assert table["response_numerator"].iloc[0] == pytest.approx(np.sum(weight * response))
    assert table["denominator_total"].iloc[0] == pytest.approx(weight.sum())
    assert table.loc[table["gam_weight"].eq(1.0), "observed_to_predicted"].item() == pytest.approx(
        1.0,
        abs=0.01,
    )
    assert (
        table.loc[table["gam_weight"].eq(0.0), "mean_tweedie_deviance"].item()
        > table.loc[table["gam_weight"].eq(1.0), "mean_tweedie_deviance"].item()
    )


def test_feature_and_risk_tables_expose_tail_calibration_with_support():
    features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    feature_tables = feature_calibration_tables(
        features,
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        categorical_features=("category",),
        n_bins=6,
    )
    risk = risk_calibration_table(
        features,
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        n_bins=8,
    )

    assert set(feature_tables) == {"x", "category", "noise"}
    assert set(feature_tables["category"]["feature_bin"]) == {"A", "B", "C"}
    assert (feature_tables["x"]["sample_weight"] > 0).all()
    assert risk["risk_bin"].tolist() == list(range(1, 9))
    assert (risk["offset_weight"] > 0).all()


def test_lorenz_curve_is_aggregate_weighted_and_reaches_one():
    _features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    table = lorenz_curve_table(
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        n_bins=20,
    )

    assert set(table["model"]) == {
        "GAM",
        "BOOSTED_BLEND",
        "BLEND_040_GAM",
        "BLEND_050_GAM",
    }
    for _, curve in table.groupby("model"):
        assert curve.iloc[0]["cumulative_weight_share"] == 0.0
        assert curve.iloc[0]["cumulative_response_share"] == 0.0
        assert curve.iloc[-1]["cumulative_weight_share"] == pytest.approx(1.0)
        assert curve.iloc[-1]["cumulative_response_share"] == pytest.approx(1.0)
        assert curve["gini"].nunique() == 1


def test_interaction_ranking_finds_unsupported_gbm_interaction_and_failure_cells():
    features, response, exposure, weight, gam_rate, gbm_rate = _diagnostic_case()

    ranking, cells = interaction_failure_tables(
        features,
        response,
        offset_exposure=exposure,
        sample_weight=weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=1.5,
        categorical_features=("category",),
        n_bins=5,
        min_cell_rows=5,
        min_cell_weight_fraction=0.0,
    )

    top_pair = {ranking.iloc[0]["feature_a"], ranking.iloc[0]["feature_b"]}
    assert top_pair == {"x", "category"}
    failure = cells.loc[
        cells["feature_a"].eq("x") & cells["feature_b"].eq("category") & cells["level_b"].eq("B")
    ]
    assert not failure.empty
    assert failure["gbm_minus_gam_mean_deviance"].max() > 0
    assert failure["interaction_log_ratio"].abs().max() > 0.1
