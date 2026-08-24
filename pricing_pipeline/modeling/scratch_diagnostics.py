"""Held-out diagnostics for governed GAM/GBM scratch comparisons.

The functions operate only on predictions already produced in memory. They do
not register models, write SQL, or retain row-level data. Aggregate outputs are
designed to show where a flexible GBM differs from, and underperforms, an
additive GAM despite a better portfolio-average objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

_PREFIX = "__scratch_diagnostic_"


class ScratchDiagnosticError(RuntimeError):
    """Raised when a scratch comparison cannot produce trustworthy evidence."""


def unit_tweedie_deviance(
    actual: Any,
    predicted: Any,
    *,
    power: float,
) -> np.ndarray:
    """Return row-level Tweedie unit deviance for Poisson or ``1 < p < 2``."""
    y = np.asarray(actual, dtype=float)
    mu = np.asarray(predicted, dtype=float)
    if y.shape != mu.shape:
        raise ValueError("actual and predicted must have the same shape")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("actual must contain finite non-negative values")
    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("predicted must contain finite positive values")
    if power == 1.0:
        out = 2.0 * (mu - y)
        positive = y > 0
        out[positive] += 2.0 * y[positive] * np.log(y[positive] / mu[positive])
        return out
    if not 1.0 < power < 2.0:
        raise ValueError("power must equal 1 or be strictly between 1 and 2")
    return 2.0 * (
        np.power(y, 2.0 - power) / ((1.0 - power) * (2.0 - power))
        - y * np.power(mu, 1.0 - power) / (1.0 - power)
        + np.power(mu, 2.0 - power) / (2.0 - power)
    )


def weighted_quantile_bins(
    values: Any,
    weights: Any,
    *,
    n_bins: int,
) -> np.ndarray:
    """Assign stable bins with approximately equal total business weight."""
    resolved_values = np.asarray(values, dtype=float)
    resolved_weights = np.asarray(weights, dtype=float)
    if resolved_values.ndim != 1 or resolved_weights.shape != resolved_values.shape:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    if not np.isfinite(resolved_values).all():
        raise ValueError("values must contain only finite values")
    if not np.isfinite(resolved_weights).all() or np.any(resolved_weights <= 0):
        raise ValueError("weights must contain finite positive values")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    order = np.argsort(resolved_values, kind="stable")
    ordered_weights = resolved_weights[order]
    midpoint = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    ordered_bins = np.floor(n_bins * midpoint / ordered_weights.sum()).astype(int)
    ordered_bins = np.clip(ordered_bins, 0, n_bins - 1)
    # A dominant observation can leave nominal bins empty. Dense numbering keeps
    # every reported bin contiguous without moving any observation.
    _, ordered_bins = np.unique(ordered_bins, return_inverse=True)
    bins = np.empty(len(order), dtype=int)
    bins[order] = ordered_bins
    return bins


def _validated_analysis_frame(
    features: pd.DataFrame,
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    governed_gam_weights: Sequence[float],
) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    collisions = [column for column in features if str(column).startswith(_PREFIX)]
    if collisions:
        raise ValueError("feature columns use the reserved scratch diagnostic prefix")
    row_count = len(features)

    def vector(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
        out = np.asarray(value, dtype=float)
        if out.ndim != 1 or len(out) != row_count:
            raise ValueError(f"{name} must be one-dimensional and match features")
        if not np.isfinite(out).all() or (positive and np.any(out <= 0)):
            qualifier = "finite positive" if positive else "finite"
            raise ValueError(f"{name} must contain {qualifier} values")
        return out

    y = vector(response, "response")
    if np.any(y < 0):
        raise ValueError("response must be non-negative")
    exposure = vector(offset_exposure, "offset_exposure", positive=True)
    weight = vector(sample_weight, "sample_weight", positive=True)
    gam = vector(gam_rate, "gam_rate", positive=True)
    gbm = vector(gbm_rate, "gbm_rate", positive=True)
    if power != 1.0 and not 1.0 < power < 2.0:
        raise ValueError("power must equal 1 or be strictly between 1 and 2")
    resolved_gam_weights = tuple(float(value) for value in governed_gam_weights)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in resolved_gam_weights):
        raise ValueError("governed_gam_weights must be finite values between 0 and 1")

    analysis = features.reset_index(drop=True).copy()
    actual_rate = y / exposure
    fit_weight = weight * np.power(exposure, 2.0 - power)
    offset_weight = weight * exposure
    gam_expected = exposure * gam
    gbm_expected = exposure * gbm
    analysis[f"{_PREFIX}response"] = y
    analysis[f"{_PREFIX}exposure"] = exposure
    analysis[f"{_PREFIX}sample_weight"] = weight
    analysis[f"{_PREFIX}fit_weight"] = fit_weight
    analysis[f"{_PREFIX}offset_weight"] = offset_weight
    analysis[f"{_PREFIX}actual_rate"] = actual_rate
    analysis[f"{_PREFIX}gam_rate"] = gam
    analysis[f"{_PREFIX}gbm_rate"] = gbm
    analysis[f"{_PREFIX}gam_expected"] = gam_expected
    analysis[f"{_PREFIX}gbm_expected"] = gbm_expected
    analysis[f"{_PREFIX}log_ratio"] = np.log(gbm / gam)
    analysis[f"{_PREFIX}gam_deviance"] = unit_tweedie_deviance(
        y,
        gam_expected,
        power=power,
    )
    analysis[f"{_PREFIX}gbm_deviance"] = unit_tweedie_deviance(
        y,
        gbm_expected,
        power=power,
    )
    for gam_weight in resolved_gam_weights:
        name = _blend_rate_column(gam_weight)
        rate = gam_weight * gam + (1.0 - gam_weight) * gbm
        expected_name = _blend_expected_column(gam_weight)
        expected = exposure * rate
        analysis[name] = rate
        analysis[expected_name] = expected
        analysis[f"{expected_name}_deviance"] = unit_tweedie_deviance(
            y,
            expected,
            power=power,
        )
    return analysis


def _blend_rate_column(gam_weight: float) -> str:
    return f"{_PREFIX}blend_{round(100 * gam_weight):03d}_rate"


def _blend_expected_column(gam_weight: float) -> str:
    return f"{_PREFIX}blend_{round(100 * gam_weight):03d}_expected"


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def _aggregate_groups(
    analysis: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    governed_gam_weights: Sequence[float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = analysis.groupby(list(group_columns), observed=True, sort=True, dropna=False)
    for raw_key, part in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        sample_weight = part[f"{_PREFIX}sample_weight"]
        fit_weight = part[f"{_PREFIX}fit_weight"]
        offset_weight = part[f"{_PREFIX}offset_weight"]
        response_total = float(np.sum(sample_weight * part[f"{_PREFIX}response"]))
        gam_prediction_total = float(np.sum(sample_weight * part[f"{_PREFIX}gam_expected"]))
        gbm_prediction_total = float(np.sum(sample_weight * part[f"{_PREFIX}gbm_expected"]))
        record = {column: value for column, value in zip(group_columns, key, strict=True)}
        record.update(
            {
                "rows": len(part),
                "positive_response_rows": int((part[f"{_PREFIX}response"] > 0).sum()),
                "sample_weight": float(sample_weight.sum()),
                "offset_weight": float(offset_weight.sum()),
                "response_total": response_total,
                "gam_prediction_total": gam_prediction_total,
                "gbm_prediction_total": gbm_prediction_total,
                "actual_response": response_total / sample_weight.sum(),
                "gam_response": gam_prediction_total / sample_weight.sum(),
                "gbm_response": gbm_prediction_total / sample_weight.sum(),
                "actual_annualized_rate": response_total / offset_weight.sum(),
                "gam_annualized_rate": gam_prediction_total / offset_weight.sum(),
                "gbm_annualized_rate": gbm_prediction_total / offset_weight.sum(),
                "geometric_mean_gbm_to_gam": float(
                    np.exp(_weighted_average(part[f"{_PREFIX}log_ratio"], sample_weight))
                ),
                "gam_mean_deviance": _weighted_average(
                    part[f"{_PREFIX}gam_deviance"], sample_weight
                ),
                "gbm_mean_deviance": _weighted_average(
                    part[f"{_PREFIX}gbm_deviance"], sample_weight
                ),
                "gbm_minus_gam_deviance_numerator": float(
                    np.sum(
                        sample_weight
                        * (part[f"{_PREFIX}gbm_deviance"] - part[f"{_PREFIX}gam_deviance"])
                    )
                ),
                "fit_weight": float(fit_weight.sum()),
            }
        )
        record["gam_observed_to_predicted"] = record["actual_response"] / record["gam_response"]
        record["gbm_observed_to_predicted"] = record["actual_response"] / record["gbm_response"]
        record["gbm_minus_gam_mean_deviance"] = (
            record["gbm_mean_deviance"] - record["gam_mean_deviance"]
        )
        for gam_weight in governed_gam_weights:
            source = _blend_expected_column(float(gam_weight))
            label = f"blend_{round(100 * gam_weight):03d}"
            prediction_total = float(np.sum(sample_weight * part[source]))
            record[f"{label}_prediction_total"] = prediction_total
            record[f"{label}_response"] = prediction_total / sample_weight.sum()
            record[f"{label}_observed_to_predicted"] = (
                record["actual_response"] / record[f"{label}_response"]
            )
            record[f"{label}_mean_deviance"] = _weighted_average(
                part[f"{source}_deviance"],
                sample_weight,
            )
        records.append(record)
    return pd.DataFrame(records)


def blend_evaluation_table(
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    gam_weights: Sequence[float] = (0.0, 0.4, 0.5, 1.0),
) -> pd.DataFrame:
    """Evaluate fixed GAM/GBM weights on one untouched comparison sample."""
    placeholder = pd.DataFrame({"row": np.arange(len(np.asarray(response)))})
    analysis = _validated_analysis_frame(
        placeholder,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=gam_weights,
    )
    sample_weight = analysis[f"{_PREFIX}sample_weight"]
    actual_response = float(
        np.sum(sample_weight * analysis[f"{_PREFIX}response"]) / sample_weight.sum()
    )
    records = []
    for gam_weight in gam_weights:
        resolved = float(gam_weight)
        if resolved == 0.0:
            expected_column = f"{_PREFIX}gbm_expected"
            deviance_column = f"{_PREFIX}gbm_deviance"
        elif resolved == 1.0:
            expected_column = f"{_PREFIX}gam_expected"
            deviance_column = f"{_PREFIX}gam_deviance"
        else:
            expected_column = _blend_expected_column(resolved)
            deviance_column = f"{expected_column}_deviance"
        predicted_response = _weighted_average(analysis[expected_column], sample_weight)
        records.append(
            {
                "gam_weight": resolved,
                "gbm_weight": 1.0 - resolved,
                "mean_tweedie_deviance": _weighted_average(
                    analysis[deviance_column], sample_weight
                ),
                "actual_response": actual_response,
                "predicted_response": predicted_response,
                "observed_to_predicted": actual_response / predicted_response,
                "response_numerator": float(np.sum(sample_weight * analysis[f"{_PREFIX}response"])),
                "prediction_numerator": float(np.sum(sample_weight * analysis[expected_column])),
                "denominator_total": float(sample_weight.sum()),
            }
        )
    return pd.DataFrame(records).sort_values("gam_weight", ignore_index=True)


def double_lift_table(
    features: pd.DataFrame,
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    n_bins: int = 10,
    governed_gam_weights: Sequence[float] = (0.4, 0.5),
) -> pd.DataFrame:
    """Build sample-weight-balanced bins ordered by ``GBM / GAM`` prediction."""
    analysis = _validated_analysis_frame(
        features,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=governed_gam_weights,
    )
    analysis[f"{_PREFIX}bin"] = weighted_quantile_bins(
        analysis[f"{_PREFIX}log_ratio"],
        analysis[f"{_PREFIX}sample_weight"],
        n_bins=n_bins,
    )
    out = _aggregate_groups(
        analysis,
        [f"{_PREFIX}bin"],
        governed_gam_weights=governed_gam_weights,
    ).rename(columns={f"{_PREFIX}bin": "double_lift_bin"})
    out["double_lift_bin"] += 1
    total_weight = out["sample_weight"].sum()
    index_sources = {
        "actual_response": "response_total",
        "gam_response": "gam_prediction_total",
        "gbm_response": "gbm_prediction_total",
        **{
            f"blend_{round(100 * value):03d}_response": (
                f"blend_{round(100 * value):03d}_prediction_total"
            )
            for value in governed_gam_weights
        },
    }
    for response_column, numerator_column in index_sources.items():
        portfolio_average = out[numerator_column].sum() / total_weight
        out[f"{response_column}_index"] = out[response_column] / portfolio_average
    return out


def risk_calibration_table(
    features: pd.DataFrame,
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    n_bins: int = 20,
    governed_gam_weights: Sequence[float] = (0.4, 0.5),
) -> pd.DataFrame:
    """Build sample-weight-balanced bins ordered by GBM predicted rate."""
    analysis = _validated_analysis_frame(
        features,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=governed_gam_weights,
    )
    analysis[f"{_PREFIX}bin"] = weighted_quantile_bins(
        analysis[f"{_PREFIX}gbm_rate"],
        analysis[f"{_PREFIX}sample_weight"],
        n_bins=n_bins,
    )
    out = _aggregate_groups(
        analysis,
        [f"{_PREFIX}bin"],
        governed_gam_weights=governed_gam_weights,
    ).rename(columns={f"{_PREFIX}bin": "risk_bin"})
    out["risk_bin"] += 1
    return out


def lorenz_curve_table(
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    n_bins: int = 100,
    governed_gam_weights: Sequence[float] = (0.4, 0.5),
) -> pd.DataFrame:
    """Return aggregate weighted Lorenz curves ordered by each model prediction."""
    placeholder = pd.DataFrame({"row": np.arange(len(np.asarray(response)))})
    analysis = _validated_analysis_frame(
        placeholder,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=governed_gam_weights,
    )
    weight = analysis[f"{_PREFIX}sample_weight"]
    response_numerator = weight * analysis[f"{_PREFIX}response"]
    total_response = float(response_numerator.sum())
    if total_response <= 0:
        raise ScratchDiagnosticError("a Lorenz curve requires positive aggregate response")
    models = {
        "GAM": f"{_PREFIX}gam_expected",
        "BOOSTED_BLEND": f"{_PREFIX}gbm_expected",
        **{
            f"BLEND_{round(100 * value):03d}_GAM": _blend_expected_column(float(value))
            for value in governed_gam_weights
        },
    }
    curves: list[pd.DataFrame] = []
    for model_name, prediction_column in models.items():
        predicted = analysis[prediction_column]
        bins = weighted_quantile_bins(predicted, weight, n_bins=n_bins)
        grouped = (
            pd.DataFrame(
                {
                    "lorenz_bin": bins,
                    "denominator": weight,
                    "response_numerator": response_numerator,
                    "prediction_numerator": weight * predicted,
                }
            )
            .groupby("lorenz_bin", sort=True, observed=True, as_index=False)
            .sum()
        )
        grouped["cumulative_weight_share"] = (
            grouped["denominator"].cumsum() / grouped["denominator"].sum()
        )
        grouped["cumulative_response_share"] = (
            grouped["response_numerator"].cumsum() / total_response
        )
        grouped["cumulative_prediction_share"] = (
            grouped["prediction_numerator"].cumsum() / grouped["prediction_numerator"].sum()
        )
        origin = pd.DataFrame(
            {
                "lorenz_bin": [0],
                "denominator": [0.0],
                "response_numerator": [0.0],
                "prediction_numerator": [0.0],
                "cumulative_weight_share": [0.0],
                "cumulative_response_share": [0.0],
                "cumulative_prediction_share": [0.0],
            }
        )
        grouped["lorenz_bin"] += 1
        curve = pd.concat([origin, grouped], ignore_index=True)
        area = float(
            np.trapezoid(
                curve["cumulative_response_share"],
                curve["cumulative_weight_share"],
            )
        )
        curve["model"] = model_name
        curve["gini"] = 1.0 - 2.0 * area
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)


def _feature_labels(
    analysis: pd.DataFrame,
    feature: str,
    *,
    categorical: bool,
    n_bins: int,
    max_categorical_levels: int,
) -> pd.Series:
    values = analysis[feature]
    if categorical:
        labels = values.astype("string").fillna("__MISSING__")
        level_weight = analysis.groupby(labels, observed=True)[f"{_PREFIX}sample_weight"].sum()
        if len(level_weight) > max_categorical_levels:
            keep = set(level_weight.nlargest(max_categorical_levels - 1).index)
            labels = labels.where(labels.isin(keep), "__OTHER__")
        return labels.astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(
            f"numeric diagnostic feature {feature!r} contains missing/non-finite values"
        )
    bins = weighted_quantile_bins(
        numeric,
        analysis[f"{_PREFIX}sample_weight"],
        n_bins=n_bins,
    )
    medians = pd.DataFrame({"bin": bins, "value": numeric}).groupby("bin")["value"].median()
    return pd.Series(
        [f"{bin_no + 1:02d} | {medians.loc[bin_no]:.6g}" for bin_no in bins],
        index=analysis.index,
    )


def feature_calibration_tables(
    features: pd.DataFrame,
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    categorical_features: Sequence[str] = (),
    n_bins: int = 10,
    max_categorical_levels: int = 20,
    governed_gam_weights: Sequence[float] = (0.4, 0.5),
) -> Mapping[str, pd.DataFrame]:
    """Return one-way held-out calibration evidence for every feature."""
    analysis = _validated_analysis_frame(
        features,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=governed_gam_weights,
    )
    categorical = set(categorical_features)
    missing = categorical - set(features.columns)
    if missing:
        raise ValueError(
            "categorical diagnostic features are missing: " + ", ".join(sorted(missing))
        )
    result: dict[str, pd.DataFrame] = {}
    for feature in features.columns:
        label_column = f"{_PREFIX}feature_bin"
        analysis[label_column] = _feature_labels(
            analysis,
            str(feature),
            categorical=str(feature) in categorical,
            n_bins=n_bins,
            max_categorical_levels=max_categorical_levels,
        )
        result[str(feature)] = _aggregate_groups(
            analysis,
            [label_column],
            governed_gam_weights=governed_gam_weights,
        ).rename(columns={label_column: "feature_bin"})
    return result


def interaction_failure_tables(
    features: pd.DataFrame,
    response: Any,
    *,
    offset_exposure: Any,
    sample_weight: Any,
    gam_rate: Any,
    gbm_rate: Any,
    power: float,
    categorical_features: Sequence[str] = (),
    n_bins: int = 6,
    max_categorical_levels: int = 12,
    min_cell_rows: int = 20,
    min_cell_weight_fraction: float = 0.001,
    governed_gam_weights: Sequence[float] = (0.4, 0.5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank pairwise GBM interactions and return credible failure cells.

    ``interaction_rms_log_ratio`` measures non-additive divergence between GBM
    and GAM prediction surfaces. Positive ``gbm_minus_gam_mean_deviance`` means
    the GBM performs worse than the GAM in that held-out cell.
    """
    if min_cell_rows < 1:
        raise ValueError("min_cell_rows must be positive")
    if not 0.0 <= min_cell_weight_fraction < 1.0:
        raise ValueError("min_cell_weight_fraction must be between 0 and 1")
    analysis = _validated_analysis_frame(
        features,
        response,
        offset_exposure=offset_exposure,
        sample_weight=sample_weight,
        gam_rate=gam_rate,
        gbm_rate=gbm_rate,
        power=power,
        governed_gam_weights=governed_gam_weights,
    )
    categorical = set(categorical_features)
    missing = categorical - set(features.columns)
    if missing:
        raise ValueError(
            "categorical diagnostic features are missing: " + ", ".join(sorted(missing))
        )
    label_columns: dict[str, str] = {}
    for feature in features.columns:
        name = str(feature)
        label = f"{_PREFIX}label_{len(label_columns)}"
        analysis[label] = _feature_labels(
            analysis,
            name,
            categorical=name in categorical,
            n_bins=n_bins,
            max_categorical_levels=max_categorical_levels,
        )
        label_columns[name] = label

    total_sample_weight = float(analysis[f"{_PREFIX}sample_weight"].sum())
    ranking_records: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    for feature_a, feature_b in combinations(label_columns, 2):
        label_a = label_columns[feature_a]
        label_b = label_columns[feature_b]
        cells = _aggregate_groups(
            analysis,
            [label_a, label_b],
            governed_gam_weights=governed_gam_weights,
        ).rename(columns={label_a: "level_a", label_b: "level_b"})
        log_ratio_global = _weighted_average(
            analysis[f"{_PREFIX}log_ratio"], analysis[f"{_PREFIX}sample_weight"]
        )
        marginal_a = (
            analysis.groupby(label_a, observed=True)
            .apply(
                lambda part: _weighted_average(
                    part[f"{_PREFIX}log_ratio"], part[f"{_PREFIX}sample_weight"]
                ),
                include_groups=False,
            )
            .to_dict()
        )
        marginal_b = (
            analysis.groupby(label_b, observed=True)
            .apply(
                lambda part: _weighted_average(
                    part[f"{_PREFIX}log_ratio"], part[f"{_PREFIX}sample_weight"]
                ),
                include_groups=False,
            )
            .to_dict()
        )
        cells["interaction_log_ratio"] = [
            np.log(row.geometric_mean_gbm_to_gam)
            - marginal_a[row.level_a]
            - marginal_b[row.level_b]
            + log_ratio_global
            for row in cells.itertuples(index=False)
        ]
        cells["feature_a"] = feature_a
        cells["feature_b"] = feature_b
        credible = cells.loc[
            cells["rows"].ge(min_cell_rows)
            & cells["sample_weight"].ge(total_sample_weight * min_cell_weight_fraction)
        ].copy()
        if credible.empty:
            continue
        interaction_rms = float(
            np.sqrt(
                np.average(
                    np.square(credible["interaction_log_ratio"]),
                    weights=credible["sample_weight"],
                )
            )
        )
        positive_failure = float(
            credible["gbm_minus_gam_deviance_numerator"].clip(lower=0.0).sum() / total_sample_weight
        )
        ranking_records.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "credible_cells": len(credible),
                "interaction_rms_log_ratio": interaction_rms,
                "gbm_failure_contribution": positive_failure,
                "worst_cell_gbm_minus_gam_mean_deviance": float(
                    credible["gbm_minus_gam_mean_deviance"].max()
                ),
            }
        )
        cell_frames.append(credible)
    if not ranking_records:
        raise ScratchDiagnosticError("no feature-pair cells met the configured support thresholds")
    ranking = pd.DataFrame(ranking_records).sort_values(
        ["gbm_failure_contribution", "interaction_rms_log_ratio"],
        ascending=False,
        ignore_index=True,
    )
    cells = pd.concat(cell_frames, ignore_index=True).sort_values(
        "gbm_minus_gam_deviance_numerator",
        ascending=False,
        ignore_index=True,
    )
    return ranking, cells


__all__ = [
    "ScratchDiagnosticError",
    "blend_evaluation_table",
    "double_lift_table",
    "feature_calibration_tables",
    "interaction_failure_tables",
    "lorenz_curve_table",
    "risk_calibration_table",
    "unit_tweedie_deviance",
    "weighted_quantile_bins",
]
