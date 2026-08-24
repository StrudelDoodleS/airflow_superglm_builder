"""Model-neutral aggregate reporting from already-scored predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pricing_pipeline.reporting._underwriter_html import render_underwriter_html
from pricing_pipeline.reporting._underwriter_movement import (
    _tie_safe_weighted_bins,
    prediction_movement_payload,
)
from pricing_pipeline.reporting.evidence import (
    MAX_MAIN_EFFECT_GRID_POINTS,
    MAX_SURFACE_AXIS_POINTS,
    EvidenceRequest,
    ExactLossEvidence,
    MainEffectEvidence,
    ModelEvidence,
    ReportContext,
    collect_model_evidence,
)

ProblemType = Literal["frequency", "severity", "burn_cost"]
ColumnOrValues = str | Sequence[float] | np.ndarray | pd.Series
ComparisonUnit = str | Sequence[Any] | np.ndarray | pd.Series

_PROBLEM_DEFAULT_POWER: Mapping[str, float] = {
    "frequency": 1.0,
    "severity": 2.0,
    "burn_cost": 1.5,
}
_PROBLEM_SEMANTICS: Mapping[str, Mapping[str, str]] = {
    "frequency": {
        "response": "Claim frequency",
        "prediction": "Predicted claim frequency",
        "volume": "Exposure",
        "curve_x": "Cumulative exposure share",
        "curve_y": "Cumulative claim-count share",
    },
    "severity": {
        "response": "Claim severity",
        "prediction": "Predicted claim severity",
        "volume": "Claim count",
        "curve_x": "Cumulative claim-count share",
        "curve_y": "Cumulative claim-cost share",
    },
    "burn_cost": {
        "response": "Burn cost",
        "prediction": "Predicted burn cost",
        "volume": "Exposure",
        "curve_x": "Cumulative exposure share",
        "curve_y": "Cumulative claim-cost share",
    },
}


class UnderwriterReportError(RuntimeError):
    """Raised when a trustworthy report cannot be produced."""


@dataclass(frozen=True)
class UnderwriterReportOptions:
    """Display and aggregation choices for one report."""

    title: str = "Pricing model review"
    problem_type: ProblemType = "burn_cost"
    tweedie_power: float | None = None
    top_k: int = 12
    double_lift_bins: int = 10
    curve_bins: int = 100
    distribution_bins: int = 200
    movement_bins: int = 10
    relativity_points: int = 200
    interaction_points: int = 80
    comparison_bootstrap_replicates: int = 200
    comparison_bootstrap_seed: int = 1729
    minimum_cell_size: int = 20

    def __post_init__(self) -> None:
        title = self.title.strip()
        if not title:
            raise ValueError("title must be non-empty")
        if self.problem_type not in _PROBLEM_DEFAULT_POWER:
            raise ValueError("problem_type must be frequency, severity, or burn_cost")
        power = self.resolved_tweedie_power
        if self.problem_type == "frequency" and power != 1.0:
            raise ValueError("frequency reports use Poisson deviance (tweedie_power=1)")
        if self.problem_type == "severity" and power != 2.0:
            raise ValueError("severity reports use Gamma deviance (tweedie_power=2)")
        if self.problem_type == "burn_cost" and not 1.0 < power < 2.0:
            raise ValueError("burn_cost tweedie_power must be strictly between 1 and 2")
        for name in (
            "top_k",
            "double_lift_bins",
            "curve_bins",
            "distribution_bins",
            "movement_bins",
            "relativity_points",
            "interaction_points",
        ):
            value = getattr(self, name)
            minimum = 2 if name != "top_k" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
            if name == "relativity_points" and value > MAX_MAIN_EFFECT_GRID_POINTS:
                raise ValueError(f"relativity_points must be at most {MAX_MAIN_EFFECT_GRID_POINTS}")
            if name == "interaction_points" and value > MAX_SURFACE_AXIS_POINTS:
                raise ValueError(f"interaction_points must be at most {MAX_SURFACE_AXIS_POINTS}")
        replicates = self.comparison_bootstrap_replicates
        if (
            not isinstance(replicates, int)
            or isinstance(replicates, bool)
            or (replicates != 0 and replicates < 100)
        ):
            raise ValueError(
                "comparison_bootstrap_replicates must be zero or an integer of at least 100"
            )
        seed = self.comparison_bootstrap_seed
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("comparison_bootstrap_seed must be a non-negative integer")
        if (
            not isinstance(self.minimum_cell_size, int)
            or isinstance(self.minimum_cell_size, bool)
            or self.minimum_cell_size < 2
        ):
            raise ValueError("minimum_cell_size must be an integer of at least 2")

    @property
    def resolved_tweedie_power(self) -> float:
        """Return the explicit power or the problem-type default."""
        if self.tweedie_power is None:
            return _PROBLEM_DEFAULT_POWER[self.problem_type]
        return float(self.tweedie_power)


@dataclass(frozen=True)
class UnderwriterReportResult:
    """Useful aggregate evidence returned alongside the HTML file."""

    output_path: Path
    metrics: pd.DataFrame
    importance: Mapping[str, pd.DataFrame]
    rows_used: int
    zero_weight_rows_ignored: int


@dataclass(frozen=True)
class _ValidatedInputs:
    frame: pd.DataFrame
    actual: np.ndarray
    predictions: Mapping[str, np.ndarray]
    weight: np.ndarray
    features: tuple[str, ...]
    comparison_unit_codes: np.ndarray
    comparison_units: int
    zero_weight_rows_ignored: int
    offset: np.ndarray | None


def _column_or_vector(
    frame: pd.DataFrame,
    value: ColumnOrValues,
    name: str,
) -> np.ndarray:
    if isinstance(value, str):
        if value not in frame.columns:
            raise KeyError(f"{name} column is missing: {value!r}")
        value = frame[value]
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or len(out) != len(frame):
        raise ValueError(f"{name} must be one-dimensional and match the frame")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_inputs(
    frame: pd.DataFrame,
    *,
    actual: ColumnOrValues,
    predictions: Mapping[str, ColumnOrValues],
    sample_weight: ColumnOrValues,
    features: Sequence[str],
    offset: ColumnOrValues | None,
    comparison_unit: ComparisonUnit | None,
    options: UnderwriterReportOptions,
) -> _ValidatedInputs:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    feature_names = tuple(str(feature) for feature in features)
    if not feature_names or any(not name.strip() for name in feature_names):
        raise ValueError("features must contain at least one non-empty column name")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("features must not contain duplicates")
    missing_features = set(feature_names) - set(frame.columns)
    if missing_features:
        raise KeyError("report features are missing: " + ", ".join(sorted(missing_features)))
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions must contain at least one named model")

    y = _column_or_vector(frame, actual, "actual")
    weight = _column_or_vector(frame, sample_weight, "sample_weight")
    if np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("sample_weight must be non-negative with at least one positive value")
    if np.any(y < 0):
        raise ValueError("actual must be non-negative")
    positive = weight > 0
    resolved_offset = None if offset is None else _column_or_vector(frame, offset, "offset")
    if options.resolved_tweedie_power >= 2.0 and np.any(y[positive] <= 0):
        raise ValueError("severity actuals must be strictly positive for Gamma deviance")

    resolved_predictions: dict[str, np.ndarray] = {}
    for raw_name, values in predictions.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("prediction model names must be non-empty")
        if name.lower() in {"__all__", "actual", "none"}:
            raise ValueError(f"prediction model name is reserved by report controls: {name!r}")
        if name in resolved_predictions:
            raise ValueError(f"duplicate prediction model name: {name!r}")
        prediction = _column_or_vector(frame, values, f"prediction {name!r}")
        if np.any(prediction[positive] <= 0):
            raise ValueError(f"prediction {name!r} must contain strictly positive values")
        resolved_predictions[name] = prediction

    ignored = int((~positive).sum())
    if comparison_unit is None:
        unit_values = np.arange(len(frame), dtype=np.int64)[positive]
    else:
        if isinstance(comparison_unit, str):
            if comparison_unit not in frame.columns:
                raise KeyError(f"comparison_unit column is missing: {comparison_unit!r}")
            if comparison_unit in feature_names:
                raise ValueError(
                    "comparison_unit identifier cannot also be a report feature: "
                    f"{comparison_unit!r}"
                )
            comparison_unit = frame[comparison_unit]
        raw_units = np.asarray(comparison_unit, dtype=object)
        if raw_units.ndim != 1 or len(raw_units) != len(frame):
            raise ValueError("comparison_unit must be one-dimensional and match the frame")
        unit_values = raw_units[positive]
        if pd.isna(unit_values).any():
            raise ValueError("comparison_unit must not contain missing values on positive rows")
    try:
        unit_codes, unique_units = pd.factorize(unit_values, sort=False)
    except TypeError as exc:
        raise ValueError("comparison_unit values must be hashable") from exc
    if len(unique_units) < options.minimum_cell_size:
        raise ValueError(
            "aggregate reporting requires at least "
            f"{options.minimum_cell_size} distinct comparison units; got {len(unique_units)}"
        )
    return _ValidatedInputs(
        frame=frame.loc[positive].reset_index(drop=True),
        actual=y[positive],
        predictions={name: values[positive] for name, values in resolved_predictions.items()},
        weight=weight[positive],
        features=feature_names,
        comparison_unit_codes=np.asarray(unit_codes, dtype=np.intp),
        comparison_units=len(unique_units),
        zero_weight_rows_ignored=ignored,
        offset=None if resolved_offset is None else resolved_offset[positive],
    )


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(values, weights=weight))


def _ordered_curve(
    actual: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    *,
    ascending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(score, kind="stable")
    if not ascending:
        order = order[::-1]
    ordered_score = score[order]
    ordered_weight = weight[order]
    ordered_actual = actual[order]
    boundaries = np.r_[True, ordered_score[1:] != ordered_score[:-1]]
    group = np.cumsum(boundaries) - 1
    grouped_weight = np.bincount(group, weights=ordered_weight)
    grouped_actual = np.bincount(group, weights=ordered_weight * ordered_actual)
    x = np.r_[0.0, np.cumsum(grouped_weight) / grouped_weight.sum()]
    total_actual = grouped_actual.sum()
    if total_actual <= 0:
        return x, np.full_like(x, np.nan)
    y = np.r_[0.0, np.cumsum(grouped_actual) / total_actual]
    return x, y


def _gini_statistics(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> tuple[float | None, float | None]:
    x, y = _ordered_curve(actual, prediction, weight, ascending=True)
    if not np.isfinite(y).all():
        return None, None
    raw = float(1.0 - 2.0 * np.trapezoid(y, x))
    perfect_x, perfect_y = _ordered_curve(actual, actual, weight, ascending=True)
    perfect = float(1.0 - 2.0 * np.trapezoid(perfect_y, perfect_x))
    normalized = None if math.isclose(perfect, 0.0, abs_tol=1e-15) else raw / perfect
    return raw, normalized


def _exact_loss_size(weight: np.ndarray, evidence: ExactLossEvidence) -> float:
    return float(len(weight) if evidence.size_basis == "row_count" else weight.sum())


def _metrics_table(
    inputs: _ValidatedInputs,
    *,
    power: float,
    exact_losses: Mapping[str, ExactLossEvidence],
    row_deviance: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    actual_mean = _weighted_mean(inputs.actual, inputs.weight)
    if actual_mean > 0:
        null_prediction = np.full(len(inputs.actual), actual_mean)
        null_deviance: float | None = _weighted_mean(
            _unit_tweedie_deviance(inputs.actual, null_prediction, power),
            inputs.weight,
        )
    else:
        null_deviance = None
    records: list[dict[str, Any]] = []
    for name, prediction in inputs.predictions.items():
        exact_loss = exact_losses.get(name)
        if exact_loss is None:
            exact_mean_nll = None
            likelihood_family = None
            likelihood_power = None
            likelihood_dispersion = None
            likelihood_source = None
        else:
            exact_mean_nll = float(
                np.sum(exact_loss.contributions) / _exact_loss_size(inputs.weight, exact_loss)
            )
            likelihood_family = exact_loss.family
            likelihood_power = exact_loss.tweedie_power
            likelihood_dispersion = exact_loss.dispersion
            likelihood_source = exact_loss.source
        mean_deviance = _weighted_mean(row_deviance[name], inputs.weight)
        gini, normalized_gini = _gini_statistics(inputs.actual, prediction, inputs.weight)
        predicted_mean = _weighted_mean(prediction, inputs.weight)
        records.append(
            {
                "model": name,
                "mean_deviance": mean_deviance,
                "null_deviance": null_deviance,
                "pseudo_r2": (
                    None
                    if null_deviance is None or math.isclose(null_deviance, 0.0, abs_tol=1e-15)
                    else 1.0 - mean_deviance / null_deviance
                ),
                "gini": gini,
                "normalized_gini": normalized_gini,
                "weighted_actual_mean": actual_mean,
                "weighted_prediction_mean": predicted_mean,
                "observed_to_predicted": actual_mean / predicted_mean,
                "exact_mean_nll": exact_mean_nll,
                "likelihood_family": likelihood_family,
                "likelihood_power": likelihood_power,
                "likelihood_dispersion": likelihood_dispersion,
                "likelihood_source": likelihood_source,
            }
        )
    return pd.DataFrame(records)


def _weighted_quantiles(
    values: np.ndarray,
    weight: np.ndarray,
    probabilities: Sequence[float],
) -> list[float]:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ordered_weight = weight[order]
    midpoint = np.cumsum(ordered_weight) - 0.5 * ordered_weight
    positions = midpoint / ordered_weight.sum()
    return [float(np.interp(probability, positions, ordered)) for probability in probabilities]


def _prediction_distributions(
    inputs: _ValidatedInputs,
    *,
    n_bins: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, prediction in inputs.predictions.items():
        x, density, bandwidth, effective_rows = _weighted_gaussian_kde(
            prediction,
            inputs.weight,
            n_points=n_bins,
        )
        result[name] = {
            "x": x.tolist(),
            "density": density.tolist(),
            "bandwidth": bandwidth,
            "effective_rows": effective_rows,
            "quantiles": dict(
                zip(
                    ("p01", "p10", "p50", "p90", "p99"),
                    _weighted_quantiles(
                        prediction,
                        inputs.weight,
                        (0.01, 0.1, 0.5, 0.9, 0.99),
                    ),
                    strict=True,
                )
            ),
        }
    return result


def _weighted_gaussian_kde(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return a smooth, business-weighted Gaussian kernel density estimate.

    The bandwidth is a weighted Silverman rule using the smaller of standard
    deviation and robust IQR scale.  Evaluation is performed from a dense
    weighted grid, avoiding an ``rows x display-points`` allocation for large
    insurance portfolios.  The final curve is normalised to unit area.
    """
    total_weight = float(weight.sum())
    effective_rows = total_weight**2 / float(np.square(weight).sum())
    mean = _weighted_mean(values, weight)
    variance = float(np.average(np.square(values - mean), weights=weight))
    standard_deviation = math.sqrt(max(variance, 0.0))
    q25, q75 = _weighted_quantiles(values, weight, (0.25, 0.75))
    robust_scale = (q75 - q25) / 1.349
    scales = [value for value in (standard_deviation, robust_scale) if value > 0]
    scale = min(scales) if scales else max(abs(mean) * 0.02, 1e-6)
    bandwidth = 0.9 * scale * max(effective_rows, 2.0) ** (-0.2)
    observed_range = float(values.max() - values.min())
    bandwidth = max(
        bandwidth,
        observed_range / max(8 * n_points, 1),
        max(abs(mean), 1.0) * 1e-9,
    )

    lower = max(0.0, float(values.min()) - 3.5 * bandwidth)
    upper = float(values.max()) + 3.5 * bandwidth
    if math.isclose(lower, upper):
        lower = max(0.0, mean - 4.0 * bandwidth)
        upper = mean + 4.0 * bandwidth

    internal_bins = min(4096, max(512, 4 * n_points))
    edges = np.linspace(lower, upper, internal_bins + 1)
    mass, _ = np.histogram(values, bins=edges, weights=weight)
    mass = mass.astype(float) / total_weight
    bin_width = float(edges[1] - edges[0])
    sigma_bins = bandwidth / bin_width
    radius = max(1, min(internal_bins - 1, math.ceil(4.5 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_bins))
    kernel /= kernel.sum()
    full_density = np.convolve(mass, kernel, mode="full")
    smoothed_mass = full_density[radius : radius + internal_bins]
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Keep enough output resolution for a smooth central-range zoom even when
    # a few extreme predictions make the full range much wider.
    x = np.linspace(lower, upper, max(n_points, 1024))
    density = np.interp(x, centres, smoothed_mass / bin_width, left=0.0, right=0.0)
    area = float(np.trapezoid(density, x))
    if area > 0:
        density /= area
    return x, density, float(bandwidth), float(effective_rows)


def _sampled_curve(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    *,
    n_bins: int,
    ascending: bool,
) -> dict[str, list[float]]:
    exact_x, exact_y = _ordered_curve(actual, prediction, weight, ascending=ascending)
    x = np.linspace(0.0, 1.0, n_bins + 1)
    y = (
        np.full_like(x, np.nan)
        if not np.isfinite(exact_y).all()
        else np.interp(x, exact_x, exact_y)
    )
    return {"x": x.tolist(), "y": y.tolist()}


def _curve_payload(
    inputs: _ValidatedInputs,
    *,
    n_bins: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}}
    for name, prediction in inputs.predictions.items():
        gini, normalized_gini = _gini_statistics(inputs.actual, prediction, inputs.weight)
        result["models"][name] = {
            "lorenz": _sampled_curve(
                inputs.actual,
                prediction,
                inputs.weight,
                n_bins=n_bins,
                ascending=True,
            ),
            "gains": _sampled_curve(
                inputs.actual,
                prediction,
                inputs.weight,
                n_bins=n_bins,
                ascending=False,
            ),
            "gini": gini,
            "normalized_gini": normalized_gini,
        }
    perfect_gini, _ = _gini_statistics(inputs.actual, inputs.actual, inputs.weight)
    result["benchmark"] = {
        "lorenz": _sampled_curve(
            inputs.actual,
            inputs.actual,
            inputs.weight,
            n_bins=n_bins,
            ascending=True,
        ),
        "gains": _sampled_curve(
            inputs.actual,
            inputs.actual,
            inputs.weight,
            n_bins=n_bins,
            ascending=False,
        ),
        "gini": perfect_gini,
    }
    return result


def _unit_tweedie_deviance(
    actual: np.ndarray,
    predicted: np.ndarray,
    power: float,
) -> np.ndarray:
    relative_difference = (actual - predicted) / predicted
    if power == 1.0:
        result = np.empty_like(relative_difference)
        positive = actual > 0.0
        delta = relative_difference[positive]
        result[positive] = 2.0 * predicted[positive] * ((1.0 + delta) * np.log1p(delta) - delta)
        result[~positive] = 2.0 * predicted[~positive]
    elif power == 2.0:
        result = 2.0 * (relative_difference - np.log1p(relative_difference))
    elif 1.0 < power < 2.0:
        exponent = 2.0 - power
        delta = relative_difference
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.expm1(exponent * np.log1p(delta)) - exponent * delta
        bracket = numerator / (exponent * (exponent - 1.0))
        near = np.abs(delta) < 1e-4
        if np.any(near):
            local = delta[near]
            bracket[near] = (
                0.5 * np.square(local)
                + (exponent - 2.0) * np.power(local, 3) / 6.0
                + (exponent - 2.0) * (exponent - 3.0) * np.power(local, 4) / 24.0
            )
        result = 2.0 * np.power(predicted, exponent) * bracket
    else:
        raise ValueError("power must be between 1 and 2")
    return np.maximum(np.asarray(result, dtype=float), 0.0)


def _weighted_line_agreement(
    actual: np.ndarray,
    predicted: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    """Return signed weighted Lin concordance and its underwriter-facing score."""
    if np.array_equal(actual, predicted):
        return 1.0, 1.0
    total_weight = float(weight.sum())
    actual_mean = float(weight @ actual / total_weight)
    predicted_mean = float(weight @ predicted / total_weight)
    actual_delta = actual - actual_mean
    predicted_delta = predicted - predicted_mean
    actual_variance = float(weight @ np.square(actual_delta) / total_weight)
    predicted_variance = float(weight @ np.square(predicted_delta) / total_weight)
    covariance = float(weight @ (actual_delta * predicted_delta) / total_weight)
    denominator = actual_variance + predicted_variance + (actual_mean - predicted_mean) ** 2
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return 0.0, 0.0
    signed = float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))
    return signed, float(np.clip(signed, 0.0, 1.0))


def _binned_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    overall_actual: float,
    power: float,
) -> dict[str, dict[str, float | None]]:
    bin_weight = np.asarray([row["weight"] for row in rows], dtype=float)
    bin_actual = np.asarray([row["actual"] for row in rows], dtype=float)
    null_prediction = np.full(len(rows), overall_actual)
    null_deviance = _weighted_mean(
        _unit_tweedie_deviance(bin_actual, null_prediction, power),
        bin_weight,
    )
    result: dict[str, dict[str, float | None]] = {}
    for name in model_names:
        prediction = np.asarray([row["predictions"][name] for row in rows], dtype=float)
        mean_deviance = _weighted_mean(
            _unit_tweedie_deviance(bin_actual, prediction, power),
            bin_weight,
        )
        signed_concordance, line_agreement = _weighted_line_agreement(
            bin_actual,
            prediction,
            bin_weight,
        )
        result[name] = {
            "mean_deviance": mean_deviance,
            "d_squared": (
                None
                if math.isclose(null_deviance, 0.0, abs_tol=1e-15)
                else 1.0 - mean_deviance / null_deviance
            ),
            "signed_concordance": signed_concordance,
            "line_agreement": line_agreement,
        }
    return result


def _privacy_safe_bins(
    values: np.ndarray,
    weight: np.ndarray,
    comparison_unit_codes: np.ndarray,
    *,
    n_bins: int,
    minimum_cell_size: int,
) -> np.ndarray:
    """Return ordered exposure bins with enough distinct comparison units."""
    maximum_bins = max(1, int(np.unique(comparison_unit_codes).size) // minimum_cell_size)
    effective_bins = min(n_bins, maximum_bins)
    if effective_bins == 1:
        return np.zeros(len(values), dtype=int)
    bins = _tie_safe_weighted_bins(values, weight, n_bins=effective_bins)
    while True:
        labels = np.unique(bins)
        cell_units = np.asarray(
            [np.unique(comparison_unit_codes[bins == label]).size for label in labels]
        )
        small = np.flatnonzero(cell_units < minimum_cell_size)
        if not len(small):
            return bins
        index = int(small[0])
        if index == 0:
            target_index = 1
        elif index == len(labels) - 1:
            target_index = index - 1
        else:
            left_weight = float(weight[bins == labels[index - 1]].sum())
            right_weight = float(weight[bins == labels[index + 1]].sum())
            target_index = index - 1 if left_weight <= right_weight else index + 1
        bins[bins == labels[index]] = labels[target_index]
        _, bins = np.unique(bins, return_inverse=True)


def _exact_losses_are_comparable(
    left: ExactLossEvidence | None,
    right: ExactLossEvidence | None,
) -> bool:
    return (
        left is not None
        and right is not None
        and left.comparison_group == right.comparison_group
        and left.size_basis == right.size_basis
        and left.score_label == right.score_label
    )


def _double_lift_payload(
    inputs: _ValidatedInputs,
    *,
    n_bins: int,
    power: float,
    exact_losses: Mapping[str, ExactLossEvidence],
    row_deviance: Mapping[str, np.ndarray],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_cell_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = list(inputs.predictions)
    total_weight = float(inputs.weight.sum())
    mean_deviance = {
        name: float(np.sum(inputs.weight * values) / total_weight)
        for name, values in row_deviance.items()
    }
    mean_exact_nll = {
        name: float(np.sum(exact_loss.contributions) / _exact_loss_size(inputs.weight, exact_loss))
        for name, exact_loss in exact_losses.items()
    }
    overall_actual = _weighted_mean(inputs.actual, inputs.weight)
    bootstrap_deviance: dict[str, np.ndarray] = {}
    bootstrap_exact_nll: dict[str, np.ndarray] = {}
    if bootstrap_replicates:
        unit_count = inputs.comparison_units
        unit_weight = np.bincount(
            inputs.comparison_unit_codes,
            weights=inputs.weight,
            minlength=unit_count,
        )
        deviance_by_unit = {
            name: np.bincount(
                inputs.comparison_unit_codes,
                weights=inputs.weight * values,
                minlength=unit_count,
            )
            for name, values in row_deviance.items()
        }
        exact_by_unit = {
            name: np.bincount(
                inputs.comparison_unit_codes,
                weights=exact_loss.contributions,
                minlength=unit_count,
            )
            for name, exact_loss in exact_losses.items()
        }
        exact_size_by_unit = {
            name: np.bincount(
                inputs.comparison_unit_codes,
                weights=(
                    np.ones(len(inputs.weight), dtype=float)
                    if exact_loss.size_basis == "row_count"
                    else inputs.weight
                ),
                minlength=unit_count,
            )
            for name, exact_loss in exact_losses.items()
        }
        bootstrap_deviance = {name: np.empty(bootstrap_replicates, dtype=float) for name in names}
        bootstrap_exact_nll = {
            name: np.empty(bootstrap_replicates, dtype=float) for name in exact_losses
        }
        generator = np.random.default_rng(bootstrap_seed)
        for replicate in range(bootstrap_replicates):
            sampled = generator.integers(0, unit_count, size=unit_count)
            counts = np.bincount(sampled, minlength=unit_count)
            sampled_weight = float(counts @ unit_weight)
            for name in names:
                bootstrap_deviance[name][replicate] = float(
                    counts @ deviance_by_unit[name] / sampled_weight
                )
            for name in bootstrap_exact_nll:
                sampled_exact_size = float(counts @ exact_size_by_unit[name])
                bootstrap_exact_nll[name][replicate] = float(
                    counts @ exact_by_unit[name] / sampled_exact_size
                )

    def paired_interval(
        numerator_name: str,
        denominator_name: str,
        *,
        exact: bool,
    ) -> tuple[float | None, float | None]:
        if bootstrap_replicates == 0:
            return None, None
        scores = bootstrap_exact_nll if exact else bootstrap_deviance
        advantages = scores[denominator_name] - scores[numerator_name]
        lower, upper = np.quantile(advantages, (0.025, 0.975))
        return float(lower), float(upper)

    for numerator_name in names:
        by_denominator: dict[str, Any] = {}
        for denominator_name in names:
            if numerator_name == denominator_name:
                continue
            numerator = inputs.predictions[numerator_name]
            denominator = inputs.predictions[denominator_name]
            log_ratio = np.log(numerator / denominator)
            bins = _privacy_safe_bins(
                log_ratio,
                inputs.weight,
                inputs.comparison_unit_codes,
                n_bins=n_bins,
                minimum_cell_size=minimum_cell_size,
            )
            numerator_exact = exact_losses.get(numerator_name)
            denominator_exact = exact_losses.get(denominator_name)
            exact_pair = _exact_losses_are_comparable(numerator_exact, denominator_exact)
            exact_denominator = (
                _exact_loss_size(inputs.weight, numerator_exact)
                if exact_pair and numerator_exact is not None
                else None
            )
            analysis = pd.DataFrame(
                {
                    "bin": bins,
                    "weight": inputs.weight,
                    "actual_numerator": inputs.weight * inputs.actual,
                    "log_ratio_numerator": inputs.weight * log_ratio,
                    "deviance_advantage_numerator": inputs.weight
                    * (row_deviance[denominator_name] - row_deviance[numerator_name]),
                    **{
                        f"prediction_{index}": inputs.weight * prediction
                        for index, prediction in enumerate(inputs.predictions.values())
                    },
                    **(
                        {
                            "exact_nll_advantage_numerator": (
                                denominator_exact.contributions - numerator_exact.contributions
                            )
                        }
                        if exact_pair
                        and numerator_exact is not None
                        and denominator_exact is not None
                        else {}
                    ),
                }
            )
            grouped = analysis.groupby("bin", sort=True, observed=True, as_index=False).sum()
            rows: list[dict[str, Any]] = []
            for record in grouped.to_dict("records"):
                bin_weight = float(record["weight"])
                rows.append(
                    {
                        "bin": int(record["bin"]) + 1,
                        "rows": int((bins == int(record["bin"])).sum()),
                        "comparison_units": int(
                            np.unique(inputs.comparison_unit_codes[bins == int(record["bin"])]).size
                        ),
                        "weight": bin_weight,
                        "weight_share": bin_weight / total_weight,
                        "actual": float(record["actual_numerator"]) / bin_weight,
                        "ranking_ratio_geometric_mean": math.exp(
                            float(record["log_ratio_numerator"]) / bin_weight
                        ),
                        "aggregate_prediction_ratio": (
                            float(record[f"prediction_{names.index(numerator_name)}"])
                            / float(record[f"prediction_{names.index(denominator_name)}"])
                        ),
                        "deviance_advantage_contribution": float(
                            record["deviance_advantage_numerator"]
                        )
                        / total_weight,
                        "predictions": {
                            name: float(record[f"prediction_{index}"]) / bin_weight
                            for index, name in enumerate(inputs.predictions)
                        },
                        **(
                            {
                                "exact_nll_advantage_contribution": float(
                                    record["exact_nll_advantage_numerator"]
                                )
                                / float(exact_denominator)
                            }
                            if exact_pair
                            else {}
                        ),
                    }
                )
            advantage = mean_deviance[denominator_name] - mean_deviance[numerator_name]
            denominator_deviance = mean_deviance[denominator_name]
            exact_advantage = (
                mean_exact_nll[denominator_name] - mean_exact_nll[numerator_name]
                if exact_pair
                else None
            )
            numerator_primary_score = (
                mean_exact_nll[numerator_name] if exact_pair else mean_deviance[numerator_name]
            )
            denominator_primary_score = (
                mean_exact_nll[denominator_name] if exact_pair else denominator_deviance
            )
            if numerator_primary_score <= denominator_primary_score:
                lower_score_model = numerator_name
                higher_score_model = denominator_name
                lower_score = numerator_primary_score
                higher_score = denominator_primary_score
            else:
                lower_score_model = denominator_name
                higher_score_model = numerator_name
                lower_score = denominator_primary_score
                higher_score = numerator_primary_score
            relative_score_reduction = (
                None
                if higher_score <= 0.0 or math.isclose(higher_score, 0.0, abs_tol=1e-15)
                else (higher_score - lower_score) / higher_score
            )
            interval_lower, interval_upper = paired_interval(
                numerator_name,
                denominator_name,
                exact=exact_pair,
            )
            if interval_lower is None or interval_upper is None:
                decision = "Interval disabled"
            elif interval_lower > 0.0:
                decision = f"{numerator_name} favoured"
            elif interval_upper < 0.0:
                decision = f"{denominator_name} favoured"
            else:
                decision = "Inconclusive"
            by_denominator[denominator_name] = {
                "bins": rows,
                "comparison": {
                    "primary_score": "exact_nll" if exact_pair else "deviance",
                    "lower_score_model": lower_score_model,
                    "higher_score_model": higher_score_model,
                    "relative_score_reduction": relative_score_reduction,
                    "mean_deviance": {
                        numerator_name: mean_deviance[numerator_name],
                        denominator_name: denominator_deviance,
                    },
                    "deviance_advantage": advantage,
                    "relative_deviance_improvement": (
                        None
                        if math.isclose(denominator_deviance, 0.0, abs_tol=1e-15)
                        else advantage / denominator_deviance
                    ),
                    "mean_exact_nll": (
                        {
                            numerator_name: mean_exact_nll[numerator_name],
                            denominator_name: mean_exact_nll[denominator_name],
                        }
                        if exact_pair
                        else None
                    ),
                    "exact_nll_advantage": exact_advantage,
                    "interval_lower": interval_lower,
                    "interval_upper": interval_upper,
                    "decision": decision,
                    "bootstrap_replicates": bootstrap_replicates,
                    "comparison_units": inputs.comparison_units,
                    "binned_calibration": _binned_calibration(
                        rows,
                        model_names=(numerator_name, denominator_name),
                        overall_actual=overall_actual,
                        power=power,
                    ),
                },
            }
        result[numerator_name] = by_denominator
    return result


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in table.to_dict("records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str | bool) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def _importance_payload(
    evidence: Mapping[str, ModelEvidence],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    columns = ["feature", "magnitude", "share", "effective_df", "method", "source"]
    for model_name, model_evidence in evidence.items():
        if model_evidence.importance is None:
            continue
        table = model_evidence.importance.table.copy(deep=True)
        if "effective_df" not in table:
            table["effective_df"] = None
        result[model_name] = table.loc[:, columns].sort_values(
            ["magnitude", "feature"],
            ascending=[False, True],
            kind="stable",
            ignore_index=True,
        )
    return result


def _main_effect_series(
    main_effect: MainEffectEvidence,
    *,
    minimum_cell_size: int,
) -> dict[str, Any]:
    effect = main_effect.effect.copy(deep=True)
    labels = effect["label"].astype(str).tolist() if "label" in effect else None
    suppressed_levels = 0
    support_basis = None
    if labels is not None:
        support = main_effect.density
        if support is None:
            raise ValueError("categorical main effects require recomputed support")
        safe = support["comparison_units"].ge(minimum_cell_size).to_numpy()
        suppressed_levels = int((~safe).sum())
        effect = effect.loc[safe].reset_index(drop=True)
        support = support.loc[safe].reset_index(drop=True)
        labels = effect["label"].astype(str).tolist()
        x = list(range(len(effect)))
        density = _records(support)
        weight = support["exposure"].astype(float).tolist()
        exposure = {"kind": "bars", "labels": labels, "y": weight}
        support_basis = {
            "privacy": "distinct_comparison_units",
            "exposure": "sample_weight_sum",
        }
    elif main_effect.density is None:
        x = effect["x"].tolist()
        density = None
        weight: list[float | None] = []
        exposure = None
    else:
        x = effect["x"].tolist()
        density = _records(main_effect.density)
        weight = main_effect.density["density"].astype(float).tolist()
        exposure = {
            "kind": "density",
            "x": main_effect.density["x"].tolist(),
            "y": weight,
        }
    suppression = main_effect.suppression
    return {
        "x": x,
        "labels": labels,
        "relativity": effect["value"].tolist(),
        "ci_lower": effect["lower"].tolist() if "lower" in effect else None,
        "ci_upper": effect["upper"].tolist() if "upper" in effect else None,
        "weight": weight,
        "exposure": exposure,
        "density": density,
        "kind": main_effect.semantic.replace("_", " "),
        "semantic": main_effect.semantic,
        "presentation": _main_effect_presentation(main_effect),
        "effective_df": main_effect.effective_df,
        "source": main_effect.source,
        "suppressed_levels": suppressed_levels,
        "support_basis": support_basis,
        "suppression": (
            None
            if suppression is None
            else {
                "status": suppression.status,
                "reason": suppression.reason,
                "presentation": suppression.presentation,
            }
        ),
    }


def _main_effect_presentation(main_effect: MainEffectEvidence) -> dict[str, Any]:
    feature = main_effect.feature
    if main_effect.semantic == "native_component":
        return {
            "title": feature,
            "axis_label": "relativity",
            "reference_value": 1.0,
            "kind_label": "Native fitted component",
            "value_label": "Fitted relativity",
            "note": (
                "Relativities are native fitted effects; exposure is descriptive context "
                "and uses the report sample for fitted objects."
            ),
        }
    if main_effect.semantic == "partial_dependence":
        return {
            "title": f"{feature} · partial dependence",
            "axis_label": "Response prediction",
            "reference_value": None,
            "kind_label": "Partial dependence",
            "value_label": "Response prediction",
            "note": (
                "Partial dependence is a model response prediction, not a fitted "
                "relativity; exposure is descriptive context."
            ),
        }
    if main_effect.semantic == "accumulated_local_effect":
        return {
            "title": f"{feature} · accumulated local effect",
            "axis_label": "Effect",
            "reference_value": 0.0,
            "kind_label": "Accumulated local effect",
            "value_label": "Effect",
            "note": (
                "Accumulated local effect is centered on zero and is not a fitted "
                "relativity; exposure is descriptive context."
            ),
        }
    if main_effect.semantic == "portfolio_aggregate":
        return {
            "title": f"{feature} · empirical portfolio aggregate",
            "axis_label": "Empirical response",
            "reference_value": None,
            "kind_label": "Empirical portfolio aggregate",
            "value_label": "Empirical aggregate",
            "note": (
                "This is an empirical portfolio aggregate from the report sample, not "
                "a fitted model effect; exposure is descriptive context."
            ),
        }
    return {
        "title": f"{feature} · effect",
        "axis_label": "Effect",
        "reference_value": 0.0,
        "kind_label": main_effect.semantic.replace("_", " ").title(),
        "value_label": "Effect",
        "note": "This signed effect is referenced to zero; exposure is descriptive context.",
    }


def _main_effect_payload(
    inputs: _ValidatedInputs,
    evidence: Mapping[str, ModelEvidence],
    *,
    minimum_cell_size: int,
) -> dict[str, dict[str, Any]]:
    by_feature: dict[str, dict[str, Any]] = {}
    for model_name, model_evidence in evidence.items():
        for feature, main_effect in model_evidence.main_effects.items():
            by_feature.setdefault(feature, {})[model_name] = _main_effect_series(
                main_effect,
                minimum_cell_size=minimum_cell_size,
            )
    for feature, by_model in by_feature.items():
        semantics = {series["semantic"] for series in by_model.values()}
        if len(semantics) > 1:
            raise ValueError(
                f"main-effect feature {feature!r} has incompatible semantics across models: "
                + ", ".join(sorted(semantics))
            )
    return {feature: by_feature[feature] for feature in inputs.features if feature in by_feature}


def _interaction_evidence_payload(interaction: Any) -> dict[str, Any]:
    return {
        "name": interaction.name,
        "parents": list(interaction.parents),
        "semantic": interaction.semantic,
        "plot_kind": interaction.plot_kind,
        "source": interaction.source,
        "effect": _records(interaction.effect),
        "grid_axes": {
            str(axis): np.asarray(values).tolist() for axis, values in interaction.grid_axes.items()
        },
        "density": _records(interaction.density) if interaction.density is not None else None,
        "support": _records(interaction.support) if interaction.support is not None else None,
        "default_levels": list(interaction.default_levels),
        "level_diagnostics": (
            _records(interaction.level_diagnostics)
            if interaction.level_diagnostics is not None
            else None
        ),
        "facts": [{"label": fact.label, "value": fact.value} for fact in interaction.facts],
        "warnings": list(interaction.warnings),
    }


def _interaction_payload(evidence: Mapping[str, ModelEvidence]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    unavailable: list[dict[str, str]] = []
    for model_name, model_evidence in evidence.items():
        if model_evidence.interactions:
            models[model_name] = {
                name: _interaction_evidence_payload(interaction)
                for name, interaction in model_evidence.interactions.items()
            }
        unavailable.extend(
            {"model": model_name, "reason": item.reason}
            for item in model_evidence.unavailable
            if item.capability == "interactions"
        )
    return {"models": models, "unavailable": unavailable}


def build_scored_model_report(
    frame: pd.DataFrame,
    *,
    actual: ColumnOrValues,
    predictions: Mapping[str, ColumnOrValues],
    sample_weight: ColumnOrValues,
    features: Sequence[str],
    output_path: str | Path,
    evidence: Mapping[str, ModelEvidence] | None = None,
    evidence_requests: Sequence[EvidenceRequest] = (),
    offset: ColumnOrValues | None = None,
    comparison_unit: ComparisonUnit | None = None,
    options: UnderwriterReportOptions | None = None,
) -> UnderwriterReportResult:
    """Write a self-contained aggregate report from scored model outputs."""
    resolved_options = options or UnderwriterReportOptions()
    inputs = _validate_inputs(
        frame,
        actual=actual,
        predictions=predictions,
        sample_weight=sample_weight,
        features=features,
        offset=offset,
        comparison_unit=comparison_unit,
        options=resolved_options,
    )
    context = ReportContext(
        frame=inputs.frame,
        actual=inputs.actual,
        predictions=inputs.predictions,
        weight=inputs.weight,
        features=inputs.features,
        comparison_unit_codes=inputs.comparison_unit_codes,
        comparison_units=inputs.comparison_units,
        minimum_cell_size=resolved_options.minimum_cell_size,
        problem_type=resolved_options.problem_type,
        deviance_power=resolved_options.resolved_tweedie_power,
        offset=inputs.offset,
    )
    resolved_evidence = collect_model_evidence(
        context,
        evidence or {},
        evidence_requests,
    )
    exact_losses = {
        model_name: model_evidence.exact_loss
        for model_name, model_evidence in resolved_evidence.items()
        if model_evidence.exact_loss is not None
    }
    row_deviance = {
        name: _unit_tweedie_deviance(
            inputs.actual,
            prediction,
            resolved_options.resolved_tweedie_power,
        )
        for name, prediction in inputs.predictions.items()
    }
    metrics = _metrics_table(
        inputs,
        power=resolved_options.resolved_tweedie_power,
        exact_losses=exact_losses,
        row_deviance=row_deviance,
    )
    importance = _importance_payload(resolved_evidence)
    relativities = _main_effect_payload(
        inputs,
        resolved_evidence,
        minimum_cell_size=resolved_options.minimum_cell_size,
    )
    interactions = _interaction_payload(resolved_evidence)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end in .html or .htm")

    payload = {
        "metadata": {
            "title": resolved_options.title,
            "problem_type": resolved_options.problem_type.replace("_", " ").title(),
            "tweedie_power": resolved_options.resolved_tweedie_power,
            "rows_used": len(inputs.frame),
            "zero_weight_rows_ignored": inputs.zero_weight_rows_ignored,
            "total_weight": float(inputs.weight.sum()),
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "top_k": resolved_options.top_k,
            "minimum_cell_size": resolved_options.minimum_cell_size,
            "semantics": dict(_PROBLEM_SEMANTICS[resolved_options.problem_type]),
        },
        "models": list(inputs.predictions),
        "metrics": _records(metrics),
        "importance": {
            model_name: _records(table.head(resolved_options.top_k))
            for model_name, table in importance.items()
        },
        "relativities": _json_safe(relativities),
        "interactions": _json_safe(interactions),
        "distributions": _prediction_distributions(
            inputs,
            n_bins=resolved_options.distribution_bins,
        ),
        "movement": prediction_movement_payload(
            inputs.predictions,
            inputs.weight,
            inputs.comparison_unit_codes,
            n_bins=resolved_options.movement_bins,
            minimum_cell_size=resolved_options.minimum_cell_size,
        ),
        "curves": _curve_payload(inputs, n_bins=resolved_options.curve_bins),
        "double_lift": _double_lift_payload(
            inputs,
            n_bins=resolved_options.double_lift_bins,
            power=resolved_options.resolved_tweedie_power,
            exact_losses=exact_losses,
            row_deviance=row_deviance,
            bootstrap_replicates=resolved_options.comparison_bootstrap_replicates,
            bootstrap_seed=resolved_options.comparison_bootstrap_seed,
            minimum_cell_size=resolved_options.minimum_cell_size,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_underwriter_html(_json_safe(payload)), encoding="utf-8")
    return UnderwriterReportResult(
        output_path=output,
        metrics=metrics,
        importance=importance,
        rows_used=len(inputs.frame),
        zero_weight_rows_ignored=inputs.zero_weight_rows_ignored,
    )


__all__ = [
    "UnderwriterReportError",
    "UnderwriterReportOptions",
    "UnderwriterReportResult",
    "build_scored_model_report",
]
