"""Privacy-safe aggregate comparisons of model prediction movement."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import permutations
from typing import Any

import numpy as np


def _tie_safe_weighted_bins(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    """Assign approximate weighted quantile bins without splitting equal values."""
    unique, inverse = np.unique(values, return_inverse=True)
    unique_weight = np.bincount(inverse, weights=weight, minlength=len(unique))
    midpoint = np.cumsum(unique_weight) - 0.5 * unique_weight
    assigned = np.floor(n_bins * midpoint / unique_weight.sum()).astype(int)
    assigned = np.clip(assigned, 0, n_bins - 1)
    _, assigned = np.unique(assigned, return_inverse=True)
    return assigned[inverse]


def _log_level_bins(values: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    logged = np.log(values)
    lower = float(logged.min())
    upper = float(logged.max())
    if math.isclose(lower, upper, abs_tol=1e-12):
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], logged, side="right")
    centres = np.exp((edges[:-1] + edges[1:]) / 2.0)
    return bins.astype(int), centres


def _weighted_quantile(
    values: np.ndarray,
    weight: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weight = weight[order]
    positions = (np.cumsum(ordered_weight) - 0.5 * ordered_weight) / ordered_weight.sum()
    return float(np.interp(probability, positions, ordered_values))


def _weighted_correlation(
    left: np.ndarray,
    right: np.ndarray,
    weight: np.ndarray,
) -> float | None:
    total = float(weight.sum())
    left_mean = float(weight @ left / total)
    right_mean = float(weight @ right / total)
    left_delta = left - left_mean
    right_delta = right - right_mean
    left_variance = float(weight @ np.square(left_delta) / total)
    right_variance = float(weight @ np.square(right_delta) / total)
    denominator = math.sqrt(left_variance * right_variance)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return float(weight @ (left_delta * right_delta) / total / denominator)


def _aggregate_cells(
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    weight: np.ndarray,
    comparison_unit_codes: np.ndarray,
    *,
    x_count: int,
    y_count: int,
    minimum_cell_size: int,
) -> tuple[list[dict[str, int | float]], float]:
    cell_codes = x_bins * y_count + y_bins
    cell_count = x_count * y_count
    rows = np.bincount(cell_codes, minlength=cell_count)
    cell_weight = np.bincount(cell_codes, weights=weight, minlength=cell_count)
    reference_total = np.bincount(
        cell_codes,
        weights=weight * reference,
        minlength=cell_count,
    )
    comparison_total = np.bincount(
        cell_codes,
        weights=weight * comparison,
        minlength=cell_count,
    )
    cell_and_unit = np.unique(
        np.column_stack((cell_codes, comparison_unit_codes)),
        axis=0,
    )
    unit_count = np.bincount(cell_and_unit[:, 0], minlength=cell_count)
    total_weight = float(weight.sum())
    safe: list[dict[str, int | float]] = []
    suppressed_weight = 0.0
    for cell_code in np.flatnonzero(rows):
        resolved_weight = float(cell_weight[cell_code])
        resolved_units = int(unit_count[cell_code])
        if resolved_units < minimum_cell_size:
            suppressed_weight += resolved_weight
            continue
        resolved_reference = float(reference_total[cell_code]) / resolved_weight
        resolved_comparison = float(comparison_total[cell_code]) / resolved_weight
        safe.append(
            {
                "x": int(cell_code // y_count) + 1,
                "y": int(cell_code % y_count) + 1,
                "rows": int(rows[cell_code]),
                "comparison_units": resolved_units,
                "weight": resolved_weight,
                "weight_share": resolved_weight / total_weight,
                "reference_prediction": resolved_reference,
                "comparison_prediction": resolved_comparison,
                "prediction_ratio": resolved_comparison / resolved_reference,
            }
        )
    return safe, suppressed_weight / total_weight


def _compact_level_axes(
    cells: list[dict[str, int | float]],
) -> tuple[list[dict[str, int | float]], list[float], list[float]]:
    """Derive level-axis positions only from privacy-safe aggregate cells."""
    active_x = sorted({int(cell["x"]) for cell in cells})
    active_y = sorted({int(cell["y"]) for cell in cells})
    x_map = {value: index + 1 for index, value in enumerate(active_x)}
    y_map = {value: index + 1 for index, value in enumerate(active_y)}

    def aggregate_axis(axis: str, prediction: str, values: list[int]) -> list[float]:
        return [
            sum(
                float(cell["weight"]) * float(cell[prediction])
                for cell in cells
                if int(cell[axis]) == value
            )
            / sum(float(cell["weight"]) for cell in cells if int(cell[axis]) == value)
            for value in values
        ]

    compact = [{**cell, "x": x_map[int(cell["x"])], "y": y_map[int(cell["y"])]} for cell in cells]
    return (
        compact,
        aggregate_axis("x", "reference_prediction", active_x),
        aggregate_axis("y", "comparison_prediction", active_y),
    )


def _pair_payload(
    reference: np.ndarray,
    comparison: np.ndarray,
    weight: np.ndarray,
    comparison_unit_codes: np.ndarray,
    *,
    n_bins: int,
    minimum_cell_size: int,
) -> dict[str, Any]:
    reference_rank = _tie_safe_weighted_bins(reference, weight, n_bins=n_bins)
    comparison_rank = _tie_safe_weighted_bins(comparison, weight, n_bins=n_bins)
    rank_x_count = int(reference_rank.max()) + 1
    rank_y_count = int(comparison_rank.max()) + 1
    rank_cells, rank_suppressed = _aggregate_cells(
        reference_rank,
        comparison_rank,
        reference,
        comparison,
        weight,
        comparison_unit_codes,
        x_count=rank_x_count,
        y_count=rank_y_count,
        minimum_cell_size=minimum_cell_size,
    )
    reference_level, _reference_centres = _log_level_bins(reference, n_bins=n_bins)
    comparison_level, _comparison_centres = _log_level_bins(comparison, n_bins=n_bins)
    level_cells, level_suppressed = _aggregate_cells(
        reference_level,
        comparison_level,
        reference,
        comparison,
        weight,
        comparison_unit_codes,
        x_count=n_bins,
        y_count=n_bins,
        minimum_cell_size=minimum_cell_size,
    )
    level_cells, safe_reference_centres, safe_comparison_centres = _compact_level_axes(level_cells)
    absolute_change = np.abs(comparison / reference - 1.0)
    total_weight = float(weight.sum())
    return {
        "rank": {
            "x_labels": [str(index) for index in range(1, rank_x_count + 1)],
            "y_labels": [str(index) for index in range(1, rank_y_count + 1)],
            "cells": rank_cells,
            "suppressed_weight_share": rank_suppressed,
        },
        "level": {
            "x_values": safe_reference_centres,
            "y_values": safe_comparison_centres,
            "cells": level_cells,
            "suppressed_weight_share": level_suppressed,
        },
        "summary": {
            "weighted_log_prediction_correlation": _weighted_correlation(
                np.log(reference),
                np.log(comparison),
                weight,
            ),
            "median_absolute_percentage_change": _weighted_quantile(
                absolute_change,
                weight,
                0.5,
            ),
            "p90_absolute_percentage_change": _weighted_quantile(
                absolute_change,
                weight,
                0.9,
            ),
            "weight_share_change_ge_10pct": float(weight[absolute_change >= 0.1].sum())
            / total_weight,
            "weight_share_moved_ge_2_bins": float(
                weight[np.abs(comparison_rank - reference_rank) >= 2].sum()
            )
            / total_weight,
        },
    }


def prediction_movement_payload(
    predictions: Mapping[str, np.ndarray],
    weight: np.ndarray,
    comparison_unit_codes: np.ndarray,
    *,
    n_bins: int,
    minimum_cell_size: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return every directed model pair as aggregate movement evidence."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    if (
        not isinstance(minimum_cell_size, int)
        or isinstance(minimum_cell_size, bool)
        or minimum_cell_size < 1
    ):
        raise ValueError("minimum_cell_size must be a positive integer")
    resolved_weight = np.asarray(weight, dtype=float)
    resolved_codes = np.asarray(comparison_unit_codes)
    if resolved_weight.ndim != 1 or resolved_codes.shape != resolved_weight.shape:
        raise ValueError("weight and comparison_unit_codes must be matching vectors")
    if not np.isfinite(resolved_weight).all() or np.any(resolved_weight <= 0.0):
        raise ValueError("weight must contain finite positive values")
    resolved: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        prediction = np.asarray(values, dtype=float)
        if prediction.shape != resolved_weight.shape:
            raise ValueError(f"prediction {name!r} must match weight")
        if not np.isfinite(prediction).all() or np.any(prediction <= 0.0):
            raise ValueError(f"prediction {name!r} must contain finite positive values")
        resolved[name] = prediction
    result: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in resolved}
    for reference_name, comparison_name in permutations(resolved, 2):
        result[reference_name][comparison_name] = _pair_payload(
            resolved[reference_name],
            resolved[comparison_name],
            resolved_weight,
            resolved_codes,
            n_bins=n_bins,
            minimum_cell_size=minimum_cell_size,
        )
    return result
