from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from pricing_pipeline.publishing.lifecycle import PredictionComparison


def compare_prediction_vectors(
    before: pd.Series,
    after: pd.Series,
    *,
    top_n: int = 25,
) -> PredictionComparison:
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")
    if len(before) != len(after):
        raise ValueError("before and after predictions must have the same length")
    if len(before) == 0:
        raise ValueError("before and after predictions must not be empty")

    before_values = _numeric_prediction_series(before, "before")
    after_values = _numeric_prediction_series(after, "after")

    with np.errstate(over="ignore", invalid="ignore"):
        absolute_change = (after_values - before_values).abs()
    relative_change = _relative_change(absolute_change, before_values)
    if not _series_is_finite(absolute_change) or not _series_is_finite(relative_change):
        raise ValueError("prediction changes must be finite")

    changed_rows = pd.DataFrame(
        {
            "row_index": range(len(before_values)),
            "before": before_values,
            "after": after_values,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
        },
    ).sort_values(["absolute_change", "row_index"], ascending=[False, True])

    summary = {
        "row_count": float(len(before_values)),
        "mean_absolute_change": _mean(absolute_change),
        "max_absolute_change": float(absolute_change.max()),
        "mean_relative_change": _mean(relative_change),
        "max_relative_change": float(relative_change.max()),
    }
    if not np.isfinite(list(summary.values())).all():
        raise ValueError("prediction summary values must be finite")

    return PredictionComparison(
        summary=summary,
        changed_rows=changed_rows.head(int(top_n)).reset_index(drop=True),
    )


def _numeric_prediction_series(values: pd.Series, label: str) -> pd.Series:
    try:
        numeric_values = pd.to_numeric(values, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} predictions must be numeric") from exc

    numeric_values = numeric_values.reset_index(drop=True)
    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError(f"{label} predictions must contain only finite values")
    return numeric_values


def _relative_change(absolute_change: pd.Series, before_values: pd.Series) -> pd.Series:
    baseline = before_values.abs()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_values = np.divide(
            absolute_change.to_numpy(),
            baseline.to_numpy(),
            out=np.zeros(len(before_values), dtype="float64"),
            where=baseline.to_numpy() != 0.0,
        )
    return pd.Series(relative_values, index=before_values.index)


def _series_is_finite(values: pd.Series) -> bool:
    return bool(np.isfinite(values.to_numpy()).all())


def _mean(values: pd.Series) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean(values.to_numpy()))
