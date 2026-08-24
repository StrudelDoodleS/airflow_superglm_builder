"""Evidence adapter for exported SuperGLM rating-table workbooks.

This module intentionally has no dependency on the SuperGLM Python package.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pricing_pipeline.reporting._core import UnderwriterReportError
from pricing_pipeline.reporting.evidence import (
    FeatureImportanceEvidence,
    MainEffectEvidence,
    ModelEvidence,
    ReportContext,
    SuppressionMetadata,
)

_RATING_SHEET = "Rating Tables"
_TERM_ROW = 4
_HEADER_ROW = 6
_DATA_START_ROW = 7
_SOURCE = "rating workbook"
_LEVEL_HEADERS = frozenset({"level", "levels", "category", "categories", "value", "values"})
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_RE = re.compile(rf"^\s*\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\)\s*$")


class RatingWorkbookAdapter:
    """Translate one exported rating workbook into neutral model evidence."""

    def collect(
        self,
        *,
        model_name: str,
        source: object,
        context: ReportContext,
    ) -> ModelEvidence:
        del model_name
        if not isinstance(source, (str, os.PathLike)):
            raise TypeError("rating workbook source must be path-like")
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"rating workbook does not exist: {path}")
        allowed = set(context.features)
        blocks = {
            feature: values
            for feature, values in _workbook_blocks(path).items()
            if feature in allowed
        }
        main_effects = {
            feature: _main_effect(feature, block, context) for feature, block in blocks.items()
        }
        return ModelEvidence(
            source=_SOURCE,
            importance=FeatureImportanceEvidence(
                table=_workbook_importance(blocks),
                method="export_log_relativity_variance",
                source=_SOURCE,
            ),
            main_effects=main_effects,
        )


def _workbook_blocks(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = pd.read_excel(path, sheet_name=_RATING_SHEET, header=None, engine="openpyxl")
    except Exception as exc:
        raise UnderwriterReportError(f"could not read rating workbook: {path}") from exc

    blocks: dict[str, dict[str, Any]] = {}
    normalized_names: set[str] = set()
    for column in range(max(raw.shape[1] - 2, 0)):
        title = raw.iat[_TERM_ROW, column] if _TERM_ROW < raw.shape[0] else None
        headers = (
            [raw.iat[_HEADER_ROW, column + offset] for offset in range(3)]
            if _HEADER_ROW < raw.shape[0]
            else [None, None, None]
        )
        if pd.isna(title) or any(pd.isna(value) for value in headers):
            continue
        normalized = [str(value).strip().lower() for value in headers]
        if "relativity" not in normalized[1] or "weight" not in normalized[2]:
            continue
        name = str(title).strip()
        normalized_name = _normalize_label(name)
        if normalized[0] not in _LEVEL_HEADERS and _normalize_label(headers[0]) != normalized_name:
            raise UnderwriterReportError(
                f"rating workbook term {name!r} has an ambiguous level header"
            )
        if normalized_name in normalized_names:
            raise UnderwriterReportError(f"rating workbook contains duplicate term {name!r}")
        normalized_names.add(normalized_name)
        levels: list[str] = []
        relativities: list[float] = []
        weights: list[float] = []
        for row in range(_DATA_START_ROW, raw.shape[0]):
            level = raw.iat[row, column]
            relativity = raw.iat[row, column + 1]
            weight = raw.iat[row, column + 2]
            if pd.isna(level) and pd.isna(relativity):
                if levels:
                    break
                continue
            if pd.isna(level) or pd.isna(relativity):
                break
            try:
                resolved_relativity = float(relativity)
                resolved_weight = 1.0 if pd.isna(weight) else float(weight)
            except (TypeError, ValueError) as exc:
                raise UnderwriterReportError(
                    f"rating workbook term {name!r} contains non-numeric relativity/weight"
                ) from exc
            if not math.isfinite(resolved_relativity) or resolved_relativity <= 0.0:
                raise UnderwriterReportError(
                    f"rating workbook term {name!r} contains an invalid relativity"
                )
            if not math.isfinite(resolved_weight) or resolved_weight < 0.0:
                raise UnderwriterReportError(
                    f"rating workbook term {name!r} contains an invalid weight"
                )
            levels.append(str(level).strip())
            relativities.append(resolved_relativity)
            weights.append(resolved_weight)
        if not levels:
            continue
        if not any(weights):
            weights = [1.0] * len(weights)
        blocks[name] = {
            "labels": levels,
            "relativity": relativities,
            "weight": weights,
        }
    if not blocks:
        raise UnderwriterReportError(f"no main-effect blocks found on {_RATING_SHEET!r} in {path}")
    return blocks


def _normalize_label(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def _main_effect(
    feature: str,
    block: dict[str, Any],
    context: ReportContext,
) -> MainEffectEvidence:
    numeric_values = _numeric_context_values(context.frame[feature])
    intervals = (
        _continuous_intervals(feature, block["labels"]) if numeric_values is not None else None
    )
    if intervals is None:
        return MainEffectEvidence(
            feature=feature,
            semantic="native_component",
            effect=pd.DataFrame(
                {
                    "label": block["labels"],
                    "value": block["relativity"],
                }
            ),
            source=_SOURCE,
        )

    codes = np.asarray(context.comparison_unit_codes)
    weights = np.asarray(context.weight, dtype=float)
    coordinates = [lower + (upper - lower) / 2.0 for lower, upper in intervals]
    masks = [
        _interval_membership(numeric_values, intervals, index) for index in range(len(intervals))
    ]
    safe = [len(np.unique(codes[mask])) >= context.minimum_cell_size for mask in masks]
    if not all(safe):
        return MainEffectEvidence(
            feature=feature,
            semantic="native_component",
            effect=pd.DataFrame({"x": [], "value": []}, dtype=float),
            source=_SOURCE,
            suppression=SuppressionMetadata(
                status="partial" if any(safe) else "all",
                reason="minimum_support",
                presentation="curve_omitted",
            ),
        )
    exposures = [float(weights[mask].sum()) for mask in masks]
    relativities = [
        float(relativity)
        for (_lower, _upper), relativity in zip(
            intervals,
            block["relativity"],
            strict=True,
        )
    ]
    return MainEffectEvidence(
        feature=feature,
        semantic="native_component",
        effect=pd.DataFrame({"x": coordinates, "value": relativities}),
        source=_SOURCE,
        density=pd.DataFrame({"x": coordinates, "density": exposures}),
    )


def _interval_membership(
    values: np.ndarray,
    intervals: list[tuple[float, float]],
    index: int,
) -> np.ndarray:
    lower, upper = intervals[index]
    if len(intervals) == 1:
        return np.isfinite(values)
    if index == 0:
        return values < upper
    if index == len(intervals) - 1:
        return values >= lower
    return (values >= lower) & (values < upper)


def _numeric_context_values(values: pd.Series) -> np.ndarray | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric[values.notna()].isna().any():
        return None
    return numeric.to_numpy(dtype=float)


def _continuous_intervals(
    feature: str,
    labels: list[str],
) -> list[tuple[float, float]] | None:
    matches = [_INTERVAL_RE.fullmatch(label) for label in labels]
    if not any(matches):
        return None
    if not all(matches):
        raise UnderwriterReportError(
            f"rating workbook term {feature!r} mixes interval and categorical levels"
        )
    intervals = [(float(match.group(1)), float(match.group(2))) for match in matches if match]
    for index, (lower, upper) in enumerate(intervals):
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise UnderwriterReportError(
                f"rating workbook term {feature!r} contains an invalid interval"
            )
        if index and lower != intervals[index - 1][1]:
            raise UnderwriterReportError(
                f"rating workbook term {feature!r} contains non-contiguous intervals"
            )
    return intervals


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(values, weights=weight))


def _workbook_importance(blocks: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature, block in blocks.items():
        log_relativity = np.log(np.asarray(block["relativity"], dtype=float))
        weight = np.asarray(block["weight"], dtype=float)
        mean = _weighted_mean(log_relativity, weight)
        variance = _weighted_mean(np.square(log_relativity - mean), weight)
        records.append({"feature": feature, "magnitude": variance})
    if not records:
        return pd.DataFrame(columns=["feature", "magnitude"])
    return pd.DataFrame(records).sort_values("magnitude", ascending=False, ignore_index=True)


__all__ = ["RatingWorkbookAdapter"]
