"""Evidence adapter for exported SuperGLM rating-table workbooks.

This module intentionally has no dependency on the SuperGLM Python package.
"""

from __future__ import annotations

import math
import os
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
)

_RATING_SHEET = "Rating Tables"
_TERM_ROW = 4
_HEADER_ROW = 6
_DATA_START_ROW = 7
_SOURCE = "rating workbook"
_LEVEL_HEADERS = frozenset({"level", "levels", "category", "categories", "value", "values"})


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
            feature: MainEffectEvidence(
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
            for feature, block in blocks.items()
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
