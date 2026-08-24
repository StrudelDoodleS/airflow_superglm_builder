"""Backward-compatible facade for aggregate model-review reports.

The canonical report builder is model-neutral. This module keeps the original
SuperGLM-oriented entry point and translates its optional inputs into evidence
adapter requests without importing model-specific packages on generic paths.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pricing_pipeline.reporting import _core
from pricing_pipeline.reporting._core import (
    UnderwriterReportError,
    UnderwriterReportOptions,
    UnderwriterReportResult,
    build_scored_model_report,
)
from pricing_pipeline.reporting.evidence import EvidenceRequest

ProblemType = Literal["frequency", "severity", "burn_cost"]
ColumnOrValues = str | Sequence[float] | np.ndarray | pd.Series
ComparisonUnit = str | Sequence[Any] | np.ndarray | pd.Series

# Compatibility re-exports used by established tests and downstream callers.
_unit_tweedie_deviance = _core._unit_tweedie_deviance
_weighted_line_agreement = _core._weighted_line_agreement


@dataclass(frozen=True)
class ModelLikelihoodSpec:
    """Training-fitted distribution metadata for exact holdout scoring."""

    tweedie_power: float
    dispersion: float

    def __post_init__(self) -> None:
        if isinstance(self.tweedie_power, bool):
            raise TypeError("tweedie_power must be numeric, not boolean")
        if isinstance(self.dispersion, bool):
            raise TypeError("dispersion must be numeric, not boolean")
        power = float(self.tweedie_power)
        dispersion = float(self.dispersion)
        if not math.isfinite(power) or not 1.0 <= power <= 2.0:
            raise ValueError("tweedie_power must be finite and between 1 and 2")
        if not math.isfinite(dispersion) or dispersion <= 0.0:
            raise ValueError("dispersion must be finite and strictly positive")
        if power == 1.0 and dispersion != 1.0:
            raise ValueError("Poisson likelihood uses fixed dispersion=1")
        object.__setattr__(self, "tweedie_power", power)
        object.__setattr__(self, "dispersion", dispersion)


def _coerce_model_likelihoods(
    values: Mapping[str, ModelLikelihoodSpec | Mapping[str, Any]],
    *,
    prediction_names: Sequence[str],
    problem_type: ProblemType,
) -> dict[str, ModelLikelihoodSpec]:
    normalized_values = [(str(raw_name).strip(), raw_spec) for raw_name, raw_spec in values.items()]
    unknown = {name for name, _ in normalized_values} - set(prediction_names)
    if unknown:
        raise ValueError(
            "model_likelihoods contains models without predictions: " + ", ".join(sorted(unknown))
        )
    expected_power = {"frequency": 1.0, "severity": 2.0}.get(problem_type)
    resolved: dict[str, ModelLikelihoodSpec] = {}
    for name, raw_spec in normalized_values:
        if isinstance(raw_spec, ModelLikelihoodSpec):
            spec = raw_spec
        elif isinstance(raw_spec, Mapping):
            unknown_fields = set(raw_spec) - {"tweedie_power", "dispersion"}
            missing_fields = {"tweedie_power", "dispersion"} - set(raw_spec)
            if unknown_fields:
                raise ValueError(
                    f"model_likelihoods[{name!r}] has unknown fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            if missing_fields:
                raise ValueError(
                    f"model_likelihoods[{name!r}] is missing fields: "
                    + ", ".join(sorted(missing_fields))
                )
            spec = ModelLikelihoodSpec(
                tweedie_power=raw_spec["tweedie_power"],
                dispersion=raw_spec["dispersion"],
            )
        else:
            raise TypeError(f"model_likelihoods[{name!r}] must be a ModelLikelihoodSpec or mapping")
        if expected_power is not None and spec.tweedie_power != expected_power:
            raise ValueError(f"model_likelihoods[{name!r}] power does not match {problem_type}")
        if problem_type == "burn_cost" and not 1.0 < spec.tweedie_power < 2.0:
            raise ValueError(
                f"model_likelihoods[{name!r}] burn-cost power must be strictly between 1 and 2"
            )
        resolved[name] = spec
    return resolved


def _validate_normalized_model_names(values: Mapping[str, object], label: str) -> None:
    seen: set[str] = set()
    for raw_name in values:
        name = str(raw_name).strip()
        if name in seen:
            raise ValueError(f"{label} contains duplicate normalized model name: {name!r}")
        seen.add(name)


def build_underwriter_report(
    frame: pd.DataFrame,
    *,
    actual: ColumnOrValues,
    predictions: Mapping[str, ColumnOrValues],
    sample_weight: ColumnOrValues,
    features: Sequence[str],
    output_path: str | Path,
    superglm_models: Mapping[str, Any] | None = None,
    rating_workbooks: Mapping[str, str | Path] | None = None,
    model_likelihoods: Mapping[
        str,
        ModelLikelihoodSpec | Mapping[str, Any],
    ]
    | None = None,
    offset: ColumnOrValues | None = None,
    comparison_unit: ComparisonUnit | None = None,
    options: UnderwriterReportOptions | None = None,
) -> UnderwriterReportResult:
    """Build the canonical report with optional legacy SuperGLM evidence.

    Supplied likelihood values are training metadata. For a fitted object they
    validate its fitted likelihood and never replace it. When the same model
    also has a rating workbook, the richer fitted-object evidence takes
    precedence, matching the historical facade behavior.
    """
    resolved_options = options or UnderwriterReportOptions()
    _validate_normalized_model_names(predictions, "predictions")
    _validate_normalized_model_names(superglm_models or {}, "superglm_models")
    _validate_normalized_model_names(rating_workbooks or {}, "rating_workbooks")
    _validate_normalized_model_names(model_likelihoods or {}, "model_likelihoods")
    resolved_likelihoods = _coerce_model_likelihoods(
        model_likelihoods or {},
        prediction_names=tuple(str(name).strip() for name in predictions),
        problem_type=resolved_options.problem_type,
    )
    requests: list[EvidenceRequest] = []
    fitted_models = {
        str(raw_name).strip(): model for raw_name, model in (superglm_models or {}).items()
    }

    if fitted_models:
        from pricing_pipeline.reporting.adapters.superglm import SuperGLMReportAdapter

        for model_name, model in fitted_models.items():
            spec = resolved_likelihoods.get(model_name)
            requests.append(
                EvidenceRequest(
                    model_name,
                    SuperGLMReportAdapter(
                        tweedie_power=None if spec is None else spec.tweedie_power,
                        dispersion=None if spec is None else spec.dispersion,
                        n_points=resolved_options.relativity_points,
                        interaction_points=resolved_options.interaction_points,
                    ),
                    model,
                )
            )

    workbook_sources = {
        str(raw_name).strip(): source
        for raw_name, source in (rating_workbooks or {}).items()
        if str(raw_name).strip() not in fitted_models
    }
    if workbook_sources:
        from pricing_pipeline.reporting.adapters.rating_workbook import RatingWorkbookAdapter

        requests.extend(
            EvidenceRequest(model_name, RatingWorkbookAdapter(), source)
            for model_name, source in workbook_sources.items()
        )

    supplied_only = {
        model_name: spec
        for model_name, spec in resolved_likelihoods.items()
        if model_name not in fitted_models
    }
    if supplied_only:
        from pricing_pipeline.reporting.adapters.superglm import SuppliedTweedieLikelihoodAdapter

        requests.extend(
            EvidenceRequest(
                model_name,
                SuppliedTweedieLikelihoodAdapter(
                    tweedie_power=spec.tweedie_power,
                    dispersion=spec.dispersion,
                ),
                None,
            )
            for model_name, spec in supplied_only.items()
        )

    return build_scored_model_report(
        frame,
        actual=actual,
        predictions=predictions,
        sample_weight=sample_weight,
        features=features,
        output_path=output_path,
        evidence_requests=tuple(requests),
        offset=offset,
        comparison_unit=comparison_unit,
        options=resolved_options,
    )


__all__ = [
    "ModelLikelihoodSpec",
    "UnderwriterReportError",
    "UnderwriterReportOptions",
    "UnderwriterReportResult",
    "build_scored_model_report",
    "build_underwriter_report",
]
