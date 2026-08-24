"""Aggregate reports for model review."""

from typing import Any

from pricing_pipeline.reporting._core import (
    UnderwriterReportError,
    UnderwriterReportOptions,
    UnderwriterReportResult,
    build_scored_model_report,
)


def __getattr__(name: str) -> Any:
    if name in {"ModelLikelihoodSpec", "build_underwriter_report"}:
        from pricing_pipeline.reporting.underwriter import (
            ModelLikelihoodSpec,
            build_underwriter_report,
        )

        legacy = {
            "ModelLikelihoodSpec": ModelLikelihoodSpec,
            "build_underwriter_report": build_underwriter_report,
        }
        globals().update(legacy)
        return legacy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelLikelihoodSpec",
    "UnderwriterReportError",
    "UnderwriterReportOptions",
    "UnderwriterReportResult",
    "build_scored_model_report",
    "build_underwriter_report",
]
