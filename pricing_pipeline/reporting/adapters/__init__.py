"""Model-specific evidence adapters for aggregate reporting."""

from typing import Any

from pricing_pipeline.reporting.adapters.rating_workbook import RatingWorkbookAdapter


def __getattr__(name: str) -> Any:
    if name in {"SuperGLMReportAdapter", "SuppliedTweedieLikelihoodAdapter"}:
        from pricing_pipeline.reporting.adapters.superglm import (
            SuperGLMReportAdapter,
            SuppliedTweedieLikelihoodAdapter,
        )

        superglm_adapters = {
            "SuperGLMReportAdapter": SuperGLMReportAdapter,
            "SuppliedTweedieLikelihoodAdapter": SuppliedTweedieLikelihoodAdapter,
        }
        globals().update(superglm_adapters)
        return superglm_adapters[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RatingWorkbookAdapter",
    "SuperGLMReportAdapter",
    "SuppliedTweedieLikelihoodAdapter",
]
