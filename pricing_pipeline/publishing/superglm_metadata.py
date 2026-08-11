from __future__ import annotations

import math
import re
from collections.abc import Mapping
from importlib.metadata import version as package_version
from typing import Any

import numpy as np
import pandas as pd
from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping
from superglm.features.interaction import CategoricalInteraction
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import (
    BSplineSmooth,
    CardinalCRSpline,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
    _SplineBase,
)
from superglm.types import LambdaPolicy

from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
)

EXTRACTOR_VERSION = "4"

_SUPERGLM_VERSION = package_version("superglm")
_SPLINE_KIND_BY_CLASS = {
    PSpline: "ps",
    BSplineSmooth: "bs",
    NaturalSpline: "ns",
    CubicRegressionSpline: "cr",
    CardinalCRSpline: "cr_cardinal",
}
_KNOT_ALPHA_STRATEGIES = {"quantile_tempered"}


def _json_value(value: Any) -> Any:
    """Convert the concrete metadata values emitted by pinned SuperGLM to JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite value in SuperGLM metadata")
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite value in SuperGLM metadata")
        return numeric
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, pd.Series | pd.Index):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("SuperGLM metadata mapping keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise ValueError(f"unsupported SuperGLM metadata value: {type(value).__name__}")


def _spline_kind(spec: _SplineBase) -> str:
    for klass, kind in _SPLINE_KIND_BY_CLASS.items():
        if isinstance(spec, klass):
            return kind
    raise ValueError(f"unsupported SuperGLM spline type: {type(spec).__name__}")


def _grouping_metadata(grouping: LevelGrouping | None) -> dict[str, Any] | None:
    if grouping is None:
        return None
    if not isinstance(grouping, LevelGrouping):
        raise ValueError("SuperGLM categorical grouping must be a LevelGrouping")
    return {
        "class_name": "LevelGrouping",
        "original_to_group": grouping.original_to_group,
        "group_to_originals": grouping.group_to_originals,
        "all_original_levels": grouping.all_original_levels,
        "grouped_levels": grouping.grouped_levels,
    }


def _lambda_policy_metadata(
    policy: LambdaPolicy | Mapping[str, LambdaPolicy] | None,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    if isinstance(policy, LambdaPolicy):
        return {"mode": policy.mode, "value": policy.value}
    if not isinstance(policy, Mapping):
        raise ValueError("SuperGLM spline lambda_policy has an unsupported shape")

    policies: dict[str, Any] = {}
    for component_name, component_policy in policy.items():
        if not isinstance(component_name, str) or not isinstance(component_policy, LambdaPolicy):
            raise ValueError("SuperGLM spline lambda_policy must map names to LambdaPolicy")
        policies[component_name] = {
            "mode": component_policy.mode,
            "value": component_policy.value,
        }
    return policies


def _base_feature_metadata(name: str, spec: Any, feature_kind: str) -> dict[str, Any]:
    return {
        "feature_kind": feature_kind,
        "superglm_class": type(spec).__name__,
        "source_term_name": name,
        "published_term_name": clean_identifier(name),
    }


def _categorical_metadata(name: str, spec: Categorical) -> dict[str, Any]:
    metadata = _base_feature_metadata(name, spec, "categorical")
    metadata.update(
        {
            "declared": {
                "base": spec.base,
                "grouping": _grouping_metadata(spec._grouping),
                "levels": spec._declared_levels,
                "unseen": spec.unseen,
            },
            "effective": {
                "level_source": spec._level_source,
                "pinned_levels": spec._pinned_levels,
                "base_fallback": spec._base_fallback,
            },
            "fitted": {
                "levels": spec._levels,
                "base_level": spec._base_level,
                "non_base_levels": spec._non_base,
            },
        }
    )
    return metadata


def _spline_metadata(name: str, spec: _SplineBase) -> dict[str, Any]:
    kind = _spline_kind(spec)
    if spec._R_inv is not None and (
        not isinstance(spec._R_inv, np.ndarray) or spec._R_inv.ndim != 2
    ):
        raise ValueError(f"SuperGLM term {name!r} has malformed fitted coefficients")
    coefficient_width = None if spec._R_inv is None else int(spec._R_inv.shape[1])
    declared = {
        "kind": kind,
        "n_knots": spec.n_knots,
        "spline_degree": spec.degree,
        "knot_strategy": spec.knot_strategy,
        "penalty": spec.penalty,
        "select": spec.select,
        "extrapolation": spec.extrapolation,
        "constraint_kind": spec.constraint_kind,
        "constraint_mode": spec.constraint_mode if spec.constraint_kind else None,
        "m": spec._m_orders,
        "knots": spec._explicit_knots,
        "boundary": spec._explicit_boundary,
        "lambda_policy": _lambda_policy_metadata(spec._lambda_policy),
    }
    if spec.knot_strategy in _KNOT_ALPHA_STRATEGIES:
        declared["knot_alpha"] = spec.knot_alpha

    metadata = _base_feature_metadata(name, spec, "spline")
    metadata.update(
        {
            "declared": declared,
            "effective": {
                "kind": kind,
                "class_name": type(spec).__name__,
                "n_knots": spec.n_knots,
                "knot_strategy_actual": spec._knot_strategy_actual,
            },
            "fitted": {
                "class_name": type(spec).__name__,
                "boundary": spec.fitted_boundary,
                "knots": spec.fitted_knots,
                "raw_basis_count": int(spec._n_basis),
                "coefficient_width": coefficient_width,
                "lower_bound": spec._lo,
                "upper_bound": spec._hi,
            },
        }
    )
    return metadata


def _ordered_categorical_metadata(name: str, spec: OrderedCategorical) -> dict[str, Any]:
    spline = spec._spline
    configured_spline = spec._spline_obj
    if not isinstance(configured_spline, _SplineBase):
        raise TypeError(
            f"SuperGLM ordered categorical {name!r} uses unsupported publication "
            f"basis {type(configured_spline).__name__}"
        )
    if not isinstance(spline, _SplineBase):
        raise TypeError(f"SuperGLM ordered categorical {name!r} has no fitted spline")
    spline_metadata = _spline_metadata(name, spline)
    spline_width = spline_metadata["fitted"]["coefficient_width"]
    if spline_width is None:
        raise ValueError(f"SuperGLM ordered categorical {name!r} has no fitted coefficients")
    special_width = int(spec._n_special_cols)

    metadata = _base_feature_metadata(name, spec, "ordered_categorical")
    metadata.update(
        {
            "declared": {
                "basis": spec.basis_kind,
                "kind": _spline_kind(configured_spline),
                "base": spec.base,
                "ordered_levels": spec._ordered_levels,
                "level_values": spec._original_level_to_value or spec._level_to_value,
                "n_knots_requested": configured_spline.n_knots,
                "degree": configured_spline.degree,
                "penalty": configured_spline.penalty,
                "select": configured_spline.select,
                "grouping": _grouping_metadata(spec._grouping),
            },
            "effective": {
                "basis": spec.basis_kind,
                "kind": _spline_kind(spline),
                "n_knots_effective": spline.n_knots,
                "n_levels": spec._n_levels,
                "ordered_levels": spec._ordered_levels,
                "level_values": spec._level_to_value,
                "base_level": spec._base_level,
                "non_base_levels": spec._non_base,
            },
            "fitted": {
                "levels": spec._ordered_levels,
                "base_level": spec._base_level,
                "non_base_levels": spec._non_base,
                "coefficient_width": spline_width + special_width,
                "spline_coefficient_width": spline_width,
                "special_coefficient_width": special_width,
            },
            "spline": spline_metadata,
        }
    )
    return metadata


def _polynomial_metadata(name: str, spec: Polynomial) -> dict[str, Any]:
    metadata = _base_feature_metadata(name, spec, "polynomial")
    metadata.update(
        {
            "declared": {"degree": spec.degree},
            "effective": {"encoding": "polynomial", "degree": spec.degree},
            "fitted": {"lower_bound": spec._lo, "upper_bound": spec._hi},
        }
    )
    return metadata


def _numeric_metadata(name: str, spec: Numeric) -> dict[str, Any]:
    metadata = _base_feature_metadata(name, spec, "numeric")
    metadata.update(
        {
            "declared": {},
            "effective": {"encoding": "identity"},
            "fitted": {},
        }
    )
    return metadata


def _feature_metadata(name: str, spec: Any) -> dict[str, Any]:
    if isinstance(spec, OrderedCategorical):
        return _ordered_categorical_metadata(name, spec)
    if isinstance(spec, Categorical):
        return _categorical_metadata(name, spec)
    if isinstance(spec, _SplineBase):
        return _spline_metadata(name, spec)
    if isinstance(spec, Polynomial):
        return _polynomial_metadata(name, spec)
    if isinstance(spec, Numeric):
        return _numeric_metadata(name, spec)
    raise ValueError(f"unsupported feature {name!r}: {type(spec).__name__}")


def _categorical_interaction_metadata(
    name: str,
    spec: Any,
    *,
    published_name: str,
    published_by_source: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(spec, CategoricalInteraction):
        raise ValueError(
            f"interaction {name!r} uses unsupported {type(spec).__name__}; "
            "only two-way categorical interactions can be published"
        )
    parent_names = tuple(spec.parent_names)
    if len(parent_names) != 2 or any(
        not isinstance(parent, str) or not parent for parent in parent_names
    ):
        raise ValueError(f"interaction {name!r} must have exactly two categorical parent features")
    missing_parents = [parent for parent in parent_names if parent not in published_by_source]
    if missing_parents:
        raise ValueError(
            f"interaction {name!r} references unpublished parent feature(s): "
            + ", ".join(missing_parents)
        )
    return {
        "feature_kind": "categorical_interaction",
        "superglm_class": type(spec).__name__,
        "source_term_name": name,
        "published_term_name": published_name,
        "parent_names": list(parent_names),
        "input_column_names": [published_by_source[parent] for parent in parent_names],
        "interaction_order": 2,
        "declared": {},
        "effective": {"encoding": "categorical_cross_product"},
        "fitted": {},
    }


def _offset_metadata(offset_contract: OffsetExportContract) -> dict[str, Any]:
    if (
        offset_contract.handling != "EXPORTED_FACTOR"
        or offset_contract.source_factor_name is None
        or offset_contract.published_factor_name is None
        or offset_contract.source_name is None
        or offset_contract.label is None
    ):
        raise ValueError("offset term metadata requires an EXPORTED_FACTOR offset contract")
    return {
        "feature_kind": "offset",
        "superglm_class": "Offset",
        "source_term_name": offset_contract.source_factor_name,
        "published_term_name": offset_contract.published_factor_name,
        "offset_handling": offset_contract.handling,
        "fixed_log_coefficient": 1.0,
        "coefficient_source": "offset",
        "offset_factor_name": offset_contract.published_factor_name,
        "offset_source_name": offset_contract.source_name,
        "offset_label": offset_contract.label,
        "declared": {
            "source_name": offset_contract.source_name,
            "label": offset_contract.label,
        },
        "effective": {"encoding": "fixed_log_coefficient", "coefficient": 1.0},
        "fitted": {},
    }


def _ordered_items(
    specs: Any,
    order: Any,
    *,
    label: str,
) -> list[tuple[str, Any]]:
    if not isinstance(specs, dict) or not isinstance(order, list):
        raise ValueError(f"SuperGLM {label} metadata has an invalid fitted shape")
    if any(not isinstance(name, str) or not name.strip() for name in order):
        raise ValueError(f"SuperGLM {label} order must contain non-empty strings")
    if len(order) != len(specs) or len(set(order)) != len(order) or set(order) != set(specs):
        raise ValueError(f"SuperGLM {label} order does not match its fitted specs")
    return [(name, specs[name]) for name in order]


def _snake_case_name(value: str) -> str:
    step_one = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step_one).lower()


def _validate_offset_contract(fit_used_offset: bool, offset_contract: OffsetExportContract) -> None:
    if fit_used_offset and offset_contract.handling == "NONE":
        raise ValueError(
            "offset contract handling must describe an exported or pre-applied offset "
            "when the SuperGLM model was fit with an offset"
        )
    if not fit_used_offset and offset_contract.handling != "NONE":
        raise ValueError(
            "offset contract handling must be NONE when the SuperGLM model was fit without an offset"
        )


def build_superglm_publication_receipt(
    model: SuperGLM,
    *,
    offset_contract: OffsetExportContract,
    fit_sample_weight_name: str | None = None,
    export_weight_name: str | None = None,
) -> SuperGLMPublicationReceipt:
    if not isinstance(model, SuperGLM):
        raise TypeError("publication metadata requires a SuperGLM model")
    if model._result is None:
        raise ValueError("SuperGLM model must be fitted before publication")
    if type(model._fit_used_offset) is not bool:
        raise ValueError("SuperGLM model has malformed fitted offset metadata")

    model_specs = _ordered_items(model._specs, model._feature_order, label="feature")
    if not model_specs:
        raise ValueError("SuperGLM model has no feature specs to publish")
    interaction_specs = _ordered_items(
        model._interaction_specs,
        model._interaction_order,
        label="interaction",
    )

    fit_used_offset = model._fit_used_offset
    _validate_offset_contract(fit_used_offset, offset_contract)

    term_metadata: dict[str, dict[str, Any]] = {}
    published_sources: dict[str, str] = {}
    published_by_source: dict[str, str] = {}
    for source_name, spec in model_specs:
        metadata = _feature_metadata(source_name, spec)
        published_name = metadata["published_term_name"]
        if published_name in published_sources:
            first_source = published_sources[published_name]
            raise ValueError(
                "canonical term name collision: "
                f"{published_name!r} from {first_source!r} and {source_name!r}"
            )
        published_sources[published_name] = source_name
        published_by_source[source_name] = published_name
        term_metadata[published_name] = _json_value(metadata)

    for source_name, spec in interaction_specs:
        published_name = clean_identifier(source_name)
        if published_name in published_sources:
            first_source = published_sources[published_name]
            raise ValueError(
                "canonical term name collision: "
                f"{published_name!r} from {first_source!r} and {source_name!r}"
            )
        metadata = _categorical_interaction_metadata(
            source_name,
            spec,
            published_name=published_name,
            published_by_source=published_by_source,
        )
        published_sources[published_name] = source_name
        published_by_source[source_name] = published_name
        term_metadata[published_name] = _json_value(metadata)

    if offset_contract.handling == "EXPORTED_FACTOR":
        offset_published_name = str(offset_contract.published_factor_name)
        if offset_published_name in published_sources:
            first_source = published_sources[offset_published_name]
            raise ValueError(
                "canonical term name collision: "
                f"{offset_published_name!r} from {first_source!r} and offset contract"
            )
        term_metadata[offset_published_name] = _json_value(_offset_metadata(offset_contract))

    distribution = model._distribution
    link = model._link
    if distribution is None or link is None:
        raise ValueError("SuperGLM model has incomplete fitted family metadata")
    family_name = _snake_case_name(type(distribution).__name__)
    family_params = {
        str(name): value
        for name, value in vars(distribution).items()
        if not str(name).startswith("_")
    }
    link_class_name = type(link).__name__
    link_name = _snake_case_name(link_class_name.removesuffix("Link"))
    if not family_name or not link_name:
        raise ValueError("SuperGLM model has malformed fitted family metadata")

    package_metadata = {
        "model": {
            "family": family_name,
            "family_params": family_params,
            "link": link_name,
            "fit_used_offset": fit_used_offset,
            "fit_sample_weight_used": fit_sample_weight_name is not None,
            "fit_sample_weight_name": fit_sample_weight_name,
            "export_weight_used": export_weight_name is not None,
            "export_weight_name": export_weight_name,
        }
    }

    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version=_SUPERGLM_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        package_metadata=_json_value(package_metadata),
        term_metadata=term_metadata,
        offset_contract=offset_contract,
    )
