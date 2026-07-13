from __future__ import annotations

import math
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import numpy as np
import pandas as pd
from superglm.features.categorical import Categorical
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

from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
)

EXTRACTOR_VERSION = "2"

_SPLINE_KIND_BY_CLASS = {
    PSpline: "ps",
    BSplineSmooth: "bs",
    NaturalSpline: "ns",
    CubicRegressionSpline: "cr",
    CardinalCRSpline: "cr_cardinal",
}
_KNOT_ALPHA_STRATEGIES = {"quantile_tempered"}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite value in SuperGLM metadata")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite value in SuperGLM metadata")
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, pd.Series | pd.Index):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("SuperGLM metadata mapping keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_value(item) for item in sorted(value, key=repr)]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except TypeError:
            pass

    try:
        if pd.isna(value):
            return None
    except TypeError, ValueError:
        pass

    raise ValueError(f"unsupported SuperGLM metadata value: {type(value).__name__}")


def _spline_kind(spec: Any) -> str:
    for klass, kind in _SPLINE_KIND_BY_CLASS.items():
        if isinstance(spec, klass):
            return kind
    return type(spec).__name__


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _shape_width(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _grouping_metadata(grouping: Any) -> Any:
    if grouping is None:
        return None
    if isinstance(grouping, Mapping):
        return _json_value(grouping)

    level_grouping_attrs = (
        "original_to_group",
        "group_to_originals",
        "all_original_levels",
        "grouped_levels",
    )
    level_grouping_metadata = {"class_name": type(grouping).__name__}
    for attr_name in level_grouping_attrs:
        candidate = _safe_getattr(grouping, attr_name)
        if candidate is not None:
            level_grouping_metadata[attr_name] = _json_value(candidate)
    if len(level_grouping_metadata) > 1:
        return level_grouping_metadata

    for attr_name in ("mapping", "groups", "grouping", "_mapping", "_groups"):
        candidate = _safe_getattr(grouping, attr_name)
        if isinstance(candidate, Mapping):
            return _json_value(candidate)

    for method_name in ("to_dict", "as_dict"):
        method = _safe_getattr(grouping, method_name)
        if callable(method):
            try:
                candidate = method()
            except Exception:
                continue
            if isinstance(candidate, Mapping):
                return _json_value(candidate)

    return {"class_name": type(grouping).__name__}


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
                "base": _safe_getattr(spec, "base"),
                "grouping": _grouping_metadata(_safe_getattr(spec, "_grouping")),
            },
            "effective": {},
            "fitted": {
                "levels": _safe_getattr(spec, "_levels"),
                "base_level": _safe_getattr(spec, "_base_level"),
                "non_base_levels": _safe_getattr(spec, "_non_base"),
            },
        }
    )
    return metadata


def _spline_metadata(name: str, spec: _SplineBase) -> dict[str, Any]:
    r_inv = _safe_getattr(spec, "_R_inv")
    constraint_kind = _safe_getattr(spec, "constraint_kind")
    knot_strategy = _safe_getattr(spec, "knot_strategy")
    declared = {
        "kind": _spline_kind(spec),
        "n_knots": _safe_getattr(spec, "n_knots"),
        "spline_degree": _safe_getattr(spec, "degree"),
        "knot_strategy": knot_strategy,
        "penalty": _safe_getattr(spec, "penalty"),
        "select": _safe_getattr(spec, "select"),
        "extrapolation": _safe_getattr(spec, "extrapolation"),
        "constraint_kind": constraint_kind,
        "constraint_mode": _safe_getattr(spec, "constraint_mode") if constraint_kind else None,
        "m": _safe_getattr(spec, "_m_orders"),
        "knots": _safe_getattr(spec, "_explicit_knots"),
        "boundary": _safe_getattr(spec, "_explicit_boundary"),
        "lambda_policy": _safe_getattr(spec, "_lambda_policy"),
    }
    if knot_strategy in _KNOT_ALPHA_STRATEGIES:
        declared["knot_alpha"] = _safe_getattr(spec, "knot_alpha")

    metadata = _base_feature_metadata(name, spec, "spline")
    metadata.update(
        {
            "declared": declared,
            "effective": {
                "kind": _spline_kind(spec),
                "class_name": type(spec).__name__,
                "n_knots": _safe_getattr(spec, "n_knots"),
                "knot_strategy_actual": _safe_getattr(spec, "_knot_strategy_actual"),
            },
            "fitted": {
                "class_name": type(spec).__name__,
                "boundary": _safe_getattr(spec, "fitted_boundary"),
                "knots": _safe_getattr(spec, "fitted_knots"),
                "raw_basis_count": _optional_int(_safe_getattr(spec, "_n_basis")),
                "coefficient_width": _shape_width(r_inv),
                "lower_bound": _safe_getattr(spec, "_lo"),
                "upper_bound": _safe_getattr(spec, "_hi"),
            },
        }
    )
    return metadata


def _ordered_basis_value(spec: OrderedCategorical) -> Any:
    basis = _safe_getattr(spec, "basis")
    if isinstance(basis, _SplineBase):
        return _spline_kind(basis)
    return basis


def _ordered_spline(spec: OrderedCategorical) -> _SplineBase | None:
    for attr_name in ("_spline", "_spline_obj"):
        candidate = _safe_getattr(spec, attr_name)
        if isinstance(candidate, _SplineBase):
            return candidate
    basis = _safe_getattr(spec, "basis")
    if isinstance(basis, _SplineBase):
        return basis
    return None


def _ordered_categorical_metadata(name: str, spec: OrderedCategorical) -> dict[str, Any]:
    spline = _ordered_spline(spec)
    r_inv = _safe_getattr(spec, "_R_inv")
    metadata = _base_feature_metadata(name, spec, "ordered_categorical")
    metadata.update(
        {
            "declared": {
                "basis": _ordered_basis_value(spec),
                "kind": _safe_getattr(spec, "kind"),
                "base": _safe_getattr(spec, "base"),
                "ordered_levels": _safe_getattr(spec, "_ordered_levels"),
                "level_values": _safe_getattr(
                    spec,
                    "_original_level_to_value",
                    _safe_getattr(spec, "_level_to_value"),
                ),
                "n_knots_requested": _safe_getattr(spec, "n_knots"),
                "degree": _safe_getattr(spec, "degree"),
                "penalty": _safe_getattr(spec, "penalty"),
                "select": _safe_getattr(spec, "select"),
                "grouping": _grouping_metadata(_safe_getattr(spec, "_grouping")),
            },
            "effective": {
                "basis": _ordered_basis_value(spec),
                "kind": _spline_kind(spline) if spline is not None else _safe_getattr(spec, "kind"),
                "n_knots_effective": _safe_getattr(spline, "n_knots")
                if spline is not None
                else None,
                "n_levels": _safe_getattr(spec, "_n_levels"),
                "ordered_levels": _safe_getattr(spec, "_ordered_levels"),
                "level_values": _safe_getattr(spec, "_level_to_value"),
                "base_level": _safe_getattr(spec, "_base_level"),
                "non_base_levels": _safe_getattr(spec, "_non_base"),
            },
            "fitted": {
                "levels": _safe_getattr(spec, "_ordered_levels"),
                "base_level": _safe_getattr(spec, "_base_level"),
                "non_base_levels": _safe_getattr(spec, "_non_base"),
                "coefficient_width": _shape_width(r_inv),
            },
        }
    )
    if spline is not None and _ordered_basis_value(spec) != "step":
        metadata["spline"] = _spline_metadata(name, spline)
    return metadata


def _polynomial_metadata(name: str, spec: Polynomial) -> dict[str, Any]:
    degree = _safe_getattr(spec, "degree")
    metadata = _base_feature_metadata(name, spec, "polynomial")
    metadata.update(
        {
            "declared": {"degree": degree},
            "effective": {"encoding": "polynomial", "degree": degree},
            "fitted": {
                "lower_bound": _safe_getattr(spec, "_lo"),
                "upper_bound": _safe_getattr(spec, "_hi"),
            },
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


def _unknown_metadata(name: str, spec: Any) -> dict[str, Any]:
    metadata = _base_feature_metadata(name, spec, "unknown")
    metadata.update({"declared": {}, "effective": {}, "fitted": {}})
    return metadata


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
        "effective": {
            "encoding": "fixed_log_coefficient",
            "coefficient": 1.0,
        },
        "fitted": {},
    }


def _superglm_version() -> str:
    try:
        return package_version("superglm")
    except PackageNotFoundError:
        return "unknown"


def _iter_model_specs(model: Any) -> list[tuple[str, Any]]:
    for specs_attr, order_attr in (
        ("_specs", "_feature_order"),
        ("specs", "feature_order"),
        ("features", "feature_order"),
    ):
        specs = _safe_getattr(model, specs_attr)
        if not isinstance(specs, Mapping):
            continue

        order = _safe_getattr(model, order_attr)
        if order is None:
            return [(str(name), spec) for name, spec in specs.items()]

        ordered: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for name in order:
            if name in specs:
                ordered.append((str(name), specs[name]))
                seen.add(str(name))
        ordered.extend((str(name), spec) for name, spec in specs.items() if str(name) not in seen)
        return ordered

    return []


def _iter_interaction_specs(model: Any) -> list[tuple[str, Any]]:
    for specs_attr, order_attr in (
        ("_interaction_specs", "_interaction_order"),
        ("interaction_specs", "interaction_order"),
    ):
        specs = _safe_getattr(model, specs_attr)
        if not isinstance(specs, Mapping):
            continue
        order = _safe_getattr(model, order_attr)
        if order is None:
            return [(str(name), spec) for name, spec in specs.items()]
        ordered: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for name in order:
            if name in specs:
                ordered.append((str(name), specs[name]))
                seen.add(str(name))
        ordered.extend((str(name), spec) for name, spec in specs.items() if str(name) not in seen)
        return ordered
    return []


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
    parent_names = tuple(str(parent).strip() for parent in spec.parent_names)
    if len(parent_names) != 2 or any(not parent for parent in parent_names):
        raise ValueError(
            f"interaction {name!r} must have exactly two categorical parent features"
        )
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
    return _unknown_metadata(name, spec)


def _model_link_name(model: Any) -> str | None:
    link = _safe_getattr(model, "link")
    if isinstance(link, str) and link:
        return link

    fitted_link = _safe_getattr(model, "_link")
    if fitted_link is not None:
        return type(fitted_link).__name__
    if link is not None:
        return type(link).__name__
    return None


def _snake_case_name(value: str) -> str:
    step_one = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step_one).lower()


def _model_family_metadata(model: Any) -> tuple[str | None, dict[str, Any]]:
    family = _safe_getattr(model, "family")
    if family is None:
        return None, {}
    if isinstance(family, str):
        return family, {}

    params = {
        str(name): value for name, value in vars(family).items() if not str(name).startswith("_")
    }
    return _snake_case_name(type(family).__name__), _json_value(params)


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
    model: Any,
    *,
    offset_contract: OffsetExportContract,
    source_to_published_names: Mapping[str, str] | None = None,
    fit_sample_weight_name: str | None = None,
    export_weight_name: str | None = None,
) -> SuperGLMPublicationReceipt:
    overrides = source_to_published_names or {}
    term_metadata: dict[str, dict[str, Any]] = {}
    published_sources: dict[str, str] = {}
    published_by_source: dict[str, str] = {}
    fit_used_offset = bool(_safe_getattr(model, "_fit_used_offset", False))
    _validate_offset_contract(fit_used_offset, offset_contract)

    model_specs = _iter_model_specs(model)
    if not model_specs:
        raise ValueError("SuperGLM model has no feature specs to publish")

    for source_name, spec in model_specs:
        metadata = _feature_metadata(source_name, spec)
        published_name = overrides.get(source_name, metadata["published_term_name"])
        metadata["published_term_name"] = published_name

        if published_name in published_sources:
            first_source = published_sources[published_name]
            raise ValueError(
                "canonical term name collision: "
                f"{published_name!r} from {first_source!r} and {source_name!r}"
            )
        published_sources[published_name] = source_name
        published_by_source[source_name] = published_name
        term_metadata[published_name] = _json_value(metadata)

    for source_name, spec in _iter_interaction_specs(model):
        published_name = overrides.get(source_name, clean_identifier(source_name))
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

    family_name, family_params = _model_family_metadata(model)
    package_metadata = {
        "model": {
            "family": family_name,
            "family_params": family_params,
            "link": _model_link_name(model),
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
        superglm_version=_superglm_version(),
        extractor_version=EXTRACTOR_VERSION,
        package_metadata=_json_value(package_metadata),
        term_metadata=term_metadata,
        offset_contract=offset_contract,
    )
