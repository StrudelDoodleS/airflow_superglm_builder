from __future__ import annotations

import copy
import hashlib
import json
import math
import types
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np
from superglm import (
    Adaptive,
    BSplineSmooth,
    Binomial,
    Categorical,
    CauchitLink,
    CloglogLink,
    CubicRegressionSpline,
    Gamma,
    Gaussian,
    GroupElasticNet,
    GroupLasso,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    LevelGrouping,
    LogitLink,
    LogLink,
    NaturalSpline,
    NegativeBinomial,
    NegativeBinomialLink,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Poisson,
    PowerLink,
    ProbitLink,
    PSpline,
    Ridge,
    SparseGroupLasso,
    SqrtLink,
    SuperGLM,
    Tweedie,
    cross_validate as _UPSTREAM_CROSS_VALIDATE,
)
from superglm.features.spline import CardinalCRSpline
from superglm.types import LambdaPolicy


SUPERGLM_VERSION = "0.12.0"
SUPERGLM_GIT_SHA = "25c06fc84b674bb2ee777ea99567772d8d57a17c"
_PINNED_CROSS_VALIDATE_CODE = _UPSTREAM_CROSS_VALIDATE.__code__
_PINNED_CLONE_MODEL = _UPSTREAM_CROSS_VALIDATE.__globals__.get("_clone_model")
_EXACT_DEEPCOPY = copy.deepcopy


class SuperGLMIdentityError(RuntimeError):
    """Raised when the pinned SuperGLM model identity cannot be trusted."""


@dataclass(frozen=True)
class SuperGLMRuntimeIdentity:
    version: str
    git_sha: str


def resolve_superglm_runtime_identity() -> SuperGLMRuntimeIdentity:
    """Resolve and enforce the one supported SuperGLM distribution pin."""
    try:
        installed_version = metadata.version("superglm")
        installed_distribution = metadata.distribution("superglm")
    except metadata.PackageNotFoundError as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution is not installed") from exc
    except Exception as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution cannot be inspected") from exc
    if type(installed_version) is not str or installed_version != SUPERGLM_VERSION:
        raise SuperGLMIdentityError(
            f"runtime requires SuperGLM version {SUPERGLM_VERSION!r}, found {installed_version!r}"
        )

    try:
        direct_url_text = installed_distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text is not None else None
    except Exception as exc:
        raise SuperGLMIdentityError(
            "installed SuperGLM direct_url.json must contain a 40-character "
            "lowercase hex git SHA in vcs_info.commit_id"
        ) from exc
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    git_sha = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git" or not _is_git_sha(git_sha):
        raise SuperGLMIdentityError(
            "installed SuperGLM direct_url.json must contain a 40-character "
            "lowercase hex git SHA in vcs_info.commit_id"
        )
    if git_sha != SUPERGLM_GIT_SHA:
        raise SuperGLMIdentityError(
            f"runtime requires SuperGLM git SHA {SUPERGLM_GIT_SHA!r}, found {git_sha!r}"
        )
    return SuperGLMRuntimeIdentity(version=installed_version, git_sha=git_sha)


def canonical_superglm_payload(model: SuperGLM) -> dict[str, Any]:
    """Return the JSON-safe semantic configuration of a pristine SuperGLM."""
    validate_pristine_superglm(model)
    before = _model_semantic_payload(model)
    runtime = resolve_superglm_runtime_identity()
    validate_pristine_superglm(model)
    after = _model_semantic_payload(model)
    if before != after:
        raise SuperGLMIdentityError("SuperGLM changed during identity capture")
    return {
        "schema": "pricing-pipeline-superglm-identity-v1",
        "runtime": {"version": runtime.version, "git_sha": runtime.git_sha},
        **after,
    }


def canonical_superglm_bytes(model: SuperGLM) -> bytes:
    """Encode the canonical SuperGLM payload deterministically."""
    return json.dumps(
        canonical_superglm_payload(model),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def superglm_semantic_sha256(model: SuperGLM) -> str:
    """Hash the canonical semantic identity of a pristine SuperGLM."""
    return hashlib.sha256(canonical_superglm_bytes(model)).hexdigest()


def validate_pristine_superglm(model: SuperGLM) -> None:
    """Reject any model outside the exact, unfitted pinned SuperGLM graph."""
    if type(model) is not SuperGLM:
        raise SuperGLMIdentityError("model must be the exact built-in SuperGLM class")
    actual_fields = frozenset(vars(model))
    if actual_fields != _MODEL_FIELDS:
        unexpected = sorted(actual_fields - _MODEL_FIELDS)
        missing = sorted(_MODEL_FIELDS - actual_fields)
        raise SuperGLMIdentityError(
            "SuperGLM is not pristine: unexpected or missing state fields "
            f"(unexpected={unexpected}, missing={missing})"
        )
    _validate_pristine_model_fields(model)
    _model_semantic_payload(model)


def exact_superglm_cross_validate(model: SuperGLM, X, y, **kwargs):
    """Run the pinned upstream CV loop with exact deep-copy model cloning."""
    snapshot, expected_identity = _pristine_superglm_copy(model)
    scoped_cross_validate = _scoped_exact_cross_validate()
    result = scoped_cross_validate(snapshot, X, y, **kwargs)
    if canonical_superglm_bytes(snapshot) != expected_identity:
        raise SuperGLMIdentityError("SuperGLM snapshot changed during cross-validation")
    if canonical_superglm_bytes(model) != expected_identity:
        raise SuperGLMIdentityError("SuperGLM changed during cross-validation")
    return result


def _scoped_exact_cross_validate() -> types.FunctionType:
    """Copy the pinned CV function with only its private clone global replaced."""
    resolve_superglm_runtime_identity()
    upstream = _UPSTREAM_CROSS_VALIDATE
    if (
        type(upstream) is not types.FunctionType
        or upstream.__code__ is not _PINNED_CROSS_VALIDATE_CODE
        or upstream.__name__ != "cross_validate"
        or upstream.__module__ != "superglm.model_selection"
        or "_clone_model" not in upstream.__code__.co_names
        or upstream.__globals__.get("_clone_model") is not _PINNED_CLONE_MODEL
        or type(_PINNED_CLONE_MODEL) is not types.FunctionType
        or _PINNED_CLONE_MODEL.__name__ != "_clone_model"
        or _PINNED_CLONE_MODEL.__module__ != "superglm.model_selection"
    ):
        raise SuperGLMIdentityError(
            "pinned SuperGLM cross_validate function structure does not match the exact adapter"
        )

    scoped_globals = upstream.__globals__.copy()
    scoped_globals["_clone_model"] = _EXACT_DEEPCOPY
    scoped = types.FunctionType(
        upstream.__code__,
        scoped_globals,
        upstream.__name__,
        upstream.__defaults__,
        upstream.__closure__,
    )
    scoped.__kwdefaults__ = (
        None if upstream.__kwdefaults__ is None else upstream.__kwdefaults__.copy()
    )
    scoped.__qualname__ = upstream.__qualname__
    scoped.__module__ = upstream.__module__
    scoped.__doc__ = upstream.__doc__
    scoped.__annotations__ = upstream.__annotations__.copy()
    scoped.__dict__.update(upstream.__dict__)
    if hasattr(upstream, "__type_params__"):
        scoped.__type_params__ = upstream.__type_params__
    return scoped


def _pristine_superglm_copy(model: SuperGLM) -> tuple[SuperGLM, bytes]:
    expected_identity = canonical_superglm_bytes(model)
    try:
        snapshot = _EXACT_DEEPCOPY(model)
    except Exception as exc:
        raise SuperGLMIdentityError(
            "model must be a pristine, copyable exact built-in SuperGLM"
        ) from exc
    if type(snapshot) is not SuperGLM or snapshot is model:
        raise SuperGLMIdentityError("model must produce a distinct exact built-in SuperGLM copy")
    snapshot_identity = canonical_superglm_bytes(snapshot)
    current_identity = canonical_superglm_bytes(model)
    if snapshot_identity != expected_identity or current_identity != expected_identity:
        raise SuperGLMIdentityError("SuperGLM changed while its exact copy was captured")
    if snapshot.penalty is model.penalty or any(
        snapshot._specs[name] is model._specs[name] for name in model._feature_order
    ):
        raise SuperGLMIdentityError("SuperGLM copy retained shared mutable configuration")
    return snapshot, expected_identity


_MODEL_FIELDS = frozenset(
    {
        "family",
        "link",
        "penalty",
        "lambda2",
        "_splines",
        "_n_knots",
        "_degree",
        "_categorical_base",
        "_active_set",
        "_direct_solve",
        "_discrete",
        "_n_bins",
        "_tol",
        "_max_iter",
        "_retain_fit_state",
        "_convergence",
        "_specs",
        "_feature_order",
        "_groups",
        "_distribution",
        "_link",
        "_result",
        "_solver_result",
        "_dm",
        "_fit_weights",
        "_fit_offset",
        "_fit_used_offset",
        "_fit_stats",
        "_runtime_canonical_state",
        "_nb_profile_result",
        "_tweedie_profile_result",
        "_last_fit_meta",
        "_monotone_repairs",
        "_prediction_plan",
        "_fast_prediction_state",
        "_fit_mu",
        "_fit_null_mu",
        "_fit_X_ref",
        "_fit_y_ref",
        "_fit_sample_weight_ref",
        "_fit_offset_ref",
        "_fit_metrics_cache",
        "_fit_metrics_cache_signature",
        "_summary_cache",
        "_interaction_specs",
        "_interaction_order",
        "_pending_interactions",
    }
)


def _model_semantic_payload(model: SuperGLM) -> dict[str, Any]:
    return {
        "family": _family_payload(model.family),
        "link": _link_payload(model.link, model.family),
        "features": _feature_plan_payload(model),
        "interactions": [list(pair) for pair in model._pending_interactions],
        "penalty": _penalty_payload(model.penalty),
        "lambda2": _finite_float(model.lambda2, "lambda2"),
        "solver": {
            "active_set": _bool_value(model._active_set, "active_set"),
            "direct_solve": model._direct_solve,
            "discrete": _bool_value(model._discrete, "discrete"),
            "n_bins": _integer_or_mapping(model._n_bins, "n_bins"),
            "tol": _finite_float(model._tol, "tol"),
            "max_iter": _integer(model._max_iter, "max_iter"),
            "convergence": model._convergence,
            "retain_fit_state": _bool_value(
                model._retain_fit_state,
                "retain_fit_state",
            ),
        },
    }


def _validate_pristine_model_fields(model: SuperGLM) -> None:
    _require_empty_list(model._groups, "SuperGLM._groups")
    _require_none_fields(
        model,
        (
            "_distribution",
            "_link",
            "_result",
            "_solver_result",
            "_dm",
            "_fit_weights",
            "_fit_offset",
            "_fit_stats",
            "_runtime_canonical_state",
            "_nb_profile_result",
            "_tweedie_profile_result",
            "_last_fit_meta",
            "_prediction_plan",
            "_fast_prediction_state",
            "_fit_mu",
            "_fit_null_mu",
            "_fit_X_ref",
            "_fit_y_ref",
            "_fit_sample_weight_ref",
            "_fit_offset_ref",
            "_fit_metrics_cache",
            "_fit_metrics_cache_signature",
            "_summary_cache",
        ),
        "SuperGLM",
    )
    if type(model._fit_used_offset) is not bool or model._fit_used_offset:
        raise SuperGLMIdentityError("SuperGLM is not pristine: _fit_used_offset is populated")
    _require_empty_dict(model._monotone_repairs, "SuperGLM._monotone_repairs")
    _require_empty_dict(model._interaction_specs, "SuperGLM._interaction_specs")
    _require_empty_list(model._interaction_order, "SuperGLM._interaction_order")

    if type(model._specs) is not dict or type(model._feature_order) is not list:
        raise SuperGLMIdentityError("SuperGLM feature plan is malformed")
    if not all(type(name) is str and name for name in model._feature_order):
        raise SuperGLMIdentityError("SuperGLM feature names must be non-empty strings")
    if list(model._specs) != model._feature_order or len(set(model._feature_order)) != len(
        model._feature_order
    ):
        raise SuperGLMIdentityError("SuperGLM feature order/spec mapping is malformed")
    feature_ids = [id(spec) for spec in model._specs.values()]
    if len(set(feature_ids)) != len(feature_ids):
        raise SuperGLMIdentityError("SuperGLM feature specs must not share object identities")
    _pending_interactions(model._pending_interactions)

    if type(model._direct_solve) is not str or model._direct_solve not in {
        "auto",
        "gram",
        "qr",
    }:
        raise SuperGLMIdentityError("direct_solve must be 'auto', 'gram', or 'qr'")
    if type(model._convergence) is not str or model._convergence not in {
        "deviance",
        "coefficients",
    }:
        raise SuperGLMIdentityError("convergence must be 'deviance' or 'coefficients'")
    if _finite_float(model.lambda2, "lambda2") < 0.0:
        raise SuperGLMIdentityError("lambda2 must be non-negative")
    if _finite_float(model._tol, "tol") <= 0.0:
        raise SuperGLMIdentityError("tol must be positive")
    if _integer(model._max_iter, "max_iter") < 1:
        raise SuperGLMIdentityError("max_iter must be positive")
    _bool_value(model._active_set, "active_set")
    _bool_value(model._discrete, "discrete")
    _bool_value(model._retain_fit_state, "retain_fit_state")
    _validate_positive_integer_config(model._n_bins, "n_bins")

    if type(model._categorical_base) is not str or not model._categorical_base:
        raise SuperGLMIdentityError("categorical_base must be a non-empty string")
    _validate_positive_integer_or_list(model._n_knots, "n_knots")
    if _integer(model._degree, "degree") < 1:
        raise SuperGLMIdentityError("degree must be positive")

    ordered_names = {
        name for name, spec in model._specs.items() if type(spec) is OrderedCategorical
    }
    if model._specs and any(
        left not in model._specs or right not in model._specs
        for left, right in model._pending_interactions
    ):
        raise SuperGLMIdentityError(
            "explicit-feature interactions must reference names in the feature plan"
        )
    if any(
        left in ordered_names or right in ordered_names
        for left, right in model._pending_interactions
    ):
        raise SuperGLMIdentityError(
            "ordered-categorical interactions are unsupported by the pinned fit dispatch"
        )


def _feature_plan_payload(model: SuperGLM) -> dict[str, Any]:
    if model._specs:
        if model._splines is not None:
            raise SuperGLMIdentityError("explicit features cannot also use auto-detect splines")
        return {
            "mode": "explicit",
            "plan": [
                {"name": name, "spec": _feature_payload(spec)}
                for name, spec in zip(
                    model._feature_order,
                    model._specs.values(),
                    strict=True,
                )
            ],
        }
    if model._splines is None:
        return {"mode": "intercept_only", "plan": []}
    if type(model._splines) is not list or not all(
        type(name) is str and name for name in model._splines
    ):
        raise SuperGLMIdentityError("auto-detect splines must be a list of column names")
    if len(model._splines) != len(set(model._splines)):
        raise SuperGLMIdentityError("auto-detect spline column names must be unique")
    if type(model._n_knots) is list and len(model._n_knots) != len(model._splines):
        raise SuperGLMIdentityError(
            "auto-detect n_knots list must align exactly with spline column order"
        )
    return {
        "mode": "auto_detect",
        "splines": list(model._splines),
        "n_knots": _integer_or_list(model._n_knots, "n_knots"),
        "degree": _integer(model._degree, "degree"),
        "categorical_base": model._categorical_base,
    }


def _feature_payload(spec: object) -> dict[str, Any]:
    if type(spec) is Numeric:
        _require_fields(spec, frozenset(), "Numeric")
        return {"kind": "numeric"}
    if type(spec) is Polynomial:
        _require_fields(spec, frozenset({"degree", "_lo", "_hi"}), "Polynomial")
        if spec._lo != 0.0 or spec._hi != 1.0:
            raise SuperGLMIdentityError("Polynomial is not pristine: fitted bounds are populated")
        degree = _integer(spec.degree, "Polynomial.degree")
        if degree < 1:
            raise SuperGLMIdentityError("Polynomial.degree must be positive")
        return {"kind": "polynomial", "degree": degree}
    if type(spec) is Categorical:
        return _categorical_payload(spec)
    if type(spec) is OrderedCategorical:
        return _ordered_categorical_payload(spec)
    if type(spec) in _SPLINE_KINDS:
        return _spline_payload(spec)
    raise SuperGLMIdentityError(
        f"unsupported feature spec type {type(spec).__module__}.{type(spec).__qualname__}"
    )


def _categorical_payload(spec: Categorical) -> dict[str, Any]:
    _require_fields(
        spec,
        frozenset({"base", "_grouping", "_levels", "_base_level", "_non_base"}),
        "Categorical",
    )
    if type(spec.base) is not str or not spec.base:
        raise SuperGLMIdentityError("Categorical.base must be a non-empty string")
    _require_empty_list(spec._levels, "Categorical._levels")
    if type(spec._base_level) is not str or spec._base_level:
        raise SuperGLMIdentityError("Categorical is not pristine: fitted base is populated")
    _require_empty_list(spec._non_base, "Categorical._non_base")
    return {
        "kind": "categorical",
        "base": spec.base,
        "grouping": _grouping_payload(spec._grouping),
    }


_ORDERED_CATEGORICAL_FIELDS = frozenset(
    {
        "_spline_obj",
        "basis",
        "kind",
        "base",
        "select",
        "penalty",
        "degree",
        "n_knots",
        "_ordered_levels",
        "_level_to_value",
        "_grouping",
        "_original_level_to_value",
        "_known_levels",
        "_n_levels",
        "_base_level",
        "_non_base",
        "_R_inv",
        "_spline",
    }
)


def _ordered_categorical_payload(spec: OrderedCategorical) -> dict[str, Any]:
    _require_fields(spec, _ORDERED_CATEGORICAL_FIELDS, "OrderedCategorical")
    if spec.basis not in {"step", "spline"}:
        raise SuperGLMIdentityError("OrderedCategorical has an unsupported basis")
    if type(spec.base) is not str or not spec.base:
        raise SuperGLMIdentityError("OrderedCategorical.base must be a non-empty string")
    if type(spec._ordered_levels) is not list or not all(
        type(level) is str for level in spec._ordered_levels
    ):
        raise SuperGLMIdentityError("OrderedCategorical ordered levels are malformed")
    if len(spec._ordered_levels) != len(set(spec._ordered_levels)) or not spec._ordered_levels:
        raise SuperGLMIdentityError(
            "OrderedCategorical ordered levels must be non-empty and unique"
        )
    if _integer(spec._n_levels, "OrderedCategorical._n_levels") != len(spec._ordered_levels):
        raise SuperGLMIdentityError("OrderedCategorical level count is malformed")
    if type(spec._known_levels) is not set or not all(
        type(level) is str for level in spec._known_levels
    ):
        raise SuperGLMIdentityError("OrderedCategorical known levels are malformed")
    if type(spec._base_level) is not str or spec._base_level:
        raise SuperGLMIdentityError("OrderedCategorical is not pristine: fitted base is populated")
    _require_empty_list(spec._non_base, "OrderedCategorical._non_base")
    if spec._R_inv is not None:
        raise SuperGLMIdentityError(
            "OrderedCategorical is not pristine: reparametrization is populated"
        )

    grouping = _grouping_payload(spec._grouping)
    level_values = _ordered_level_values(spec._level_to_value, spec._ordered_levels)
    if spec._grouping is None:
        if spec._original_level_to_value is not None:
            raise SuperGLMIdentityError("OrderedCategorical original level state is malformed")
        if spec._known_levels != set(spec._ordered_levels):
            raise SuperGLMIdentityError("OrderedCategorical known levels are malformed")
    else:
        if spec._known_levels != set(spec._grouping.all_original_levels):
            raise SuperGLMIdentityError("OrderedCategorical grouped known levels are malformed")
        _ordered_level_values(
            spec._original_level_to_value,
            spec._grouping.all_original_levels,
        )

    payload: dict[str, Any] = {
        "kind": "ordered_categorical",
        "basis": spec.basis,
        "base": spec.base,
        "ordered_levels": list(spec._ordered_levels),
        "grouping": grouping,
    }
    if spec.basis == "step":
        if spec._spline_obj is not None or spec._spline is not None or bool(spec.select):
            raise SuperGLMIdentityError("OrderedCategorical step state is malformed")
        return payload

    if type(spec._spline) not in _SPLINE_KINDS:
        raise SuperGLMIdentityError("OrderedCategorical spline must be an exact built-in spline")
    if spec._spline_obj is not None:
        if type(spec._spline_obj) not in _SPLINE_KINDS:
            raise SuperGLMIdentityError(
                "OrderedCategorical source spline must be an exact built-in spline"
            )
        _spline_payload(spec._spline_obj)
    spline_payload = _spline_payload(spec._spline)
    expected_kind = _SPLINE_KINDS[type(spec._spline)]
    if (
        spec.kind != expected_kind
        or bool(spec.select) != bool(spec._spline.select)
        or spec.penalty != spec._spline.penalty
        or _integer(spec.degree, "OrderedCategorical.degree") != int(spec._spline.degree)
        or _integer(spec.n_knots, "OrderedCategorical.n_knots") != int(spec._spline.n_knots)
    ):
        raise SuperGLMIdentityError("OrderedCategorical spline metadata is malformed")
    payload["level_values"] = level_values
    payload["spline"] = spline_payload
    return payload


def _ordered_level_values(value: object, levels: list[str]) -> dict[str, float]:
    if type(value) is not dict or set(value) != set(levels):
        raise SuperGLMIdentityError("OrderedCategorical level values are malformed")
    return {
        level: _finite_float(value[level], f"OrderedCategorical value {level!r}")
        for level in levels
    }


def _grouping_payload(grouping: object) -> dict[str, Any] | None:
    if grouping is None:
        return None
    if type(grouping) is not LevelGrouping:
        raise SuperGLMIdentityError("grouping must be the exact built-in LevelGrouping")
    _require_fields(
        grouping,
        frozenset(
            {
                "original_to_group",
                "group_to_originals",
                "all_original_levels",
                "grouped_levels",
            }
        ),
        "LevelGrouping",
    )
    if type(grouping.all_original_levels) is not list or type(grouping.grouped_levels) is not list:
        raise SuperGLMIdentityError("LevelGrouping level order is malformed")
    originals = grouping.all_original_levels
    groups = grouping.grouped_levels
    if (
        not originals
        or not groups
        or not all(type(level) is str for level in originals + groups)
        or len(originals) != len(set(originals))
        or len(groups) != len(set(groups))
    ):
        raise SuperGLMIdentityError("LevelGrouping levels must be non-empty unique strings")
    if type(grouping.original_to_group) is not dict or set(grouping.original_to_group) != set(
        originals
    ):
        raise SuperGLMIdentityError("LevelGrouping original_to_group is malformed")
    if type(grouping.group_to_originals) is not dict or set(grouping.group_to_originals) != set(
        groups
    ):
        raise SuperGLMIdentityError("LevelGrouping group_to_originals is malformed")
    for group in groups:
        members = grouping.group_to_originals[group]
        if type(members) is not list or not members:
            raise SuperGLMIdentityError("LevelGrouping group members are malformed")
        if any(
            member not in originals or grouping.original_to_group.get(member) != group
            for member in members
        ):
            raise SuperGLMIdentityError("LevelGrouping mappings are inconsistent")
    flattened = [member for group in groups for member in grouping.group_to_originals[group]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(originals):
        raise SuperGLMIdentityError("LevelGrouping mappings are inconsistent")
    return {
        "original_to_group": dict(grouping.original_to_group),
        "group_to_originals": {group: list(grouping.group_to_originals[group]) for group in groups},
        "all_original_levels": list(originals),
        "grouped_levels": list(groups),
    }


_SPLINE_KINDS = {
    PSpline: "ps",
    BSplineSmooth: "bs",
    NaturalSpline: "ns",
    CubicRegressionSpline: "cr",
    CardinalCRSpline: "cr_cardinal",
}
_SPLINE_PAYLOAD_KINDS = {
    PSpline: "p_spline",
    BSplineSmooth: "b_spline_smooth",
    NaturalSpline: "natural_spline",
    CubicRegressionSpline: "cubic_regression_spline",
    CardinalCRSpline: "cardinal_cr_spline",
}
_SPLINE_COMMON_FIELDS = frozenset(
    {
        "constraint_kind",
        "constraint_mode",
        "monotone",
        "monotone_mode",
        "select",
        "_m_orders",
        "n_knots",
        "_explicit_knots",
        "degree",
        "knot_strategy",
        "penalty",
        "discrete",
        "n_bins",
        "extrapolation",
        "knot_alpha",
        "_explicit_boundary",
        "_knots",
        "_n_basis",
        "_lo",
        "_hi",
        "_knot_strategy_actual",
        "_R_inv",
        "_interaction_projection",
        "_basis_lo",
        "_basis_hi",
        "_basis_d1_lo",
        "_basis_d1_hi",
        "_U_null",
        "_U_range",
        "_omega_range",
        "_penalty_components",
        "_lambda_policy",
    }
)
_SPLINE_EXTRA_FIELDS = {
    PSpline: frozenset(),
    BSplineSmooth: frozenset(),
    NaturalSpline: frozenset({"_Z"}),
    CubicRegressionSpline: frozenset({"_Z"}),
    CardinalCRSpline: frozenset({"_cr_knots", "_cr_M", "_cr_S"}),
}


def _spline_payload(spec: object) -> dict[str, Any]:
    spline_type = type(spec)
    if spline_type not in _SPLINE_KINDS:
        raise SuperGLMIdentityError("spline must be an exact pinned built-in spline")
    expected_fields = _SPLINE_COMMON_FIELDS | _SPLINE_EXTRA_FIELDS[spline_type]
    _require_fields(spec, expected_fields, spline_type.__name__)

    if type(spec._knots) is not np.ndarray or spec._knots.ndim != 1 or spec._knots.size:
        raise SuperGLMIdentityError(
            f"{spline_type.__name__} is not pristine: fitted knots are populated"
        )
    if _integer(spec._n_basis, f"{spline_type.__name__}._n_basis") != 0:
        raise SuperGLMIdentityError(
            f"{spline_type.__name__} is not pristine: fitted basis is populated"
        )
    if spec._lo != 0.0 or spec._hi != 1.0:
        raise SuperGLMIdentityError(
            f"{spline_type.__name__} is not pristine: fitted bounds are populated"
        )
    if spec._knot_strategy_actual != spec.knot_strategy:
        raise SuperGLMIdentityError(
            f"{spline_type.__name__} is not pristine: fitted knot strategy is populated"
        )
    pristine_none_fields = (
        "_R_inv",
        "_interaction_projection",
        "_basis_lo",
        "_basis_hi",
        "_basis_d1_lo",
        "_basis_d1_hi",
        "_U_null",
        "_U_range",
        "_omega_range",
        "_penalty_components",
        *tuple(_SPLINE_EXTRA_FIELDS[spline_type]),
    )
    _require_none_fields(spec, pristine_none_fields, spline_type.__name__)

    n_knots = _integer(spec.n_knots, f"{spline_type.__name__}.n_knots")
    degree = _integer(spec.degree, f"{spline_type.__name__}.degree")
    if n_knots < 1 or degree < 1:
        raise SuperGLMIdentityError("spline n_knots and degree must be positive")
    if spec.knot_strategy not in {
        "uniform",
        "quantile",
        "quantile_rows",
        "quantile_tempered",
    }:
        raise SuperGLMIdentityError("spline knot_strategy is unsupported")
    if spec.penalty not in {"ssp", "none"}:
        raise SuperGLMIdentityError("spline penalty must be 'ssp' or 'none'")
    if spec.extrapolation not in {"clip", "extend", "error"}:
        raise SuperGLMIdentityError("spline extrapolation is unsupported")
    knot_alpha = _finite_float(spec.knot_alpha, f"{spline_type.__name__}.knot_alpha")
    if knot_alpha < 0.0:
        raise SuperGLMIdentityError("spline knot_alpha must be non-negative")
    if spec.discrete is not None:
        _bool_value(spec.discrete, f"{spline_type.__name__}.discrete")
    if spec.n_bins is not None and _integer(spec.n_bins, f"{spline_type.__name__}.n_bins") < 1:
        raise SuperGLMIdentityError("spline n_bins must be positive")

    explicit_knots = _explicit_knots_payload(spec._explicit_knots, n_knots)
    boundary = _boundary_payload(spec._explicit_boundary)
    if explicit_knots is not None and boundary is not None:
        if explicit_knots[0] <= boundary[0] or explicit_knots[-1] >= boundary[1]:
            raise SuperGLMIdentityError("explicit spline knots must lie inside boundary")

    if type(spec._m_orders) is not tuple or not spec._m_orders:
        raise SuperGLMIdentityError("spline derivative penalty orders are malformed")
    m_orders = [_integer(order, "spline m") for order in spec._m_orders]
    if any(order < 1 for order in m_orders) or len(m_orders) != len(set(m_orders)):
        raise SuperGLMIdentityError("spline derivative penalty orders must be unique and positive")

    constraint = _constraint_payload(spec)
    lambda_policy = _lambda_policy_payload(spec._lambda_policy)
    return {
        "kind": _SPLINE_PAYLOAD_KINDS[spline_type],
        "n_knots": n_knots,
        "degree": degree,
        "knot_strategy": spec.knot_strategy,
        "knot_alpha": knot_alpha,
        "penalty": spec.penalty,
        "select": _bool_value(spec.select, f"{spline_type.__name__}.select"),
        "knots": explicit_knots,
        "boundary": boundary,
        "discrete": (
            None
            if spec.discrete is None
            else _bool_value(spec.discrete, f"{spline_type.__name__}.discrete")
        ),
        "n_bins": (
            None if spec.n_bins is None else _integer(spec.n_bins, f"{spline_type.__name__}.n_bins")
        ),
        "extrapolation": spec.extrapolation,
        "constraint": constraint,
        "m": m_orders,
        "lambda_policy": lambda_policy,
    }


def _explicit_knots_payload(value: object, n_knots: int) -> list[float] | None:
    if value is None:
        return None
    if type(value) is not np.ndarray or value.ndim != 1 or len(value) != n_knots:
        raise SuperGLMIdentityError("explicit spline knots are malformed")
    knots = [_finite_float(item, "explicit spline knot") for item in value]
    if any(right <= left for left, right in zip(knots, knots[1:], strict=False)):
        raise SuperGLMIdentityError("explicit spline knots must be strictly increasing")
    return knots


def _boundary_payload(value: object) -> list[float] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise SuperGLMIdentityError("spline boundary is malformed")
    boundary = [_finite_float(item, "spline boundary") for item in value]
    if boundary[0] >= boundary[1]:
        raise SuperGLMIdentityError("spline boundary must satisfy lower < upper")
    return boundary


def _constraint_payload(spec: object) -> dict[str, str] | None:
    if spec.constraint_kind is None:
        if (
            spec.constraint_mode != "postfit"
            or spec.monotone is not None
            or spec.monotone_mode != "postfit"
        ):
            raise SuperGLMIdentityError("spline constraint state is malformed")
        return None
    if spec.constraint_kind not in {"increasing", "decreasing", "convex", "concave"}:
        raise SuperGLMIdentityError("spline constraint kind is unsupported")
    if spec.constraint_mode not in {"fit", "postfit"}:
        raise SuperGLMIdentityError("spline constraint mode is unsupported")
    if spec.monotone != spec.constraint_kind or spec.monotone_mode != spec.constraint_mode:
        raise SuperGLMIdentityError("spline constraint state is malformed")
    return {"kind": spec.constraint_kind, "mode": spec.constraint_mode}


def _lambda_policy_payload(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is LambdaPolicy:
        return _one_lambda_policy_payload(value)
    if type(value) is dict:
        if not all(type(key) is str and key for key in value):
            raise SuperGLMIdentityError("lambda policy keys must be non-empty strings")
        return {key: _one_lambda_policy_payload(policy) for key, policy in value.items()}
    raise SuperGLMIdentityError("lambda policy must use exact built-in LambdaPolicy values")


def _one_lambda_policy_payload(policy: object) -> dict[str, Any]:
    if type(policy) is not LambdaPolicy:
        raise SuperGLMIdentityError("lambda policy must be the exact built-in LambdaPolicy")
    _require_fields(policy, frozenset({"mode", "value"}), "LambdaPolicy")
    if policy.mode == "estimate" and policy.value is None:
        return {"mode": "estimate", "value": None}
    if policy.mode == "fixed" and policy.value is not None:
        fixed = _finite_float(policy.value, "fixed lambda policy")
        if fixed < 0.0:
            raise SuperGLMIdentityError("fixed lambda policy must be non-negative")
        return {"mode": "fixed", "value": fixed}
    raise SuperGLMIdentityError("lambda policy state is malformed")


_PENALTY_KINDS = {
    GroupLasso: "group_lasso",
    SparseGroupLasso: "sparse_group_lasso",
    GroupElasticNet: "group_elastic_net",
    Ridge: "ridge",
}


def _penalty_payload(penalty: object) -> dict[str, Any]:
    penalty_type = type(penalty)
    if penalty_type not in _PENALTY_KINDS:
        raise SuperGLMIdentityError("penalty must be an exact pinned built-in penalty")
    expected_fields = {"lambda1", "flavor", "features"}
    if penalty_type in {SparseGroupLasso, GroupElasticNet}:
        expected_fields.add("alpha")
    _require_fields(penalty, frozenset(expected_fields), penalty_type.__name__)

    lambda1 = None
    if penalty.lambda1 is not None:
        lambda1 = _finite_float(penalty.lambda1, "penalty.lambda1")
        if lambda1 < 0.0:
            raise SuperGLMIdentityError("penalty.lambda1 must be non-negative")
    alpha = None
    if "alpha" in expected_fields:
        alpha = _finite_float(penalty.alpha, "penalty.alpha")
        if not 0.0 <= alpha <= 1.0:
            raise SuperGLMIdentityError("penalty.alpha must be between zero and one")
    flavor = _flavor_payload(penalty.flavor, allow_adaptive=penalty_type is not Ridge)
    targets = _penalty_targets_payload(penalty.features)
    payload = {
        "kind": _PENALTY_KINDS[penalty_type],
        "lambda1": lambda1,
        "flavor": flavor,
        "targets": targets,
    }
    if alpha is not None:
        payload["alpha"] = alpha
    return payload


def _flavor_payload(flavor: object, *, allow_adaptive: bool) -> dict[str, Any] | None:
    if flavor is None:
        return None
    if not allow_adaptive or type(flavor) is not Adaptive:
        raise SuperGLMIdentityError("penalty flavor must be the exact built-in Adaptive flavor")
    _require_fields(flavor, frozenset({"expon", "eps"}), "Adaptive")
    expon = _finite_float(flavor.expon, "Adaptive.expon")
    eps = _finite_float(flavor.eps, "Adaptive.eps")
    if expon <= 0.0 or eps <= 0.0:
        raise SuperGLMIdentityError("Adaptive expon and eps must be positive")
    return {"kind": "adaptive", "expon": expon, "eps": eps}


def _penalty_targets_payload(value: object) -> list[str] | None:
    if value is None:
        return None
    if type(value) is not frozenset or not all(type(item) is str and item for item in value):
        raise SuperGLMIdentityError("penalty targets are malformed")
    return sorted(value)


def _family_payload(family: object) -> dict[str, Any]:
    parameter_free = {
        "poisson": Poisson,
        "gaussian": Gaussian,
        "gamma": Gamma,
        "binomial": Binomial,
    }
    if type(family) is str:
        if family not in parameter_free:
            raise SuperGLMIdentityError("unsupported SuperGLM family shortcut")
        return {"kind": family}
    for kind, family_type in parameter_free.items():
        if type(family) is family_type:
            _require_fields(family, frozenset(), family_type.__name__)
            return {"kind": kind}
    if type(family) is NegativeBinomial:
        _require_fields(family, frozenset({"theta"}), "NegativeBinomial")
        if type(family.theta) is str:
            if family.theta != "auto":
                raise SuperGLMIdentityError("NegativeBinomial.theta must be positive or 'auto'")
            theta: float | str = "auto"
        else:
            theta = _finite_float(family.theta, "NegativeBinomial.theta")
            if theta <= 0.0:
                raise SuperGLMIdentityError("NegativeBinomial.theta must be positive")
        return {"kind": "negative_binomial", "theta": theta}
    if type(family) is Tweedie:
        _require_fields(family, frozenset({"p"}), "Tweedie")
        power = _finite_float(family.p, "Tweedie.p")
        if not 1.0 < power < 2.0:
            raise SuperGLMIdentityError("Tweedie.p must be strictly between one and two")
        return {"kind": "tweedie", "p": power}
    raise SuperGLMIdentityError("unsupported SuperGLM family")


def _link_payload(link: object, family: object) -> dict[str, Any]:
    parameter_free = {
        "log": LogLink,
        "identity": IdentityLink,
        "logit": LogitLink,
        "probit": ProbitLink,
        "cloglog": CloglogLink,
        "cauchit": CauchitLink,
        "inverse": InverseLink,
        "inverse_squared": InverseSquaredLink,
        "sqrt": SqrtLink,
    }
    if link is None:
        family_kind = _family_payload(family)["kind"]
        default_kind = {
            "gaussian": "identity",
            "binomial": "logit",
            "poisson": "log",
            "gamma": "log",
            "negative_binomial": "log",
            "tweedie": "log",
        }[family_kind]
        return {"kind": default_kind}
    if type(link) is str:
        if link not in parameter_free:
            raise SuperGLMIdentityError("unsupported SuperGLM link shortcut")
        return {"kind": link}
    for kind, link_type in parameter_free.items():
        if type(link) is link_type:
            _require_fields(link, frozenset(), link_type.__name__)
            return {"kind": kind}
    if type(link) is PowerLink:
        _require_fields(link, frozenset({"power"}), "PowerLink")
        power = _finite_float(link.power, "PowerLink.power")
        if power == 0.0:
            raise SuperGLMIdentityError("PowerLink.power must be non-zero")
        return {"kind": "power", "power": power}
    if type(link) is NegativeBinomialLink:
        _require_fields(link, frozenset({"theta"}), "NegativeBinomialLink")
        theta = _finite_float(link.theta, "NegativeBinomialLink.theta")
        if theta <= 0.0:
            raise SuperGLMIdentityError("NegativeBinomialLink.theta must be positive")
        return {"kind": "negative_binomial", "theta": theta}
    raise SuperGLMIdentityError("unsupported SuperGLM link")


def _pending_interactions(value: object) -> None:
    if type(value) is not list:
        raise SuperGLMIdentityError("pending interactions must be a list")
    for pair in value:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or not all(type(name) is str and name for name in pair)
        ):
            raise SuperGLMIdentityError("pending interactions must contain name pairs")


def _require_fields(value: object, expected: frozenset[str], label: str) -> None:
    actual = frozenset(vars(value))
    if actual != expected:
        raise SuperGLMIdentityError(
            f"unexpected {label} state fields: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _require_none_fields(value: object, fields: tuple[str, ...], label: str) -> None:
    populated = [field for field in fields if getattr(value, field) is not None]
    if populated:
        raise SuperGLMIdentityError(
            f"{label} is not pristine: populated state fields {sorted(populated)}"
        )


def _require_empty_list(value: object, label: str) -> None:
    if type(value) is not list or value:
        raise SuperGLMIdentityError(f"{label} is not pristine: expected an empty list")


def _require_empty_dict(value: object, label: str) -> None:
    if type(value) is not dict or value:
        raise SuperGLMIdentityError(f"{label} is not pristine: expected an empty mapping")


def _validate_positive_integer_or_list(value: object, label: str) -> None:
    values = value if type(value) is list else [value]
    if not values or any(_integer(item, label) < 1 for item in values):
        raise SuperGLMIdentityError(f"{label} must contain positive integers")


def _validate_positive_integer_config(value: object, label: str) -> None:
    if type(value) is dict:
        if not all(type(key) is str and key for key in value):
            raise SuperGLMIdentityError(f"{label} mapping keys must be non-empty strings")
        values = value.values()
    else:
        values = (value,)
    if any(_integer(item, label) < 1 for item in values):
        raise SuperGLMIdentityError(f"{label} must contain positive integers")


def _integer(value: object, label: str) -> int:
    if type(value) is bool or isinstance(value, np.bool_):
        raise SuperGLMIdentityError(f"{label} must be an integer")
    if type(value) is not int and not isinstance(value, np.integer):
        raise SuperGLMIdentityError(f"{label} must be an integer")
    return int(value)


def _finite_float(value: object, label: str) -> float:
    if type(value) is bool or isinstance(value, np.bool_):
        raise SuperGLMIdentityError(f"{label} must be a finite number")
    if type(value) not in {int, float} and not isinstance(value, np.integer | np.floating):
        raise SuperGLMIdentityError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SuperGLMIdentityError(f"{label} must be a finite number")
    return 0.0 if normalized == 0.0 else normalized


def _bool_value(value: object, label: str) -> bool:
    if type(value) is not bool and not isinstance(value, np.bool_):
        raise SuperGLMIdentityError(f"{label} must be boolean")
    return bool(value)


def _integer_or_list(value: object, label: str) -> int | list[int]:
    if type(value) is list:
        return [_integer(item, label) for item in value]
    return _integer(value, label)


def _integer_or_mapping(value: object, label: str) -> int | dict[str, int]:
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise SuperGLMIdentityError(f"{label} mapping keys must be strings")
        return {key: _integer(item, label) for key, item in value.items()}
    return _integer(value, label)


def _is_git_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )
