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


_FUNCTION_METADATA_MAX_DEPTH = 64


def _function_metadata(
    value: object,
    *,
    _seen: frozenset[int] = frozenset(),
    _depth: int = 0,
) -> tuple[Any, ...]:
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0.0 else "-inf")
        return ("float", value)
    if type(value) is str:
        return ("str", value)
    if _depth >= _FUNCTION_METADATA_MAX_DEPTH:
        return ("unsupported-depth",)
    value_id = id(value)
    if value_id in _seen:
        return ("recursive",)
    nested_seen = _seen | {value_id}
    nested_depth = _depth + 1
    if type(value) is tuple:
        return (
            "tuple",
            tuple(
                _function_metadata(item, _seen=nested_seen, _depth=nested_depth) for item in value
            ),
        )
    if type(value) is dict and all(type(key) is str for key in value):
        return (
            "dict",
            tuple(
                (
                    key,
                    _function_metadata(
                        value[key],
                        _seen=nested_seen,
                        _depth=nested_depth,
                    ),
                )
                for key in sorted(value)
            ),
        )
    if type(value) is types.FunctionType:
        code = value.__code__
        return (
            "function",
            ("code", id(code), code),
            _function_metadata(value.__name__, _seen=nested_seen, _depth=nested_depth),
            _function_metadata(value.__qualname__, _seen=nested_seen, _depth=nested_depth),
            _function_metadata(value.__module__, _seen=nested_seen, _depth=nested_depth),
            _function_metadata(value.__defaults__, _seen=nested_seen, _depth=nested_depth),
            _function_metadata(value.__kwdefaults__, _seen=nested_seen, _depth=nested_depth),
            _closure_metadata(value.__closure__, _seen=nested_seen, _depth=nested_depth),
        )
    return ("unsupported",)


def _closure_metadata(
    value: object,
    *,
    _seen: frozenset[int] = frozenset(),
    _depth: int = 0,
) -> tuple[Any, ...]:
    if value is None:
        return ("none",)
    if type(value) is not tuple or not all(type(cell) is types.CellType for cell in value):
        return ("unsupported",)
    if _depth >= _FUNCTION_METADATA_MAX_DEPTH:
        return ("unsupported-depth",)
    value_id = id(value)
    if value_id in _seen:
        return ("recursive",)
    nested_seen = _seen | {value_id}
    contents = []
    for cell in value:
        try:
            contents.append(
                _function_metadata(
                    cell.cell_contents,
                    _seen=nested_seen,
                    _depth=_depth + 1,
                )
            )
        except ValueError:
            contents.append(("empty",))
    return ("closure", tuple(contents))


SUPERGLM_VERSION = "0.12.0"
SUPERGLM_GIT_SHA = "25c06fc84b674bb2ee777ea99567772d8d57a17c"
_PINNED_CROSS_VALIDATE_CODE = _UPSTREAM_CROSS_VALIDATE.__code__
_PINNED_CLONE_MODEL = _UPSTREAM_CROSS_VALIDATE.__globals__.get("_clone_model")
_PINNED_CROSS_VALIDATE_DEFAULTS_METADATA = _function_metadata(_UPSTREAM_CROSS_VALIDATE.__defaults__)
_PINNED_CROSS_VALIDATE_KWDEFAULTS_METADATA = _function_metadata(
    _UPSTREAM_CROSS_VALIDATE.__kwdefaults__
)
_PINNED_CROSS_VALIDATE_CLOSURE_METADATA = _closure_metadata(_UPSTREAM_CROSS_VALIDATE.__closure__)
_CROSS_VALIDATE_GLOBAL_NAMES = (
    "np",
    "_resolve_scorers",
    "_POOLED_PARTS",
    "copy",
    "_clone_model",
    "time",
    "_RESERVED_COLUMNS",
    "logger",
    "pd",
    "CrossValidationResult",
)
_PINNED_CROSS_VALIDATE_GLOBALS = tuple(
    (
        name,
        _UPSTREAM_CROSS_VALIDATE.__globals__[name],
        _function_metadata(_UPSTREAM_CROSS_VALIDATE.__globals__[name]),
    )
    for name in _CROSS_VALIDATE_GLOBAL_NAMES
)
_PINNED_POOLED_PARTS = tuple(
    (name, value, _function_metadata(value))
    for name, value in sorted(_UPSTREAM_CROSS_VALIDATE.__globals__["_POOLED_PARTS"].items())
)
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
    except metadata.PackageNotFoundError as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution is not installed") from exc
    except Exception as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution cannot be inspected") from exc
    if type(installed_version) is not str:
        raise SuperGLMIdentityError("metadata.version returned unsupported type")
    if installed_version != SUPERGLM_VERSION:
        raise SuperGLMIdentityError(
            f"runtime requires SuperGLM version {SUPERGLM_VERSION!r}, found {installed_version!r}"
        )
    try:
        installed_distribution = metadata.distribution("superglm")
    except metadata.PackageNotFoundError as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution is not installed") from exc
    except Exception as exc:
        raise SuperGLMIdentityError("the pinned SuperGLM distribution cannot be inspected") from exc

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
    try:
        result = scoped_cross_validate(snapshot, X, y, **kwargs)
    except BaseException as upstream_error:
        try:
            _verify_cross_validate_models(snapshot, model, expected_identity)
        except SuperGLMIdentityError as mutation_error:
            raise mutation_error from upstream_error
        raise
    _verify_cross_validate_models(snapshot, model, expected_identity)
    return result


def _verify_cross_validate_models(
    snapshot: SuperGLM,
    model: SuperGLM,
    expected_identity: bytes,
) -> None:
    snapshot_identity, snapshot_error = _capture_model_semantic_bytes(snapshot)
    model_identity, model_error = _capture_model_semantic_bytes(model)
    if snapshot_error is not None:
        raise SuperGLMIdentityError(
            "SuperGLM snapshot identity verification failed"
        ) from snapshot_error
    if model_error is not None:
        raise SuperGLMIdentityError("SuperGLM identity verification failed") from model_error
    if snapshot_identity != expected_identity:
        raise SuperGLMIdentityError("SuperGLM snapshot changed during cross-validation")
    if model_identity != expected_identity:
        raise SuperGLMIdentityError("SuperGLM changed during cross-validation")


def _capture_model_semantic_bytes(
    model: SuperGLM,
) -> tuple[bytes | None, BaseException | None]:
    try:
        return _model_semantic_bytes(model), None
    except BaseException as error:
        return None, error


def _scoped_exact_cross_validate() -> types.FunctionType:
    """Copy the pinned CV function with only its private clone global replaced."""
    resolve_superglm_runtime_identity()
    upstream = _UPSTREAM_CROSS_VALIDATE
    if type(upstream) is not types.FunctionType:
        raise SuperGLMIdentityError(
            "pinned SuperGLM cross_validate function structure does not match the exact adapter"
        )
    defaults = upstream.__defaults__
    kwdefaults = (
        upstream.__kwdefaults__.copy()
        if type(upstream.__kwdefaults__) is dict
        else upstream.__kwdefaults__
    )
    closure = upstream.__closure__
    if (
        upstream.__code__ is not _PINNED_CROSS_VALIDATE_CODE
        or upstream.__name__ != "cross_validate"
        or upstream.__module__ != "superglm.model_selection"
        or "_clone_model" not in upstream.__code__.co_names
        or upstream.__globals__.get("_clone_model") is not _PINNED_CLONE_MODEL
        or type(_PINNED_CLONE_MODEL) is not types.FunctionType
        or _PINNED_CLONE_MODEL.__name__ != "_clone_model"
        or _PINNED_CLONE_MODEL.__module__ != "superglm.model_selection"
        or _function_metadata(defaults) != _PINNED_CROSS_VALIDATE_DEFAULTS_METADATA
        or _function_metadata(kwdefaults) != _PINNED_CROSS_VALIDATE_KWDEFAULTS_METADATA
        or _closure_metadata(closure) != _PINNED_CROSS_VALIDATE_CLOSURE_METADATA
        or not _cross_validate_globals_match(upstream.__globals__)
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
        defaults,
        closure,
    )
    scoped.__kwdefaults__ = kwdefaults
    scoped.__qualname__ = upstream.__qualname__
    scoped.__module__ = upstream.__module__
    scoped.__doc__ = upstream.__doc__
    scoped.__annotations__ = upstream.__annotations__.copy()
    scoped.__dict__.update(upstream.__dict__)
    if hasattr(upstream, "__type_params__"):
        scoped.__type_params__ = upstream.__type_params__
    return scoped


def _cross_validate_globals_match(upstream_globals: dict[str, Any]) -> bool:
    for name, expected, expected_metadata in _PINNED_CROSS_VALIDATE_GLOBALS:
        actual = upstream_globals.get(name)
        if actual is not expected or _function_metadata(actual) != expected_metadata:
            return False
    pooled_parts = upstream_globals.get("_POOLED_PARTS")
    if (
        type(pooled_parts) is not dict
        or len(pooled_parts) != len(_PINNED_POOLED_PARTS)
        or not all(type(name) is str for name in pooled_parts)
    ):
        return False
    return all(
        name in pooled_parts
        and pooled_parts[name] is expected
        and _function_metadata(pooled_parts[name]) == expected_metadata
        for name, expected, expected_metadata in _PINNED_POOLED_PARTS
    )


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
    return snapshot, _model_semantic_bytes(model)


def _model_semantic_bytes(model: SuperGLM) -> bytes:
    validate_pristine_superglm(model)
    return json.dumps(
        _model_semantic_payload(model),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
        "interactions": [list(pair) for pair in _pending_interactions(model._pending_interactions)],
        "penalty": _penalty_payload(model.penalty),
        "lambda2": _finite_float(model.lambda2, "lambda2"),
        "solver": {
            "active_set": _bool_value(model._active_set, "active_set"),
            "direct_solve": _native_string(model._direct_solve, "direct_solve"),
            "discrete": _bool_value(model._discrete, "discrete"),
            "n_bins": _integer_or_mapping(model._n_bins, "n_bins"),
            "tol": _finite_float(model._tol, "tol"),
            "max_iter": _integer(model._max_iter, "max_iter"),
            "convergence": _native_string(model._convergence, "convergence"),
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
    feature_order = _native_string_values(model._feature_order, "SuperGLM feature name")
    spec_order = _native_string_values(model._specs, "SuperGLM feature name")
    if spec_order != feature_order or len(set(feature_order)) != len(feature_order):
        raise SuperGLMIdentityError("SuperGLM feature order/spec mapping is malformed")
    feature_ids = [id(spec) for spec in model._specs.values()]
    if len(set(feature_ids)) != len(feature_ids):
        raise SuperGLMIdentityError("SuperGLM feature specs must not share object identities")
    pending_interactions = _pending_interactions(model._pending_interactions)

    direct_solve = _native_string(model._direct_solve, "direct_solve")
    if direct_solve not in {
        "auto",
        "gram",
        "qr",
    }:
        raise SuperGLMIdentityError("direct_solve must be 'auto', 'gram', or 'qr'")
    convergence = _native_string(model._convergence, "convergence")
    if convergence not in {
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

    _native_string(model._categorical_base, "categorical_base")
    _validate_positive_integer_or_list(model._n_knots, "n_knots")
    if _integer(model._degree, "degree") < 1:
        raise SuperGLMIdentityError("degree must be positive")

    ordered_names = {
        name
        for name, spec in zip(feature_order, model._specs.values(), strict=True)
        if type(spec) is OrderedCategorical
    }
    if model._specs and any(
        left not in feature_order or right not in feature_order
        for left, right in pending_interactions
    ):
        raise SuperGLMIdentityError(
            "explicit-feature interactions must reference names in the feature plan"
        )
    if any(left in ordered_names or right in ordered_names for left, right in pending_interactions):
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
                {
                    "name": _native_string(name, "SuperGLM feature name"),
                    "spec": _feature_payload(spec),
                }
                for name, spec in zip(
                    model._feature_order,
                    model._specs.values(),
                    strict=True,
                )
            ],
        }
    if model._splines is None:
        return {"mode": "intercept_only", "plan": []}
    if type(model._splines) is not list:
        raise SuperGLMIdentityError("auto-detect splines must be a list of column names")
    spline_names = _native_string_values(model._splines, "auto-detect spline column name")
    if len(spline_names) != len(set(spline_names)):
        raise SuperGLMIdentityError("auto-detect spline column names must be unique")
    if type(model._n_knots) is list and len(model._n_knots) != len(model._splines):
        raise SuperGLMIdentityError(
            "auto-detect n_knots list must align exactly with spline column order"
        )
    return {
        "mode": "auto_detect",
        "splines": spline_names,
        "n_knots": _integer_or_list(model._n_knots, "n_knots"),
        "degree": _integer(model._degree, "degree"),
        "categorical_base": _native_string(model._categorical_base, "categorical_base"),
    }


def _feature_payload(spec: object) -> dict[str, Any]:
    if type(spec) is Numeric:
        _require_fields(spec, frozenset(), "Numeric")
        return {"kind": "numeric"}
    if type(spec) is Polynomial:
        _require_fields(spec, frozenset({"degree", "_lo", "_hi"}), "Polynomial")
        try:
            lower_bound = _finite_float(spec._lo, "Polynomial._lo")
            upper_bound = _finite_float(spec._hi, "Polynomial._hi")
        except SuperGLMIdentityError as exc:
            raise SuperGLMIdentityError("Polynomial fitted bounds are malformed") from exc
        if lower_bound != 0.0 or upper_bound != 1.0:
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
    base = _native_string(spec.base, "Categorical.base")
    _require_empty_list(spec._levels, "Categorical._levels")
    if _native_string(spec._base_level, "Categorical._base_level", allow_empty=True):
        raise SuperGLMIdentityError("Categorical is not pristine: fitted base is populated")
    _require_empty_list(spec._non_base, "Categorical._non_base")
    return {
        "kind": "categorical",
        "base": base,
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
    basis = _native_string(spec.basis, "OrderedCategorical.basis")
    if basis not in {"step", "spline"}:
        raise SuperGLMIdentityError("OrderedCategorical has an unsupported basis")
    base = _native_string(spec.base, "OrderedCategorical.base")
    select = _bool_value(spec.select, "OrderedCategorical.select")
    if type(spec._ordered_levels) is not list:
        raise SuperGLMIdentityError("OrderedCategorical ordered levels are malformed")
    ordered_levels = _native_string_values(
        spec._ordered_levels,
        "OrderedCategorical ordered level",
    )
    if len(ordered_levels) != len(set(ordered_levels)) or not ordered_levels:
        raise SuperGLMIdentityError(
            "OrderedCategorical ordered levels must be non-empty and unique"
        )
    if _integer(spec._n_levels, "OrderedCategorical._n_levels") != len(ordered_levels):
        raise SuperGLMIdentityError("OrderedCategorical level count is malformed")
    if type(spec._known_levels) is not set:
        raise SuperGLMIdentityError("OrderedCategorical known levels are malformed")
    known_levels = set(_native_string_values(spec._known_levels, "OrderedCategorical known level"))
    if _native_string(spec._base_level, "OrderedCategorical._base_level", allow_empty=True):
        raise SuperGLMIdentityError("OrderedCategorical is not pristine: fitted base is populated")
    _require_empty_list(spec._non_base, "OrderedCategorical._non_base")
    if spec._R_inv is not None:
        raise SuperGLMIdentityError(
            "OrderedCategorical is not pristine: reparametrization is populated"
        )

    grouping = _grouping_payload(spec._grouping)
    level_values = _ordered_level_values(spec._level_to_value, ordered_levels)
    original_level_values = None
    if spec._grouping is None:
        if spec._original_level_to_value is not None:
            raise SuperGLMIdentityError("OrderedCategorical original level state is malformed")
        if known_levels != set(ordered_levels):
            raise SuperGLMIdentityError("OrderedCategorical known levels are malformed")
    else:
        original_levels = _native_string_values(
            spec._grouping.all_original_levels,
            "LevelGrouping original level",
        )
        if known_levels != set(original_levels):
            raise SuperGLMIdentityError("OrderedCategorical grouped known levels are malformed")
        original_level_values = _grouped_ordered_level_values(
            spec._original_level_to_value,
            spec._grouping,
        )

    payload: dict[str, Any] = {
        "kind": "ordered_categorical",
        "basis": basis,
        "base": base,
        "ordered_levels": ordered_levels,
        "grouping": grouping,
    }
    if basis == "step":
        if spec._spline_obj is not None or spec._spline is not None or select:
            raise SuperGLMIdentityError("OrderedCategorical step state is malformed")
        return payload

    if type(spec._spline) not in _SPLINE_KINDS:
        raise SuperGLMIdentityError("OrderedCategorical spline must be an exact built-in spline")
    source_spline_payload = None
    if spec._spline_obj is not None:
        if type(spec._spline_obj) not in _SPLINE_KINDS:
            raise SuperGLMIdentityError(
                "OrderedCategorical source spline must be an exact built-in spline"
            )
        source_spline_payload = _spline_payload(spec._spline_obj)
    spline_payload = _spline_payload(spec._spline)
    expected_kind = _SPLINE_KINDS[type(spec._spline)]
    requested_n_knots = _integer(spec.n_knots, "OrderedCategorical.n_knots")
    expected_n_knots = (
        requested_n_knots
        if spec._spline_obj is not None
        else min(requested_n_knots, len(ordered_levels) - 1)
    )
    if (
        _native_string(spec.kind, "OrderedCategorical.kind") != expected_kind
        or select != spline_payload["select"]
        or _native_string(spec.penalty, "OrderedCategorical.penalty") != spline_payload["penalty"]
        or _integer(spec.degree, "OrderedCategorical.degree") != spline_payload["degree"]
        or expected_n_knots != spline_payload["n_knots"]
    ):
        raise SuperGLMIdentityError("OrderedCategorical spline metadata is malformed")
    payload["level_values"] = level_values
    if original_level_values is not None:
        payload["original_level_values"] = original_level_values
    payload["spline"] = spline_payload
    if source_spline_payload is not None:
        payload["source_spline"] = source_spline_payload
    return payload


def _ordered_level_values(value: object, levels: list[str]) -> dict[str, float]:
    try:
        normalized = _string_keyed_mapping(value, "OrderedCategorical level values")
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError("OrderedCategorical level values are malformed") from exc
    if set(normalized) != set(levels):
        raise SuperGLMIdentityError("OrderedCategorical level values are malformed")
    return {
        level: _finite_float(normalized[level], f"OrderedCategorical value {level!r}")
        for level in levels
    }


def _grouped_ordered_level_values(
    value: object,
    grouping: LevelGrouping,
) -> dict[str, Any]:
    try:
        normalized = _string_keyed_mapping(
            value,
            "OrderedCategorical original level values",
        )
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError(
            "OrderedCategorical original level values are malformed"
        ) from exc
    original_levels = _native_string_values(
        grouping.all_original_levels,
        "LevelGrouping original level",
    )
    grouped_levels = _native_string_values(
        grouping.grouped_levels,
        "LevelGrouping grouped level",
    )
    value_keys = set(normalized)
    if value_keys == set(original_levels):
        scope = "original"
        levels = original_levels
    elif value_keys == set(grouped_levels):
        scope = "grouped"
        levels = grouped_levels
    else:
        raise SuperGLMIdentityError("OrderedCategorical original level values are malformed")
    return {
        "scope": scope,
        "values": _ordered_level_values(normalized, levels),
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
    originals = _native_string_values(
        grouping.all_original_levels,
        "LevelGrouping original level",
    )
    groups = _native_string_values(
        grouping.grouped_levels,
        "LevelGrouping grouped level",
    )
    if (
        not originals
        or not groups
        or len(originals) != len(set(originals))
        or len(groups) != len(set(groups))
    ):
        raise SuperGLMIdentityError("LevelGrouping levels must be non-empty unique strings")
    try:
        original_to_group_raw = _string_keyed_mapping(
            grouping.original_to_group,
            "LevelGrouping original_to_group",
        )
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError("LevelGrouping original_to_group is malformed") from exc
    if set(original_to_group_raw) != set(originals):
        raise SuperGLMIdentityError("LevelGrouping original_to_group is malformed")
    try:
        original_to_group = {
            original: _native_string(group, "LevelGrouping original_to_group value")
            for original, group in original_to_group_raw.items()
        }
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError("LevelGrouping original_to_group values are malformed") from exc
    if any(group not in groups for group in original_to_group.values()):
        raise SuperGLMIdentityError("LevelGrouping original_to_group values are malformed")
    try:
        group_to_originals_raw = _string_keyed_mapping(
            grouping.group_to_originals,
            "LevelGrouping group_to_originals",
        )
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError("LevelGrouping group_to_originals is malformed") from exc
    if set(group_to_originals_raw) != set(groups):
        raise SuperGLMIdentityError("LevelGrouping group_to_originals is malformed")
    group_to_originals: dict[str, list[str]] = {}
    for group in groups:
        raw_members = group_to_originals_raw[group]
        if type(raw_members) is not list or not raw_members:
            raise SuperGLMIdentityError("LevelGrouping group members are malformed")
        try:
            members = _native_string_values(raw_members, "LevelGrouping group member")
        except SuperGLMIdentityError as exc:
            raise SuperGLMIdentityError("LevelGrouping group members are malformed") from exc
        group_to_originals[group] = members
        if any(
            member not in originals or original_to_group.get(member) != group for member in members
        ):
            raise SuperGLMIdentityError("LevelGrouping mappings are inconsistent")
    flattened = [member for group in groups for member in group_to_originals[group]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(originals):
        raise SuperGLMIdentityError("LevelGrouping mappings are inconsistent")
    return {
        "original_to_group": original_to_group,
        "group_to_originals": group_to_originals,
        "all_original_levels": originals,
        "grouped_levels": groups,
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


def _validate_pinned_spline_invariants(
    spline_type: type[object],
    degree: int,
    m_orders: tuple[int, ...],
    select: bool,
    constraint: dict[str, str] | None,
) -> None:
    if spline_type is NaturalSpline:
        if select:
            raise SuperGLMIdentityError("NaturalSpline does not support select")
        if constraint is not None:
            raise SuperGLMIdentityError("NaturalSpline does not support constraints")
    elif spline_type is CubicRegressionSpline:
        if degree != 3:
            raise SuperGLMIdentityError("CubicRegressionSpline degree must be 3")
        if any(order > 3 for order in m_orders):
            raise SuperGLMIdentityError("CubicRegressionSpline penalty order must not exceed 3")
    elif spline_type is CardinalCRSpline:
        if degree != 3:
            raise SuperGLMIdentityError("CardinalCRSpline degree must be 3")
        if len(m_orders) > 1:
            raise SuperGLMIdentityError("CardinalCRSpline does not support multi-order penalties")
        if any(order > 2 for order in m_orders):
            raise SuperGLMIdentityError("CardinalCRSpline penalty order must not exceed 2")
        if select and m_orders != (2,):
            raise SuperGLMIdentityError("CardinalCRSpline select requires penalty order 2")
    elif spline_type is PSpline:
        if select and max(m_orders) > 2:
            raise SuperGLMIdentityError("PSpline select requires penalty orders at most 2")
    elif spline_type is BSplineSmooth and select and max(m_orders) > 2:
        raise SuperGLMIdentityError("BSplineSmooth select requires penalty orders at most 2")


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
    try:
        lower_bound = _finite_float(spec._lo, f"{spline_type.__name__}._lo")
        upper_bound = _finite_float(spec._hi, f"{spline_type.__name__}._hi")
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError(f"{spline_type.__name__} fitted bounds are malformed") from exc
    if lower_bound != 0.0 or upper_bound != 1.0:
        raise SuperGLMIdentityError(
            f"{spline_type.__name__} is not pristine: fitted bounds are populated"
        )
    knot_strategy = _native_string(spec.knot_strategy, f"{spline_type.__name__}.knot_strategy")
    actual_knot_strategy = _native_string(
        spec._knot_strategy_actual,
        f"{spline_type.__name__}._knot_strategy_actual",
    )
    if actual_knot_strategy != knot_strategy:
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
    if knot_strategy not in {
        "uniform",
        "quantile",
        "quantile_rows",
        "quantile_tempered",
    }:
        raise SuperGLMIdentityError("spline knot_strategy is unsupported")
    penalty = _native_string(spec.penalty, f"{spline_type.__name__}.penalty")
    if penalty not in {"ssp", "none"}:
        raise SuperGLMIdentityError("spline penalty must be 'ssp' or 'none'")
    extrapolation = _native_string(
        spec.extrapolation,
        f"{spline_type.__name__}.extrapolation",
    )
    if extrapolation not in {"clip", "extend", "error"}:
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
    m_orders = tuple(_integer(order, "spline m") for order in spec._m_orders)
    if any(order < 1 for order in m_orders) or len(m_orders) != len(set(m_orders)):
        raise SuperGLMIdentityError("spline derivative penalty orders must be unique and positive")
    select = _bool_value(spec.select, f"{spline_type.__name__}.select")
    constraint = _constraint_payload(spec)
    _validate_pinned_spline_invariants(spline_type, degree, m_orders, select, constraint)
    lambda_policy = _lambda_policy_payload(spec._lambda_policy)
    return {
        "kind": _SPLINE_PAYLOAD_KINDS[spline_type],
        "n_knots": n_knots,
        "degree": degree,
        "knot_strategy": knot_strategy,
        "knot_alpha": knot_alpha,
        "penalty": penalty,
        "select": select,
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
        "extrapolation": extrapolation,
        "constraint": constraint,
        "m": list(m_orders),
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
            _native_string(spec.constraint_mode, "spline constraint mode") != "postfit"
            or spec.monotone is not None
            or _native_string(spec.monotone_mode, "spline monotone mode") != "postfit"
        ):
            raise SuperGLMIdentityError("spline constraint state is malformed")
        return None
    constraint_kind = _native_string(spec.constraint_kind, "spline constraint kind")
    constraint_mode = _native_string(spec.constraint_mode, "spline constraint mode")
    monotone = _native_string(spec.monotone, "spline monotone kind")
    monotone_mode = _native_string(spec.monotone_mode, "spline monotone mode")
    if constraint_kind not in {"increasing", "decreasing", "convex", "concave"}:
        raise SuperGLMIdentityError("spline constraint kind is unsupported")
    if constraint_mode not in {"fit", "postfit"}:
        raise SuperGLMIdentityError("spline constraint mode is unsupported")
    if monotone != constraint_kind or monotone_mode != constraint_mode:
        raise SuperGLMIdentityError("spline constraint state is malformed")
    return {"kind": constraint_kind, "mode": constraint_mode}


def _lambda_policy_payload(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is LambdaPolicy:
        return _one_lambda_policy_payload(value)
    if type(value) is dict:
        policies = _string_keyed_mapping(value, "lambda policy")
        return {key: _one_lambda_policy_payload(policy) for key, policy in policies.items()}
    raise SuperGLMIdentityError("lambda policy must use exact built-in LambdaPolicy values")


def _one_lambda_policy_payload(policy: object) -> dict[str, Any]:
    if type(policy) is not LambdaPolicy:
        raise SuperGLMIdentityError("lambda policy must be the exact built-in LambdaPolicy")
    _require_fields(policy, frozenset({"mode", "value"}), "LambdaPolicy")
    mode = _native_string(policy.mode, "LambdaPolicy.mode")
    if mode == "estimate" and policy.value is None:
        return {"mode": "estimate", "value": None}
    if mode == "fixed" and policy.value is not None:
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
    if type(value) is not frozenset:
        raise SuperGLMIdentityError("penalty targets are malformed")
    try:
        targets = _native_string_values(value, "penalty target")
    except SuperGLMIdentityError as exc:
        raise SuperGLMIdentityError("penalty targets are malformed") from exc
    if len(targets) != len(set(targets)):
        raise SuperGLMIdentityError("penalty targets are malformed")
    return sorted(targets)


def _family_payload(family: object) -> dict[str, Any]:
    parameter_free = {
        "poisson": Poisson,
        "gaussian": Gaussian,
        "gamma": Gamma,
        "binomial": Binomial,
    }
    if type(family) in {str, np.str_}:
        family_name = _native_string(family, "SuperGLM family")
        if family_name not in parameter_free:
            raise SuperGLMIdentityError("unsupported SuperGLM family shortcut")
        return {"kind": family_name}
    for kind, family_type in parameter_free.items():
        if type(family) is family_type:
            _require_fields(family, frozenset(), family_type.__name__)
            return {"kind": kind}
    if type(family) is NegativeBinomial:
        _require_fields(family, frozenset({"theta"}), "NegativeBinomial")
        if type(family.theta) in {str, np.str_}:
            theta_name = _native_string(family.theta, "NegativeBinomial.theta")
            if theta_name != "auto":
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
    if type(link) in {str, np.str_}:
        link_name = _native_string(link, "SuperGLM link")
        if link_name not in parameter_free:
            raise SuperGLMIdentityError("unsupported SuperGLM link shortcut")
        return {"kind": link_name}
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


def _pending_interactions(value: object) -> list[tuple[str, str]]:
    if type(value) is not list:
        raise SuperGLMIdentityError("pending interactions must be a list")
    normalized: list[tuple[str, str]] = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise SuperGLMIdentityError("pending interactions must contain name pairs")
        try:
            left, right = (_native_string(name, "pending interaction name") for name in pair)
        except SuperGLMIdentityError as exc:
            raise SuperGLMIdentityError("pending interactions must contain name pairs") from exc
        normalized.append((left, right))
    return normalized


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


def _native_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is str:
        normalized = value
    elif type(value) is np.str_:
        normalized = str(value)
    else:
        raise SuperGLMIdentityError(f"{label} must be a string scalar")
    if not allow_empty and not normalized:
        raise SuperGLMIdentityError(f"{label} must be a non-empty string")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SuperGLMIdentityError(f"{label} must be UTF-8 encodable") from exc
    return normalized


def _native_string_values(values: object, label: str) -> list[str]:
    return [_native_string(value, label) for value in values]


def _string_keyed_mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SuperGLMIdentityError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = _native_string(raw_key, f"{label} key")
        if key in normalized:
            raise SuperGLMIdentityError(f"{label} contains duplicate string keys")
        normalized[key] = item
    return normalized


def _validate_positive_integer_or_list(value: object, label: str) -> None:
    values = value if type(value) is list else [value]
    if not values or any(_integer(item, label) < 1 for item in values):
        raise SuperGLMIdentityError(f"{label} must contain positive integers")


def _validate_positive_integer_config(value: object, label: str) -> None:
    if type(value) is dict:
        values = _string_keyed_mapping(value, f"{label} mapping").values()
    else:
        values = (value,)
    if any(_integer(item, label) < 1 for item in values):
        raise SuperGLMIdentityError(f"{label} must contain positive integers")


_NUMPY_INTEGER_TYPES = frozenset(
    {
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.intp,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uintp,
        np.longlong,
        np.ulonglong,
    }
)
_NUMPY_FLOAT_TYPES = frozenset({np.float16, np.float32, np.float64, np.longdouble})


def _integer(value: object, label: str) -> int:
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif value_type not in _NUMPY_INTEGER_TYPES:
        raise SuperGLMIdentityError(f"{label} must be an integer")
    else:
        try:
            normalized = int(value)
        except Exception as exc:
            raise SuperGLMIdentityError(f"{label} must be an integer") from exc
    try:
        json.dumps(normalized)
    except Exception as exc:
        raise SuperGLMIdentityError(f"{label} must be a JSON-encodable integer") from exc
    return normalized


def _finite_float(value: object, label: str) -> float:
    value_type = type(value)
    if value_type not in {int, float} | _NUMPY_INTEGER_TYPES | _NUMPY_FLOAT_TYPES:
        raise SuperGLMIdentityError(f"{label} must be a finite number")
    try:
        normalized = float(value)
    except Exception as exc:
        raise SuperGLMIdentityError(f"{label} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise SuperGLMIdentityError(f"{label} must be a finite number")
    return 0.0 if normalized == 0.0 else normalized


def _bool_value(value: object, label: str) -> bool:
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is not np.bool_:
        raise SuperGLMIdentityError(f"{label} must be boolean")
    return bool(value)


def _integer_or_list(value: object, label: str) -> int | list[int]:
    if type(value) is list:
        return [_integer(item, label) for item in value]
    return _integer(value, label)


def _integer_or_mapping(value: object, label: str) -> int | dict[str, int]:
    if type(value) is dict:
        normalized = _string_keyed_mapping(value, f"{label} mapping")
        return {key: _integer(item, label) for key, item in normalized.items()}
    return _integer(value, label)


def _is_git_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )
