"""Controlled SuperGLM refits for model and feature-drift monitoring.

The deployed package remains the production authority.  Monitoring runs are
lightweight observations against one exact deployment and dataset manifest;
they are never publishable rate packages.

This module owns the narrow SuperGLM 0.26 compatibility seam needed to turn a
fitted model into a controlled refit.  Groupings, categorical universes,
reporting bases, ordered special levels, basis types, dimensions, penalty
orders, and shape constraints are frozen for every automatic variant.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version as package_version
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.constraint import ConstraintSpec
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline, _SplineBase
from superglm.types import LambdaPolicy

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.superglm_metadata import (
    _spline_kind,
    build_superglm_publication_receipt,
)
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract

FIT_CONTRACT_SCHEMA = "superglm_monitoring_fit_contract"
FIT_CONTRACT_SCHEMA_VERSION = 1
INVARIANT_EVIDENCE_SCHEMA = "superglm_monitoring_invariant_evidence"
INVARIANT_EVIDENCE_SCHEMA_VERSION = 1
PRIVATE_SUPERGLM_MONITORING_API = "SuperGLM._config/_specs plus fitted spline and categorical state"


class MonitoringError(RuntimeError):
    """Raised when a monitoring contract, refit, or persistence write is unsafe."""


class MonitoringVariant(StrEnum):
    """The only supported, interpretable monitoring comparisons."""

    STATIC_SCORE = "STATIC_SCORE"
    FROZEN_REFIT = "FROZEN_REFIT"
    REESTIMATE_LAMBDA = "REESTIMATE_LAMBDA"
    FULL_ADAPTIVE = "FULL_ADAPTIVE"


@dataclass(frozen=True)
class MonitoringVariantPolicy:
    refit_coefficients: bool
    reestimate_lambdas: bool
    reposition_data_driven_knots: bool


MONITORING_VARIANT_POLICIES: Mapping[MonitoringVariant, MonitoringVariantPolicy] = MappingProxyType(
    {
        MonitoringVariant.STATIC_SCORE: MonitoringVariantPolicy(False, False, False),
        MonitoringVariant.FROZEN_REFIT: MonitoringVariantPolicy(True, False, False),
        MonitoringVariant.REESTIMATE_LAMBDA: MonitoringVariantPolicy(True, True, False),
        MonitoringVariant.FULL_ADAPTIVE: MonitoringVariantPolicy(True, True, True),
    }
)


@dataclass(frozen=True)
class ModelFitContract:
    contract_json: str
    contract_sha256: str
    structure_sha256: str
    superglm_version: str

    def payload(self) -> dict[str, Any]:
        """Return a new mutable decoding of the immutable canonical JSON."""
        return json.loads(self.contract_json)


@dataclass(frozen=True)
class MonitoringTerm:
    term_name: str
    term_kind: str
    sequence_no: int
    metadata_json: str
    structure_sha256: str


@dataclass(frozen=True)
class MonitoringLambda:
    term_name: str | None
    component_name: str
    lambda_value: float
    lambda_mode: str


@dataclass(frozen=True)
class MonitoringRelativity:
    term_name: str
    term_kind: str
    point_key: str
    point_label: str | None
    point_numeric: float | None
    relativity: float
    log_relativity: float
    is_reference: bool


@dataclass(frozen=True)
class MonitoringInvariantEvidence:
    status: str
    evidence_json: str
    evidence_sha256: str

    def payload(self) -> dict[str, Any]:
        """Return a new mutable decoding of the canonical evidence JSON."""
        return json.loads(self.evidence_json)


@dataclass(frozen=True)
class MonitoringFitResult:
    variant: MonitoringVariant
    contract: ModelFitContract
    fitted_model: SuperGLM
    terms: tuple[MonitoringTerm, ...]
    lambdas: tuple[MonitoringLambda, ...]
    relativities: tuple[MonitoringRelativity, ...]
    metrics: Mapping[str, float]
    invariant_evidence: MonitoringInvariantEvidence


@dataclass(frozen=True)
class PersistedMonitoringRun:
    monitor_run_id: str
    fit_contract_id: str
    run_signature_sha256: str
    deduplicated: bool


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if type(value) is int:
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MonitoringError("monitoring evidence contains a non-finite number")
        return numeric
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, pd.Series | pd.Index):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise MonitoringError(f"unsupported monitoring evidence value: {type(value).__name__}")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_fitted_superglm(model: Any) -> SuperGLM:
    if not isinstance(model, SuperGLM):
        raise TypeError("monitoring requires a fitted SuperGLM model")
    try:
        _ = model.result
    except RuntimeError as exc:
        raise MonitoringError("monitoring requires a fitted SuperGLM model") from exc
    if not hasattr(model, "_config") or not isinstance(getattr(model, "_specs", None), dict):
        raise MonitoringError(
            "installed SuperGLM no longer exposes the pinned monitoring compatibility seam: "
            f"{PRIVATE_SUPERGLM_MONITORING_API}"
        )
    return model


def _evaluation_grid(
    model: SuperGLM,
    term_metadata: Mapping[str, Mapping[str, Any]],
    *,
    continuous_points: int,
) -> dict[str, dict[str, Any]]:
    if continuous_points < 2:
        raise ValueError("continuous_points must be at least 2")

    grids: dict[str, dict[str, Any]] = {}
    relativity_frames = model.relativities(with_se=False, centering="native")
    for metadata in term_metadata.values():
        kind = str(metadata["feature_kind"])
        if kind == "offset":
            continue
        source_name = str(metadata["source_term_name"])
        if kind in {"spline", "polynomial"}:
            fitted = metadata["fitted"]
            if kind == "spline":
                boundary = fitted.get("boundary")
            else:
                boundary = [fitted.get("lower_bound"), fitted.get("upper_bound")]
            if boundary is None or any(value is None for value in boundary):
                raise MonitoringError(f"term {source_name!r} has no fitted continuous boundary")
            points = np.linspace(float(boundary[0]), float(boundary[1]), continuous_points)
            grids[source_name] = {"kind": "continuous", "points": points.tolist()}
        elif kind in {"categorical", "ordered_categorical"}:
            inference = model.term_inference(source_name, with_se=False, centering="native")
            grids[source_name] = {
                "kind": "categorical",
                # Public inference expands a fitted grouping back to original
                # levels.  Those are the stable business-facing points to track,
                # not the internal group labels.
                "points": list(inference.levels or []),
            }
        elif kind == "numeric":
            grids[source_name] = {"kind": "numeric", "points": ["per_unit"]}
        elif kind == "categorical_interaction":
            frame = relativity_frames.get(source_name)
            if frame is None or "level" not in frame:
                raise MonitoringError(
                    f"categorical interaction {source_name!r} has no stable level grid"
                )
            grids[source_name] = {
                "kind": "categorical_interaction",
                "points": frame["level"].tolist(),
            }
        else:
            raise MonitoringError(f"term {source_name!r} uses unsupported monitoring kind {kind!r}")
    return grids


def build_model_fit_contract(
    model: SuperGLM,
    *,
    offset_contract: OffsetExportContract | None = None,
    fit_sample_weight_name: str | None = None,
    export_weight_name: str | None = None,
    continuous_points: int = 101,
) -> ModelFitContract:
    """Capture one fitted model's immutable structural and smoothing contract."""
    fitted = _require_fitted_superglm(model)
    resolved_offset = offset_contract or OffsetExportContract(handling="NONE")
    receipt = build_superglm_publication_receipt(
        fitted,
        offset_contract=resolved_offset,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )
    telemetry = fitted.training_telemetry()
    lambdas = fitted.reml_diagnostics().get("lambdas", {})
    term_metadata = receipt.model_dump(mode="json")["term_metadata"]
    structure = {
        "model": telemetry["model"],
        "feature_schema": telemetry["features"],
        "package_metadata": receipt.model_dump(mode="json")["package_metadata"],
        "term_metadata": term_metadata,
        "always_frozen": [
            "family_and_link",
            "feature_order_and_types",
            "categorical_level_universes",
            "categorical_groupings",
            "categorical_bases_and_unseen_policy",
            "ordered_level_values_and_special_levels",
            "spline_kind_degree_dimension_and_penalty_order",
            "shape_and_monotonic_constraints",
            "caller_declared_explicit_knots_and_boundaries",
        ],
    }
    structure_json = _canonical_json(structure)
    payload = {
        "schema_name": FIT_CONTRACT_SCHEMA,
        "schema_version": FIT_CONTRACT_SCHEMA_VERSION,
        "superglm_version": package_version("superglm"),
        "structure_sha256": _sha256_text(structure_json),
        "structure": structure,
        "fitted_lambdas": dict(sorted((str(k), float(v)) for k, v in lambdas.items())),
        "evaluation_grid": _evaluation_grid(
            fitted,
            term_metadata,
            continuous_points=continuous_points,
        ),
        "variants": {
            variant.value: {
                "refit_coefficients": policy.refit_coefficients,
                "reestimate_lambdas": policy.reestimate_lambdas,
                "reposition_data_driven_knots": policy.reposition_data_driven_knots,
            }
            for variant, policy in MONITORING_VARIANT_POLICIES.items()
        },
    }
    contract_json = _canonical_json(payload)
    return ModelFitContract(
        contract_json=contract_json,
        contract_sha256=_sha256_text(contract_json),
        structure_sha256=payload["structure_sha256"],
        superglm_version=payload["superglm_version"],
    )


def _constraint(spec: _SplineBase) -> ConstraintSpec | None:
    kind = getattr(spec, "constraint_kind", None)
    if kind is None:
        return None
    return ConstraintSpec(mode=str(spec.constraint_mode), kind=str(kind))


def _fixed_lambda_policy(
    term_name: str,
    fitted_lambdas: Mapping[str, float],
    configured_policy: Any,
) -> LambdaPolicy | dict[str, LambdaPolicy] | None:
    direct = fitted_lambdas.get(term_name)
    if direct is not None:
        return LambdaPolicy.fixed(float(direct))
    components = {
        name.removeprefix(f"{term_name}:"): LambdaPolicy.fixed(float(value))
        for name, value in fitted_lambdas.items()
        if name.startswith(f"{term_name}:")
    }
    if components:
        return components
    global_lambda = fitted_lambdas.get("lambda2")
    if global_lambda is not None:
        return LambdaPolicy.fixed(float(global_lambda))
    return copy.deepcopy(configured_policy)


def _rebuild_spline(
    configured: _SplineBase,
    fitted: _SplineBase,
    *,
    term_name: str,
    freeze_geometry: bool,
    freeze_lambdas: bool,
    fitted_lambdas: Mapping[str, float],
) -> _SplineBase:
    if freeze_geometry:
        knots = fitted.fitted_knots
        boundary = fitted.fitted_boundary
    else:
        named_knots = getattr(configured, "_named_knots", None)
        explicit_knots = getattr(configured, "_explicit_knots", None)
        knots = named_knots if named_knots is not None else explicit_knots
        boundary = getattr(configured, "_explicit_boundary", None)

    configured_policy = getattr(configured, "_lambda_policy", None)
    lambda_policy = (
        _fixed_lambda_policy(term_name, fitted_lambdas, configured_policy)
        if freeze_lambdas
        else copy.deepcopy(configured_policy)
    )
    m_orders = tuple(int(value) for value in configured._m_orders)
    m: int | tuple[int, ...] = m_orders[0] if len(m_orders) == 1 else m_orders
    return Spline(
        kind=_spline_kind(configured),
        n_knots=int(configured.n_knots),
        degree=int(configured.degree),
        knot_strategy=str(configured.knot_strategy),
        penalty=str(configured.penalty),
        select=bool(configured.select),
        knots=None if knots is None else copy.deepcopy(knots),
        discrete=configured.discrete,
        n_bins=configured.n_bins,
        extrapolation=str(configured.extrapolation),
        boundary=None if boundary is None else tuple(float(value) for value in boundary),
        knot_alpha=float(configured.knot_alpha),
        constraint=_constraint(configured),
        m=m,
        lambda_policy=lambda_policy,
    )


def _freeze_categorical(configured: Categorical, fitted: Categorical) -> Categorical:
    grouping = getattr(fitted, "_grouping", None)
    levels = (
        list(grouping.all_original_levels)
        if grouping is not None
        else list(getattr(fitted, "_levels", ()))
    )
    return Categorical(
        base=str(fitted._base_level),
        grouping=copy.deepcopy(grouping),
        levels=levels,
        unseen=str(configured.unseen),
    )


def _freeze_ordered_categorical(
    configured: OrderedCategorical,
    fitted: OrderedCategorical,
    *,
    term_name: str,
    freeze_geometry: bool,
    freeze_lambdas: bool,
    fitted_lambdas: Mapping[str, float],
) -> OrderedCategorical:
    configured_basis = getattr(configured, "_spline_obj", None)
    fitted_basis = getattr(fitted, "_spline", None)
    if not isinstance(configured_basis, _SplineBase) or not isinstance(fitted_basis, _SplineBase):
        raise MonitoringError(
            f"ordered categorical {term_name!r} must use a spline basis for controlled refits"
        )
    basis = _rebuild_spline(
        configured_basis,
        fitted_basis,
        term_name=term_name,
        freeze_geometry=freeze_geometry,
        freeze_lambdas=freeze_lambdas,
        fitted_lambdas=fitted_lambdas,
    )
    values = copy.deepcopy(
        getattr(fitted, "_original_level_to_value", None)
        or getattr(fitted, "_level_to_value", None)
    )
    if not values:
        raise MonitoringError(
            f"ordered categorical {term_name!r} has no fitted original-level values"
        )
    return OrderedCategorical(
        values=values,
        basis=basis,
        base=str(fitted._base_level),
        grouping=copy.deepcopy(getattr(fitted, "_grouping", None)),
        specials=copy.deepcopy(getattr(fitted, "_special_raw", None)),
    )


def materialize_monitoring_model(
    baseline_model: SuperGLM,
    variant: MonitoringVariant | str,
) -> SuperGLM:
    """Create an unfitted model obeying one validated monitoring preset."""
    baseline = _require_fitted_superglm(baseline_model)
    resolved_variant = MonitoringVariant(variant)
    if resolved_variant is MonitoringVariant.STATIC_SCORE:
        raise MonitoringError("STATIC_SCORE uses the fitted baseline and has no refit model")

    policy = MONITORING_VARIANT_POLICIES[resolved_variant]
    fitted_lambdas = {
        str(name): float(value)
        for name, value in baseline.reml_diagnostics().get("lambdas", {}).items()
    }
    configured_by_name = dict(baseline._config.feature_templates)
    if set(configured_by_name) != set(baseline._specs):
        raise MonitoringError("SuperGLM configured and fitted feature sets do not match")

    templates: list[tuple[Any, Any]] = []
    for name in baseline._feature_order:
        configured = configured_by_name[name]
        fitted = baseline._specs[name]
        if isinstance(fitted, OrderedCategorical) and isinstance(configured, OrderedCategorical):
            replacement = _freeze_ordered_categorical(
                configured,
                fitted,
                term_name=str(name),
                freeze_geometry=not policy.reposition_data_driven_knots,
                freeze_lambdas=not policy.reestimate_lambdas,
                fitted_lambdas=fitted_lambdas,
            )
        elif isinstance(fitted, Categorical) and isinstance(configured, Categorical):
            replacement = _freeze_categorical(configured, fitted)
        elif isinstance(fitted, _SplineBase) and isinstance(configured, _SplineBase):
            replacement = _rebuild_spline(
                configured,
                fitted,
                term_name=str(name),
                freeze_geometry=not policy.reposition_data_driven_knots,
                freeze_lambdas=not policy.reestimate_lambdas,
                fitted_lambdas=fitted_lambdas,
            )
        elif isinstance(fitted, Polynomial) and isinstance(configured, Polynomial):
            if not policy.reposition_data_driven_knots:
                raise MonitoringError(
                    f"term {name!r} is a data-orthogonal Polynomial whose fitted QR basis "
                    "cannot currently be frozen by SuperGLM; use FULL_ADAPTIVE or replace "
                    "the term with an explicitly governed Spline/Numeric basis"
                )
            replacement = copy.deepcopy(configured)
        elif isinstance(fitted, Numeric) and isinstance(configured, Numeric):
            replacement = copy.deepcopy(configured)
        else:
            raise MonitoringError(
                f"term {name!r} uses unsupported controlled-refit type {type(fitted).__name__}"
            )
        templates.append((name, replacement))

    selected = float(getattr(baseline, "selection_penalty_", 0.0) or 0.0)
    if not math.isclose(selected, 0.0, abs_tol=1e-15):
        raise MonitoringError(
            "controlled REML monitoring requires a baseline with no group-selection "
            "penalty; selection changes are a separate model-spec decision"
        )
    config = baseline._config.with_value(
        feature_templates=tuple(templates),
        features_explicit=True,
        level_bindings=None,
    )
    materialized = config.materialize(type(baseline))
    materialized.selection_penalty = 0.0
    return materialized


def _result_terms(
    model: SuperGLM,
    *,
    offset_contract: OffsetExportContract,
    fit_sample_weight_name: str | None,
    export_weight_name: str | None,
) -> tuple[MonitoringTerm, ...]:
    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=offset_contract,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )
    terms: list[MonitoringTerm] = []
    for sequence_no, metadata in enumerate(receipt.term_metadata.values(), start=1):
        metadata_json = _canonical_json(metadata)
        terms.append(
            MonitoringTerm(
                term_name=str(metadata["source_term_name"]),
                term_kind=str(metadata["feature_kind"]),
                sequence_no=sequence_no,
                metadata_json=metadata_json,
                structure_sha256=_sha256_text(metadata_json),
            )
        )
    return tuple(terms)


def _lambda_term(component_name: str, term_names: tuple[str, ...]) -> str | None:
    matches = [
        term_name
        for term_name in term_names
        if component_name == term_name or component_name.startswith(f"{term_name}:")
    ]
    return max(matches, key=len) if matches else None


def _component_policy(model: SuperGLM, term_name: str | None, component: str) -> Any:
    if term_name is None:
        return None
    spec = model._specs.get(term_name)
    if isinstance(spec, OrderedCategorical):
        spec = getattr(spec, "_spline", None)
    policy = getattr(spec, "_lambda_policy", None)
    if isinstance(policy, Mapping):
        suffix = component.removeprefix(f"{term_name}:")
        return policy.get(suffix)
    return policy


def _canonical_component_names(
    model: SuperGLM,
    raw: Mapping[str, Any],
) -> dict[str, str]:
    term_names = tuple(str(name) for name in model._feature_order)
    component_terms = {
        str(component): _lambda_term(str(component), term_names) for component in raw
    }
    canonical: dict[str, str] = {}
    for component in raw:
        raw_component = str(component)
        term_name = component_terms[raw_component]
        sibling_count = sum(sibling_term == term_name for sibling_term in component_terms.values())
        # SuperGLM names a lone estimated component ``term`` but the same
        # component ``term:wiggle`` when LambdaPolicy.fixed is explicit.  One
        # canonical name keeps week-to-week joins stable; multi-component
        # terms retain their meaningful suffixes.
        canonical[raw_component] = (
            term_name
            if term_name is not None
            and sibling_count == 1
            and raw_component in {term_name, f"{term_name}:wiggle"}
            else raw_component
        )
    if len(set(canonical.values())) != len(canonical):
        raise MonitoringError("SuperGLM returned ambiguous canonical lambda component names")
    return canonical


def _canonical_lambda_values(
    model: SuperGLM,
    raw: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    resolved = model.reml_diagnostics().get("lambdas", {}) if raw is None else raw
    names = _canonical_component_names(model, resolved)
    return {
        names[str(component)]: float(value)
        for component, value in sorted(resolved.items(), key=lambda item: str(item[0]))
    }


def _result_lambdas(
    model: SuperGLM,
    variant: MonitoringVariant,
) -> tuple[MonitoringLambda, ...]:
    raw = model.reml_diagnostics().get("lambdas", {})
    term_names = tuple(str(name) for name in model._feature_order)
    component_terms = {
        str(component): _lambda_term(str(component), term_names) for component in raw
    }
    canonical_names = _canonical_component_names(model, raw)
    rows: list[MonitoringLambda] = []
    for component, value in sorted(raw.items()):
        raw_component = str(component)
        term_name = component_terms[raw_component]
        component_name = canonical_names[raw_component]
        if variant is MonitoringVariant.STATIC_SCORE:
            mode = "BASELINE"
        elif variant is MonitoringVariant.FROZEN_REFIT:
            mode = "FIXED"
        else:
            component_policy = _component_policy(model, term_name, raw_component)
            mode = (
                "FIXED"
                if isinstance(component_policy, LambdaPolicy) and component_policy.mode == "fixed"
                else "ESTIMATED"
            )
        rows.append(
            MonitoringLambda(
                term_name=term_name,
                component_name=component_name,
                lambda_value=float(value),
                lambda_mode=mode,
            )
        )
    return tuple(rows)


def _requested_level_values(
    inference: Any,
    points: list[Any],
) -> list[tuple[Any, float, float]]:
    levels = list(inference.levels or [])
    by_text = {str(level): index for index, level in enumerate(levels)}
    rows: list[tuple[Any, float, float]] = []
    for point in points:
        index = by_text.get(str(point))
        if index is None:
            raise MonitoringError(f"controlled refit is missing frozen categorical level {point!r}")
        rows.append(
            (
                point,
                float(np.asarray(inference.relativity)[index]),
                float(np.asarray(inference.log_relativity)[index]),
            )
        )
    return rows


def _result_relativities(
    model: SuperGLM,
    evaluation_grid: Mapping[str, Mapping[str, Any]],
) -> tuple[MonitoringRelativity, ...]:
    rows: list[MonitoringRelativity] = []
    generic_frames = model.relativities(with_se=False, centering="native")
    for term_name, grid in evaluation_grid.items():
        kind = str(grid["kind"])
        points = list(grid["points"])
        if kind == "categorical_interaction":
            frame = generic_frames.get(term_name)
            if frame is None or "level" not in frame:
                raise MonitoringError(f"interaction {term_name!r} has no comparable levels")
            indexed = frame.assign(_key=frame["level"].astype(str)).set_index("_key")
            values = []
            for point in points:
                if str(point) not in indexed.index:
                    raise MonitoringError(
                        f"interaction {term_name!r} is missing frozen point {point!r}"
                    )
                record = indexed.loc[str(point)]
                values.append((point, float(record["relativity"]), float(record["log_relativity"])))
        else:
            inference = model.term_inference(
                term_name,
                with_se=False,
                n_points=max(501, len(points)),
                centering="native",
            )
            if kind == "continuous":
                source_x = np.asarray(inference.x, dtype=float)
                source_log = np.asarray(inference.log_relativity, dtype=float)
                requested_x = np.asarray(points, dtype=float)
                requested_log = np.interp(requested_x, source_x, source_log)
                values = [
                    (point, float(np.exp(log_value)), float(log_value))
                    for point, log_value in zip(points, requested_log, strict=True)
                ]
            elif kind == "categorical":
                values = _requested_level_values(inference, points)
            elif kind == "numeric":
                values = [
                    (
                        "per_unit",
                        float(np.asarray(inference.relativity).ravel()[0]),
                        float(np.asarray(inference.log_relativity).ravel()[0]),
                    )
                ]
            else:
                raise MonitoringError(f"unsupported evaluation-grid kind {kind!r}")

        for point, relativity, log_relativity in values:
            point_numeric = float(point) if kind == "continuous" else None
            point_label = None if point_numeric is not None else str(point)
            point_key = _canonical_json(
                {"x": point_numeric} if point_numeric is not None else {"level": point_label}
            )
            rows.append(
                MonitoringRelativity(
                    term_name=term_name,
                    term_kind=kind,
                    point_key=point_key,
                    point_label=point_label,
                    point_numeric=point_numeric,
                    relativity=relativity,
                    log_relativity=log_relativity,
                    is_reference=math.isclose(log_relativity, 0.0, abs_tol=1e-12),
                )
            )
    return tuple(rows)


def _result_metrics(
    model: SuperGLM,
    X: pd.DataFrame,
    y: Any,
    sample_weight: Any,
    offset: Any,
) -> Mapping[str, float]:
    predictions = np.asarray(model.predict(X, offset=offset), dtype=float)
    diagnostics = model.metrics(X, np.asarray(y), sample_weight, offset)
    metrics = {
        "row_count": float(len(X)),
        "mean_observed": float(np.mean(np.asarray(y, dtype=float))),
        "mean_prediction": float(np.mean(predictions)),
        "sum_prediction": float(np.sum(predictions)),
    }
    for name in ("deviance", "null_deviance", "explained_deviance", "log_likelihood"):
        value = getattr(diagnostics, name, None)
        if value is not None and np.isscalar(value) and math.isfinite(float(value)):
            metrics[name] = float(value)
    return MappingProxyType(dict(sorted(metrics.items())))


def _publication_receipt_payload(
    model: SuperGLM,
    *,
    offset_contract: OffsetExportContract,
    fit_sample_weight_name: str | None,
    export_weight_name: str | None,
) -> dict[str, Any]:
    return build_superglm_publication_receipt(
        model,
        offset_contract=offset_contract,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    ).model_dump(mode="json")


def _normalize_spline_structure(metadata: dict[str, Any]) -> None:
    declared = metadata.get("declared", {})
    for field_name in ("boundary", "knots", "lambda_policy"):
        declared.pop(field_name, None)
    metadata.get("effective", {}).pop("knot_strategy_actual", None)
    fitted = metadata.get("fitted", {})
    for field_name in ("boundary", "knots", "lower_bound", "upper_bound"):
        fitted.pop(field_name, None)


def _normalized_runtime_structure(
    model: SuperGLM,
    receipt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    terms = copy.deepcopy(dict(receipt_payload["term_metadata"]))
    for metadata in terms.values():
        kind = str(metadata["feature_kind"])
        if kind == "categorical":
            metadata.get("declared", {}).pop("levels", None)
            effective = metadata.get("effective", {})
            effective.pop("level_source", None)
            effective.pop("pinned_levels", None)
            fitted = metadata.get("fitted", {})
            fitted.pop("non_base_levels", None)
            fitted["levels"] = sorted(fitted["levels"], key=_canonical_json)
        elif kind == "ordered_categorical":
            fitted = metadata.get("fitted", {})
            for field_name in (
                "coefficient_width",
                "non_base_levels",
                "pinned_special_levels",
                "special_coefficient_width",
            ):
                fitted.pop(field_name, None)
            _normalize_spline_structure(metadata["spline"])
        elif kind == "spline":
            _normalize_spline_structure(metadata)
        elif kind == "polynomial":
            fitted = metadata.get("fitted", {})
            fitted.pop("lower_bound", None)
            fitted.pop("upper_bound", None)

    telemetry = model.training_telemetry()
    feature_schema = copy.deepcopy(telemetry["features"])
    # Active design-matrix groups may contract when a governed level has no
    # effective rows in a particular snapshot. SuperGLM keeps that level known
    # and pins it to zero/base; the persisted raw term metadata records this.
    # It is observation availability, not a structural contract change.
    feature_schema.pop("groups", None)
    return {
        "model": telemetry["model"],
        "feature_schema": feature_schema,
        "package_metadata": receipt_payload["package_metadata"],
        "term_metadata": terms,
    }


def _geometry_from_receipt(receipt_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    geometry: dict[str, dict[str, Any]] = {}
    for metadata in receipt_payload["term_metadata"].values():
        term_name = str(metadata["source_term_name"])
        kind = str(metadata["feature_kind"])
        if kind == "ordered_categorical":
            fitted = metadata["spline"]["fitted"]
            geometry[term_name] = {
                "boundary": fitted["boundary"],
                "knots": fitted["knots"],
            }
        elif kind == "spline":
            fitted = metadata["fitted"]
            geometry[term_name] = {
                "boundary": fitted["boundary"],
                "knots": fitted["knots"],
            }
        elif kind == "polynomial":
            fitted = metadata["fitted"]
            geometry[term_name] = {"boundary": [fitted["lower_bound"], fitted["upper_bound"]]}
    return geometry


def _protected_geometry_fields(
    baseline: SuperGLM,
    variant: MonitoringVariant,
    baseline_geometry: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if variant is not MonitoringVariant.FULL_ADAPTIVE:
        return tuple(
            f"{term_name}.{field_name}"
            for term_name, fields in sorted(baseline_geometry.items())
            for field_name in sorted(fields)
        )

    protected: list[str] = []
    for term_name, configured in baseline._config.feature_templates:
        spline = (
            getattr(configured, "_spline_obj", None)
            if isinstance(configured, OrderedCategorical)
            else configured
        )
        if not isinstance(spline, _SplineBase):
            continue
        if (
            getattr(spline, "_explicit_knots", None) is not None
            or getattr(spline, "_named_knots", None) is not None
        ):
            protected.append(f"{term_name}.knots")
        if getattr(spline, "_explicit_boundary", None) is not None:
            protected.append(f"{term_name}.boundary")
    return tuple(sorted(protected))


def _geometry_value(
    geometry: Mapping[str, Mapping[str, Any]],
    path: str,
) -> Any:
    term_name, field_name = path.rsplit(".", 1)
    if term_name not in geometry or field_name not in geometry[term_name]:
        raise MonitoringError(f"protected spline geometry is missing after refit: {path}")
    return geometry[term_name][field_name]


def _verify_monitoring_invariants(
    baseline: SuperGLM,
    fitted: SuperGLM,
    *,
    variant: MonitoringVariant,
    contract: ModelFitContract,
    offset_contract: OffsetExportContract,
    fit_sample_weight_name: str | None,
    export_weight_name: str | None,
) -> MonitoringInvariantEvidence:
    baseline_receipt = _publication_receipt_payload(
        baseline,
        offset_contract=offset_contract,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )
    fitted_receipt = _publication_receipt_payload(
        fitted,
        offset_contract=offset_contract,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )

    baseline_structure_json = _canonical_json(
        _normalized_runtime_structure(baseline, baseline_receipt)
    )
    fitted_structure_json = _canonical_json(_normalized_runtime_structure(fitted, fitted_receipt))
    baseline_structure_sha256 = _sha256_text(baseline_structure_json)
    fitted_structure_sha256 = _sha256_text(fitted_structure_json)
    if fitted_structure_sha256 != baseline_structure_sha256:
        raise MonitoringError(
            "post-fit invariant guard rejected a structural change in the model, feature "
            "universe/grouping, basis, or constraint contract"
        )

    baseline_geometry = _geometry_from_receipt(baseline_receipt)
    fitted_geometry = _geometry_from_receipt(fitted_receipt)
    if {term_name: tuple(sorted(fields)) for term_name, fields in baseline_geometry.items()} != {
        term_name: tuple(sorted(fields)) for term_name, fields in fitted_geometry.items()
    }:
        raise MonitoringError("post-fit invariant guard rejected changed geometry components")
    protected_geometry = _protected_geometry_fields(baseline, variant, baseline_geometry)
    changed_geometry = [
        path
        for path in protected_geometry
        if _geometry_value(baseline_geometry, path) != _geometry_value(fitted_geometry, path)
    ]
    if changed_geometry:
        raise MonitoringError(
            "post-fit invariant guard rejected changed protected knot/boundary geometry: "
            + ", ".join(changed_geometry)
        )

    baseline_lambda_rows = _result_lambdas(baseline, MonitoringVariant.REESTIMATE_LAMBDA)
    fitted_lambda_rows = _result_lambdas(fitted, variant)
    baseline_lambdas = {row.component_name: row.lambda_value for row in baseline_lambda_rows}
    fitted_lambdas = {row.component_name: row.lambda_value for row in fitted_lambda_rows}
    if set(fitted_lambdas) != set(baseline_lambdas):
        raise MonitoringError("post-fit invariant guard rejected changed lambda components")
    baseline_modes = {row.component_name: row.lambda_mode for row in baseline_lambda_rows}
    fitted_modes = {row.component_name: row.lambda_mode for row in fitted_lambda_rows}
    protected_lambdas = (
        tuple(sorted(baseline_lambdas))
        if variant in {MonitoringVariant.STATIC_SCORE, MonitoringVariant.FROZEN_REFIT}
        else tuple(
            sorted(component for component, mode in baseline_modes.items() if mode == "FIXED")
        )
    )
    changed_lambdas = [
        component
        for component in protected_lambdas
        if fitted_lambdas[component] != baseline_lambdas[component]
    ]
    if changed_lambdas:
        raise MonitoringError(
            "post-fit invariant guard rejected changed fixed lambda values: "
            + ", ".join(changed_lambdas)
        )
    if variant is MonitoringVariant.FROZEN_REFIT and any(
        fitted_modes[component] != "FIXED" for component in fitted_modes
    ):
        raise MonitoringError("post-fit invariant guard found a non-fixed frozen lambda policy")
    if variant is not MonitoringVariant.STATIC_SCORE and any(
        fitted_modes[component] != "FIXED" for component in protected_lambdas
    ):
        raise MonitoringError("post-fit invariant guard found a protected lambda was not fixed")

    diagnostics = fitted.reml_diagnostics()
    termination_reason = diagnostics.get("termination_reason")
    if (
        variant is MonitoringVariant.FROZEN_REFIT
        and baseline_lambdas
        and termination_reason != "fixed_lambdas"
    ):
        raise MonitoringError(
            "post-fit invariant guard expected SuperGLM termination_reason='fixed_lambdas'"
        )
    history = (
        []
        if variant is MonitoringVariant.STATIC_SCORE
        else [
            _canonical_lambda_values(fitted, raw_step)
            for raw_step in diagnostics.get("lambda_history", [])
        ]
    )
    if protected_lambdas and variant is not MonitoringVariant.STATIC_SCORE and not history:
        raise MonitoringError("post-fit invariant guard found no fixed-lambda history evidence")
    for step_no, step in enumerate(history):
        changed_at_step = [
            component
            for component in protected_lambdas
            if component not in step or step[component] != baseline_lambdas[component]
        ]
        if changed_at_step:
            raise MonitoringError(
                "post-fit invariant guard rejected a fixed lambda change in REML history "
                f"step {step_no}: " + ", ".join(changed_at_step)
            )

    policy = MONITORING_VARIANT_POLICIES[variant]
    payload = {
        "schema_name": INVARIANT_EVIDENCE_SCHEMA,
        "schema_version": INVARIANT_EVIDENCE_SCHEMA_VERSION,
        "status": "VERIFIED",
        "variant": variant.value,
        "contract_sha256": contract.contract_sha256,
        "contract_structure_sha256": contract.structure_sha256,
        "policy": {
            "refit_coefficients": policy.refit_coefficients,
            "reestimate_lambdas": policy.reestimate_lambdas,
            "reposition_data_driven_knots": policy.reposition_data_driven_knots,
        },
        "structure": {
            "baseline_sha256": baseline_structure_sha256,
            "fitted_sha256": fitted_structure_sha256,
            "exact_match": True,
        },
        "geometry": {
            "baseline": baseline_geometry,
            "fitted": fitted_geometry,
            "protected_fields": list(protected_geometry),
            "protected_exact_match": True,
        },
        "lambdas": {
            "baseline": baseline_lambdas,
            "fitted": fitted_lambdas,
            "baseline_modes": baseline_modes,
            "fitted_modes": fitted_modes,
            "protected_components": list(protected_lambdas),
            "protected_exact_match": True,
            "history": history,
            "history_checked": variant is not MonitoringVariant.STATIC_SCORE,
            "history_exact_for_protected_components": True,
            "termination_reason": termination_reason,
        },
    }
    evidence_json = _canonical_json(payload)
    return MonitoringInvariantEvidence(
        status="VERIFIED",
        evidence_json=evidence_json,
        evidence_sha256=_sha256_text(evidence_json),
    )


def _validate_persistable_invariant_evidence(
    result: MonitoringFitResult,
) -> MonitoringInvariantEvidence:
    evidence = result.invariant_evidence
    if evidence.status != "VERIFIED":
        raise MonitoringError("monitoring persistence requires VERIFIED invariant evidence")
    try:
        payload = evidence.payload()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MonitoringError("monitoring invariant evidence is not valid JSON") from exc
    if _canonical_json(payload) != evidence.evidence_json:
        raise MonitoringError("monitoring invariant evidence is not canonical JSON")
    if _sha256_text(evidence.evidence_json) != evidence.evidence_sha256:
        raise MonitoringError("monitoring invariant evidence digest does not match its JSON")
    if (
        payload.get("status") != "VERIFIED"
        or payload.get("variant") != result.variant.value
        or payload.get("contract_sha256") != result.contract.contract_sha256
    ):
        raise MonitoringError("monitoring invariant evidence does not identify this fit result")
    return evidence


def run_monitoring_fit(
    baseline_model: SuperGLM,
    X: pd.DataFrame,
    y: Any,
    *,
    variant: MonitoringVariant | str,
    sample_weight: Any = None,
    offset: Any = None,
    offset_contract: OffsetExportContract | None = None,
    fit_sample_weight_name: str | None = None,
    export_weight_name: str | None = None,
    continuous_points: int = 101,
    max_reml_iter: int = 20,
    reml_tol: float | None = None,
    runtime_validation: str | bool = "auto",
) -> MonitoringFitResult:
    """Score or refit one preset and return SQL-ready lightweight evidence."""
    baseline = _require_fitted_superglm(baseline_model)
    resolved_variant = MonitoringVariant(variant)
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if len(X) != len(y):
        raise ValueError("X and y must have the same row count")
    resolved_offset = offset_contract or OffsetExportContract(handling="NONE")
    contract = build_model_fit_contract(
        baseline,
        offset_contract=resolved_offset,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
        continuous_points=continuous_points,
    )
    if resolved_variant is MonitoringVariant.STATIC_SCORE:
        fitted = baseline
    else:
        fitted = materialize_monitoring_model(baseline, resolved_variant)
        fitted.fit_reml(
            X,
            np.asarray(y),
            sample_weight=sample_weight,
            offset=offset,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            runtime_validation=runtime_validation,
        )
    invariant_evidence = _verify_monitoring_invariants(
        baseline,
        fitted,
        variant=resolved_variant,
        contract=contract,
        offset_contract=resolved_offset,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )
    payload = contract.payload()
    return MonitoringFitResult(
        variant=resolved_variant,
        contract=contract,
        fitted_model=fitted,
        terms=_result_terms(
            fitted,
            offset_contract=resolved_offset,
            fit_sample_weight_name=fit_sample_weight_name,
            export_weight_name=export_weight_name,
        ),
        lambdas=_result_lambdas(fitted, resolved_variant),
        relativities=_result_relativities(fitted, payload["evaluation_grid"]),
        metrics=_result_metrics(fitted, X, y, sample_weight, offset),
        invariant_evidence=invariant_evidence,
    )


def _run_signature(
    *,
    baseline_deployment_id: int,
    manifest_id: str,
    variant: MonitoringVariant,
    contract_sha256: str,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "baseline_deployment_id": int(baseline_deployment_id),
                "manifest_id": manifest_id,
                "variant": variant.value,
                "contract_sha256": contract_sha256,
            }
        )
    )


def persist_monitoring_fit(
    engine,
    result: MonitoringFitResult,
    *,
    baseline_model_run_id: str | int,
    baseline_deployment_id: int,
    manifest_id: str,
    created_by: str,
    component_role: str = "OTHER",
) -> PersistedMonitoringRun:
    """Persist one completed observation, deduplicating an exact retry.

    SQLite stores the local mirror in ``pricing`` so persistent views remain
    usable when ``pricing.sqlite`` is opened directly.  SQL Server stores the
    same logical tables under ``mlops``.
    """
    invariant_evidence = _validate_persistable_invariant_evidence(result)
    if not (
        isinstance(baseline_model_run_id, int)
        and not isinstance(baseline_model_run_id, bool)
        and baseline_model_run_id > 0
    ) and not (isinstance(baseline_model_run_id, str) and baseline_model_run_id.strip()):
        raise ValueError("baseline_model_run_id is required")
    for value, label in (
        (manifest_id, "manifest_id"),
        (created_by, "created_by"),
        (component_role, "component_role"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is required")
    component = component_role.strip().upper()
    if component not in {"FREQUENCY", "SEVERITY", "OTHER"}:
        raise ValueError("component_role must be FREQUENCY, SEVERITY, or OTHER")

    schemas = schema_names_from_connectable(engine)
    pricing_schema = schemas.pricing
    monitor_schema = pricing_schema if engine.dialect.name == "sqlite" else schemas.mlops
    signature = _run_signature(
        baseline_deployment_id=baseline_deployment_id,
        manifest_id=manifest_id,
        variant=result.variant,
        contract_sha256=result.contract.contract_sha256,
    )

    with engine.begin() as connection:
        baseline = (
            connection.execute(
                text(
                    f"""
                    SELECT mr.model_id, mr.rate_package_id
                    FROM {pricing_schema}.MODEL_RUN AS mr
                    JOIN {pricing_schema}.PRICING_RATE_PACKAGE AS rp
                      ON rp.rate_package_id = mr.rate_package_id
                    WHERE mr.model_run_id = :model_run_id
                      AND mr.run_status = 'SUCCESS'
                      AND rp.package_status = 'PUBLISHED'
                    """
                ),
                {"model_run_id": baseline_model_run_id},
            )
            .mappings()
            .one_or_none()
        )
        if baseline is None:
            raise MonitoringError(
                "baseline_model_run_id must identify a successful published model run"
            )
        deployment = connection.execute(
            text(
                f"""
                    SELECT deployment_id
                    FROM {pricing_schema}.PRICING_MODEL_DEPLOYMENT
                    WHERE deployment_id = :deployment_id
                      AND model_id = :model_id
                      AND rate_package_id = :rate_package_id
                    """
            ),
            {
                "deployment_id": int(baseline_deployment_id),
                "model_id": baseline["model_id"],
                "rate_package_id": baseline["rate_package_id"],
            },
        ).scalar_one_or_none()
        if deployment is None:
            raise MonitoringError(
                "baseline_deployment_id does not identify the supplied published model run"
            )
        manifest_exists = connection.execute(
            text(
                f"SELECT 1 FROM {pricing_schema}.DATASET_MANIFEST WHERE manifest_id = :manifest_id"
            ),
            {"manifest_id": manifest_id},
        ).scalar_one_or_none()
        if manifest_exists is None:
            raise MonitoringError("manifest_id does not exist")

        existing_contract = (
            connection.execute(
                text(
                    f"""
                    SELECT fit_contract_id, contract_sha256
                    FROM {monitor_schema}.MODEL_FIT_CONTRACT
                    WHERE baseline_model_run_id = :model_run_id
                    """
                ),
                {"model_run_id": baseline_model_run_id},
            )
            .mappings()
            .one_or_none()
        )
        if existing_contract is None:
            fit_contract_id = str(uuid.uuid4())
            connection.execute(
                text(
                    f"""
                    INSERT INTO {monitor_schema}.MODEL_FIT_CONTRACT (
                        fit_contract_id, baseline_model_run_id, model_id,
                        rate_package_id, contract_schema_version,
                        contract_sha256, structure_sha256, contract_json,
                        superglm_version, created_by
                    ) VALUES (
                        :fit_contract_id, :baseline_model_run_id, :model_id,
                        :rate_package_id, :schema_version,
                        :contract_sha256, :structure_sha256, :contract_json,
                        :superglm_version, :created_by
                    )
                    """
                ),
                {
                    "fit_contract_id": fit_contract_id,
                    "baseline_model_run_id": baseline_model_run_id,
                    "model_id": baseline["model_id"],
                    "rate_package_id": baseline["rate_package_id"],
                    "schema_version": FIT_CONTRACT_SCHEMA_VERSION,
                    "contract_sha256": result.contract.contract_sha256,
                    "structure_sha256": result.contract.structure_sha256,
                    "contract_json": result.contract.contract_json,
                    "superglm_version": result.contract.superglm_version,
                    "created_by": created_by.strip(),
                },
            )
        else:
            fit_contract_id = str(existing_contract["fit_contract_id"])
            if existing_contract["contract_sha256"] != result.contract.contract_sha256:
                raise MonitoringError(
                    "the immutable fit contract for baseline_model_run_id has changed; "
                    "publish/deploy a new baseline rather than mutating its contract"
                )

        existing_run = connection.execute(
            text(
                f"""
                    SELECT monitor_run_id
                    FROM {monitor_schema}.MODEL_MONITOR_RUN
                    WHERE run_signature_sha256 = :signature
                    """
            ),
            {"signature": signature},
        ).scalar_one_or_none()
        if existing_run is not None:
            return PersistedMonitoringRun(
                monitor_run_id=str(existing_run),
                fit_contract_id=fit_contract_id,
                run_signature_sha256=signature,
                deduplicated=True,
            )

        monitor_run_id = str(uuid.uuid4())
        connection.execute(
            text(
                f"""
                INSERT INTO {monitor_schema}.MODEL_MONITOR_RUN (
                    monitor_run_id, fit_contract_id, baseline_deployment_id,
                    model_id, rate_package_id, manifest_id, component_role,
                    variant_code, run_signature_sha256, run_status,
                    invariant_status, invariant_evidence_sha256,
                    invariant_evidence_json, created_by
                ) VALUES (
                    :monitor_run_id, :fit_contract_id, :baseline_deployment_id,
                    :model_id, :rate_package_id, :manifest_id, :component_role,
                    :variant_code, :signature, 'SUCCESS',
                    :invariant_status, :invariant_evidence_sha256,
                    :invariant_evidence_json, :created_by
                )
                """
            ),
            {
                "monitor_run_id": monitor_run_id,
                "fit_contract_id": fit_contract_id,
                "baseline_deployment_id": int(baseline_deployment_id),
                "model_id": baseline["model_id"],
                "rate_package_id": baseline["rate_package_id"],
                "manifest_id": manifest_id,
                "component_role": component,
                "variant_code": result.variant.value,
                "signature": signature,
                "invariant_status": invariant_evidence.status,
                "invariant_evidence_sha256": invariant_evidence.evidence_sha256,
                "invariant_evidence_json": invariant_evidence.evidence_json,
                "created_by": created_by.strip(),
            },
        )
        if result.terms:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {monitor_schema}.MODEL_MONITOR_TERM (
                        monitor_run_id, term_name, term_kind, sequence_no,
                        term_structure_sha256, term_metadata_json
                    ) VALUES (
                        :monitor_run_id, :term_name, :term_kind, :sequence_no,
                        :structure_sha256, :metadata_json
                    )
                    """
                ),
                [
                    {
                        "monitor_run_id": monitor_run_id,
                        "term_name": row.term_name,
                        "term_kind": row.term_kind,
                        "sequence_no": row.sequence_no,
                        "structure_sha256": row.structure_sha256,
                        "metadata_json": row.metadata_json,
                    }
                    for row in result.terms
                ],
            )
        if result.lambdas:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {monitor_schema}.MODEL_MONITOR_LAMBDA (
                        monitor_run_id, component_name, term_name,
                        lambda_value, lambda_mode
                    ) VALUES (
                        :monitor_run_id, :component_name, :term_name,
                        :lambda_value, :lambda_mode
                    )
                    """
                ),
                [
                    {
                        "monitor_run_id": monitor_run_id,
                        "component_name": row.component_name,
                        "term_name": row.term_name,
                        "lambda_value": row.lambda_value,
                        "lambda_mode": row.lambda_mode,
                    }
                    for row in result.lambdas
                ],
            )
        if result.relativities:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {monitor_schema}.MODEL_MONITOR_RELATIVITY (
                        monitor_run_id, term_name, term_kind, point_key,
                        point_label, point_numeric, relativity,
                        log_relativity, is_reference
                    ) VALUES (
                        :monitor_run_id, :term_name, :term_kind, :point_key,
                        :point_label, :point_numeric, :relativity,
                        :log_relativity, :is_reference
                    )
                    """
                ),
                [
                    {
                        "monitor_run_id": monitor_run_id,
                        "term_name": row.term_name,
                        "term_kind": row.term_kind,
                        "point_key": row.point_key,
                        "point_label": row.point_label,
                        "point_numeric": row.point_numeric,
                        "relativity": row.relativity,
                        "log_relativity": row.log_relativity,
                        "is_reference": int(row.is_reference),
                    }
                    for row in result.relativities
                ],
            )
        if result.metrics:
            connection.execute(
                text(
                    f"""
                    INSERT INTO {monitor_schema}.MODEL_MONITOR_METRIC (
                        monitor_run_id, metric_name, metric_value
                    ) VALUES (
                        :monitor_run_id, :metric_name, :metric_value
                    )
                    """
                ),
                [
                    {
                        "monitor_run_id": monitor_run_id,
                        "metric_name": name,
                        "metric_value": value,
                    }
                    for name, value in result.metrics.items()
                ],
            )

    return PersistedMonitoringRun(
        monitor_run_id=monitor_run_id,
        fit_contract_id=fit_contract_id,
        run_signature_sha256=signature,
        deduplicated=False,
    )


__all__ = [
    "FIT_CONTRACT_SCHEMA",
    "FIT_CONTRACT_SCHEMA_VERSION",
    "INVARIANT_EVIDENCE_SCHEMA",
    "INVARIANT_EVIDENCE_SCHEMA_VERSION",
    "MONITORING_VARIANT_POLICIES",
    "ModelFitContract",
    "MonitoringError",
    "MonitoringFitResult",
    "MonitoringInvariantEvidence",
    "MonitoringLambda",
    "MonitoringRelativity",
    "MonitoringTerm",
    "MonitoringVariant",
    "MonitoringVariantPolicy",
    "PersistedMonitoringRun",
    "build_model_fit_contract",
    "materialize_monitoring_model",
    "persist_monitoring_fit",
    "run_monitoring_fit",
]
