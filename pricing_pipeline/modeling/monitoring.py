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
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.metadata import version as package_version
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.constraint import ConstraintSpec
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline, _SplineBase
from superglm.types import LambdaPolicy

from pricing_pipeline.data.manifest import model_frame_evidence
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.superglm_metadata import (
    _spline_kind,
    build_superglm_publication_receipt,
)
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    canonical_receipt_bytes,
)
from pricing_pipeline.workbench.artifacts import (
    CandidateArtifactError,
    CandidateBundle,
    load_candidate_bundle,
)
from pricing_pipeline.workbench.core import Candidate, CandidateLineageError

FIT_CONTRACT_SCHEMA = "superglm_monitoring_fit_contract"
FIT_CONTRACT_SCHEMA_VERSION = 1
INVARIANT_EVIDENCE_SCHEMA = "superglm_monitoring_invariant_evidence"
INVARIANT_EVIDENCE_SCHEMA_VERSION = 1
RESULT_EVIDENCE_SCHEMA = "superglm_monitoring_result_evidence"
RESULT_EVIDENCE_SCHEMA_VERSION = 1
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
    model_frame_sha256: str | None
    fit_configuration_json: str
    result_evidence_sha256: str


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


def _required_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MonitoringError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _verified_candidate_baseline(
    candidate: Candidate,
) -> tuple[SuperGLM, dict[str, Any], CandidateBundle]:
    try:
        refreshed = candidate.workbench.open(
            candidate.model_name,
            package_version=candidate.package_version,
        )
    except (CandidateArtifactError, CandidateLineageError, TypeError, ValueError) as exc:
        raise MonitoringError(
            "monitoring baseline candidate could not be refreshed from SQL"
        ) from exc
    if not isinstance(refreshed, Candidate):
        raise MonitoringError("monitoring baseline refresh did not return a Candidate")
    candidate = refreshed
    technical = candidate.technical
    if str(technical.get("package_status") or "").upper() != "PUBLISHED":
        raise MonitoringError("monitoring baseline candidate must be PUBLISHED")
    if str(technical.get("run_status") or "").upper() != "SUCCESS":
        raise MonitoringError("monitoring baseline candidate run must be SUCCESS")

    expected_identity = {
        "model_run_id": candidate.model_run_id,
        "rate_package_id": candidate.rate_package_id,
        "model_name": candidate.model_name,
    }
    for field_name, expected in expected_identity.items():
        if technical.get(field_name) != expected:
            raise MonitoringError(
                f"monitoring baseline candidate {field_name} does not match its SQL evidence"
            )
    current_rate_package_id = technical.get("current_rate_package_id")
    current_deployment_id = technical.get("current_deployment_id")
    if current_rate_package_id != candidate.rate_package_id or current_deployment_id is None:
        raise MonitoringError("monitoring baseline candidate is not the current deployment")

    try:
        bundle = load_candidate_bundle(
            technical.get("candidate_artifact_path"),
            expected_sha256=_required_sha256(
                technical.get("candidate_artifact_sha256"),
                "candidate_artifact_sha256",
            ),
            expected_size_bytes=int(technical.get("candidate_artifact_size_bytes")),
            expected_format=str(technical.get("candidate_artifact_format") or ""),
            expected_python_version=str(technical.get("candidate_python_version") or ""),
            expected_superglm_version=str(technical.get("candidate_superglm_version") or ""),
            allowed_root=candidate.workbench.settings.workbench_artifact_root,
        )
    except (CandidateArtifactError, TypeError, ValueError) as exc:
        raise MonitoringError(
            "monitoring baseline candidate artifact could not be re-verified"
        ) from exc
    if bundle.model_name != candidate.model_name:
        raise MonitoringError("monitoring baseline artifact model_name does not match SQL")
    if bundle.model_version != technical.get("model_version"):
        raise MonitoringError("monitoring baseline artifact model_version does not match SQL")
    if bundle.export_id != technical.get("export_id"):
        raise MonitoringError("monitoring baseline artifact export_id does not match SQL")
    if bundle.manifest_id != technical.get("manifest_id"):
        raise MonitoringError("monitoring baseline artifact manifest_id does not match SQL")
    if bundle.model_source_sha256 != technical.get("model_source_sha256"):
        raise MonitoringError("monitoring baseline artifact source digest does not match SQL")
    if bundle.model_frame_sha256 != technical.get("model_frame_sha256"):
        raise MonitoringError("monitoring baseline artifact frame digest does not match SQL")
    if bundle.split_set_id != technical.get("split_set_id"):
        raise MonitoringError("monitoring baseline artifact split_set_id does not match SQL")
    receipt_sha256 = _required_sha256(
        technical.get("publication_receipt_sha256"),
        "publication_receipt_sha256",
    )
    package_receipt_sha256 = _required_sha256(
        technical.get("package_publication_receipt_sha256"),
        "package_publication_receipt_sha256",
    )
    if receipt_sha256 != package_receipt_sha256:
        raise MonitoringError(
            "monitoring baseline run and package publication receipts do not match"
        )
    fitted_model = _require_fitted_superglm(bundle.fitted_model)
    try:
        rebuilt_receipt = build_superglm_publication_receipt(
            fitted_model,
            offset_contract=bundle.offset_contract,
            fit_sample_weight_name=bundle.fit_sample_weight_name,
            export_weight_name=bundle.export_weight_name,
        )
    except (TypeError, ValueError) as exc:
        raise MonitoringError(
            "monitoring baseline candidate publication receipt could not be rebuilt"
        ) from exc
    rebuilt_receipt_sha256 = hashlib.sha256(canonical_receipt_bytes(rebuilt_receipt)).hexdigest()
    if rebuilt_receipt_sha256 != receipt_sha256 or rebuilt_receipt_sha256 != package_receipt_sha256:
        raise MonitoringError(
            "monitoring baseline candidate publication receipt does not match SQL lineage"
        )

    artifact_format = str(technical.get("candidate_artifact_format") or "").strip()
    python_version = str(technical.get("candidate_python_version") or "").strip()
    superglm_version = str(technical.get("candidate_superglm_version") or "").strip()
    deployment_slot = str(candidate.workbench.model_config.deployment_slot or "").strip()
    data_as_of_date = str(technical.get("data_as_of_date") or "").strip()
    for value, field_name in (
        (artifact_format, "candidate_artifact_format"),
        (python_version, "candidate_python_version"),
        (superglm_version, "candidate_superglm_version"),
        (deployment_slot, "deployment_slot"),
        (data_as_of_date, "data_as_of_date"),
    ):
        if not value:
            raise MonitoringError(f"monitoring baseline {field_name} is required")
    artifact_size = int(technical.get("candidate_artifact_size_bytes"))
    if artifact_size <= 0:
        raise MonitoringError("monitoring baseline candidate artifact size must be positive")

    baseline = {
        "candidate_artifact_format": artifact_format,
        "candidate_artifact_sha256": _required_sha256(
            technical.get("candidate_artifact_sha256"),
            "candidate_artifact_sha256",
        ),
        "candidate_artifact_size_bytes": artifact_size,
        "candidate_python_version": python_version,
        "candidate_superglm_version": superglm_version,
        "data_as_of_date": data_as_of_date,
        "deployment_id": int(current_deployment_id),
        "deployment_slot": deployment_slot,
        "export_id": bundle.export_id,
        "manifest_id": str(technical.get("manifest_id") or ""),
        "model_equivalence_sha256": _required_sha256(
            technical.get("model_equivalence_sha256"),
            "model_equivalence_sha256",
        ),
        "model_frame_sha256": _required_sha256(
            technical.get("model_frame_sha256"),
            "baseline model_frame_sha256",
        ),
        "model_id": int(technical.get("model_id")),
        "model_run_id": candidate.model_run_id,
        "model_source_sha256": _required_sha256(
            technical.get("model_source_sha256"),
            "model_source_sha256",
        ),
        "model_version": bundle.model_version,
        "package_version": int(candidate.package_version),
        "package_publication_receipt_sha256": receipt_sha256,
        "publication_receipt_sha256": receipt_sha256,
        "rate_package_id": candidate.rate_package_id,
        "row_order_sha256": _required_sha256(
            bundle.row_order_sha256,
            "row_order_sha256",
        ),
        "split_set_id": bundle.split_set_id,
    }
    return fitted_model, baseline, bundle


def _resolve_monitoring_baseline(
    value: SuperGLM | Candidate,
) -> tuple[SuperGLM, dict[str, Any] | None, CandidateBundle | None]:
    if isinstance(value, Candidate):
        return _verified_candidate_baseline(value)
    return _require_fitted_superglm(value), None, None


def _ordered_series_matches(left: Any, right: pd.Series) -> bool:
    values = np.asarray(left)
    if values.ndim != 1 or len(values) != len(right):
        return False
    return pd.Series(values).reset_index(drop=True).equals(right.reset_index(drop=True))


def _bind_monitoring_model_frame(
    X: pd.DataFrame,
    y: Any,
    *,
    model_frame: pd.DataFrame | None,
    target_column: str | None,
    sample_weight: Any,
    fit_sample_weight_name: str | None,
    offset: Any,
    offset_column: str | None,
) -> str | None:
    if model_frame is None:
        if target_column is not None or offset_column is not None:
            raise MonitoringError("target_column and offset_column require the ordered model_frame")
        return None
    if not isinstance(model_frame, pd.DataFrame) or model_frame.empty:
        raise ValueError("model_frame must be a non-empty pandas DataFrame")
    if not isinstance(target_column, str) or not target_column.strip():
        raise MonitoringError("target_column is required with model_frame")
    target_name = target_column.strip()
    required_columns = [*X.columns, target_name]
    if sample_weight is not None:
        if not isinstance(fit_sample_weight_name, str) or not fit_sample_weight_name.strip():
            raise MonitoringError(
                "fit_sample_weight_name is required to bind sample_weight to model_frame"
            )
        required_columns.append(fit_sample_weight_name.strip())
    if offset is not None:
        if not isinstance(offset_column, str) or not offset_column.strip():
            raise MonitoringError("offset_column is required to bind offset to model_frame")
        required_columns.append(offset_column.strip())
    missing = [str(column) for column in required_columns if column not in model_frame.columns]
    if missing:
        raise MonitoringError(
            "ordered model frame is missing monitoring columns: " + ", ".join(missing)
        )
    if len(model_frame) != len(X):
        raise MonitoringError("X does not match the ordered model frame row count")
    actual_X = X.reset_index(drop=True)
    expected_X = model_frame.loc[:, list(X.columns)].reset_index(drop=True)
    if not actual_X.equals(expected_X):
        raise MonitoringError("X does not match the ordered model frame")
    if not _ordered_series_matches(y, model_frame[target_name]):
        raise MonitoringError("y does not match target_column in the ordered model frame")
    if sample_weight is not None:
        weight_name = str(fit_sample_weight_name).strip()
        if not _ordered_series_matches(sample_weight, model_frame[weight_name]):
            raise MonitoringError(
                "sample_weight does not match fit_sample_weight_name in the ordered model frame"
            )
    if offset is not None:
        resolved_offset_column = str(offset_column).strip()
        if not _ordered_series_matches(offset, model_frame[resolved_offset_column]):
            raise MonitoringError("offset does not match offset_column in the ordered model frame")
    return model_frame_evidence(model_frame)[0]


def _monitoring_fit_configuration_json(
    *,
    variant: MonitoringVariant,
    baseline_identity: Mapping[str, Any] | None,
    model_frame_sha256: str | None,
    target_column: str | None,
    fit_sample_weight_name: str | None,
    offset_column: str | None,
    offset_contract: OffsetExportContract,
    continuous_points: int,
    max_reml_iter: int,
    reml_tol: float | None,
    runtime_validation: str | bool,
) -> str:
    refits = variant is not MonitoringVariant.STATIC_SCORE
    return _canonical_json(
        {
            "schema_name": "superglm_monitoring_fit_configuration",
            "schema_version": 1,
            "variant": variant.value,
            "baseline": baseline_identity,
            "model_frame_sha256": model_frame_sha256,
            "target_column": target_column,
            "fit_sample_weight_name": fit_sample_weight_name,
            "offset_column": offset_column,
            "offset_contract": offset_contract.model_dump(mode="json"),
            "continuous_points": continuous_points,
            "max_reml_iter": max_reml_iter if refits else None,
            "reml_tol": reml_tol if refits else None,
            "runtime_validation": runtime_validation if refits else None,
        }
    )


def _monitoring_result_evidence_sha256(result: MonitoringFitResult) -> str:
    payload = {
        "schema_name": RESULT_EVIDENCE_SCHEMA,
        "schema_version": RESULT_EVIDENCE_SCHEMA_VERSION,
        "variant": result.variant.value,
        "contract_sha256": result.contract.contract_sha256,
        "model_frame_sha256": result.model_frame_sha256,
        "fit_configuration": json.loads(result.fit_configuration_json),
        "terms": [
            {
                "term_name": row.term_name,
                "term_kind": row.term_kind,
                "sequence_no": row.sequence_no,
                "metadata_json": json.loads(row.metadata_json),
                "structure_sha256": row.structure_sha256,
            }
            for row in result.terms
        ],
        "lambdas": [
            {
                "term_name": row.term_name,
                "component_name": row.component_name,
                "lambda_value": row.lambda_value,
                "lambda_mode": row.lambda_mode,
            }
            for row in result.lambdas
        ],
        "relativities": [
            {
                "term_name": row.term_name,
                "term_kind": row.term_kind,
                "point_key": row.point_key,
                "point_label": row.point_label,
                "point_numeric": row.point_numeric,
                "relativity": row.relativity,
                "log_relativity": row.log_relativity,
                "is_reference": row.is_reference,
            }
            for row in result.relativities
        ],
        "metrics": dict(result.metrics),
        "invariant_evidence_sha256": result.invariant_evidence.evidence_sha256,
    }
    return _sha256_text(_canonical_json(payload))


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


def _requested_continuous_values(
    model: SuperGLM,
    term_name: str,
    points: list[Any],
) -> list[tuple[Any, float, float]]:
    requested_x = np.asarray(points, dtype=float)
    if requested_x.ndim != 1 or not np.isfinite(requested_x).all():
        raise MonitoringError(f"term {term_name!r} has an invalid frozen numeric grid")
    spec = model._specs.get(term_name)
    feature_groups = [group for group in model._groups if group.feature_name == term_name]
    if spec is None or not feature_groups:
        raise MonitoringError(f"term {term_name!r} has no fitted continuous contribution")
    beta_combined = np.concatenate(
        [np.asarray(model.result.beta[group.sl], dtype=float).ravel() for group in feature_groups]
    )
    transformed = np.asarray(spec.transform(requested_x), dtype=float)
    expected_shape = (len(requested_x), len(beta_combined))
    if transformed.shape != expected_shape:
        raise MonitoringError(
            f"term {term_name!r} transform returned {transformed.shape}, expected {expected_shape}"
        )
    if not np.isfinite(beta_combined).all() or not np.isfinite(transformed).all():
        raise MonitoringError(f"term {term_name!r} continuous contribution is not finite")
    requested_log = transformed @ beta_combined
    with np.errstate(over="ignore", invalid="ignore"):
        requested_relativity = np.exp(requested_log)
    if not np.isfinite(requested_log).all() or not np.isfinite(requested_relativity).all():
        raise MonitoringError(f"term {term_name!r} continuous relativity is not finite")
    return [
        (point, float(relativity), float(log_relativity))
        for point, relativity, log_relativity in zip(
            points,
            requested_relativity,
            requested_log,
            strict=True,
        )
    ]


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
        elif kind == "continuous":
            values = _requested_continuous_values(model, term_name, points)
        else:
            inference = model.term_inference(
                term_name,
                with_se=False,
                n_points=max(501, len(points)),
                centering="native",
            )
            if kind == "categorical":
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
    observed = np.asarray(y, dtype=float)
    diagnostics = model.metrics(X, observed, sample_weight, offset)
    weights = (
        np.ones(len(X), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    weight_sum = float(np.sum(weights))
    weighted_observed_sum = float(np.dot(weights, observed))
    weighted_prediction_sum = float(np.dot(weights, predictions))
    metrics = {
        "row_count": float(len(X)),
        "sample_weight_sum": weight_sum,
        "sample_weighted_mean_observed": weighted_observed_sum / weight_sum,
        "sample_weighted_mean_prediction": weighted_prediction_sum / weight_sum,
        "sample_weighted_sum_observed": weighted_observed_sum,
        "sample_weighted_sum_prediction": weighted_prediction_sum,
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


def _validate_persistable_result_evidence(
    result: MonitoringFitResult,
) -> tuple[str, dict[str, Any]]:
    frame_digest = result.model_frame_sha256
    if (
        not isinstance(frame_digest, str)
        or len(frame_digest) != 64
        or frame_digest != frame_digest.lower()
        or any(character not in "0123456789abcdef" for character in frame_digest)
    ):
        raise MonitoringError(
            "monitoring persistence requires an exact model_frame_sha256 from run_monitoring_fit"
        )
    try:
        fit_configuration = json.loads(result.fit_configuration_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MonitoringError("monitoring fit configuration is not valid JSON") from exc
    if _canonical_json(fit_configuration) != result.fit_configuration_json:
        raise MonitoringError("monitoring fit configuration is not canonical JSON")
    baseline = fit_configuration.get("baseline")
    if not isinstance(baseline, dict):
        raise MonitoringError(
            "monitoring persistence requires a verified deployed candidate artifact"
        )
    for field_name in (
        "candidate_artifact_sha256",
        "model_equivalence_sha256",
        "model_frame_sha256",
        "model_source_sha256",
        "package_publication_receipt_sha256",
        "publication_receipt_sha256",
        "row_order_sha256",
    ):
        _required_sha256(baseline.get(field_name), f"baseline {field_name}")
    for field_name in (
        "deployment_id",
        "model_id",
        "model_run_id",
        "rate_package_id",
    ):
        value = baseline.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int | str) or not str(value).strip():
            raise MonitoringError(f"baseline {field_name} is required")
    if not isinstance(baseline.get("manifest_id"), str) or not baseline["manifest_id"].strip():
        raise MonitoringError("baseline manifest_id is required")
    expected = _monitoring_result_evidence_sha256(result)
    if result.result_evidence_sha256 != expected:
        raise MonitoringError("monitoring result evidence digest does not match its details")
    return expected, baseline


def run_monitoring_fit(
    baseline_model: SuperGLM | Candidate,
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
    model_frame: pd.DataFrame | None = None,
    target_column: str | None = None,
    offset_column: str | None = None,
) -> MonitoringFitResult:
    """Score or refit one preset and return SQL-ready lightweight evidence."""
    baseline, baseline_identity, baseline_bundle = _resolve_monitoring_baseline(baseline_model)
    resolved_variant = MonitoringVariant(variant)
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if len(X) != len(y):
        raise ValueError("X and y must have the same row count")
    if baseline_bundle is None:
        resolved_offset = offset_contract or OffsetExportContract(handling="NONE")
    else:
        resolved_offset = baseline_bundle.offset_contract
        if offset_contract is not None and offset_contract != resolved_offset:
            raise MonitoringError(
                "offset_contract does not match the verified baseline candidate artifact"
            )
        for supplied, expected, field_name in (
            (
                fit_sample_weight_name,
                baseline_bundle.fit_sample_weight_name,
                "fit_sample_weight_name",
            ),
            (
                export_weight_name,
                baseline_bundle.export_weight_name,
                "export_weight_name",
            ),
        ):
            if supplied is not None and supplied != expected:
                raise MonitoringError(
                    f"{field_name} does not match the verified baseline candidate artifact"
                )
        fit_sample_weight_name = baseline_bundle.fit_sample_weight_name
        export_weight_name = baseline_bundle.export_weight_name
        if fit_sample_weight_name is not None and sample_weight is None:
            raise MonitoringError(
                "sample_weight is required by the verified baseline candidate fit contract"
            )
        if resolved_offset.handling != "NONE" and offset is None:
            raise MonitoringError(
                "offset is required by the verified baseline candidate fit contract"
            )
        if resolved_offset.handling == "NONE" and offset is not None:
            raise MonitoringError(
                "offset was not used by the verified baseline candidate fit contract"
            )
    model_frame_sha256 = _bind_monitoring_model_frame(
        X,
        y,
        model_frame=model_frame,
        target_column=target_column,
        sample_weight=sample_weight,
        fit_sample_weight_name=fit_sample_weight_name,
        offset=offset,
        offset_column=offset_column,
    )
    fit_configuration_json = _monitoring_fit_configuration_json(
        variant=resolved_variant,
        baseline_identity=baseline_identity,
        model_frame_sha256=model_frame_sha256,
        target_column=(None if target_column is None else target_column.strip()),
        fit_sample_weight_name=fit_sample_weight_name,
        offset_column=None if offset_column is None else offset_column.strip(),
        offset_contract=resolved_offset,
        continuous_points=continuous_points,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        runtime_validation=runtime_validation,
    )
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
    result = MonitoringFitResult(
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
        model_frame_sha256=model_frame_sha256,
        fit_configuration_json=fit_configuration_json,
        result_evidence_sha256="",
    )
    return replace(
        result,
        result_evidence_sha256=_monitoring_result_evidence_sha256(result),
    )


def _run_signature(
    *,
    baseline_deployment_id: int,
    manifest_id: str,
    variant: MonitoringVariant,
    contract_sha256: str,
    component_role: str,
    result_evidence_sha256: str,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "baseline_deployment_id": int(baseline_deployment_id),
                "manifest_id": manifest_id,
                "variant": variant.value,
                "contract_sha256": contract_sha256,
                "component_role": component_role,
                "result_evidence_sha256": result_evidence_sha256,
            }
        )
    )


def _persist_monitoring_fit_once(
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
    result_evidence_sha256, baseline_identity = _validate_persistable_result_evidence(result)
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
    if str(baseline_identity["model_run_id"]) != str(baseline_model_run_id):
        raise MonitoringError("baseline_model_run_id does not match the verified candidate")
    if int(baseline_identity["deployment_id"]) != int(baseline_deployment_id):
        raise MonitoringError("baseline_deployment_id does not match the verified candidate")

    schemas = schema_names_from_connectable(engine)
    pricing_schema = schemas.pricing
    monitor_schema = pricing_schema if engine.dialect.name == "sqlite" else schemas.mlops
    signature = _run_signature(
        baseline_deployment_id=baseline_deployment_id,
        manifest_id=manifest_id,
        variant=result.variant,
        contract_sha256=result.contract.contract_sha256,
        component_role=component,
        result_evidence_sha256=result_evidence_sha256,
    )

    with engine.begin() as connection:
        baseline = (
            connection.execute(
                text(
                    f"""
                    SELECT
                        mr.model_id,
                        mr.rate_package_id,
                        mr.model_version,
                        mr.export_id,
                        mr.manifest_id AS baseline_manifest_id,
                        mr.candidate_artifact_format,
                        mr.candidate_artifact_sha256,
                        mr.candidate_artifact_size_bytes,
                        mr.candidate_python_version,
                        mr.candidate_superglm_version,
                        mr.model_equivalence_sha256,
                        mr.model_source_sha256,
                        mr.publication_receipt_sha256,
                        rp.package_version,
                        rp.publication_receipt_sha256
                            AS package_publication_receipt_sha256,
                        baseline_manifest.model_frame_sha256 AS baseline_model_frame_sha256,
                        baseline_manifest.data_as_of_date AS baseline_data_as_of_date
                    FROM {pricing_schema}.MODEL_RUN AS mr
                    JOIN {pricing_schema}.PRICING_RATE_PACKAGE AS rp
                      ON rp.rate_package_id = mr.rate_package_id
                    JOIN {pricing_schema}.DATASET_MANIFEST AS baseline_manifest
                      ON baseline_manifest.manifest_id = mr.manifest_id
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
        expected_baseline = {
            "candidate_artifact_format": baseline["candidate_artifact_format"],
            "candidate_artifact_sha256": baseline["candidate_artifact_sha256"],
            "candidate_artifact_size_bytes": baseline["candidate_artifact_size_bytes"],
            "candidate_python_version": baseline["candidate_python_version"],
            "candidate_superglm_version": baseline["candidate_superglm_version"],
            "data_as_of_date": baseline["baseline_data_as_of_date"],
            "export_id": baseline["export_id"],
            "manifest_id": baseline["baseline_manifest_id"],
            "model_equivalence_sha256": baseline["model_equivalence_sha256"],
            "model_frame_sha256": baseline["baseline_model_frame_sha256"],
            "model_id": baseline["model_id"],
            "model_run_id": baseline_model_run_id,
            "model_source_sha256": baseline["model_source_sha256"],
            "model_version": baseline["model_version"],
            "package_version": baseline["package_version"],
            "package_publication_receipt_sha256": baseline["package_publication_receipt_sha256"],
            "publication_receipt_sha256": baseline["publication_receipt_sha256"],
            "rate_package_id": baseline["rate_package_id"],
        }
        mismatches = [
            field_name
            for field_name, expected in expected_baseline.items()
            if str(baseline_identity.get(field_name)) != str(expected)
        ]
        if mismatches:
            raise MonitoringError(
                "verified baseline candidate does not match SQL lineage: " + ", ".join(mismatches)
            )
        split_rows = (
            connection.execute(
                text(
                    f"""
                    SELECT split_set_id
                    FROM {schemas.mlops}.MODEL_RUN_SPLIT_SET
                    WHERE model_run_id = :model_run_id
                      AND manifest_id = :manifest_id
                      AND dataset_role = 'training'
                      AND split_role = 'validation'
                    """
                ),
                {
                    "model_run_id": baseline_model_run_id,
                    "manifest_id": baseline["baseline_manifest_id"],
                },
            )
            .scalars()
            .all()
        )
        if len(split_rows) > 1:
            raise MonitoringError(
                "baseline_model_run_id has ambiguous training/validation split lineage"
            )
        sql_split_set_id = None if not split_rows else str(split_rows[0])
        if baseline_identity.get("split_set_id") != sql_split_set_id:
            raise MonitoringError(
                "verified baseline candidate does not match SQL lineage: split_set_id"
            )
        deployment = (
            connection.execute(
                text(
                    f"""
                    SELECT deployment_id, deployment_slot
                    FROM {pricing_schema}.PRICING_MODEL_DEPLOYMENT
                    WHERE deployment_id = :deployment_id
                      AND model_id = :model_id
                      AND rate_package_id = :rate_package_id
                      AND effective_to_ts IS NULL
                    """
                ),
                {
                    "deployment_id": int(baseline_deployment_id),
                    "model_id": baseline["model_id"],
                    "rate_package_id": baseline["rate_package_id"],
                },
            )
            .mappings()
            .one_or_none()
        )
        if deployment is None:
            raise MonitoringError(
                "baseline_deployment_id does not identify the supplied published model run"
            )
        if str(deployment["deployment_slot"]) != str(baseline_identity.get("deployment_slot")):
            raise MonitoringError(
                "verified baseline candidate does not match SQL lineage: deployment_slot"
            )
        manifest = (
            connection.execute(
                text(
                    f"""
                    SELECT
                        model_frame_sha256,
                        row_count,
                        target_column,
                        weight_column,
                        offset_column,
                        offset_source_column,
                        offset_label,
                        export_weight_column
                    FROM {pricing_schema}.DATASET_MANIFEST
                    WHERE manifest_id = :manifest_id
                    """
                ),
                {"manifest_id": manifest_id},
            )
            .mappings()
            .one_or_none()
        )
        if manifest is None:
            raise MonitoringError("manifest_id does not exist")
        if manifest["model_frame_sha256"] != result.model_frame_sha256:
            raise MonitoringError("monitoring model frame does not match manifest_id")
        if int(manifest["row_count"]) != int(result.metrics.get("row_count", -1)):
            raise MonitoringError("monitoring row count does not match manifest_id")
        fit_configuration = json.loads(result.fit_configuration_json)
        if manifest["target_column"] != fit_configuration.get("target_column"):
            raise MonitoringError("monitoring target column does not match manifest_id")
        try:
            offset_contract = OffsetExportContract.model_validate(
                fit_configuration["offset_contract"]
            )
            contract_model_metadata = result.contract.payload()["structure"]["package_metadata"][
                "model"
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitoringError("monitoring fit/export role contract is malformed") from exc
        expected_manifest_roles = {
            "weight_column": fit_configuration.get("fit_sample_weight_name"),
            "offset_column": fit_configuration.get("offset_column"),
            "offset_source_column": (
                offset_contract.source_name
                if offset_contract.handling == "EXPORTED_FACTOR"
                else None
            ),
            "offset_label": offset_contract.label,
            "export_weight_column": contract_model_metadata.get("export_weight_name"),
        }
        for field_name, expected in expected_manifest_roles.items():
            if manifest[field_name] != expected:
                raise MonitoringError(f"monitoring {field_name} does not match manifest_id")

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

        existing_run = (
            connection.execute(
                text(
                    f"""
                    SELECT monitor_run_id, run_signature_sha256
                    FROM {monitor_schema}.MODEL_MONITOR_RUN
                    WHERE baseline_deployment_id = :baseline_deployment_id
                      AND manifest_id = :manifest_id
                      AND component_role = :component_role
                      AND variant_code = :variant_code
                    """
                ),
                {
                    "baseline_deployment_id": int(baseline_deployment_id),
                    "manifest_id": manifest_id,
                    "component_role": component,
                    "variant_code": result.variant.value,
                },
            )
            .mappings()
            .one_or_none()
        )
        if existing_run is not None:
            if existing_run["run_signature_sha256"] != signature:
                raise MonitoringError(
                    "a monitoring observation already exists with different fit evidence"
                )
            return PersistedMonitoringRun(
                monitor_run_id=str(existing_run["monitor_run_id"]),
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
                    invariant_evidence_json, model_frame_sha256,
                    fit_configuration_json, result_evidence_sha256,
                    created_by
                ) VALUES (
                    :monitor_run_id, :fit_contract_id, :baseline_deployment_id,
                    :model_id, :rate_package_id, :manifest_id, :component_role,
                    :variant_code, :signature, 'SUCCESS',
                    :invariant_status, :invariant_evidence_sha256,
                    :invariant_evidence_json, :model_frame_sha256,
                    :fit_configuration_json, :result_evidence_sha256,
                    :created_by
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
                "model_frame_sha256": result.model_frame_sha256,
                "fit_configuration_json": result.fit_configuration_json,
                "result_evidence_sha256": result_evidence_sha256,
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


def _recover_concurrent_monitoring_retry(
    engine,
    result: MonitoringFitResult,
    *,
    baseline_deployment_id: int,
    manifest_id: str,
    component_role: str,
) -> PersistedMonitoringRun | None:
    result_evidence_sha256, _ = _validate_persistable_result_evidence(result)
    component = component_role.strip().upper()
    signature = _run_signature(
        baseline_deployment_id=baseline_deployment_id,
        manifest_id=manifest_id,
        variant=result.variant,
        contract_sha256=result.contract.contract_sha256,
        component_role=component,
        result_evidence_sha256=result_evidence_sha256,
    )
    schemas = schema_names_from_connectable(engine)
    monitor_schema = schemas.pricing if engine.dialect.name == "sqlite" else schemas.mlops
    with engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    f"""
                    SELECT
                        monitor_run_id,
                        fit_contract_id,
                        run_signature_sha256
                    FROM {monitor_schema}.MODEL_MONITOR_RUN
                    WHERE baseline_deployment_id = :baseline_deployment_id
                      AND manifest_id = :manifest_id
                      AND component_role = :component_role
                      AND variant_code = :variant_code
                    """
                ),
                {
                    "baseline_deployment_id": int(baseline_deployment_id),
                    "manifest_id": manifest_id,
                    "component_role": component,
                    "variant_code": result.variant.value,
                },
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        return None
    if row["run_signature_sha256"] != signature:
        raise MonitoringError(
            "a concurrent monitoring observation committed different fit evidence"
        )
    return PersistedMonitoringRun(
        monitor_run_id=str(row["monitor_run_id"]),
        fit_contract_id=str(row["fit_contract_id"]),
        run_signature_sha256=signature,
        deduplicated=True,
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
    """Persist one observation and exactly recover a concurrent identical retry."""
    try:
        return _persist_monitoring_fit_once(
            engine,
            result,
            baseline_model_run_id=baseline_model_run_id,
            baseline_deployment_id=baseline_deployment_id,
            manifest_id=manifest_id,
            created_by=created_by,
            component_role=component_role,
        )
    except IntegrityError, OperationalError:
        recovered = _recover_concurrent_monitoring_retry(
            engine,
            result,
            baseline_deployment_id=baseline_deployment_id,
            manifest_id=manifest_id,
            component_role=component_role,
        )
        if recovered is None:
            raise
        return recovered


__all__ = [
    "FIT_CONTRACT_SCHEMA",
    "FIT_CONTRACT_SCHEMA_VERSION",
    "INVARIANT_EVIDENCE_SCHEMA",
    "INVARIANT_EVIDENCE_SCHEMA_VERSION",
    "MONITORING_VARIANT_POLICIES",
    "RESULT_EVIDENCE_SCHEMA",
    "RESULT_EVIDENCE_SCHEMA_VERSION",
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
