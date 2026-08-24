"""SuperGLM-specific evidence extraction for aggregate reports."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from superglm import SuperGLM
from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.editor import EditorSession
from superglm.editor.payloads import session_payload
from superglm.profiling.tweedie import tweedie_logpdf

from pricing_pipeline.reporting._core import UnderwriterReportError
from pricing_pipeline.reporting.evidence import (
    MAX_INTERACTION_ROWS,
    MAX_MAIN_EFFECT_GRID_POINTS,
    MAX_SURFACE_AXIS_POINTS,
    CapabilityUnavailable,
    EvidenceFact,
    ExactLossEvidence,
    FeatureImportanceEvidence,
    InteractionEvidence,
    MainEffectEvidence,
    ModelEvidence,
    ReportContext,
)

_MODEL_SOURCE = "SuperGLM object"
_LIKELIHOOD_SOURCE = "fitted SuperGLM object"
_SUPPLIED_SOURCE = "supplied training metadata"
_MAIN_EFFECTS_UNAVAILABLE_REASON = "native main effects are not supported"
_IMPORTANCE_UNAVAILABLE_REASON = "native term importance is not supported"
_EXACT_LOSS_UNAVAILABLE_REASON = "fitted distribution does not expose a supported exact likelihood"
_INTERACTIONS_UNAVAILABLE_REASON = "native interaction reporting is not supported"
_PREDICTION_BINDING_RTOL = 1e-8
_PREDICTION_BINDING_ATOL = 1e-10
_PLOT_KINDS = frozenset(
    {
        "surface",
        "categorical_heatmap",
        "varying_coefficient",
        "numeric_categorical",
        "numeric_numeric",
    }
)
_INTERACTION_CLASS_PLOT_KINDS = {
    "TensorInteraction": "surface",
    "CategoricalInteraction": "categorical_heatmap",
    "SplineCategorical": "varying_coefficient",
    "NumericCategorical": "numeric_categorical",
    "NumericInteraction": "numeric_numeric",
}


@dataclass(frozen=True)
class _LikelihoodSpec:
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


@dataclass(frozen=True)
class SuperGLMReportAdapter:
    """Collect fitted native effects, importance, and exact likelihood evidence."""

    tweedie_power: float | None = None
    dispersion: float | None = None
    n_points: int = 200
    interaction_points: int = 160

    def __post_init__(self) -> None:
        supplied = (self.tweedie_power, self.dispersion)
        if (supplied[0] is None) != (supplied[1] is None):
            raise ValueError("tweedie_power and dispersion must be supplied together")
        if supplied[0] is not None:
            _LikelihoodSpec(supplied[0], supplied[1])
        if not isinstance(self.n_points, int) or isinstance(self.n_points, bool):
            raise TypeError("n_points must be an integer")
        if self.n_points < 2:
            raise ValueError("n_points must be at least 2")
        if self.n_points > MAX_MAIN_EFFECT_GRID_POINTS:
            raise ValueError(f"n_points must be at most {MAX_MAIN_EFFECT_GRID_POINTS}")
        if not isinstance(self.interaction_points, int) or isinstance(
            self.interaction_points, bool
        ):
            raise TypeError("interaction_points must be an integer")
        if not 2 <= self.interaction_points <= MAX_SURFACE_AXIS_POINTS:
            raise ValueError(
                f"interaction_points must be at least 2 and at most {MAX_SURFACE_AXIS_POINTS}"
            )

    def collect(
        self,
        *,
        model_name: str,
        source: object,
        context: ReportContext,
    ) -> ModelEvidence:
        fitted = _likelihood_from_superglm(source)
        if self.tweedie_power is not None:
            supplied = _LikelihoodSpec(self.tweedie_power, self.dispersion)
            if fitted is None or supplied != fitted:
                raise ValueError(
                    f"supplied likelihood metadata for {model_name!r} does not match "
                    "the fitted SuperGLM object"
                )
        _bind_fitted_predictions(model_name, source, context)

        unavailable: list[CapabilityUnavailable] = []
        try:
            main_effects = _model_main_effects(
                source,
                context,
                n_points=self.n_points,
            )
        except NotImplementedError:
            main_effects = {}
            unavailable.append(
                CapabilityUnavailable(
                    "main_effects",
                    _MAIN_EFFECTS_UNAVAILABLE_REASON,
                )
            )

        try:
            importance = _model_importance(source, context)
        except NotImplementedError:
            importance = None
            unavailable.append(
                CapabilityUnavailable(
                    "importance",
                    _IMPORTANCE_UNAVAILABLE_REASON,
                )
            )

        interactions, unavailable_interactions = _model_interactions(
            source,
            context,
            n_points=self.interaction_points,
        )
        interaction_warnings: tuple[str, ...] = ()
        if interactions:
            interaction_warnings = tuple(item.reason for item in unavailable_interactions)
        else:
            unavailable.extend(unavailable_interactions)

        if fitted is None:
            exact_loss = None
            unavailable.append(
                CapabilityUnavailable(
                    "exact_loss",
                    _EXACT_LOSS_UNAVAILABLE_REASON,
                )
            )
        else:
            exact_loss = _exact_loss_evidence(
                model_name=model_name,
                context=context,
                spec=fitted,
                source=_LIKELIHOOD_SOURCE,
            )
        return ModelEvidence(
            source=_MODEL_SOURCE,
            importance=importance,
            main_effects=main_effects,
            interactions=interactions,
            exact_loss=exact_loss,
            warnings=interaction_warnings,
            unavailable=tuple(unavailable),
        )


def _bind_fitted_predictions(
    model_name: str,
    model: object,
    context: ReportContext,
) -> None:
    if not isinstance(model, SuperGLM) or getattr(model, "_result", None) is None:
        return
    message = (
        f"fitted SuperGLM object for {model_name!r} does not match the supplied prediction series"
    )
    try:
        declared = np.asarray(context.predictions[model_name], dtype=float)
        offset = _aligned_fitted_offset(model, context)
        fitted = np.asarray(model.predict(context.frame, offset=offset), dtype=float)
    except Exception:  # noqa: BLE001 - sanitize the fitted-model adapter boundary
        raise UnderwriterReportError(message) from None
    if (
        declared.ndim != 1
        or fitted.ndim != 1
        or declared.shape != fitted.shape
        or not np.isfinite(declared).all()
        or not np.isfinite(fitted).all()
        or not np.allclose(
            fitted,
            declared,
            rtol=_PREDICTION_BINDING_RTOL,
            atol=_PREDICTION_BINDING_ATOL,
        )
    ):
        raise UnderwriterReportError(message)


def _aligned_fitted_offset(model: SuperGLM, context: ReportContext) -> np.ndarray | None:
    if not bool(getattr(model, "_fit_used_offset", False)):
        return None
    offset = np.asarray(getattr(model, "_fit_offset", None), dtype=float)
    fit_weight = np.asarray(getattr(model, "_fit_weights", None), dtype=float)
    if (
        offset.ndim != 1
        or fit_weight.ndim != 1
        or offset.shape != fit_weight.shape
        or not np.isfinite(offset).all()
        or not np.isfinite(fit_weight).all()
        or (fit_weight < 0.0).any()
    ):
        raise ValueError("retained fitted offset state is unavailable")
    if len(offset) == len(context.frame):
        aligned_offset = offset
        aligned_weight = fit_weight
    else:
        positive = fit_weight > 0.0
        if int(positive.sum()) != len(context.frame):
            raise ValueError("retained fitted offset state is not row-aligned")
        aligned_offset = offset[positive]
        aligned_weight = fit_weight[positive]
    if not np.array_equal(aligned_weight, np.asarray(context.weight, dtype=float)):
        raise ValueError("retained fitted offset weights are not row-aligned")
    return aligned_offset


@dataclass(frozen=True)
class SuppliedTweedieLikelihoodAdapter:
    """Collect exact loss from explicit training-fitted Tweedie metadata."""

    tweedie_power: float
    dispersion: float

    def __post_init__(self) -> None:
        _LikelihoodSpec(self.tweedie_power, self.dispersion)

    def collect(
        self,
        *,
        model_name: str,
        source: object,
        context: ReportContext,
    ) -> ModelEvidence:
        del source
        exact_loss = _exact_loss_evidence(
            model_name=model_name,
            context=context,
            spec=_LikelihoodSpec(self.tweedie_power, self.dispersion),
            source=_SUPPLIED_SOURCE,
        )
        return ModelEvidence(source=_SUPPLIED_SOURCE, exact_loss=exact_loss)


def _likelihood_from_superglm(model: object) -> _LikelihoodSpec | None:
    distribution = getattr(model, "_distribution", None)
    if isinstance(distribution, Poisson):
        return _LikelihoodSpec(tweedie_power=1.0, dispersion=1.0)
    result = getattr(model, "result", None)
    if result is None:
        return None
    if isinstance(distribution, Gamma):
        return _LikelihoodSpec(tweedie_power=2.0, dispersion=result.phi)
    if isinstance(distribution, Tweedie):
        return _LikelihoodSpec(tweedie_power=distribution.p, dispersion=result.phi)
    return None


def _exact_loss_evidence(
    *,
    model_name: str,
    context: ReportContext,
    spec: _LikelihoodSpec,
    source: str,
) -> ExactLossEvidence:
    try:
        prediction = context.predictions[model_name]
    except KeyError as exc:
        raise KeyError(f"unknown model name: {model_name!r}") from exc
    contributions = _exact_nll_contributions(
        context.actual,
        prediction,
        context.weight,
        spec,
    )
    family, comparison_group = _family_metadata(spec.tweedie_power)
    return ExactLossEvidence(
        contributions=contributions,
        size_basis=("row_count" if 1.0 < spec.tweedie_power < 2.0 else "weight_sum"),
        comparison_group=comparison_group,
        score_label="Exact NLL",
        source=source,
        family=family,
        tweedie_power=spec.tweedie_power,
        dispersion=spec.dispersion,
    )


def _family_metadata(power: float) -> tuple[str, str]:
    if power == 1.0:
        return "Poisson", "poisson"
    if power == 2.0:
        return "Gamma", "gamma"
    return "Tweedie", f"tweedie:{power:g}"


def _exact_nll_contributions(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    spec: _LikelihoodSpec,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    power = spec.tweedie_power
    if power == 1.0:
        log_likelihood = weight * (
            actual * np.log(np.maximum(prediction, 1e-300)) - prediction - gammaln(actual + 1.0)
        )
    elif power == 2.0:
        shape = 1.0 / spec.dispersion
        log_likelihood = weight * (
            shape * np.log(shape * actual / prediction)
            - shape * actual / prediction
            - np.log(actual)
            - gammaln(shape)
        )
    else:
        log_likelihood = tweedie_logpdf(
            actual,
            prediction,
            spec.dispersion,
            power,
            weights=weight,
        )
    contributions = -np.asarray(log_likelihood, dtype=float)
    if contributions.ndim != 1 or len(contributions) != len(actual):
        raise UnderwriterReportError("exact likelihood returned a non-row-aligned series")
    if not np.isfinite(contributions).all():
        raise UnderwriterReportError("exact likelihood returned non-finite contributions")
    return contributions


def _model_interactions(
    model: object,
    context: ReportContext,
    *,
    n_points: int,
) -> tuple[dict[str, InteractionEvidence], list[CapabilityUnavailable]]:
    """Extract interaction terms through SuperGLM's public reporting APIs."""
    telemetry, telemetry_unavailable = _interaction_telemetry(model)
    if telemetry_unavailable is not None:
        return {}, [telemetry_unavailable]
    interactions: dict[str, InteractionEvidence] = {}
    unavailable: list[CapabilityUnavailable] = []
    for name, interaction_class, parents, expected_plot_kind in telemetry:
        if not set(parents).issubset(context.features):
            continue
        try:
            if interaction_class == "FactorSmooth":
                term = _factor_smooth_interaction(model, name, parents, context, n_points)
            else:
                raw = model.plot_data(
                    name,
                    X=context.frame,
                    sample_weight=context.weight,
                    n_points=n_points,
                )
                term = _interaction_from_plot_data(
                    name,
                    parents,
                    raw,
                    expected_plot_kind=expected_plot_kind,
                )
        except Exception as exc:
            if _is_unavailable_interaction_reporting(exc, operation=interaction_class):
                unavailable.append(
                    CapabilityUnavailable(
                        "interactions",
                        f"{name}: {_INTERACTIONS_UNAVAILABLE_REASON}",
                    )
                )
                continue
            if isinstance(exc, UnderwriterReportError):
                raise
            raise UnderwriterReportError(
                f"could not extract SuperGLM interaction {name!r}"
            ) from exc
        interactions[name] = term
    return interactions, unavailable


def _interaction_telemetry(
    model: object,
) -> tuple[list[tuple[str, str, tuple[str, str], str | None]], CapabilityUnavailable | None]:
    if not callable(getattr(model, "training_telemetry", None)):
        return [], CapabilityUnavailable("interactions", _INTERACTIONS_UNAVAILABLE_REASON)
    try:
        telemetry = model.training_telemetry()
    except Exception as exc:
        if _is_unavailable_interaction_reporting(exc, operation="training_telemetry"):
            return [], CapabilityUnavailable("interactions", _INTERACTIONS_UNAVAILABLE_REASON)
        raise UnderwriterReportError("could not read SuperGLM interaction telemetry") from exc
    if not isinstance(telemetry, Mapping):
        raise UnderwriterReportError("SuperGLM training telemetry must be a mapping")
    features = telemetry.get("features")
    if not isinstance(features, Mapping):
        raise UnderwriterReportError("SuperGLM training telemetry has no features table")
    order = features.get("interaction_order")
    specs = features.get("interactions")
    if not isinstance(order, list) or not isinstance(specs, Mapping):
        raise UnderwriterReportError("SuperGLM interaction telemetry has an unexpected shape")
    if any(not isinstance(name, str) or not name for name in order):
        raise UnderwriterReportError("SuperGLM interaction telemetry has an invalid name")
    if len(order) != len(set(order)):
        raise UnderwriterReportError("SuperGLM interaction telemetry has duplicate order names")
    if set(order) != set(specs):
        raise UnderwriterReportError("SuperGLM interaction telemetry order and specs disagree")

    result: list[tuple[str, str, tuple[str, str], str | None]] = []
    for raw_name in order:
        if not isinstance(raw_name, str) or not raw_name:
            raise UnderwriterReportError("SuperGLM interaction telemetry has an invalid name")
        spec = specs.get(raw_name)
        if not isinstance(spec, Mapping):
            raise UnderwriterReportError(
                f"SuperGLM interaction telemetry has no specification for {raw_name!r}"
            )
        interaction_class = spec.get("class")
        raw_parents = spec.get("parents")
        if (
            not isinstance(interaction_class, str)
            or not interaction_class
            or not isinstance(raw_parents, list)
            or len(raw_parents) != 2
            or not all(isinstance(parent, str) and parent for parent in raw_parents)
        ):
            raise UnderwriterReportError(
                f"SuperGLM interaction telemetry for {raw_name!r} has an unexpected shape"
            )
        if interaction_class == "FactorSmooth":
            expected_plot_kind = None
        else:
            try:
                expected_plot_kind = _INTERACTION_CLASS_PLOT_KINDS[interaction_class]
            except KeyError as exc:
                raise UnderwriterReportError(
                    f"SuperGLM interaction telemetry has an unknown class for {raw_name!r}"
                ) from exc
        result.append(
            (raw_name, interaction_class, (raw_parents[0], raw_parents[1]), expected_plot_kind)
        )
    return result, None


def _is_unavailable_interaction_reporting(exc: Exception, *, operation: str) -> bool:
    message = str(exc)
    if operation == "training_telemetry":
        return (
            isinstance(exc, NotImplementedError)
            and message == "training telemetry is not supported"
        )
    if operation == "FactorSmooth":
        return isinstance(exc, RuntimeError) and (
            message == "factor_smooth() requires a fitted model"
            or message == "factor_smooth() requires a fit_reml() result"
            or re.fullmatch(
                r"FactorSmooth support for .+ is unavailable; refit with "
                r"retain_fit_state=True or direct_solve='structured'\.",
                message,
            )
            is not None
        )
    return (
        isinstance(exc, NotImplementedError)
        and message == "native interaction reporting is not supported"
    )


def _interaction_from_plot_data(
    name: str,
    parents: tuple[str, str],
    raw: object,
    *,
    expected_plot_kind: str | None,
) -> InteractionEvidence:
    if not isinstance(raw, Mapping):
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} returned an unexpected payload"
        )
    if raw.get("kind") != "interaction":
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} returned an invalid payload kind"
        )
    if raw.get("name") != name:
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} returned an invalid payload name"
        )
    if raw.get("parents") != list(parents):
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} returned invalid payload parents"
        )
    plot_kind = raw.get("plot_kind")
    if plot_kind not in _PLOT_KINDS or plot_kind != expected_plot_kind:
        raise UnderwriterReportError(f"SuperGLM interaction {name!r} returned an unknown plot kind")
    raw_effect = raw.get("effect")
    if not isinstance(raw_effect, pd.DataFrame):
        raise UnderwriterReportError(f"SuperGLM interaction {name!r} returned no effect table")
    if len(raw_effect) > MAX_INTERACTION_ROWS:
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} exceeds the reporting row limit"
        )

    if plot_kind == "surface":
        effect = _surface_effect(name, parents, raw_effect)
        axes = _surface_axes(name, parents, raw.get("grid_axes"))
        density = _surface_density(name, parents, raw.get("density"))
        return InteractionEvidence(
            name=name,
            parents=parents,
            semantic="native_component",
            plot_kind="surface",
            effect=effect,
            source=_MODEL_SOURCE,
            grid_axes=axes,
            density=density,
        )
    if plot_kind == "categorical_heatmap":
        effect = _interaction_frame(
            name,
            raw_effect,
            {"left": parents[0], "right": parents[1], "value": "relativity"},
        )
    elif plot_kind == "varying_coefficient":
        effect = _interaction_frame(
            name,
            raw_effect,
            {"x": parents[0], "level": parents[1], "value": "relativity"},
        )
    elif plot_kind == "numeric_categorical":
        effect = _interaction_frame(
            name,
            raw_effect,
            {"level": parents[1], "value": "relativity_per_unit"},
        )
    else:
        effect = _interaction_frame(
            name,
            raw_effect,
            {"value": "relativity_per_unit_unit"},
        )
    return InteractionEvidence(
        name=name,
        parents=parents,
        semantic="native_component",
        plot_kind=plot_kind,
        effect=effect,
        source=_MODEL_SOURCE,
    )


def _interaction_frame(
    name: str,
    raw: pd.DataFrame,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    missing = set(columns.values()) - set(raw.columns)
    if missing:
        raise UnderwriterReportError(
            f"SuperGLM interaction {name!r} effect table is missing columns: "
            + ", ".join(sorted(missing))
        )
    return (
        raw.loc[:, list(columns.values())]
        .copy(deep=True)
        .rename(columns={value: key for key, value in columns.items()})
    )


def _surface_effect(
    name: str,
    parents: tuple[str, str],
    raw: pd.DataFrame,
) -> pd.DataFrame:
    return _interaction_frame(
        name,
        raw,
        {"x": parents[0], "y": parents[1], "value": "relativity"},
    )


def _surface_axes(
    name: str,
    parents: tuple[str, str],
    raw: object,
) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != set(parents):
        raise UnderwriterReportError(f"SuperGLM interaction {name!r} returned invalid surface axes")
    axes: dict[str, np.ndarray] = {}
    for output_name, parent in zip(("x", "y"), parents, strict=True):
        values = _finite_vector(raw[parent], f"interaction {name!r} surface axis {output_name}")
        if len(values) > MAX_SURFACE_AXIS_POINTS:
            raise UnderwriterReportError(
                f"SuperGLM interaction {name!r} exceeds the surface axis limit"
            )
        axes[output_name] = values.copy()
    return axes


def _surface_density(
    name: str,
    parents: tuple[str, str],
    raw: object,
) -> pd.DataFrame | None:
    if raw is None:
        return None
    if not isinstance(raw, pd.DataFrame):
        raise UnderwriterReportError(f"SuperGLM interaction {name!r} returned invalid density")
    return _interaction_frame(
        name,
        raw,
        {
            "x": parents[0],
            "y": parents[1],
            "density": "density",
            "hdr_mass": "hdr_mass",
        },
    )


def _factor_smooth_interaction(
    model: object,
    name: str,
    parents: tuple[str, str],
    context: ReportContext,
    n_points: int,
) -> InteractionEvidence:
    safe_levels = _safe_factor_smooth_levels(parents[1], context)
    result = model.factor_smooth(name, grid=n_points, levels=safe_levels)
    basis = getattr(result, "basis", None)
    if basis not in {"fs", "sz"}:
        raise UnderwriterReportError(f"SuperGLM factor smooth {name!r} returned an invalid basis")
    curves = getattr(result, "curves", None)
    diagnostics = getattr(result, "table", None)
    if not isinstance(curves, pd.DataFrame) or not isinstance(diagnostics, pd.DataFrame):
        raise UnderwriterReportError(f"SuperGLM factor smooth {name!r} returned unexpected tables")
    effect = _factor_smooth_effect(name, parents, curves)
    if len(effect) > MAX_INTERACTION_ROWS:
        raise UnderwriterReportError(
            f"SuperGLM factor smooth {name!r} exceeds the reporting row limit"
        )
    level_diagnostics = _factor_smooth_diagnostics(name, diagnostics, result, safe_levels)
    facts = _factor_smooth_facts(name, result, basis)
    warnings = _factor_smooth_warnings(name, result, basis)
    return InteractionEvidence(
        name=name,
        parents=parents,
        semantic="native_component",
        plot_kind="factor_smooth",
        effect=effect,
        source=_MODEL_SOURCE,
        level_diagnostics=level_diagnostics,
        facts=facts,
        warnings=warnings,
    )


def _safe_factor_smooth_levels(feature: str, context: ReportContext) -> list[object]:
    values = context.frame[feature]
    codes = np.asarray(context.comparison_unit_codes)
    safe: list[object] = []
    for level in pd.unique(values):
        mask = values.eq(level).to_numpy()
        if len(np.unique(codes[mask])) >= context.minimum_cell_size:
            safe.append(level)
    return safe


def _factor_smooth_effect(
    name: str,
    parents: tuple[str, str],
    curves: pd.DataFrame,
) -> pd.DataFrame:
    required = {"level", parents[0], "effect", "lower", "upper"}
    missing = required - set(curves.columns)
    if missing:
        raise UnderwriterReportError(
            f"SuperGLM factor smooth {name!r} is missing columns: " + ", ".join(sorted(missing))
        )
    effect = pd.DataFrame(
        {
            "x": _finite_vector(curves[parents[0]], f"factor smooth {name!r} x"),
            "level": curves["level"].astype(str),
            "value": _response_scale_vector(
                _finite_vector(curves["effect"], f"factor smooth {name!r} effect"),
                f"factor smooth {name!r} effect",
            ),
            "lower": _response_scale_vector(
                _finite_vector(curves["lower"], f"factor smooth {name!r} lower"),
                f"factor smooth {name!r} lower",
            ),
            "upper": _response_scale_vector(
                _finite_vector(curves["upper"], f"factor smooth {name!r} upper"),
                f"factor smooth {name!r} upper",
            ),
        }
    )
    return effect


def _factor_smooth_diagnostics(
    name: str,
    table: pd.DataFrame,
    result: object,
    safe_levels: list[object],
) -> pd.DataFrame:
    required = {"level", "effective_df", "has_information", "sufficient_support"}
    missing = required - set(table.columns)
    if missing:
        raise UnderwriterReportError(
            f"SuperGLM factor smooth {name!r} is missing diagnostics: " + ", ".join(sorted(missing))
        )
    selected = table.loc[table["level"].isin(safe_levels)].reset_index(drop=True)
    data: dict[str, object] = {
        "level": selected["level"].astype(str),
        "effective_df": selected["effective_df"],
        "has_information": selected["has_information"],
        "sufficient_support": selected["sufficient_support"],
        "collapsed": np.full(len(selected), bool(getattr(result, "collapsed", False)), dtype=bool),
    }
    if "credibility" in selected:
        data["credibility"] = selected["credibility"]
    return pd.DataFrame(data)


def _factor_smooth_facts(name: str, result: object, basis: str) -> tuple[EvidenceFact, ...]:
    lambdas = getattr(result, "lambdas", None)
    if not isinstance(lambdas, Mapping):
        raise UnderwriterReportError(f"SuperGLM factor smooth {name!r} returned invalid lambdas")
    facts: list[EvidenceFact] = [
        EvidenceFact("Basis", basis),
        EvidenceFact(
            "Interpretation",
            (
                "Level-specific fitted effect"
                if basis == "fs"
                else "Sum-to-zero deviation from the common smooth"
            ),
        ),
    ]
    for component, value in lambdas.items():
        facts.append(
            EvidenceFact(
                f"Lambda ({component})",
                _optional_finite_number(value, f"{name} lambda {component}"),
            )
        )
    facts.append(
        EvidenceFact(
            "Effective DF",
            _optional_finite_number(getattr(result, "effective_df", None), f"{name} effective_df"),
        )
    )
    return tuple(facts)


def _factor_smooth_warnings(name: str, result: object, basis: str) -> tuple[str, ...]:
    warnings: list[str] = []
    for boundary, label in (("at_lower_boundary", "lower"), ("at_upper_boundary", "upper")):
        values = getattr(result, boundary, None)
        if not isinstance(values, Mapping):
            raise UnderwriterReportError(
                f"SuperGLM factor smooth {name!r} returned invalid boundaries"
            )
        for component, at_boundary in values.items():
            if at_boundary is True:
                warnings.append(f"{name}: lambda {component} is at the {label} boundary")
    if basis == "sz":
        warnings.append("Not a standalone rating relativity")
    return tuple(warnings)


def _model_main_effects(
    model: object,
    context: ReportContext,
    *,
    n_points: int,
) -> dict[str, MainEffectEvidence]:
    try:
        session = EditorSession.from_model(
            model,
            n_points=n_points,
            centering="native",
            with_se=True,
            train_data=(context.frame, context.actual, context.weight),
        )
        payload = session_payload(session)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise UnderwriterReportError(
            "could not extract SuperGLM main-effect relativities; ensure the object is "
            "fitted and the report frame contains its feature columns"
        ) from exc

    if not isinstance(getattr(session, "terms", None), Mapping):
        raise UnderwriterReportError("SuperGLM editor returned an unexpected terms table")
    if not isinstance(payload, Mapping):
        raise UnderwriterReportError("SuperGLM editor returned an unexpected payload table")

    allowed = set(context.features)
    result: dict[str, MainEffectEvidence] = {}
    for raw_name, term in session.terms.items():
        name = str(raw_name)
        if name not in allowed:
            continue
        try:
            result[name] = _main_effect_from_editor_term(name, term, payload)
        except UnderwriterReportError:
            raise
        except (
            AttributeError,
            KeyError,
            NotImplementedError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise UnderwriterReportError(
                f"SuperGLM main effect {name!r} returned an unexpected table"
            ) from exc
    return result


def _main_effect_from_editor_term(
    name: str,
    term: object,
    payload: Mapping[object, object],
) -> MainEffectEvidence:
    log_effect = _finite_vector(term.original_log_effect, f"main effect {name!r}")
    levels = term.levels
    x = term.x
    values: dict[str, Any]
    if levels is not None:
        level_values = np.asarray(levels, dtype=object)
        if level_values.ndim != 1:
            raise UnderwriterReportError(
                f"SuperGLM main effect {name!r} levels must be one-dimensional"
            )
        values = {"label": [str(value) for value in level_values]}
        density = None
    elif x is not None:
        values = {"x": _finite_vector(x, f"main effect {name!r} x")}
        term_payload = payload[name]
        if not isinstance(term_payload, Mapping):
            raise UnderwriterReportError(
                f"SuperGLM main effect {name!r} returned an unexpected payload table"
            )
        density = _numeric_density(name, term_payload.get("exposure"))
    else:
        values = {"label": ["per unit"]}
        density = None
    values["value"] = _response_scale_vector(log_effect, f"main effect {name!r}")
    lower = _optional_response_scale_vector(
        term.ci_lower_log_effect,
        f"main effect {name!r} lower bound",
    )
    upper = _optional_response_scale_vector(
        term.ci_upper_log_effect,
        f"main effect {name!r} upper bound",
    )
    if (lower is None) != (upper is None):
        raise UnderwriterReportError(
            f"SuperGLM main effect {name!r} returned incomplete confidence bounds"
        )
    if lower is not None:
        values["lower"] = lower
        values["upper"] = upper
    lengths = {len(np.asarray(value)) for value in values.values()}
    if len(lengths) != 1:
        raise UnderwriterReportError(f"SuperGLM main effect {name!r} returned an unexpected table")
    metadata = term.metadata
    if not isinstance(metadata, Mapping):
        raise UnderwriterReportError(f"SuperGLM main effect {name!r} returned invalid metadata")
    effective_df = _optional_finite_number(
        metadata.get("edf"),
        f"main effect {name!r} effective_df",
    )
    return MainEffectEvidence(
        feature=name,
        semantic="native_component",
        effect=pd.DataFrame(values),
        source=_MODEL_SOURCE,
        density=density,
        effective_df=effective_df,
    )


def _numeric_density(name: str, exposure: object) -> pd.DataFrame | None:
    if exposure is None:
        return None
    if not isinstance(exposure, dict) or exposure.get("kind") != "density":
        raise UnderwriterReportError(
            f"SuperGLM main effect {name!r} returned an unexpected exposure table"
        )
    x = _finite_vector(exposure.get("x"), f"main effect {name!r} exposure x")
    density = _finite_vector(exposure.get("y"), f"main effect {name!r} exposure density")
    if len(x) != len(density) or np.any(density < 0.0):
        raise UnderwriterReportError(
            f"SuperGLM main effect {name!r} returned an invalid exposure table"
        )
    return pd.DataFrame({"x": x, "density": density})


def _optional_response_scale_vector(value: object, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _response_scale_vector(_finite_vector(value, name), name)


def _response_scale_vector(log_values: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(log_values)
    if not np.isfinite(result).all():
        raise UnderwriterReportError(f"{name} is non-finite on the response scale")
    return result


def _finite_vector(value: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise UnderwriterReportError(f"{name} must be numeric") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise UnderwriterReportError(f"{name} must be one-dimensional and finite")
    return result


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnderwriterReportError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise UnderwriterReportError(f"{name} must be finite")
    return number


def _model_importance(
    model: object,
    context: ReportContext,
) -> FeatureImportanceEvidence:
    try:
        raw = model.term_importance(context.frame, context.weight)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise UnderwriterReportError(
            "could not calculate SuperGLM term importance; ensure the object is fitted "
            "and the report frame contains its feature columns"
        ) from exc
    required = {"term", "feature", "variance_eta", "sd_eta", "edf"}
    if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
        raise UnderwriterReportError("SuperGLM term_importance returned an unexpected table")
    allowed = set(context.features)
    filtered = raw.loc[raw["feature"].astype(str).isin(allowed)].copy()
    if filtered.empty:
        table = pd.DataFrame(columns=["feature", "magnitude", "effective_df"])
    else:
        filtered["feature"] = filtered["feature"].astype(str)
        for column in ("variance_eta", "edf"):
            try:
                numeric = pd.to_numeric(filtered[column], errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise UnderwriterReportError(
                    f"SuperGLM term_importance column {column!r} must be numeric"
                ) from exc
            if not np.isfinite(numeric).all():
                raise UnderwriterReportError(
                    f"SuperGLM term_importance column {column!r} must be finite"
                )
            if np.any(numeric < 0.0):
                raise UnderwriterReportError(
                    f"SuperGLM term_importance column {column!r} must be non-negative"
                )
            filtered[column] = numeric
        grouped = (
            filtered.groupby("feature", sort=False, observed=True)
            .agg(
                magnitude=("variance_eta", "sum"),
                effective_df=("edf", "sum"),
            )
            .reset_index()
        )
        table = grouped.sort_values("magnitude", ascending=False, ignore_index=True)
    return FeatureImportanceEvidence(
        table=table,
        method="native_link_variance",
        source=_MODEL_SOURCE,
    )


__all__ = ["SuperGLMReportAdapter", "SuppliedTweedieLikelihoodAdapter"]
