"""Library-neutral evidence supplied to scored-data reporting.

Adapters translate fitted-model artifacts into these immutable value objects;
the report core only consumes normalized, copied evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
import pandas as pd

EvidenceSemantic = Literal[
    "native_component",
    "partial_dependence",
    "accumulated_local_effect",
    "shap_interaction",
    "portfolio_aggregate",
]
InteractionPlotKind = Literal[
    "surface",
    "categorical_heatmap",
    "varying_coefficient",
    "numeric_categorical",
    "numeric_numeric",
    "factor_smooth",
]
ExactLossSizeBasis = Literal["row_count", "weight_sum"]

MAX_MAIN_EFFECT_GRID_POINTS = 512
MAX_SURFACE_AXIS_POINTS = 160
MAX_SURFACE_CELLS = 25_600
MAX_INTERACTION_ROWS = 25_600

REQUIRED_INTERACTION_COLUMNS = {
    "surface": {"x", "y", "value"},
    "categorical_heatmap": {"left", "right", "value"},
    "varying_coefficient": {"x", "level", "value"},
    "numeric_categorical": {"level", "value"},
    "numeric_numeric": {"value"},
    "factor_smooth": {"x", "level", "value"},
}
_INTERACTION_COLUMN_ORDER = {
    "surface": ("x", "y", "value"),
    "categorical_heatmap": ("left", "right", "value"),
    "varying_coefficient": ("x", "level", "value"),
    "numeric_categorical": ("level", "value"),
    "numeric_numeric": ("value",),
    "factor_smooth": ("x", "level", "value"),
}
_CURVE_INTERACTION_KINDS = frozenset(
    {"varying_coefficient", "numeric_categorical", "numeric_numeric", "factor_smooth"}
)
_LEVEL_INTERACTION_KINDS = frozenset(
    {"varying_coefficient", "numeric_categorical", "factor_smooth"}
)
_LEVEL_DIAGNOSTIC_COLUMNS = (
    "level",
    "effective_df",
    "credibility",
    "has_information",
    "sufficient_support",
    "collapsed",
)

_SEMANTICS = frozenset(
    {
        "native_component",
        "partial_dependence",
        "accumulated_local_effect",
        "shap_interaction",
        "portfolio_aggregate",
    }
)
_INTERACTION_PLOT_KINDS = frozenset(
    {
        "surface",
        "categorical_heatmap",
        "varying_coefficient",
        "numeric_categorical",
        "numeric_numeric",
        "factor_smooth",
    }
)
_CAPABILITIES = frozenset({"importance", "main_effects", "interactions", "exact_loss"})
_SIZE_BASES = frozenset({"row_count", "weight_sum"})
_PROBLEM_POWERS = {"frequency": 1.0, "severity": 2.0}


@dataclass(frozen=True)
class ReportContext:
    frame: pd.DataFrame
    actual: np.ndarray
    predictions: Mapping[str, np.ndarray]
    weight: np.ndarray
    features: tuple[str, ...]
    comparison_unit_codes: np.ndarray
    comparison_units: int
    minimum_cell_size: int
    problem_type: Literal["frequency", "severity", "burn_cost"]
    deviance_power: float


@dataclass(frozen=True)
class EvidenceFact:
    label: str
    value: str | int | float | bool | None


@dataclass(frozen=True)
class CapabilityUnavailable:
    capability: Literal["importance", "main_effects", "interactions", "exact_loss"]
    reason: str


@dataclass(frozen=True)
class FeatureImportanceEvidence:
    table: pd.DataFrame
    method: str
    source: str


@dataclass(frozen=True)
class MainEffectEvidence:
    feature: str
    semantic: EvidenceSemantic
    effect: pd.DataFrame
    source: str
    density: pd.DataFrame | None = None
    effective_df: float | None = None
    facts: tuple[EvidenceFact, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionEvidence:
    name: str
    parents: tuple[str, str]
    semantic: EvidenceSemantic
    plot_kind: InteractionPlotKind
    effect: pd.DataFrame
    source: str
    grid_axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    density: pd.DataFrame | None = None
    support: pd.DataFrame | None = None
    default_levels: tuple[str, ...] = ()
    level_diagnostics: pd.DataFrame | None = None
    facts: tuple[EvidenceFact, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExactLossEvidence:
    contributions: np.ndarray
    size_basis: ExactLossSizeBasis
    comparison_group: str
    score_label: str
    source: str
    family: str
    tweedie_power: float | None
    dispersion: float | None
    facts: tuple[EvidenceFact, ...] = ()


@dataclass(frozen=True)
class ModelEvidence:
    source: str
    importance: FeatureImportanceEvidence | None = None
    main_effects: Mapping[str, MainEffectEvidence] = field(default_factory=dict)
    interactions: Mapping[str, InteractionEvidence] = field(default_factory=dict)
    exact_loss: ExactLossEvidence | None = None
    facts: tuple[EvidenceFact, ...] = ()
    warnings: tuple[str, ...] = ()
    unavailable: tuple[CapabilityUnavailable, ...] = ()


class ModelEvidenceAdapter(Protocol):
    def collect(
        self,
        *,
        model_name: str,
        source: object,
        context: ReportContext,
    ) -> ModelEvidence:
        raise NotImplementedError


@dataclass(frozen=True)
class EvidenceRequest:
    model_name: str
    adapter: ModelEvidenceAdapter
    source: object


def collect_model_evidence(
    context: ReportContext,
    direct: Mapping[str, ModelEvidence],
    requests: Sequence[EvidenceRequest],
) -> dict[str, ModelEvidence]:
    """Collect and compose direct evidence with adapter-produced evidence."""
    _validate_context(context)
    if not isinstance(direct, Mapping):
        raise TypeError("direct must be a mapping of model names to ModelEvidence")

    grouped: dict[str, list[ModelEvidence]] = {}
    for model_name, evidence in direct.items():
        _validate_model_name(model_name, context)
        _require_model_evidence(evidence)
        grouped.setdefault(model_name, []).append(
            normalize_model_evidence(model_name, evidence, context)
        )

    for request in requests:
        if not isinstance(request, EvidenceRequest):
            raise TypeError("requests must contain EvidenceRequest values")
        _validate_model_name(request.model_name, context)
        evidence = request.adapter.collect(
            model_name=request.model_name,
            source=request.source,
            context=context,
        )
        _require_model_evidence(evidence)
        grouped.setdefault(request.model_name, []).append(
            normalize_model_evidence(request.model_name, evidence, context)
        )

    return {
        model_name: normalize_model_evidence(
            model_name,
            _prepare_evidence_for_renormalization(_merge_evidence(model_name, values)),
            context,
        )
        for model_name, values in grouped.items()
    }


def normalize_model_evidence(
    model_name: str,
    evidence: ModelEvidence,
    context: ReportContext,
) -> ModelEvidence:
    """Validate evidence and detach all tabular and array values from callers."""
    _validate_context(context)
    _validate_model_name(model_name, context)
    _require_model_evidence(evidence)
    source = _plain_text(evidence.source, "source")
    importance = _normalize_importance(evidence.importance)
    main_effects = _normalize_main_effects(evidence.main_effects, context)
    interactions, unavailable_interactions = _normalize_interactions(
        model_name, evidence.interactions, context
    )
    if interactions:
        unavailable_interactions = ()
    exact_loss = _normalize_exact_loss(evidence.exact_loss, context)
    normalized = ModelEvidence(
        source=source,
        importance=importance,
        main_effects=MappingProxyType(main_effects),
        interactions=MappingProxyType(interactions),
        exact_loss=exact_loss,
        facts=_normalize_facts(evidence.facts),
        warnings=_normalize_warnings(evidence.warnings),
        unavailable=_normalize_unavailable(evidence.unavailable),
    )
    _validate_availability_consistency(normalized)
    return replace(
        normalized,
        unavailable=normalized.unavailable + unavailable_interactions,
    )


def _merge_evidence(model_name: str, values: Sequence[ModelEvidence]) -> ModelEvidence:
    if not values:
        raise ValueError(f"no evidence supplied for model {model_name!r}")
    merged = values[0]
    for next_evidence in values[1:]:
        for capability in ("importance", "main_effects", "interactions", "exact_loss"):
            if _is_populated(merged, capability) and _is_populated(next_evidence, capability):
                raise ValueError(
                    f"conflicting evidence for model {model_name!r} capability {capability!r}"
                )
        candidate = ModelEvidence(
            source=merged.source,
            importance=merged.importance or next_evidence.importance,
            main_effects=merged.main_effects or next_evidence.main_effects,
            interactions=merged.interactions or next_evidence.interactions,
            exact_loss=merged.exact_loss or next_evidence.exact_loss,
            facts=tuple(merged.facts) + tuple(next_evidence.facts),
            warnings=tuple(merged.warnings) + tuple(next_evidence.warnings),
            unavailable=tuple(merged.unavailable) + tuple(next_evidence.unavailable),
        )
        merged = replace(
            candidate,
            unavailable=tuple(
                item
                for item in candidate.unavailable
                if not _is_populated(candidate, item.capability)
            ),
        )
    return merged


def _prepare_evidence_for_renormalization(evidence: ModelEvidence) -> ModelEvidence:
    interactions = {
        name: replace(interaction, support=None, default_levels=())
        for name, interaction in evidence.interactions.items()
    }
    return replace(evidence, interactions=interactions)


def _is_populated(evidence: ModelEvidence, capability: str) -> bool:
    value = getattr(evidence, capability)
    return bool(value) if isinstance(value, Mapping) else value is not None


def _normalize_importance(
    importance: FeatureImportanceEvidence | None,
) -> FeatureImportanceEvidence | None:
    if importance is None:
        return None
    if not isinstance(importance, FeatureImportanceEvidence):
        raise TypeError("importance must be FeatureImportanceEvidence")
    table = _data_frame(importance.table, "importance.table")
    required = {"feature", "magnitude"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError("importance.table is missing columns: " + ", ".join(sorted(missing)))
    permitted = ["feature", "magnitude", "effective_df"]
    table = table.loc[:, [column for column in permitted if column in table.columns]].copy(
        deep=True
    )
    table["feature"] = _text_column(table["feature"], "importance.table.feature")
    magnitudes = _numeric_column(table["magnitude"], "importance.table.magnitude")
    if (magnitudes < 0.0).any():
        raise ValueError("importance.table.magnitude must be non-negative")
    table["magnitude"] = magnitudes
    if "effective_df" in table:
        table["effective_df"] = _numeric_column(
            table["effective_df"], "importance.table.effective_df"
        )
    total = float(magnitudes.sum())
    table["share"] = magnitudes / total if total else np.zeros(len(table), dtype=float)
    table["method"] = _plain_text(importance.method, "importance.method")
    table["source"] = _plain_text(importance.source, "importance.source")
    return FeatureImportanceEvidence(
        table=table,
        method=table["method"].iat[0]
        if not table.empty
        else _plain_text(importance.method, "importance.method"),
        source=table["source"].iat[0]
        if not table.empty
        else _plain_text(importance.source, "importance.source"),
    )


def _normalize_main_effects(
    main_effects: Mapping[str, MainEffectEvidence], context: ReportContext
) -> dict[str, MainEffectEvidence]:
    if not isinstance(main_effects, Mapping):
        raise TypeError("main_effects must be a mapping")
    normalized: dict[str, MainEffectEvidence] = {}
    for name, main_effect in main_effects.items():
        key = _plain_text(name, "main_effects key")
        if not isinstance(main_effect, MainEffectEvidence):
            raise TypeError("main_effects values must be MainEffectEvidence")
        feature = _plain_text(main_effect.feature, "main_effect.feature")
        if feature not in context.features:
            raise ValueError(f"main-effect feature {feature!r} is not an allowed feature")
        if key != feature:
            raise ValueError("main_effects keys must match MainEffectEvidence.feature")
        semantic = _semantic(main_effect.semantic, "main_effect.semantic")
        effect = _data_frame(main_effect.effect, "main_effect.effect")
        if "value" not in effect:
            raise ValueError("main_effect.effect must include value")
        has_x = "x" in effect
        has_label = "label" in effect
        if has_x == has_label:
            raise ValueError("main_effect.effect requires exactly one of x or label")
        values = _numeric_column(effect["value"], "main_effect.effect.value")
        if semantic == "native_component" and (values <= 0.0).any():
            raise ValueError("native_component values must be finite and positive")
        coordinate = "x" if has_x else "label"
        permitted = [coordinate, "value", "lower", "upper"]
        effect = effect.loc[:, [column for column in permitted if column in effect]].copy(deep=True)
        effect["value"] = values
        density: pd.DataFrame | None
        if has_x:
            _validate_grid_size(len(effect), "main_effect.effect")
            effect["x"] = _numeric_column(effect["x"], "main_effect.effect.x")
            density = _normalize_numeric_density(main_effect.density)
        else:
            effect["label"] = _text_column(effect["label"], "main_effect.effect.label")
            density = _categorical_support(feature, effect["label"], context)
        _validate_bounds(effect)
        effective_df = _optional_finite_number(main_effect.effective_df, "main_effect.effective_df")
        normalized[key] = MainEffectEvidence(
            feature=feature,
            semantic=semantic,
            effect=effect,
            source=_plain_text(main_effect.source, "main_effect.source"),
            density=density,
            effective_df=effective_df,
            facts=_normalize_facts(main_effect.facts),
            warnings=_normalize_warnings(main_effect.warnings),
        )
    return normalized


def _normalize_numeric_density(density: pd.DataFrame | None) -> pd.DataFrame | None:
    if density is None:
        return None
    density = _data_frame(density, "main_effect.density")
    if set(density.columns) != {"x", "density"}:
        raise ValueError("main_effect.density must contain exactly x and density")
    density = density.loc[:, ["x", "density"]].copy(deep=True)
    _validate_grid_size(len(density), "main_effect.density")
    density["x"] = _numeric_column(density["x"], "main_effect.density.x")
    values = _numeric_column(density["density"], "main_effect.density.density")
    if (values < 0.0).any():
        raise ValueError("main_effect.density.density must be non-negative")
    density["density"] = values
    return density


def _categorical_support(feature: str, labels: pd.Series, context: ReportContext) -> pd.DataFrame:
    values = context.frame[feature].astype(str)
    codes = np.asarray(context.comparison_unit_codes)
    weights = np.asarray(context.weight, dtype=float)
    masks = [values.eq(label).to_numpy() for label in labels]
    support = [int(np.unique(codes[mask]).size) for mask in masks]
    exposure = [float(weights[mask].sum()) for mask in masks]
    return pd.DataFrame(
        {
            "label": labels.to_numpy(copy=True),
            "comparison_units": support,
            "exposure": exposure,
        }
    )


def _validate_grid_size(size: int, name: str) -> None:
    if size > MAX_MAIN_EFFECT_GRID_POINTS:
        raise ValueError(f"{name} must contain at most {MAX_MAIN_EFFECT_GRID_POINTS} points")


def _validate_bounds(effect: pd.DataFrame) -> None:
    has_lower = "lower" in effect
    has_upper = "upper" in effect
    if has_lower != has_upper:
        raise ValueError("main_effect.effect lower and upper must appear together")
    if not has_lower:
        return
    lower = _numeric_column(effect["lower"], "main_effect.effect.lower")
    upper = _numeric_column(effect["upper"], "main_effect.effect.upper")
    value = effect["value"].to_numpy(dtype=float, copy=False)
    if (lower > value).any() or (value > upper).any():
        raise ValueError("main_effect.effect lower and upper must bracket value")
    effect["lower"] = lower
    effect["upper"] = upper


def normalize_interaction_evidence(
    model_name: str,
    evidence: InteractionEvidence,
    context: ReportContext,
) -> InteractionEvidence:
    """Validate one interaction and derive privacy-safe reporting support."""
    _validate_context(context)
    _validate_model_name(model_name, context)
    if not isinstance(evidence, InteractionEvidence):
        raise TypeError("evidence must be InteractionEvidence")
    if evidence.support is not None:
        raise ValueError("interaction.support must be None before normalization")
    if not isinstance(evidence.default_levels, tuple):
        raise TypeError("interaction.default_levels must be a tuple")
    if evidence.default_levels:
        raise ValueError("interaction.default_levels must be empty before normalization")

    name = _plain_text(evidence.name, "interaction.name")
    parents = _interaction_parents(evidence.parents, context)
    semantic = _semantic(evidence.semantic, "interaction.semantic")
    plot_kind = _plot_kind(evidence.plot_kind)
    _validate_interaction_compatibility(plot_kind, semantic, parents, context)
    _validate_interaction_input_sizes(evidence, plot_kind)
    effect = _normalize_interaction_effect(evidence.effect, plot_kind, semantic)
    grid_axes, density = _normalize_interaction_grid(
        evidence.grid_axes,
        evidence.density,
        effect,
        plot_kind,
    )
    level_diagnostics = _normalize_level_diagnostics(
        evidence.level_diagnostics,
        plot_kind,
    )

    support = None
    default_levels: tuple[str, ...] = ()
    if plot_kind == "categorical_heatmap":
        effect, support = _suppress_categorical_pairs(effect, parents, context)
    elif plot_kind in _LEVEL_INTERACTION_KINDS:
        effect, support, default_levels, safe_levels = _suppress_categorical_levels(
            effect,
            parents,
            context,
        )
        if level_diagnostics is not None:
            level_diagnostics = level_diagnostics.loc[
                level_diagnostics["level"].isin(safe_levels)
            ].reset_index(drop=True)

    return replace(
        evidence,
        name=name,
        parents=parents,
        semantic=semantic,
        plot_kind=plot_kind,
        effect=effect,
        source=_plain_text(evidence.source, "interaction.source"),
        grid_axes=MappingProxyType(grid_axes),
        density=density,
        support=support,
        default_levels=default_levels,
        level_diagnostics=level_diagnostics,
        facts=_normalize_facts(evidence.facts),
        warnings=_normalize_warnings(evidence.warnings),
    )


def _interaction_parents(
    value: object,
    context: ReportContext,
) -> tuple[str, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("interaction.parents must contain exactly two features")
    parents = tuple(_plain_text(parent, "interaction.parent") for parent in value)
    for parent in parents:
        if parent not in context.features:
            raise ValueError(f"interaction parent {parent!r} is not allowed")
    if parents[0] == parents[1]:
        raise ValueError("interaction parents must be distinct")
    return parents


def _validate_interaction_compatibility(
    plot_kind: InteractionPlotKind,
    semantic: EvidenceSemantic,
    parents: tuple[str, str],
    context: ReportContext,
) -> None:
    if plot_kind == "factor_smooth" and semantic != "native_component":
        raise ValueError("factor_smooth interactions require native_component semantic")
    if plot_kind in {"surface", "numeric_numeric"}:
        if not all(_is_numeric_feature(parent, context) for parent in parents):
            raise ValueError(f"{plot_kind} interaction parents must be numeric")
    elif plot_kind in _LEVEL_INTERACTION_KINDS and not _is_numeric_feature(parents[0], context):
        raise ValueError(f"{plot_kind} interaction first parent must be numeric")


def _is_numeric_feature(feature: str, context: ReportContext) -> bool:
    values = context.frame[feature]
    return not pd.api.types.is_bool_dtype(values) and pd.api.types.is_numeric_dtype(values)


def _validate_interaction_input_sizes(
    evidence: InteractionEvidence,
    plot_kind: InteractionPlotKind,
) -> None:
    _validate_interaction_table_size(evidence.effect, "interaction.effect")
    if evidence.density is not None:
        _validate_interaction_table_size(evidence.density, "interaction.density")
    if plot_kind != "surface" or not isinstance(evidence.grid_axes, Mapping):
        return
    lengths: dict[str, int] = {}
    for axis in ("x", "y"):
        if axis not in evidence.grid_axes:
            continue
        try:
            size = len(evidence.grid_axes[axis])
        except TypeError:
            continue
        if size > MAX_SURFACE_AXIS_POINTS:
            raise ValueError(
                f"surface grid_axes must contain at most {MAX_SURFACE_AXIS_POINTS} points per axis"
            )
        lengths[axis] = size
    if set(lengths) == {"x", "y"} and lengths["x"] * lengths["y"] > MAX_SURFACE_CELLS:
        raise ValueError(f"surface grid must contain at most {MAX_SURFACE_CELLS:,} cells")


def _validate_interaction_table_size(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if len(value) > MAX_INTERACTION_ROWS:
        raise ValueError(f"{name} must contain at most {MAX_INTERACTION_ROWS:,} rows")


def _normalize_interaction_effect(
    value: object,
    plot_kind: InteractionPlotKind,
    semantic: EvidenceSemantic,
) -> pd.DataFrame:
    effect = _data_frame(value, "interaction.effect")
    required = REQUIRED_INTERACTION_COLUMNS[plot_kind]
    missing = set(required) - set(effect.columns)
    if missing:
        raise ValueError("interaction.effect is missing columns: " + ", ".join(sorted(missing)))
    permitted = list(_INTERACTION_COLUMN_ORDER[plot_kind])
    if plot_kind in _CURVE_INTERACTION_KINDS:
        permitted.extend(("lower", "upper"))
    unknown = set(effect.columns) - set(permitted)
    if unknown:
        raise ValueError("interaction.effect has unknown columns: " + ", ".join(sorted(unknown)))
    effect = effect.loc[:, [column for column in permitted if column in effect]].copy(deep=True)

    for column in ("left", "right", "level"):
        if column in effect:
            effect[column] = _text_column(effect[column], f"interaction.effect.{column}")
    for column in ("x", "y", "value"):
        if column in effect:
            effect[column] = _numeric_column(effect[column], f"interaction.effect.{column}")

    _validate_interaction_bounds(effect)
    _validate_interaction_response(effect, semantic)
    _validate_interaction_coordinates(effect, plot_kind)
    return effect


def _validate_interaction_coordinates(
    effect: pd.DataFrame,
    plot_kind: InteractionPlotKind,
) -> None:
    if (
        plot_kind in {"varying_coefficient", "factor_smooth"}
        and effect.duplicated(["x", "level"]).any()
    ):
        raise ValueError(f"{plot_kind} interaction.effect must have unique coordinates")
    if plot_kind == "numeric_categorical" and effect["level"].duplicated().any():
        raise ValueError("numeric_categorical interaction.effect must have unique coordinates")
    if plot_kind == "numeric_numeric" and len(effect) != 1:
        raise ValueError("numeric_numeric interaction.effect must contain exactly one row")


def _validate_interaction_bounds(effect: pd.DataFrame) -> None:
    has_lower = "lower" in effect
    has_upper = "upper" in effect
    if has_lower != has_upper:
        raise ValueError("interaction.effect lower and upper must appear together")
    if not has_lower:
        return
    lower = _numeric_column(effect["lower"], "interaction.effect.lower")
    upper = _numeric_column(effect["upper"], "interaction.effect.upper")
    values = effect["value"].to_numpy(dtype=float, copy=False)
    if (lower > values).any() or (values > upper).any():
        raise ValueError("interaction.effect lower and upper must bracket value")
    effect["lower"] = lower
    effect["upper"] = upper


def _validate_interaction_response(
    effect: pd.DataFrame,
    semantic: EvidenceSemantic,
) -> None:
    columns = [column for column in ("value", "lower", "upper") if column in effect]
    values = effect.loc[:, columns].to_numpy(dtype=float, copy=False)
    if semantic == "native_component" and (values <= 0.0).any():
        raise ValueError("native_component interaction response values must be positive")
    if semantic in {"partial_dependence", "portfolio_aggregate"} and (values < 0.0).any():
        raise ValueError(f"{semantic} interaction response values must be non-negative")


def _normalize_interaction_grid(
    grid_axes: object,
    density: pd.DataFrame | None,
    effect: pd.DataFrame,
    plot_kind: InteractionPlotKind,
) -> tuple[dict[str, np.ndarray], pd.DataFrame | None]:
    if not isinstance(grid_axes, Mapping):
        raise TypeError("interaction.grid_axes must be a mapping")
    if plot_kind != "surface":
        if grid_axes:
            raise ValueError("interaction.grid_axes is only valid for surface")
        if density is not None:
            raise ValueError("interaction.density is only valid for surface")
        return {}, None
    if set(grid_axes) != {"x", "y"}:
        raise ValueError("surface grid_axes must contain exactly x and y")

    axes: dict[str, np.ndarray] = {}
    for axis in ("x", "y"):
        raw = pd.Series(_array(grid_axes[axis], f"interaction.grid_axes.{axis}"))
        normalized = _numeric_column(raw, f"interaction.grid_axes.{axis}")
        if len(normalized) > MAX_SURFACE_AXIS_POINTS:
            raise ValueError(
                f"surface grid_axes must contain at most {MAX_SURFACE_AXIS_POINTS} points per axis"
            )
        if len(normalized) == 0:
            raise ValueError("surface grid_axes must not be empty")
        if len(np.unique(normalized)) != len(normalized):
            raise ValueError("surface grid_axes must contain unique coordinates")
        axes[axis] = normalized

    cells = len(axes["x"]) * len(axes["y"])
    if cells > MAX_SURFACE_CELLS:
        raise ValueError(f"surface grid must contain at most {MAX_SURFACE_CELLS:,} cells")
    if effect.duplicated(["x", "y"]).any():
        raise ValueError("surface interaction.effect has duplicate grid coordinates")
    if (
        len(effect) != cells
        or set(effect["x"]) != set(axes["x"])
        or set(effect["y"]) != set(axes["y"])
    ):
        raise ValueError("surface grid_axes disagree with interaction.effect")

    return axes, _normalize_surface_density(density, effect)


def _normalize_surface_density(
    density: pd.DataFrame | None,
    effect: pd.DataFrame,
) -> pd.DataFrame | None:
    if density is None:
        return None
    density = _data_frame(density, "interaction.density")
    required = ["x", "y", "density", "hdr_mass"]
    if set(density.columns) != set(required):
        raise ValueError("interaction.density must contain exactly x, y, density, and hdr_mass")
    density = density.loc[:, required].copy(deep=True)
    for column in required:
        density[column] = _numeric_column(density[column], f"interaction.density.{column}")
    if (density["density"] < 0.0).any():
        raise ValueError("interaction.density.density must be non-negative")
    hdr_tolerance = 8.0 * np.finfo(float).eps
    if ((density["hdr_mass"] < -hdr_tolerance) | (density["hdr_mass"] > 1.0 + hdr_tolerance)).any():
        raise ValueError("interaction.density.hdr_mass must be between 0 and 1")
    density["hdr_mass"] = density["hdr_mass"].clip(0.0, 1.0)
    if density.duplicated(["x", "y"]).any() or _coordinate_pairs(density) != _coordinate_pairs(
        effect
    ):
        raise ValueError("interaction.density grid must match interaction.effect grid")
    return density


def _coordinate_pairs(frame: pd.DataFrame) -> set[tuple[float, float]]:
    return set(zip(frame["x"].tolist(), frame["y"].tolist(), strict=True))


def _normalize_level_diagnostics(
    value: pd.DataFrame | None,
    plot_kind: InteractionPlotKind,
) -> pd.DataFrame | None:
    if value is None:
        return None
    if plot_kind != "factor_smooth":
        raise ValueError("interaction.level_diagnostics is only valid for factor_smooth")
    diagnostics = _data_frame(value, "interaction.level_diagnostics")
    if "level" not in diagnostics:
        raise ValueError("interaction.level_diagnostics is missing columns: level")
    unknown = set(diagnostics.columns) - set(_LEVEL_DIAGNOSTIC_COLUMNS)
    if unknown:
        raise ValueError(
            "interaction.level_diagnostics has unknown columns: " + ", ".join(sorted(unknown))
        )
    columns = [column for column in _LEVEL_DIAGNOSTIC_COLUMNS if column in diagnostics]
    diagnostics = diagnostics.loc[:, columns].copy(deep=True)
    diagnostics["level"] = _text_column(diagnostics["level"], "interaction.level_diagnostics.level")
    if diagnostics["level"].duplicated().any():
        raise ValueError("interaction.level_diagnostics must contain unique levels")
    for column in ("effective_df", "credibility"):
        if column in diagnostics:
            diagnostics[column] = _numeric_column(
                diagnostics[column], f"interaction.level_diagnostics.{column}"
            )
    if "effective_df" in diagnostics and (diagnostics["effective_df"] < 0.0).any():
        raise ValueError("interaction.level_diagnostics.effective_df must be non-negative")
    if (
        "credibility" in diagnostics
        and ((diagnostics["credibility"] < 0.0) | (diagnostics["credibility"] > 1.0)).any()
    ):
        raise ValueError("interaction.level_diagnostics.credibility must be between 0 and 1")
    for column in ("has_information", "sufficient_support", "collapsed"):
        if column in diagnostics:
            if not pd.api.types.is_bool_dtype(diagnostics[column]):
                raise TypeError(f"interaction.level_diagnostics.{column} must be boolean")
            diagnostics[column] = diagnostics[column].to_numpy(dtype=bool, copy=True)
    return diagnostics


def _suppress_categorical_pairs(
    effect: pd.DataFrame,
    parents: tuple[str, str],
    context: ReportContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if effect.duplicated(["left", "right"]).any():
        raise ValueError("interaction.effect has duplicate categorical cells")
    left_values = _context_category_labels(parents[0], context)
    right_values = _context_category_labels(parents[1], context)
    masks = [
        (left_values.eq(left) & right_values.eq(right)).to_numpy()
        for left, right in effect.loc[:, ["left", "right"]].itertuples(index=False, name=None)
    ]
    support = _interaction_support(effect.loc[:, ["left", "right"]], masks, context)
    safe = support["comparison_units"] >= context.minimum_cell_size
    return (
        effect.loc[safe.to_numpy()].reset_index(drop=True),
        support.loc[safe].reset_index(drop=True),
    )


def _suppress_categorical_levels(
    effect: pd.DataFrame,
    parents: tuple[str, str],
    context: ReportContext,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], set[str]]:
    levels = pd.Series(pd.unique(effect["level"]), dtype="object")
    parent_values = _context_category_labels(parents[1], context)
    masks = [parent_values.eq(level).to_numpy() for level in levels]
    coordinates = pd.DataFrame({"level": levels.to_numpy(copy=True)})
    support = _interaction_support(coordinates, masks, context)
    support = support.loc[support["comparison_units"] >= context.minimum_cell_size].reset_index(
        drop=True
    )
    safe_levels = set(support["level"])
    effect = effect.loc[effect["level"].isin(safe_levels)].reset_index(drop=True)
    ranked = support.sort_values("weight", ascending=False, kind="mergesort")
    default_levels = tuple(ranked["level"].head(6).tolist())
    return effect, support, default_levels, safe_levels


def _context_category_labels(feature: str, context: ReportContext) -> pd.Series:
    labels: list[str] = []
    identities: dict[str, tuple[object, ...]] = {}
    for raw_value in context.frame[feature].tolist():
        label = _plain_text(str(raw_value), f"context feature {feature!r} category")
        identity = _raw_category_identity(raw_value)
        prior_identity = identities.get(label)
        if prior_identity is not None and prior_identity != identity:
            raise ValueError(
                f"context feature {feature!r}: distinct categories have an ambiguous "
                "text representation"
            )
        identities[label] = identity
        labels.append(label)
    return pd.Series(labels, index=context.frame.index, dtype="object")


def _raw_category_identity(value: object) -> tuple[object, ...]:
    try:
        missing = pd.isna(value)
    except TypeError, ValueError:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ("missing",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Integral):
        return ("integer", int(value))
    if isinstance(value, Real):
        return ("real", float(value))
    if isinstance(value, str):
        return ("text", value)
    return (type(value).__qualname__, repr(value))


def _interaction_support(
    coordinates: pd.DataFrame,
    masks: list[np.ndarray],
    context: ReportContext,
) -> pd.DataFrame:
    codes = np.asarray(context.comparison_unit_codes)
    weights = np.asarray(context.weight, dtype=float)
    total_weight = float(weights.sum())
    support = coordinates.copy(deep=True)
    support["rows"] = [int(mask.sum()) for mask in masks]
    support["comparison_units"] = [int(np.unique(codes[mask]).size) for mask in masks]
    support["weight"] = [float(weights[mask].sum()) for mask in masks]
    support["weight_share"] = support["weight"] / total_weight
    return support


def _normalize_interactions(
    model_name: str,
    interactions: Mapping[str, InteractionEvidence],
    context: ReportContext,
) -> tuple[dict[str, InteractionEvidence], tuple[CapabilityUnavailable, ...]]:
    if not isinstance(interactions, Mapping):
        raise TypeError("interactions must be a mapping")
    normalized: dict[str, InteractionEvidence] = {}
    unavailable: list[CapabilityUnavailable] = []
    for key, interaction in interactions.items():
        key = _plain_text(key, "interactions key")
        if not isinstance(interaction, InteractionEvidence):
            raise TypeError("interactions values must be InteractionEvidence")
        name = _plain_text(interaction.name, "interaction.name")
        if name != key:
            raise ValueError("interactions keys must match InteractionEvidence.name")
        normalized_interaction = normalize_interaction_evidence(model_name, interaction, context)
        if (
            normalized_interaction.plot_kind in _LEVEL_INTERACTION_KINDS | {"categorical_heatmap"}
            and normalized_interaction.effect.empty
        ):
            unavailable.append(
                CapabilityUnavailable(
                    capability="interactions",
                    reason=f"{name}: no cells meet minimum support",
                )
            )
            continue
        normalized[key] = normalized_interaction
    return normalized, tuple(unavailable)


def _normalize_exact_loss(
    exact_loss: ExactLossEvidence | None, context: ReportContext
) -> ExactLossEvidence | None:
    if exact_loss is None:
        return None
    if not isinstance(exact_loss, ExactLossEvidence):
        raise TypeError("exact_loss must be ExactLossEvidence")
    contributions = _numeric_vector(exact_loss.contributions, "exact_loss.contributions")
    if len(contributions) != len(context.actual):
        raise ValueError("exact_loss.contributions must match context.actual length")
    if exact_loss.size_basis not in _SIZE_BASES:
        raise ValueError("exact_loss.size_basis must be row_count or weight_sum")
    power = _optional_finite_number(exact_loss.tweedie_power, "exact_loss.tweedie_power")
    dispersion = _optional_finite_number(exact_loss.dispersion, "exact_loss.dispersion")
    if dispersion is not None and dispersion <= 0.0:
        raise ValueError("exact_loss.dispersion must be positive")
    family = _plain_text(exact_loss.family, "exact_loss.family")
    _validate_loss_compatibility(family, power, context)
    return ExactLossEvidence(
        contributions=contributions,
        size_basis=exact_loss.size_basis,
        comparison_group=_plain_text(exact_loss.comparison_group, "exact_loss.comparison_group"),
        score_label=_plain_text(exact_loss.score_label, "exact_loss.score_label"),
        source=_plain_text(exact_loss.source, "exact_loss.source"),
        family=family,
        tweedie_power=power,
        dispersion=dispersion,
        facts=_normalize_facts(exact_loss.facts),
    )


def _validate_loss_compatibility(family: str, power: float | None, context: ReportContext) -> None:
    expected_family = {
        "frequency": "poisson",
        "severity": "gamma",
        "burn_cost": "tweedie",
    }[context.problem_type]
    if family.casefold() != expected_family:
        raise ValueError(
            f"exact_loss.family {family!r} is incompatible with {context.problem_type!r}"
        )
    if context.problem_type == "burn_cost":
        if power is None or not 1.0 < power < 2.0:
            raise ValueError("burn_cost exact_loss.tweedie_power must be between 1 and 2")
        return
    expected_power = _PROBLEM_POWERS[context.problem_type]
    if power is not None and not math.isclose(power, expected_power):
        raise ValueError(
            f"{context.problem_type} exact_loss.tweedie_power must equal {expected_power:g}"
        )


def _normalize_facts(facts: Sequence[EvidenceFact]) -> tuple[EvidenceFact, ...]:
    normalized: list[EvidenceFact] = []
    for fact in facts:
        if not isinstance(fact, EvidenceFact):
            raise TypeError("facts must contain EvidenceFact values")
        value = fact.value
        if isinstance(value, str):
            value = _plain_text(value, "fact.value")
        elif value is not None and not isinstance(value, (bool, Integral, Real)):
            raise TypeError("fact.value must be a scalar display value")
        elif (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise ValueError("fact.value must be finite")
        normalized.append(EvidenceFact(_plain_text(fact.label, "fact.label"), value))
    return tuple(normalized)


def _normalize_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(warnings, tuple):
        raise TypeError("warnings must be a tuple of plain text values")
    return tuple(_plain_text(warning, "warnings") for warning in warnings)


def _normalize_unavailable(
    unavailable: Sequence[CapabilityUnavailable],
) -> tuple[CapabilityUnavailable, ...]:
    normalized: list[CapabilityUnavailable] = []
    for item in unavailable:
        if not isinstance(item, CapabilityUnavailable):
            raise TypeError("unavailable must contain CapabilityUnavailable values")
        if item.capability not in _CAPABILITIES:
            raise ValueError("unavailable capability is invalid")
        normalized.append(
            CapabilityUnavailable(
                capability=item.capability,
                reason=_plain_text(item.reason, "unavailable.reason"),
            )
        )
    return tuple(normalized)


def _validate_availability_consistency(evidence: ModelEvidence) -> None:
    for item in evidence.unavailable:
        if _is_populated(evidence, item.capability):
            raise ValueError(f"{item.capability} capability is populated and declared unavailable")


def _validate_context(context: ReportContext) -> None:
    if not isinstance(context, ReportContext):
        raise TypeError("context must be ReportContext")
    if not isinstance(context.frame, pd.DataFrame):
        raise TypeError("context.frame must be a pandas DataFrame")
    rows = len(context.frame)
    if rows == 0:
        raise ValueError("context.frame must not be empty")
    actual = _numeric_vector(context.actual, "context.actual")
    weight = _numeric_vector(context.weight, "context.weight")
    codes = np.asarray(context.comparison_unit_codes)
    if len(actual) != rows or len(weight) != rows:
        raise ValueError("context vectors must match context.frame length")
    if (weight <= 0.0).any():
        raise ValueError("context.weight must be positive")
    if codes.ndim != 1 or len(codes) != rows or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("context.comparison_unit_codes must be one-dimensional integer codes")
    if not isinstance(context.predictions, Mapping) or not context.predictions:
        raise ValueError("context.predictions must be a non-empty mapping")
    for model_name, prediction in context.predictions.items():
        _plain_text(model_name, "context prediction name")
        if len(_numeric_vector(prediction, "context.prediction")) != rows:
            raise ValueError("context predictions must match context.frame length")
    if not isinstance(context.features, tuple) or not context.features:
        raise ValueError("context.features must be a non-empty tuple")
    for feature in context.features:
        feature = _plain_text(feature, "context feature")
        if feature not in context.frame.columns:
            raise ValueError(f"context feature {feature!r} is missing from context.frame")
    if len(set(context.features)) != len(context.features):
        raise ValueError("context.features must not contain duplicates")
    if not isinstance(context.comparison_units, int) or isinstance(context.comparison_units, bool):
        raise TypeError("context.comparison_units must be an integer")
    if context.comparison_units <= 0:
        raise ValueError("context.comparison_units must be positive")
    if not isinstance(context.minimum_cell_size, int) or isinstance(
        context.minimum_cell_size, bool
    ):
        raise TypeError("context.minimum_cell_size must be an integer")
    if context.minimum_cell_size <= 0:
        raise ValueError("context.minimum_cell_size must be positive")
    if context.problem_type not in {"frequency", "severity", "burn_cost"}:
        raise ValueError("context.problem_type is invalid")
    power = _finite_number(context.deviance_power, "context.deviance_power")
    if context.problem_type in _PROBLEM_POWERS and power != _PROBLEM_POWERS[context.problem_type]:
        raise ValueError("context.deviance_power is incompatible with problem_type")
    if context.problem_type == "burn_cost" and not 1.0 < power < 2.0:
        raise ValueError("burn_cost context.deviance_power must be between 1 and 2")


def _validate_model_name(model_name: str, context: ReportContext) -> None:
    normalized_name = _plain_text(model_name, "model_name")
    if model_name != normalized_name or model_name not in context.predictions:
        raise KeyError(f"unknown model name: {model_name!r}")


def _require_model_evidence(evidence: object) -> None:
    if not isinstance(evidence, ModelEvidence):
        raise TypeError("evidence must be ModelEvidence")


def _semantic(value: object, name: str) -> EvidenceSemantic:
    if value not in _SEMANTICS:
        raise ValueError(f"{name} is invalid")
    return value


def _plot_kind(value: object) -> InteractionPlotKind:
    if value not in _INTERACTION_PLOT_KINDS:
        raise ValueError("interaction.plot_kind is invalid")
    return value


def _plain_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    value = value.strip()
    if not value or "<" in value or ">" in value:
        raise ValueError(f"{name} must be plain non-empty text without HTML tags")
    return value


def _data_frame(value: object, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value.copy(deep=True)


def _array(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result.copy()


def _numeric_vector(value: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _numeric_column(value: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(value) or not pd.api.types.is_numeric_dtype(value):
        raise TypeError(f"{name} must be numeric")
    result = value.to_numpy(dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _text_column(value: pd.Series, name: str) -> pd.Series:
    normalized = [_plain_text(item, name) for item in value.tolist()]
    return pd.Series(normalized, index=value.index, dtype="object")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _optional_finite_number(value: object, name: str) -> float | None:
    return None if value is None else _finite_number(value, name)
