"""Run local GAM/GBM diagnostics from an external TOML file."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from superglm import (
    Categorical,
    Constraint,
    Numeric,
    OrderedCategorical,
    Spline,
    SuperGLM,
    Tweedie,
    collapse_levels,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.modeling.scratch_benchmark import (
    _convex_tweedie_weights,
    fit_boosted_blend,
    superglm_edf_table,
    unconstrained_superglm_features,
)
from pricing_pipeline.modeling.scratch_diagnostics import (
    blend_evaluation_table,
    blend_weight_label,
    double_lift_table,
    feature_calibration_tables,
    interaction_failure_tables,
    lorenz_curve_table,
    risk_calibration_table,
    unit_tweedie_deviance,
)

_ALLOWED_SECTIONS = {
    "run",
    "data",
    "columns",
    "model",
    "simplified_gam",
    "split",
    "diagnostics",
}
_ALLOWED_KEYS = {
    "run": {"output_dir", "random_seed"},
    "data": {"path", "sample_rows"},
    "columns": {
        "target",
        "offset_source",
        "offset_divisor",
        "sample_weight",
        "features",
        "categorical_features",
        "linear_features",
    },
    "model": {
        "tweedie_power",
        "spline_kind",
        "spline_k",
        "knot_strategy",
        "knot_alpha",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "thread_count",
        "cv_folds",
        "max_reml_iter",
    },
    "simplified_gam": {
        "ordered_spline_features",
        "monotone_increasing_features",
        "special_levels",
        "collapse_from",
        "level_groups",
        "spline_k",
    },
    "split": {"train_fraction", "validation_fraction"},
    "diagnostics": {
        "governed_gam_weights",
        "double_lift_bins",
        "risk_bins",
        "lorenz_bins",
        "feature_bins",
        "interaction_bins",
        "max_categorical_levels",
        "min_cell_rows",
        "min_cell_weight_fraction",
        "top_interactions",
    },
}


@dataclass(frozen=True)
class SimplifiedGamConfig:
    """Generic structural decisions for a second GAM."""

    ordered_spline_features: tuple[str, ...]
    monotone_increasing_features: tuple[str, ...]
    special_levels: Mapping[str, tuple[str, ...]]
    collapse_from: Mapping[str, str]
    level_groups: Mapping[str, Mapping[str, tuple[str, ...]]]
    spline_k: int


@dataclass(frozen=True)
class DiagnosticRunConfig:
    data_path: Path
    output_dir: Path
    random_seed: int
    sample_rows: int | None
    target: str
    offset_source: str
    offset_divisor: float
    sample_weight: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    linear_features: tuple[str, ...]
    tweedie_power: float
    spline_kind: str
    spline_k: int
    knot_strategy: str
    knot_alpha: float
    n_estimators: int
    learning_rate: float
    max_depth: int
    thread_count: int
    cv_folds: int
    max_reml_iter: int
    simplified_gam: SimplifiedGamConfig | None
    train_fraction: float
    validation_fraction: float
    governed_gam_weights: tuple[float, ...]
    double_lift_bins: int
    risk_bins: int
    lorenz_bins: int
    feature_bins: int
    interaction_bins: int
    max_categorical_levels: int
    min_cell_rows: int
    min_cell_weight_fraction: float
    top_interactions: int


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"TOML section [{name}] must be a table")
    unknown = set(value) - _ALLOWED_KEYS[name]
    if unknown:
        raise ValueError(f"unknown [{name}] keys: {', '.join(sorted(unknown))}")
    return value


def _strings(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty TOML string array")
    out = tuple(value)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must not contain duplicates")
    return out


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) and key and item
        for key, item in value.items()
    ):
        raise ValueError(f"{name} must be a TOML table of non-empty string values")
    return dict(value)


def _string_list_mapping(value: Any, name: str) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a TOML table of string arrays")
    result: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _strings(items, f"{name}.{key}", required=True)
    return result


def _level_group_mapping(
    value: Any,
    name: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a TOML table")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for feature, raw_groups in value.items():
        if not isinstance(feature, str) or not feature or not isinstance(raw_groups, dict):
            raise ValueError(f"{name} must map feature names to grouping tables")
        groups: dict[str, tuple[str, ...]] = {}
        for label, members in raw_groups.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"{name}.{feature} group labels must be non-empty strings")
            groups[label] = _strings(
                members,
                f"{name}.{feature}.{label}",
                required=True,
            )
        result[feature] = groups
    return result


def load_diagnostic_config(path: Path) -> DiagnosticRunConfig:
    """Load strict local-only diagnostic configuration."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    unknown_sections = set(payload) - _ALLOWED_SECTIONS
    if unknown_sections:
        raise ValueError("unknown TOML sections: " + ", ".join(sorted(unknown_sections)))
    run = _mapping(payload.get("run"), "run")
    data = _mapping(payload.get("data"), "data")
    columns = _mapping(payload.get("columns"), "columns")
    model = _mapping(payload.get("model"), "model")
    simplified = _mapping(payload.get("simplified_gam"), "simplified_gam")
    split = _mapping(payload.get("split"), "split")
    diagnostics = _mapping(payload.get("diagnostics"), "diagnostics")

    data_value = data.get("path")
    if not isinstance(data_value, str) or not data_value.strip():
        raise ValueError("[data].path must be a non-empty local parquet path")
    data_path = Path(data_value).expanduser().resolve()
    if data_path.suffix.lower() != ".parquet":
        raise ValueError("[data].path must identify a parquet file")
    output_value = run.get("output_dir", "state/scratch_blend_diagnostics")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("[run].output_dir must be a non-empty path")
    output_dir = Path(output_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    state_root = (ROOT / "state").resolve()
    if not output_dir.is_relative_to(state_root):
        raise ValueError("diagnostic output_dir must remain under the ignored repository state/")

    required_column_names = ("target", "offset_source", "sample_weight")
    for name in required_column_names:
        if not isinstance(columns.get(name), str) or not columns[name].strip():
            raise ValueError(f"[columns].{name} must be a non-empty column name")
    features = _strings(columns.get("features"), "[columns].features", required=True)
    categorical = _strings(
        columns.get("categorical_features"),
        "[columns].categorical_features",
    )
    linear = _strings(columns.get("linear_features"), "[columns].linear_features")
    if set(categorical) - set(features) or set(linear) - set(features):
        raise ValueError("categorical_features and linear_features must be subsets of features")
    if set(categorical) & set(linear):
        raise ValueError("categorical_features and linear_features must not overlap")
    structural = {str(columns[name]) for name in required_column_names}
    if structural & set(features):
        raise ValueError("target, offset_source, and sample_weight must not also be features")

    simplified_config: SimplifiedGamConfig | None = None
    if simplified:
        ordered_spline_features = _strings(
            simplified.get("ordered_spline_features"),
            "[simplified_gam].ordered_spline_features",
            required=True,
        )
        monotone_increasing_features = _strings(
            simplified.get("monotone_increasing_features"),
            "[simplified_gam].monotone_increasing_features",
        )
        special_levels = _string_list_mapping(
            simplified.get("special_levels"),
            "[simplified_gam].special_levels",
        )
        collapse_from = _string_mapping(
            simplified.get("collapse_from"),
            "[simplified_gam].collapse_from",
        )
        level_groups = _level_group_mapping(
            simplified.get("level_groups"),
            "[simplified_gam].level_groups",
        )
        declared = (
            set(ordered_spline_features)
            | set(monotone_increasing_features)
            | set(special_levels)
            | set(collapse_from)
            | set(level_groups)
        )
        if declared - set(features):
            raise ValueError("[simplified_gam] declarations must name configured features")
        if set(special_levels) - set(ordered_spline_features):
            raise ValueError("special_levels may only target ordered_spline_features")
        if set(monotone_increasing_features) - set(ordered_spline_features):
            raise ValueError("monotone_increasing_features may only target ordered_spline_features")
        if set(collapse_from) - set(ordered_spline_features):
            raise ValueError("collapse_from may only target ordered_spline_features")
        overlap = set(collapse_from) & set(level_groups)
        if overlap:
            raise ValueError(
                "collapse_from and level_groups cannot both target: " + ", ".join(sorted(overlap))
            )
        simplified_config = SimplifiedGamConfig(
            ordered_spline_features=ordered_spline_features,
            monotone_increasing_features=monotone_increasing_features,
            special_levels=special_levels,
            collapse_from=collapse_from,
            level_groups=level_groups,
            spline_k=_positive_int(
                simplified.get("spline_k", model.get("spline_k", 10)),
                "[simplified_gam].spline_k",
            ),
        )

    power = float(model.get("tweedie_power", 1.5))
    if not 1.0 < power < 2.0:
        raise ValueError("[model].tweedie_power must be strictly between 1 and 2")
    train_fraction = float(split.get("train_fraction", 0.6))
    validation_fraction = float(split.get("validation_fraction", 0.2))
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave an untouched test set")
    governed_weights = tuple(
        float(value) for value in diagnostics.get("governed_gam_weights", [0.4, 0.5])
    )
    if not governed_weights or any(not 0.0 < value < 1.0 for value in governed_weights):
        raise ValueError("governed_gam_weights must contain values strictly between 0 and 1")
    offset_divisor = float(columns.get("offset_divisor", 1.0))
    if not np.isfinite(offset_divisor) or offset_divisor <= 0:
        raise ValueError("[columns].offset_divisor must be finite and positive")

    raw_sample_rows = data.get("sample_rows")
    sample_rows = (
        None if raw_sample_rows in {None, 0} else _positive_int(raw_sample_rows, "sample_rows")
    )
    return DiagnosticRunConfig(
        data_path=data_path,
        output_dir=output_dir,
        random_seed=int(run.get("random_seed", 1729)),
        sample_rows=sample_rows,
        target=str(columns["target"]),
        offset_source=str(columns["offset_source"]),
        offset_divisor=offset_divisor,
        sample_weight=str(columns["sample_weight"]),
        features=features,
        categorical_features=categorical,
        linear_features=linear,
        tweedie_power=power,
        spline_kind=str(model.get("spline_kind", "ps")),
        spline_k=_positive_int(model.get("spline_k", 10), "spline_k"),
        knot_strategy=str(model.get("knot_strategy", "quantile_tempered")),
        knot_alpha=float(model.get("knot_alpha", 0.2)),
        n_estimators=_positive_int(model.get("n_estimators", 300), "n_estimators"),
        learning_rate=float(model.get("learning_rate", 0.05)),
        max_depth=_positive_int(model.get("max_depth", 5), "max_depth"),
        thread_count=int(model.get("thread_count", -1)),
        cv_folds=_positive_int(model.get("cv_folds", 5), "cv_folds"),
        max_reml_iter=_positive_int(model.get("max_reml_iter", 25), "max_reml_iter"),
        simplified_gam=simplified_config,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        governed_gam_weights=governed_weights,
        double_lift_bins=_positive_int(diagnostics.get("double_lift_bins", 10), "double_lift_bins"),
        risk_bins=_positive_int(diagnostics.get("risk_bins", 20), "risk_bins"),
        lorenz_bins=_positive_int(diagnostics.get("lorenz_bins", 100), "lorenz_bins"),
        feature_bins=_positive_int(diagnostics.get("feature_bins", 10), "feature_bins"),
        interaction_bins=_positive_int(diagnostics.get("interaction_bins", 6), "interaction_bins"),
        max_categorical_levels=_positive_int(
            diagnostics.get("max_categorical_levels", 12), "max_categorical_levels"
        ),
        min_cell_rows=_positive_int(diagnostics.get("min_cell_rows", 30), "min_cell_rows"),
        min_cell_weight_fraction=float(diagnostics.get("min_cell_weight_fraction", 0.001)),
        top_interactions=_positive_int(diagnostics.get("top_interactions", 6), "top_interactions"),
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "feature"


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_blend_curve(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    power: float,
    optimum_weight: float,
    governed_weights: tuple[float, ...],
    output_dir: Path,
) -> None:
    grid = np.linspace(0.0, 1.0, 101)

    def curve(frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                np.average(
                    unit_tweedie_deviance(
                        frame["actual_response"],
                        weight * frame["gam_expected"] + (1.0 - weight) * frame["gbm_expected"],
                        power=power,
                    ),
                    weights=frame["sample_weight"],
                )
                for weight in grid
            ]
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(100 * grid, curve(validation), label="Validation — weight selected here", lw=2)
    axis.plot(100 * grid, curve(test), label="Untouched test — diagnostic only", lw=2)
    axis.axvline(
        100 * optimum_weight,
        color="#d62728",
        ls="--",
        label=f"Technical optimum {optimum_weight:.1%}",
    )
    colors = ("#9467bd", "#8c564b", "#2ca02c")
    for weight, color in zip(governed_weights, colors, strict=False):
        axis.axvline(
            100 * weight,
            color=color,
            ls="--",
            label=f"Governed GAM weight {weight:.0%}",
        )
    axis.set(
        title="Predictive optimum versus governed GAM weight",
        xlabel="GAM weight (%)",
        ylabel=f"Mean Tweedie deviance (p={power:g})",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_dir / "01_blend_weight_curve.png")


def _plot_calibration_table(
    table: pd.DataFrame,
    *,
    bin_column: str,
    title: str,
    output_path: Path,
    governed_weights: tuple[float, ...],
    gam_label: str = "GAM",
    indexed: bool = False,
) -> None:
    figure, (axis, support_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(1, len(table) + 1)
    suffix = "_index" if indexed else ""
    axis.plot(
        x,
        table[f"actual_response{suffix}"],
        color="black",
        marker="o",
        label="Observed",
    )
    axis.plot(
        x,
        table[f"gam_response{suffix}"],
        color="#1f77b4",
        marker="s",
        label=gam_label,
    )
    axis.plot(
        x,
        table[f"gbm_response{suffix}"],
        color="#d62728",
        marker="^",
        label="Boosted blend",
    )
    for index, weight in enumerate(governed_weights):
        column = f"{blend_weight_label(weight)}_response{suffix}"
        axis.plot(
            x,
            table[column],
            marker="D",
            label=f"{weight:.0%} {gam_label}",
            color=("#9467bd", "#2ca02c", "#8c564b")[index % 3],
        )
    axis.set(
        title=title,
        ylabel=(
            "Index to each curve's portfolio average"
            if indexed
            else "Aggregate response / declared weight"
        ),
    )
    if indexed:
        axis.axhline(1.0, color="#666666", ls=":", lw=1)
    axis.legend(frameon=False, ncols=2)
    axis.grid(alpha=0.2)
    support_axis.bar(x, table["sample_weight"], color="#999999")
    support_axis.set(xlabel=bin_column, ylabel="Sample weight")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_feature_table(
    feature: str,
    table: pd.DataFrame,
    *,
    governed_weights: tuple[float, ...],
    gam_label: str = "GAM",
    output_path: Path,
) -> None:
    figure, (axis, support_axis) = plt.subplots(
        2,
        1,
        figsize=(max(10, min(18, 0.65 * len(table) + 6)), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(len(table))
    axis.plot(x, table["actual_response"], color="black", marker="o", label="Observed")
    axis.plot(x, table["gam_response"], color="#1f77b4", marker="s", label=gam_label)
    axis.plot(
        x,
        table["gbm_response"],
        color="#d62728",
        marker="^",
        label="Boosted blend",
    )
    for index, weight in enumerate(governed_weights):
        axis.plot(
            x,
            table[f"{blend_weight_label(weight)}_response"],
            marker="D",
            label=f"{weight:.0%} {gam_label}",
            color=("#9467bd", "#2ca02c", "#8c564b")[index % 3],
        )
    axis.set(
        title=f"Held-out calibration by {feature}",
        ylabel="Aggregate response / declared weight",
    )
    axis.legend(frameon=False, ncols=2)
    axis.grid(alpha=0.2)
    support_axis.bar(x, table["sample_weight"], color="#999999")
    support_axis.set(ylabel="Sample weight")
    support_axis.set_xticks(x, table["feature_bin"].astype(str), rotation=45, ha="right")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_lorenz(
    table: pd.DataFrame,
    *,
    output_path: Path,
    gam_label: str = "Additive SuperGLM",
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    labels = {
        "GAM": gam_label,
        "BOOSTED_BLEND": "Boosted blend",
    }
    for model_name, part in table.groupby("model", sort=False):
        label = labels.get(str(model_name), str(model_name).replace("_", " ").title())
        axis.plot(
            part["cumulative_weight_share"],
            part["cumulative_response_share"],
            lw=2,
            label=f"{label} — Gini {part['gini'].iloc[0]:.3f}",
        )
    axis.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="No discrimination")
    axis.set(
        title="Weighted Lorenz curves on untouched test data",
        xlabel="Cumulative declared weight, ordered low → high prediction",
        ylabel="Cumulative aggregate response numerator",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_interaction(
    feature_a: str,
    feature_b: str,
    cells: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    selected = cells.loc[cells["feature_a"].eq(feature_a) & cells["feature_b"].eq(feature_b)]
    interaction = selected.pivot(index="level_a", columns="level_b", values="interaction_log_ratio")
    failure = selected.pivot(
        index="level_a",
        columns="level_b",
        values="gbm_minus_gam_mean_deviance",
    ).reindex(index=interaction.index, columns=interaction.columns)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, matrix, title, cmap in (
        (
            axes[0],
            interaction,
            "Non-additive log(boosted blend / GAM)",
            "coolwarm",
        ),
        (
            axes[1],
            failure,
            "Boosted blend minus GAM held-out deviance\n(red means the blend is worse)",
            "RdYlGn_r",
        ),
    ):
        limit = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
        limit = max(limit, np.finfo(float).eps)
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        axis.set_xticks(
            range(len(matrix.columns)), matrix.columns.astype(str), rotation=45, ha="right"
        )
        axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
        axis.set(title=title, xlabel=feature_b, ylabel=feature_a)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Where the boosted interaction helps or fails: {feature_a} × {feature_b}")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _typed_level_identity(value: Any) -> tuple[str, str]:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ("missing", "")
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _level_display_label(value: Any) -> str:
    missing = pd.isna(value)
    return "__MISSING__" if isinstance(missing, (bool, np.bool_)) and bool(missing) else str(value)


def _scalar_values_compare_equal(left: Any, right: Any) -> bool:
    comparison = left == right
    return isinstance(comparison, (bool, np.bool_)) and bool(comparison)


def _observed_levels(series: pd.Series) -> list[Any]:
    if series.isna().any():
        raise ValueError(f"simplified GAM feature {series.name!r} contains missing levels")
    levels: list[Any] = []
    identities: set[tuple[str, str]] = set()
    for value in series.tolist():
        identity = _typed_level_identity(value)
        if identity in identities:
            continue
        if any(_scalar_values_compare_equal(value, existing) for existing in levels):
            raise ValueError(
                f"simplified GAM feature {series.name!r} has distinct typed levels that compare equal"
            )
        identities.add(identity)
        levels.append(value)
    try:
        return sorted(levels)
    except TypeError as exc:
        raise ValueError(
            f"simplified GAM feature {series.name!r} has levels that cannot be ordered safely"
        ) from exc


def _validate_holdout_categorical_levels(
    frame: pd.DataFrame,
    train: np.ndarray,
    holdout: np.ndarray,
    *,
    categorical_features: tuple[str, ...],
    partition: str,
) -> None:
    for feature in categorical_features:
        training_levels = {
            _typed_level_identity(value) for value in frame.iloc[train][feature].tolist()
        }
        holdout_levels = {
            _typed_level_identity(value): _level_display_label(value)
            for value in frame.iloc[holdout][feature].tolist()
        }
        unseen = sorted(set(holdout_levels) - training_levels)
        if unseen:
            raise ValueError(
                f"{partition} partition contains unseen levels for categorical feature "
                f"{feature!r}: {', '.join(sorted(holdout_levels[level] for level in unseen))}"
            )


def _simplified_superglm_features(
    frame: pd.DataFrame,
    *,
    config: DiagnosticRunConfig,
    apply_monotone_constraints: bool = True,
) -> dict[str, Any]:
    simplified = config.simplified_gam
    if simplified is None:
        raise ValueError("a simplified GAM configuration is required")
    categorical = set(config.categorical_features)
    linear = set(config.linear_features)
    ordered = set(simplified.ordered_spline_features)
    features: dict[str, Any] = {}
    for column in frame.columns:
        grouping = None
        if column in simplified.collapse_from or column in simplified.level_groups:
            levels = _observed_levels(frame[column])
            if column in simplified.collapse_from:
                grouping = collapse_levels(
                    frame[column],
                    from_level=simplified.collapse_from[column],
                    order=levels,
                )
            else:
                grouping = collapse_levels(
                    frame[column],
                    groups={
                        label: list(members)
                        for label, members in simplified.level_groups[column].items()
                    },
                    order=levels,
                )
        if column in ordered:
            levels = _observed_levels(frame[column])
            ordered_spec = OrderedCategorical(
                order=levels,
                basis=Spline(
                    kind=config.spline_kind,
                    k=simplified.spline_k,
                    knot_strategy=config.knot_strategy,
                    knot_alpha=config.knot_alpha,
                    constraint=(
                        Constraint.fit.increasing
                        if apply_monotone_constraints
                        and column in simplified.monotone_increasing_features
                        else None
                    ),
                ),
                grouping=grouping,
                specials=list(simplified.special_levels.get(column, ())),
            )
            features[column] = ordered_spec
        elif column in categorical:
            features[column] = Categorical(grouping=grouping)
        elif column in linear:
            features[column] = Numeric()
        else:
            features[column] = Spline(
                kind=config.spline_kind,
                k=simplified.spline_k,
                knot_strategy=config.knot_strategy,
                knot_alpha=config.knot_alpha,
            )
    return features


def _fit_superglm(
    frame: pd.DataFrame,
    target: np.ndarray,
    exposure: np.ndarray,
    sample_weight: np.ndarray,
    train: np.ndarray,
    *,
    features: Mapping[str, Any],
    config: DiagnosticRunConfig,
    fixed_lambdas: Mapping[str, float] | None = None,
) -> SuperGLM:
    has_fit_constraint = any(
        getattr(getattr(spec, "_spline_obj", spec), "constraint_mode", None) == "fit"
        and getattr(getattr(spec, "_spline_obj", spec), "constraint_kind", None) is not None
        for spec in features.values()
    )
    model = SuperGLM(
        family=Tweedie(p=config.tweedie_power),
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        tol=1e-6,
        max_iter=500,
        convergence="deviance",
        features=dict(features),
    ).bind_levels(frame.iloc[train], sample_weight=sample_weight[train])
    if fixed_lambdas is not None:
        if not has_fit_constraint:
            raise ValueError("fixed_lambdas is reserved for a fit-constrained scratch GAM")
        model.lambda2 = {str(name): float(value) for name, value in fixed_lambdas.items()}
        return model.fit(
            frame.iloc[train],
            target[train],
            sample_weight=sample_weight[train],
            offset=np.log(exposure[train]),
            max_iter=500,
        )
    return model.fit_reml(
        frame.iloc[train],
        target[train],
        sample_weight=sample_weight[train],
        offset=np.log(exposure[train]),
        max_reml_iter=config.max_reml_iter,
        max_pirls_iter=500,
        runtime_validation="skip",
    )


def _prediction_frame(
    rows: np.ndarray,
    *,
    frame: pd.DataFrame,
    target: np.ndarray,
    exposure: np.ndarray,
    sample_weight: np.ndarray,
    raw_gam: SuperGLM,
    gam: SuperGLM,
    gbm,
    power: float,
) -> pd.DataFrame:
    X = frame.iloc[rows]
    resolved_exposure = exposure[rows]
    raw_gam_rate = (
        np.asarray(raw_gam.predict(X, offset=np.log(resolved_exposure)), dtype=float)
        / resolved_exposure
    )
    gam_rate = (
        np.asarray(gam.predict(X, offset=np.log(resolved_exposure)), dtype=float)
        / resolved_exposure
    )
    gbm_components = gbm.predict_components(X)
    component_weights = np.array([gbm.weights[column] for column in gbm_components.columns])
    gbm_rate = np.asarray(gbm_components @ component_weights, dtype=float)
    payload: dict[str, Any] = {
        "actual_rate": target[rows] / resolved_exposure,
        "actual_response": target[rows],
        "exposure": resolved_exposure,
        "sample_weight": sample_weight[rows],
        "fit_weight": sample_weight[rows] * np.power(resolved_exposure, 2.0 - power),
        "raw_gam_rate": raw_gam_rate,
        "gam_rate": gam_rate,
        "gbm_rate": gbm_rate,
        "raw_gam_expected": resolved_exposure * raw_gam_rate,
        "gam_expected": resolved_exposure * gam_rate,
        "gbm_expected": resolved_exposure * gbm_rate,
    }
    for column in gbm_components:
        payload[f"{column}_rate"] = gbm_components[column].to_numpy(dtype=float)
        payload[f"{column}_expected"] = resolved_exposure * gbm_components[column].to_numpy(
            dtype=float
        )
    return pd.DataFrame(payload)


def _all_model_expected(
    features: pd.DataFrame,
    exposure: np.ndarray,
    *,
    raw_gam: SuperGLM,
    gam: SuperGLM,
    gbm,
    governed_weights: tuple[float, ...],
    technical_gam_weight: float | None = None,
) -> dict[str, np.ndarray]:
    offset = np.log(exposure)
    raw_expected = np.asarray(raw_gam.predict(features, offset=offset), dtype=float)
    gam_expected = np.asarray(gam.predict(features, offset=offset), dtype=float)
    component_rates = gbm.predict_components(features)
    component_expected = {
        str(column): exposure * component_rates[column].to_numpy(dtype=float)
        for column in component_rates
    }
    component_weights = np.array([gbm.weights[column] for column in component_rates])
    boosted_expected = exposure * np.asarray(component_rates @ component_weights, dtype=float)
    predictions = {
        "raw_additive_superglm": raw_expected,
        "simplified_gam": gam_expected,
        **component_expected,
        "boosted_ensemble": boosted_expected,
    }
    for weight in governed_weights:
        predictions[f"{blend_weight_label(weight)}_simplified_gam"] = (
            weight * gam_expected + (1.0 - weight) * boosted_expected
        )
    if technical_gam_weight is not None and not any(
        np.isclose(technical_gam_weight, weight) for weight in governed_weights
    ):
        predictions["blend_technical_simplified_gam"] = (
            technical_gam_weight * gam_expected + (1.0 - technical_gam_weight) * boosted_expected
        )
    return predictions


def _all_model_summary(
    response: np.ndarray,
    sample_weight: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    power: float,
) -> pd.DataFrame:
    actual_average = float(np.average(response, weights=sample_weight))
    records = []
    for model_name, expected in predictions.items():
        predicted_average = float(np.average(expected, weights=sample_weight))
        records.append(
            {
                "model": model_name,
                "mean_tweedie_deviance": float(
                    np.average(
                        unit_tweedie_deviance(response, expected, power=power),
                        weights=sample_weight,
                    )
                ),
                "actual_response": actual_average,
                "predicted_response": predicted_average,
                "observed_to_predicted": actual_average / predicted_average,
            }
        )
    return pd.DataFrame(records).sort_values("mean_tweedie_deviance", ignore_index=True)


def _marginal_relativity_tables(
    reference_features: pd.DataFrame,
    full_features: pd.DataFrame,
    response: np.ndarray,
    exposure: np.ndarray,
    sample_weight: np.ndarray,
    *,
    raw_gam: SuperGLM,
    gam: SuperGLM,
    gbm,
    categorical_features: tuple[str, ...],
    governed_weights: tuple[float, ...],
    technical_gam_weight: float,
) -> dict[str, pd.DataFrame]:
    """Portfolio-standardised marginal relativities on one held-out population."""
    baseline_predictions = _all_model_expected(
        reference_features,
        exposure,
        raw_gam=raw_gam,
        gam=gam,
        gbm=gbm,
        governed_weights=governed_weights,
        technical_gam_weight=technical_gam_weight,
    )
    baseline_average = {
        name: float(np.average(values, weights=sample_weight))
        for name, values in baseline_predictions.items()
    }
    observed_average = float(np.average(response, weights=sample_weight))
    categorical = set(categorical_features)
    tables: dict[str, pd.DataFrame] = {}
    for feature in reference_features.columns:
        if feature not in categorical:
            continue
        levels = _observed_levels(full_features[feature])
        held_out_levels = reference_features[feature]
        records: list[dict[str, Any]] = []
        for level in levels:
            mask = held_out_levels.eq(level).to_numpy()
            support = float(sample_weight[mask].sum())
            scenario = reference_features.copy()
            scenario[feature] = level
            predictions = _all_model_expected(
                scenario,
                exposure,
                raw_gam=raw_gam,
                gam=gam,
                gbm=gbm,
                governed_weights=governed_weights,
                technical_gam_weight=technical_gam_weight,
            )
            record: dict[str, Any] = {
                "feature_level": level,
                "rows": int(mask.sum()),
                "sample_weight": support,
                "observed_unadjusted_relativity": (
                    float(np.average(response[mask], weights=sample_weight[mask]))
                    / observed_average
                    if support > 0
                    else np.nan
                ),
            }
            record.update(
                {
                    name: float(np.average(values, weights=sample_weight)) / baseline_average[name]
                    for name, values in predictions.items()
                }
            )
            records.append(record)
        tables[str(feature)] = pd.DataFrame(records)
    return tables


def _plot_relativity_table(
    feature: str,
    table: pd.DataFrame,
    *,
    governed_weights: tuple[float, ...],
    output_path: Path,
) -> None:
    figure, (headline_axis, component_axis, support_axis) = plt.subplots(
        3,
        1,
        figsize=(max(12, min(22, 0.7 * len(table) + 7)), 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 2.2, 1]},
    )
    x = np.arange(len(table))
    headline_axis.plot(
        x,
        table["observed_unadjusted_relativity"],
        color="black",
        marker="o",
        ls="--",
        label="Observed one-way (unadjusted)",
    )
    headline = (
        ("raw_additive_superglm", "Raw additive SuperGLM", "#7f7f7f", "s"),
        ("simplified_gam", "Simplified governed GAM", "#1f77b4", "o"),
        ("boosted_ensemble", "Boosted ensemble", "#d62728", "^"),
    )
    for column, label, color, marker in headline:
        headline_axis.plot(x, table[column], color=color, marker=marker, label=label)
    blend_colors = ("#9467bd", "#2ca02c", "#8c564b")
    for index, weight in enumerate(governed_weights):
        column = f"{blend_weight_label(weight)}_simplified_gam"
        headline_axis.plot(
            x,
            table[column],
            color=blend_colors[index % len(blend_colors)],
            marker="D",
            label=f"{weight:.0%} simplified GAM blend",
        )
    if "blend_technical_simplified_gam" in table:
        headline_axis.plot(
            x,
            table["blend_technical_simplified_gam"],
            color="#bcbd22",
            ls=":",
            lw=2,
            label="Validation-optimal blend",
        )
    component_style = (
        ("catboost", "CatBoost", "#ff7f0e"),
        ("lightgbm", "LightGBM", "#17becf"),
        ("xgboost", "XGBoost", "#e377c2"),
        ("boosted_ensemble", "Weighted boosted ensemble", "#d62728"),
    )
    for column, label, color in component_style:
        component_axis.plot(x, table[column], color=color, marker="o", label=label)
    for axis in (headline_axis, component_axis):
        axis.axhline(1.0, color="#555555", ls=":", lw=1)
        axis.set_ylabel("Relativity\n(portfolio = 1)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncols=2)
    headline_axis.set_title(
        f"Held-out portfolio-standardised model relativities: {feature}\n"
        "Model curves hold every other feature distribution fixed"
    )
    component_axis.set_title("Boosted component relativities", loc="left", fontsize=10)
    support_axis.bar(x, table["sample_weight"], color="#999999")
    support_axis.set(ylabel="Declared\nweight")
    support_axis.set_xticks(x, table["feature_level"].astype(str), rotation=45, ha="right")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _simplified_gam_invariants(
    config: DiagnosticRunConfig,
    full_features: pd.DataFrame,
    relativity_tables: Mapping[str, pd.DataFrame],
    *,
    model: SuperGLM | None = None,
    fitted_lambdas: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    simplified = config.simplified_gam
    if simplified is None:
        return pd.DataFrame()
    tolerance = 1e-8
    records: list[dict[str, Any]] = []

    def record(check: str, feature: str, detail: str, error: float) -> None:
        records.append(
            {
                "check": check,
                "feature": feature,
                "detail": detail,
                "max_violation": float(error),
                "status": "PASS" if error <= tolerance else "FAIL",
            }
        )

    for feature in simplified.monotone_increasing_features:
        table = relativity_tables[feature].set_index("feature_level")
        levels = _observed_levels(full_features[feature])
        specials = set(simplified.special_levels.get(feature, ()))
        values = table.loc[[level for level in levels if level not in specials], "simplified_gam"]
        minimum_step = float(np.min(np.diff(values.to_numpy(dtype=float))))
        record(
            "MONOTONE_INCREASING_RELATIVITY",
            feature,
            "raw ordered levels excluding configured specials",
            max(0.0, -minimum_step),
        )

    for feature, groups in simplified.level_groups.items():
        table = relativity_tables[feature].set_index("feature_level")
        for label, members in groups.items():
            values = table.loc[list(members), "simplified_gam"].to_numpy(dtype=float)
            record(
                "EXPLICIT_LEVEL_GROUP_EQUALITY",
                feature,
                label,
                float(np.max(values) - np.min(values)),
            )

    for feature, from_level in simplified.collapse_from.items():
        levels = _observed_levels(full_features[feature])
        tail = levels[levels.index(from_level) :]
        values = (
            relativity_tables[feature]
            .set_index("feature_level")
            .loc[tail, "simplified_gam"]
            .to_numpy(dtype=float)
        )
        record(
            "COLLAPSED_TAIL_EQUALITY",
            feature,
            f"{from_level} and above",
            float(np.max(values) - np.min(values)),
        )

    if model is not None:
        specs = model.features
        for feature in simplified.monotone_increasing_features:
            spline = specs[feature]._spline_obj
            valid = spline.constraint_kind == "increasing" and spline.constraint_mode == "fit"
            record(
                "FIT_TIME_CONSTRAINT_METADATA",
                feature,
                "Constraint.fit.increasing",
                0.0 if valid else 1.0,
            )
        if fitted_lambdas:
            configured_lambdas = model.lambda2
            for feature, expected in fitted_lambdas.items():
                actual = float(configured_lambdas[feature])
                record(
                    "FROZEN_LAMBDA_EQUALITY",
                    feature,
                    "unconstrained REML estimate equals constrained-fit lambda",
                    abs(actual - float(expected)),
                )
        record(
            "COEFFICIENT_SOLVER_CONVERGENCE",
            "__MODEL__",
            "final constrained coefficient fit",
            0.0 if bool(model.result.converged) else 1.0,
        )

    result = pd.DataFrame(records)
    failures = result.loc[result["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(
            "simplified GAM invariant checks failed: "
            + "; ".join(f"{row.check}:{row.feature}:{row.detail}" for row in failures.itertuples())
        )
    return result


def run_diagnostics(config: DiagnosticRunConfig) -> dict[str, Any]:
    """Fit locally and write aggregate-only diagnostics under ignored state."""
    if not config.data_path.is_file():
        raise FileNotFoundError(f"configured local parquet does not exist: {config.data_path}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = tuple(
        dict.fromkeys(
            (
                config.target,
                config.offset_source,
                config.sample_weight,
                *config.features,
            )
        )
    )
    source = pd.read_parquet(config.data_path, columns=list(required_columns))
    if config.sample_rows is not None and len(source) > config.sample_rows:
        source = source.sample(n=config.sample_rows, random_state=config.random_seed)
    source = source.reset_index(drop=True)
    target = source[config.target].to_numpy(dtype=float)
    exposure = source[config.offset_source].to_numpy(dtype=float) / config.offset_divisor
    sample_weight = source[config.sample_weight].to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("target must contain finite non-negative values")
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("offset exposure must contain finite positive values")
    if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
        raise ValueError("sample weight must contain finite positive values")
    frame = source.loc[:, list(config.features)].copy()
    categorical = set(config.categorical_features)
    for column in frame.columns:
        if column in categorical:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValueError(
                f"non-categorical feature {column!r} must contain finite numeric values"
            )
        frame[column] = numeric
    indices = np.arange(len(source))
    stratify = target > 0 if len(np.unique(target > 0)) == 2 else None
    train, remainder = train_test_split(
        indices,
        train_size=config.train_fraction,
        random_state=config.random_seed,
        stratify=stratify,
    )
    relative_validation = config.validation_fraction / (1.0 - config.train_fraction)
    remainder_stratify = target[remainder] > 0 if stratify is not None else None
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation,
        random_state=config.random_seed + 1,
        stratify=remainder_stratify,
    )
    _validate_holdout_categorical_levels(
        frame,
        train,
        validation,
        categorical_features=config.categorical_features,
        partition="validation",
    )
    _validate_holdout_categorical_levels(
        frame,
        train,
        test,
        categorical_features=config.categorical_features,
        partition="test",
    )

    raw_gam_features = unconstrained_superglm_features(
        frame.iloc[train],
        categorical_columns=config.categorical_features,
        linear_columns=config.linear_features,
        spline_kind=config.spline_kind,
        k=config.spline_k,
        knot_strategy=config.knot_strategy,
        knot_alpha=config.knot_alpha,
    )
    raw_gam = _fit_superglm(
        frame,
        target,
        exposure,
        sample_weight,
        train,
        features=raw_gam_features,
        config=config,
    )
    fitted_lambdas: dict[str, float] = {}
    if config.simplified_gam is None:
        gam = raw_gam
    else:
        if config.simplified_gam.monotone_increasing_features:
            smoothing_source = _fit_superglm(
                frame,
                target,
                exposure,
                sample_weight,
                train,
                features=_simplified_superglm_features(
                    frame.iloc[train],
                    config=config,
                    apply_monotone_constraints=False,
                ),
                config=config,
            )
            fitted_lambdas = {
                str(name): float(value)
                for name, value in smoothing_source.reml_diagnostics().get("lambdas", {}).items()
            }
            missing_lambdas = set(config.simplified_gam.ordered_spline_features) - set(
                fitted_lambdas
            )
            if missing_lambdas:
                raise RuntimeError(
                    "unconstrained smoothing fit did not estimate lambdas for: "
                    + ", ".join(sorted(missing_lambdas))
                )
            gam = _fit_superglm(
                frame,
                target,
                exposure,
                sample_weight,
                train,
                features=_simplified_superglm_features(
                    frame.iloc[train],
                    config=config,
                ),
                config=config,
                fixed_lambdas=fitted_lambdas,
            )
            pd.DataFrame(
                [
                    {
                        "feature_name": name,
                        "lambda_value": value,
                        "lambda_source": "UNCONSTRAINED_REML_SAME_TRAINING_SAMPLE",
                        "final_policy": "FIXED_DURING_CONSTRAINED_COEFFICIENT_REFIT",
                    }
                    for name, value in sorted(fitted_lambdas.items())
                ]
            ).to_csv(config.output_dir / "simplified_gam_lambdas.csv", index=False)
            superglm_edf_table(smoothing_source).to_csv(
                config.output_dir / "simplified_gam_smoothing_source_edf.csv",
                index=False,
            )
        else:
            gam = _fit_superglm(
                frame,
                target,
                exposure,
                sample_weight,
                train,
                features=_simplified_superglm_features(frame.iloc[train], config=config),
                config=config,
            )
        if not config.simplified_gam.monotone_increasing_features:
            superglm_edf_table(gam).to_csv(
                config.output_dir / "simplified_gam_edf.csv",
                index=False,
            )
    gbm = fit_boosted_blend(
        frame.iloc[train],
        target[train],
        categorical_columns=config.categorical_features,
        sample_weight=sample_weight[train],
        exposure=exposure[train],
        n_splits=config.cv_folds,
        random_state=config.random_seed,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        thread_count=config.thread_count,
        tweedie_power=(
            config.tweedie_power
            if config.simplified_gam is not None
            and config.simplified_gam.monotone_increasing_features
            else None
        ),
        reference_superglm=(
            None
            if config.simplified_gam is not None
            and config.simplified_gam.monotone_increasing_features
            else gam
        ),
    )
    gbm.metrics.to_csv(config.output_dir / "gbm_component_summary.csv", index=False)
    validation_predictions = _prediction_frame(
        validation,
        frame=frame,
        target=target,
        exposure=exposure,
        sample_weight=sample_weight,
        raw_gam=raw_gam,
        gam=gam,
        gbm=gbm,
        power=config.tweedie_power,
    )
    test_predictions = _prediction_frame(
        test,
        frame=frame,
        target=target,
        exposure=exposure,
        sample_weight=sample_weight,
        raw_gam=raw_gam,
        gam=gam,
        gbm=gbm,
        power=config.tweedie_power,
    )
    technical_gam_weight = float(
        _convex_tweedie_weights(
            validation_predictions["actual_response"].to_numpy(),
            validation_predictions[["gam_expected", "gbm_expected"]].to_numpy(),
            validation_predictions["sample_weight"].to_numpy(),
            power=config.tweedie_power,
        )[0]
    )
    diagnostic_weights = tuple(
        sorted({0.0, 1.0, technical_gam_weight, *config.governed_gam_weights})
    )
    summary = blend_evaluation_table(
        target[test],
        offset_exposure=exposure[test],
        sample_weight=sample_weight[test],
        gam_rate=test_predictions["gam_rate"],
        gbm_rate=test_predictions["gbm_rate"],
        power=config.tweedie_power,
        gam_weights=diagnostic_weights,
    )
    summary["weight_kind"] = [
        "TECHNICAL_VALIDATION_OPTIMUM"
        if np.isclose(value, technical_gam_weight)
        else "GOVERNED"
        if any(np.isclose(value, item) for item in config.governed_gam_weights)
        else "ENDPOINT"
        for value in summary["gam_weight"]
    ]
    summary.to_csv(config.output_dir / "blend_summary.csv", index=False)

    all_predictions = _all_model_expected(
        frame.iloc[test],
        exposure[test],
        raw_gam=raw_gam,
        gam=gam,
        gbm=gbm,
        governed_weights=config.governed_gam_weights,
        technical_gam_weight=technical_gam_weight,
    )
    _all_model_summary(
        target[test],
        sample_weight[test],
        all_predictions,
        power=config.tweedie_power,
    ).to_csv(config.output_dir / "all_model_summary.csv", index=False)

    test_features = frame.iloc[test].reset_index(drop=True)
    diagnostic_arguments = {
        "offset_exposure": exposure[test],
        "sample_weight": sample_weight[test],
        "gam_rate": test_predictions["gam_rate"],
        "gbm_rate": test_predictions["gbm_rate"],
        "power": config.tweedie_power,
        "governed_gam_weights": config.governed_gam_weights,
    }
    double_lift = double_lift_table(
        test_features,
        target[test],
        n_bins=config.double_lift_bins,
        **diagnostic_arguments,
    )
    risk = risk_calibration_table(
        test_features,
        target[test],
        n_bins=config.risk_bins,
        **diagnostic_arguments,
    )
    lorenz = lorenz_curve_table(
        target[test],
        n_bins=config.lorenz_bins,
        **diagnostic_arguments,
    )
    feature_tables = feature_calibration_tables(
        test_features,
        target[test],
        categorical_features=config.categorical_features,
        n_bins=config.feature_bins,
        max_categorical_levels=config.max_categorical_levels,
        **diagnostic_arguments,
    )
    interaction_ranking, interaction_cells = interaction_failure_tables(
        test_features,
        target[test],
        categorical_features=config.categorical_features,
        n_bins=config.interaction_bins,
        max_categorical_levels=config.max_categorical_levels,
        min_cell_rows=config.min_cell_rows,
        min_cell_weight_fraction=config.min_cell_weight_fraction,
        **diagnostic_arguments,
    )
    double_lift.to_csv(config.output_dir / "double_lift.csv", index=False)
    risk.to_csv(config.output_dir / "risk_calibration.csv", index=False)
    lorenz.to_csv(config.output_dir / "lorenz_curve.csv", index=False)
    pd.concat(
        [table.assign(feature=feature) for feature, table in feature_tables.items()],
        ignore_index=True,
    ).to_csv(config.output_dir / "feature_calibration.csv", index=False)
    interaction_ranking.to_csv(config.output_dir / "interaction_ranking.csv", index=False)
    interaction_cells.to_csv(config.output_dir / "interaction_failure_cells.csv", index=False)

    relativity_tables: dict[str, pd.DataFrame] = {}
    if config.simplified_gam is not None:
        relativity_tables = _marginal_relativity_tables(
            test_features,
            frame,
            target[test],
            exposure[test],
            sample_weight[test],
            raw_gam=raw_gam,
            gam=gam,
            gbm=gbm,
            categorical_features=config.categorical_features,
            governed_weights=config.governed_gam_weights,
            technical_gam_weight=technical_gam_weight,
        )
        pd.concat(
            [table.assign(feature=feature) for feature, table in relativity_tables.items()],
            ignore_index=True,
        ).to_csv(config.output_dir / "model_relativities.csv", index=False)
        _simplified_gam_invariants(
            config,
            frame,
            relativity_tables,
            model=gam,
            fitted_lambdas=fitted_lambdas,
        ).to_csv(config.output_dir / "simplified_gam_invariants.csv", index=False)

    _plot_blend_curve(
        validation_predictions,
        test_predictions,
        power=config.tweedie_power,
        optimum_weight=technical_gam_weight,
        governed_weights=config.governed_gam_weights,
        output_dir=config.output_dir,
    )
    gam_label = (
        "Simplified governed GAM" if config.simplified_gam is not None else "Additive SuperGLM"
    )
    _plot_calibration_table(
        double_lift,
        bin_column="Boosted blend / GAM ratio bin: low → high",
        title="Double lift ordered by boosted blend / GAM prediction ratio",
        output_path=config.output_dir / "02_double_lift.png",
        governed_weights=config.governed_gam_weights,
        gam_label=gam_label,
        indexed=True,
    )
    _plot_calibration_table(
        risk,
        bin_column="Boosted-blend predicted-risk bin: low → high",
        title="Held-out calibration through the boosted-blend risk tail",
        output_path=config.output_dir / "03_risk_calibration.png",
        governed_weights=config.governed_gam_weights,
        gam_label=gam_label,
    )
    _plot_lorenz(
        lorenz,
        output_path=config.output_dir / "04_lorenz.png",
        gam_label=gam_label,
    )
    for index, (feature, table) in enumerate(feature_tables.items(), start=1):
        _plot_feature_table(
            feature,
            table,
            governed_weights=config.governed_gam_weights,
            gam_label=gam_label,
            output_path=config.output_dir / f"05_feature_{index:02d}_{_slug(feature)}.png",
        )
    for index, row in interaction_ranking.head(config.top_interactions).iterrows():
        feature_a = str(row["feature_a"])
        feature_b = str(row["feature_b"])
        _plot_interaction(
            feature_a,
            feature_b,
            interaction_cells,
            output_path=config.output_dir
            / f"06_interaction_{index + 1:02d}_{_slug(feature_a)}__{_slug(feature_b)}.png",
        )
    for index, (feature, table) in enumerate(relativity_tables.items(), start=1):
        _plot_relativity_table(
            feature,
            table,
            governed_weights=config.governed_gam_weights,
            output_path=config.output_dir / f"07_relativity_{index:02d}_{_slug(feature)}.png",
        )

    report_lines = [
        "# Scratch GAM/boosted-blend failure diagnostic",
        "",
        f"- Rows: train {len(train):,}; validation {len(validation):,}; test {len(test):,}",
        f"- Tweedie power: {config.tweedie_power:g}",
        "- GAM endpoint: "
        + ("simplified governed GAM" if config.simplified_gam is not None else "raw additive"),
        f"- Technical GAM weight selected on validation: {technical_gam_weight:.2%}",
        f"- Governed GAM weights inspected: {', '.join(f'{value:.0%}' for value in config.governed_gam_weights)}",
        "- Boosted component weights: "
        + ", ".join(f"{name} {weight:.2%}" for name, weight in gbm.weights.items()),
        "- Double-lift bins: balanced by declared sample weight and ordered by boosted-blend/GAM",
        "- Every plotted mean: sum(weight × response) / sum(weight); predictions use the same denominator",
        "- Row-level data and predictions: not retained",
        "- Relativity curves: held-out portfolio-standardised marginal predictions; observed is unadjusted one-way experience",
        "",
        "## Top interaction/failure pairs",
        "",
        "```text",
        interaction_ranking.head(config.top_interactions).to_string(index=False),
        "```",
        "",
        "Positive boosted-blend-minus-GAM deviance means the boosted blend is worse in that held-out cell.",
    ]
    (config.output_dir / "diagnostic_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(source),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "technical_gam_weight": technical_gam_weight,
        "output_dir": str(config.output_dir),
        "top_failure_pair": interaction_ranking.iloc[0][["feature_a", "feature_b"]].tolist(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--allow-local-input",
        action="store_true",
        help="Required acknowledgement that the configured parquet is local and must not be copied.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.allow_local_input:
        raise SystemExit("Refusing to read configured input without --allow-local-input")
    config = load_diagnostic_config(args.config.expanduser().resolve())
    result = run_diagnostics(config)
    print("Local aggregate-only diagnostics complete.")
    print(f"Rows: {result['rows']:,}")
    print(f"Technical GAM weight: {result['technical_gam_weight']:.2%}")
    print(f"Top failure pair: {' × '.join(result['top_failure_pair'])}")
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
