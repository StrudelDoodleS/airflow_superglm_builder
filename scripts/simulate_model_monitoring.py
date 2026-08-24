"""Run a reproducible synthetic streaming-drift monitoring demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.metrics import mean_poisson_deviance
from superglm import Categorical, Numeric, Spline, SuperGLM, collapse_levels
from superglm.features import Constraint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.modeling.monitoring import (
    MonitoringVariant,
    run_monitoring_fit,
)

VARIANT_COLORS = {
    MonitoringVariant.STATIC_SCORE.value: "#4c566a",
    MonitoringVariant.FROZEN_REFIT.value: "#2f6f9f",
    MonitoringVariant.REESTIMATE_LAMBDA.value: "#d08770",
    MonitoringVariant.FULL_ADAPTIVE.value: "#a3be8c",
}
WINDOWS = (60, 70, 80, 90, 100)
STREAM_VARIANTS = tuple(MonitoringVariant)


def _synthetic_stream(row_count: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if row_count < 1000 or row_count % 10:
        raise ValueError("row_count must be at least 1000 and divisible by 10")

    rng = np.random.default_rng(seed)
    tenth = row_count // 10
    window = np.repeat(WINDOWS, (6 * tenth, tenth, tenth, tenth, tenth))
    drift_step = np.repeat(range(5), (6 * tenth, tenth, tenth, tenth, tenth))
    risk_score = np.empty(row_count, dtype=float)
    segment = np.empty(row_count, dtype="U1")
    tenure = np.empty(row_count, dtype=float)
    segment_levels = np.array(["A", "B", "C", "D"])

    probabilities = (
        (0.34, 0.30, 0.23, 0.13),
        (0.30, 0.28, 0.25, 0.17),
        (0.27, 0.26, 0.26, 0.21),
        (0.24, 0.24, 0.27, 0.25),
        (0.21, 0.22, 0.28, 0.29),
    )
    for step in range(5):
        mask = drift_step == step
        count = int(mask.sum())
        risk_score[mask] = rng.beta(2.0 + 0.45 * step, 4.8 - 0.35 * step, count)
        segment[mask] = rng.choice(segment_levels, size=count, p=probabilities[step])
        tenure[mask] = np.clip(rng.normal(0.45 + 0.035 * step, 0.22, count), 0.0, 1.0)

    segment_effect = np.select(
        [segment == "C", segment == "D"],
        [0.16 + 0.025 * drift_step, 0.32 + 0.075 * drift_step],
        default=0.0,
    )
    log_mean = (
        -1.15
        + 0.035 * drift_step
        + (0.85 + 0.10 * drift_step) * risk_score
        + (0.55 + 0.06 * drift_step) * risk_score**2
        + (0.18 - 0.015 * drift_step) * tenure
        + segment_effect
    )
    y = rng.poisson(np.exp(log_mean))
    frame = pd.DataFrame(
        {
            "stream_window": window,
            "drift_step": drift_step,
            "segment": segment,
            "risk_score": risk_score,
            "tenure": tenure,
        }
    )
    return frame, y


def _fit_baseline(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_reml_iter: int,
) -> SuperGLM:
    grouping = collapse_levels(X["segment"], groups={"CORE": ["A", "B"]})
    return SuperGLM(
        family="poisson",
        features={
            "segment": Categorical(grouping=grouping, base="CORE"),
            "risk_score": Spline(
                kind="ps",
                k=9,
                knot_strategy="quantile",
                constraint=Constraint.postfit.increasing,
            ),
            "tenure": Numeric(),
        },
        selection_penalty=0.0,
    ).fit_reml(
        X,
        y,
        max_reml_iter=max_reml_iter,
        runtime_validation="skip",
    )


def _append_lambda_rows(
    rows: list[dict[str, object]],
    *,
    percent: int,
    variant: MonitoringVariant,
    result,
    source: str,
) -> None:
    for item in result.lambdas:
        rows.append(
            {
                "available_percent": percent,
                "variant": variant.value,
                "component_name": item.component_name,
                "term_name": item.term_name,
                "lambda_value": item.lambda_value,
                "lambda_mode": item.lambda_mode,
                "source": source,
            }
        )


def _append_knot_rows(
    rows: list[dict[str, object]],
    *,
    percent: int,
    variant: MonitoringVariant,
    model: SuperGLM,
    source: str,
) -> None:
    spline = model._specs["risk_score"]
    for knot_index, knot_value in enumerate(spline.fitted_knots, start=1):
        rows.append(
            {
                "available_percent": percent,
                "variant": variant.value,
                "knot_index": knot_index,
                "knot_value": float(knot_value),
                "boundary_lower": float(spline.fitted_boundary[0]),
                "boundary_upper": float(spline.fitted_boundary[1]),
                "source": source,
            }
        )


def _append_relativity_rows(
    rows: list[dict[str, object]],
    *,
    percent: int,
    variant: MonitoringVariant,
    result,
    source: str,
) -> None:
    for item in result.relativities:
        if item.term_name != "risk_score":
            continue
        rows.append(
            {
                "available_percent": percent,
                "variant": variant.value,
                "risk_score": item.point_numeric,
                "relativity": item.relativity,
                "log_relativity": item.log_relativity,
                "source": source,
            }
        )


def _save_figure(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_feature_drift(frame: pd.DataFrame, output_dir: Path) -> None:
    aggregates = frame.groupby("stream_window")["risk_score"].agg(
        mean="mean",
        p10=lambda values: values.quantile(0.1),
        median="median",
        p90=lambda values: values.quantile(0.9),
    )
    mix = pd.crosstab(frame["stream_window"], frame["segment"], normalize="index")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x_values = aggregates.index.to_numpy(dtype=float)
    axes[0].fill_between(
        x_values,
        aggregates["p10"].to_numpy(),
        aggregates["p90"].to_numpy(),
        color="#88c0d0",
        alpha=0.28,
        label="10th–90th percentile",
    )
    axes[0].plot(x_values, aggregates["mean"], marker="o", color="#2f6f9f", label="mean")
    axes[0].plot(
        x_values,
        aggregates["median"],
        marker="s",
        color="#5e81ac",
        label="median",
    )
    axes[0].set(title="Continuous feature drift", xlabel="Data available (%)", ylabel="Risk score")
    axes[0].legend(frameon=False)
    mix.plot(
        kind="bar", stacked=True, ax=axes[1], color=["#5e81ac", "#88c0d0", "#d08770", "#bf616a"]
    )
    axes[1].set(title="Categorical mix drift", xlabel="Stream window (%)", ylabel="Portfolio share")
    axes[1].legend(title="Segment", frameon=False, ncols=2)
    axes[1].tick_params(axis="x", rotation=0)
    figure.suptitle("Known synthetic drift injected after the 60% baseline", fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, output_dir, "01_feature_drift.png")


def _plot_lambda_paths(lambda_frame: pd.DataFrame, output_dir: Path) -> None:
    selected = lambda_frame[lambda_frame["term_name"].eq("risk_score")]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    for variant, rows in selected.groupby("variant", sort=False):
        axis.plot(
            rows["available_percent"],
            rows["lambda_value"],
            marker="o",
            linewidth=2,
            linestyle="--" if variant == MonitoringVariant.STATIC_SCORE.value else "-",
            label=variant,
            color=VARIANT_COLORS[variant],
            zorder=4 if variant == MonitoringVariant.STATIC_SCORE.value else 2,
        )
    if (selected["lambda_value"] > 0).all():
        axis.set_yscale("log")
    axis.set(
        title=("REML smoothing path for risk_score\nSTATIC_SCORE and FROZEN_REFIT overlap exactly"),
        xlabel="Cumulative data available (%)",
        ylabel="Lambda (log scale)",
        xticks=WINDOWS,
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_dir, "02_lambda_paths.png")


def _plot_knot_paths(knot_frame: pd.DataFrame, output_dir: Path) -> None:
    panels = (MonitoringVariant.FROZEN_REFIT.value, MonitoringVariant.FULL_ADAPTIVE.value)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, variant in zip(axes, panels, strict=True):
        selected = knot_frame[knot_frame["variant"].eq(variant)]
        for knot_index, rows in selected.groupby("knot_index"):
            axis.plot(
                rows["available_percent"],
                rows["knot_value"],
                marker="o",
                linewidth=1.7,
                label=f"knot {knot_index}",
            )
        title = (
            "Frozen geometry\n(lambda-only geometry is identical)"
            if variant == panels[0]
            else "Adaptive geometry"
        )
        axis.set(title=title, xlabel="Cumulative data available (%)", xticks=WINDOWS)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Interior knot location")
    axes[1].legend(frameon=False, ncols=2, fontsize=8)
    figure.suptitle(
        "Protected knots remain exact; adaptive quantile knots follow feature drift",
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, output_dir, "03_knot_paths.png")


def _plot_relativity_paths(relativity_frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    color_map = plt.get_cmap("viridis")
    normalization = Normalize(vmin=min(WINDOWS), vmax=max(WINDOWS))
    for axis, variant in zip(axes.flat, STREAM_VARIANTS, strict=True):
        selected = relativity_frame[relativity_frame["variant"].eq(variant.value)]
        for percent, rows in selected.groupby("available_percent"):
            axis.plot(
                rows["risk_score"],
                rows["relativity"],
                color=color_map(normalization(percent)),
                linewidth=2 if percent in {60, 100} else 1.2,
                alpha=1.0 if percent in {60, 100} else 0.65,
            )
        axis.set_title(variant.value)
        axis.grid(alpha=0.16)
        if variant is MonitoringVariant.STATIC_SCORE:
            axis.text(
                0.03,
                0.95,
                "All windows overlap exactly",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("Risk score on deployed grid")
    for axis in axes[:, 0]:
        axis.set_ylabel("Relativity")
    legend = [
        Line2D([0], [0], color=color_map(normalization(percent)), lw=2, label=f"{percent}%")
        for percent in WINDOWS
    ]
    figure.legend(handles=legend, loc="center right", frameon=False, title="Data available")
    figure.suptitle("Relativity drift on one fixed business comparison grid", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 0.91, 0.96))
    _save_figure(figure, output_dir, "04_relativity_drift.png")


def _plot_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for variant, rows in summary.groupby("variant", sort=False):
        color = VARIANT_COLORS[variant]
        axes[0].plot(
            rows["available_percent"],
            rows["new_window_mean_poisson_deviance"],
            marker="o",
            linewidth=2,
            label=variant,
            color=color,
        )
        axes[1].plot(
            rows["available_percent"],
            rows["new_window_calibration_ratio"],
            marker="o",
            linewidth=2,
            label=variant,
            color=color,
        )
    axes[0].set(
        title="Out-of-time predictive deviance",
        xlabel="Newest 10% arrival window",
        ylabel="Mean Poisson deviance (lower is better)",
        xticks=WINDOWS[1:],
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Out-of-time calibration",
        xlabel="Newest 10% arrival window",
        ylabel="Observed / predicted mean",
        xticks=WINDOWS[1:],
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Each arrival is scored before update; variants then refit cumulatively",
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, output_dir, "05_out_of_time_performance.png")


def _max_path_delta(
    frame: pd.DataFrame,
    *,
    variant: str,
    value_column: str,
    key_columns: list[str],
) -> float:
    selected = frame[frame["variant"].eq(variant)]
    baseline = selected[selected["available_percent"].eq(60)].set_index(key_columns)[value_column]
    later = selected[selected["available_percent"].gt(60)].copy()
    later["baseline"] = later.set_index(key_columns).index.map(baseline)
    return float((later[value_column] - later["baseline"]).abs().max())


def run_simulation(
    output_dir: Path,
    *,
    row_count: int = 10_000,
    seed: int = 1729,
    max_reml_iter: int = 12,
) -> dict[str, object]:
    plt.switch_backend("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, y = _synthetic_stream(row_count, seed)
    features = ["segment", "risk_score", "tenure"]
    baseline_mask = frame["stream_window"].eq(60).to_numpy()
    baseline_X = frame.loc[baseline_mask, features].reset_index(drop=True)
    baseline_y = y[baseline_mask]
    baseline = _fit_baseline(baseline_X, baseline_y, max_reml_iter=max_reml_iter)
    baseline_result = run_monitoring_fit(
        baseline,
        baseline_X,
        baseline_y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=81,
    )

    lambda_rows: list[dict[str, object]] = []
    knot_rows: list[dict[str, object]] = []
    relativity_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    invariant_evidence: dict[str, object] = {
        "60_STATIC_SCORE": baseline_result.invariant_evidence.payload()
    }
    previous_models = {variant: baseline for variant in STREAM_VARIANTS}
    for variant in STREAM_VARIANTS:
        _append_lambda_rows(
            lambda_rows,
            percent=60,
            variant=variant,
            result=baseline_result,
            source="DEPLOYED_BASELINE",
        )
        _append_knot_rows(
            knot_rows,
            percent=60,
            variant=variant,
            model=baseline,
            source="DEPLOYED_BASELINE",
        )
        _append_relativity_rows(
            relativity_rows,
            percent=60,
            variant=variant,
            result=baseline_result,
            source="DEPLOYED_BASELINE",
        )

    for percent in WINDOWS[1:]:
        cumulative_mask = frame["stream_window"].le(percent).to_numpy()
        new_window_mask = frame["stream_window"].eq(percent).to_numpy()
        cumulative_X = frame.loc[cumulative_mask, features].reset_index(drop=True)
        cumulative_y = y[cumulative_mask]
        new_X = frame.loc[new_window_mask, features].reset_index(drop=True)
        new_y = y[new_window_mask]
        for variant in STREAM_VARIANTS:
            scoring_model = previous_models[variant]
            predictions = np.asarray(scoring_model.predict(new_X), dtype=float)
            result = run_monitoring_fit(
                baseline,
                cumulative_X,
                cumulative_y,
                variant=variant,
                continuous_points=81,
                max_reml_iter=max_reml_iter,
                runtime_validation="skip",
            )
            previous_models[variant] = result.fitted_model
            summary_rows.append(
                {
                    "available_percent": percent,
                    "scoring_model_trained_through_percent": percent - 10,
                    "cumulative_rows": int(cumulative_mask.sum()),
                    "new_window_rows": int(new_window_mask.sum()),
                    "variant": variant.value,
                    "invariant_status": result.invariant_evidence.status,
                    "invariant_evidence_sha256": result.invariant_evidence.evidence_sha256,
                    "new_window_mean_poisson_deviance": mean_poisson_deviance(
                        new_y,
                        predictions,
                    ),
                    "new_window_observed_mean": float(np.mean(new_y)),
                    "new_window_predicted_mean": float(np.mean(predictions)),
                    "new_window_calibration_ratio": float(np.mean(new_y) / np.mean(predictions)),
                }
            )
            invariant_evidence[f"{percent}_{variant.value}"] = result.invariant_evidence.payload()
            _append_lambda_rows(
                lambda_rows,
                percent=percent,
                variant=variant,
                result=result,
                source="MONITORING_RUN",
            )
            _append_knot_rows(
                knot_rows,
                percent=percent,
                variant=variant,
                model=result.fitted_model,
                source="MONITORING_RUN",
            )
            _append_relativity_rows(
                relativity_rows,
                percent=percent,
                variant=variant,
                result=result,
                source="MONITORING_RUN",
            )

    summary = pd.DataFrame(summary_rows)
    lambda_frame = pd.DataFrame(lambda_rows)
    knot_frame = pd.DataFrame(knot_rows)
    relativity_frame = pd.DataFrame(relativity_rows)
    feature_drift = (
        frame.groupby("stream_window")
        .agg(
            rows=("risk_score", "size"),
            risk_score_mean=("risk_score", "mean"),
            risk_score_p10=("risk_score", lambda values: values.quantile(0.1)),
            risk_score_p50=("risk_score", "median"),
            risk_score_p90=("risk_score", lambda values: values.quantile(0.9)),
            tenure_mean=("tenure", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "stream_summary.csv", index=False)
    lambda_frame.to_csv(output_dir / "lambda_paths.csv", index=False)
    knot_frame.to_csv(output_dir / "knot_paths.csv", index=False)
    relativity_frame.to_csv(output_dir / "relativity_paths.csv", index=False)
    feature_drift.to_csv(output_dir / "feature_drift.csv", index=False)
    (output_dir / "invariant_evidence.json").write_text(
        json.dumps(invariant_evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _plot_feature_drift(frame, output_dir)
    _plot_lambda_paths(lambda_frame, output_dir)
    _plot_knot_paths(knot_frame, output_dir)
    _plot_relativity_paths(relativity_frame, output_dir)
    _plot_performance(summary, output_dir)

    checks = {
        "verified_monitoring_runs": int(summary["invariant_status"].eq("VERIFIED").sum()),
        "total_monitoring_runs": len(summary),
        "frozen_lambda_max_absolute_delta": _max_path_delta(
            lambda_frame[lambda_frame["term_name"].eq("risk_score")],
            variant=MonitoringVariant.FROZEN_REFIT.value,
            value_column="lambda_value",
            key_columns=["component_name"],
        ),
        "frozen_knot_max_absolute_delta": _max_path_delta(
            knot_frame,
            variant=MonitoringVariant.FROZEN_REFIT.value,
            value_column="knot_value",
            key_columns=["knot_index"],
        ),
        "lambda_only_knot_max_absolute_delta": _max_path_delta(
            knot_frame,
            variant=MonitoringVariant.REESTIMATE_LAMBDA.value,
            value_column="knot_value",
            key_columns=["knot_index"],
        ),
        "adaptive_knot_max_absolute_delta": _max_path_delta(
            knot_frame,
            variant=MonitoringVariant.FULL_ADAPTIVE.value,
            value_column="knot_value",
            key_columns=["knot_index"],
        ),
    }
    report_lines = [
        "# Synthetic streaming monitoring report",
        "",
        f"Rows: {row_count:,}; baseline: 60%; arrivals: four windows of 10%.",
        "All data are generated by this script.",
        "",
        "## Guard checks",
        "",
        *(f"- {name}: {value}" for name, value in checks.items()),
        "",
        (
            "Each arriving 10% window is scored using models fitted only through the previous "
            "window. After scoring, every refit variant is updated on all currently available "
            "rows. The 100% refit is retained for parameter evidence and future scoring."
        ),
    ]
    (output_dir / "simulation_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir.resolve()), **checks}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("state/monitoring_simulation"),
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-reml-iter", type=int, default=12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_simulation(
        args.output_dir,
        row_count=args.rows,
        seed=args.seed,
        max_reml_iter=args.max_reml_iter,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
