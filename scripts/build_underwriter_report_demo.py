"""Build a richer public freMTPL preview of the underwriter HTML report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from superglm import Categorical, Spline, SuperGLM, TensorInteraction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.data.fremtpl import fetch_fremtpl
from pricing_pipeline.reporting import UnderwriterReportOptions, build_underwriter_report


def _core_features():
    return {
        "Area": Categorical(),
        "VehPower": Spline(k=5),
        "VehAge": Spline(k=6),
        "DrivAge": Spline(k=7),
        "VehBrand": Categorical(),
        "VehGas": Categorical(),
        "Region": Categorical(),
        "LogDensity": Spline(k=6),
    }


def _new_interactions():
    interaction = TensorInteraction("DrivAge", "VehAge", n_knots=(4, 4))
    interaction.name = "DrivAge:VehAge"
    return [interaction]


def _fit(frame, rows: np.ndarray, features, *, interactions=None):
    names = list(features)
    return SuperGLM(
        family="poisson",
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        spline_penalty=1.0,
    ).fit(
        frame.loc[rows, names],
        frame.loc[rows, "ClaimRate"].to_numpy(),
        sample_weight=frame.loc[rows, "Exposure"].to_numpy(),
    )


def _resolved_row_count(available: int, requested: int | None) -> int:
    if requested is None:
        return available
    if requested < 2_000:
        raise ValueError("rows must be at least 2,000")
    return min(requested, available)


def build_demo(output_path: Path, *, rows: int | None, seed: int):
    """Fit three vintages on public data and write a common-holdout report."""
    raw = fetch_fremtpl()
    rows = _resolved_row_count(len(raw), rows)
    rng = np.random.default_rng(seed)
    frame = (
        raw.reset_index(drop=True)
        if rows == len(raw)
        else raw.iloc[rng.choice(len(raw), size=rows, replace=False)].reset_index(drop=True)
    )
    frame["Exposure"] = frame["Exposure"].astype(float).clip(lower=1e-6, upper=1.0)
    frame["ClaimNb"] = frame["ClaimNb"].astype(float).clip(upper=4.0)
    frame["ClaimRate"] = frame["ClaimNb"] / frame["Exposure"]
    frame["LogDensity"] = np.log1p(frame["Density"].astype(float))

    permutation = rng.permutation(len(frame))
    test_size = max(500, len(frame) // 4)
    test_rows = permutation[:test_size]
    training_rows = permutation[test_size:]
    old_rows = training_rows[: max(1, round(0.75 * len(training_rows)))]

    old_name = "Old · 75% · core"
    refresh_name = "Refresh · 100% · core"
    new_name = "New · + BonusMalus"
    old_model = _fit(frame, old_rows, _core_features())
    refresh_model = _fit(frame, training_rows, _core_features())
    new_features = {**_core_features(), "BonusMalus": Spline(k=6)}
    new_model = _fit(
        frame,
        training_rows,
        new_features,
        interactions=_new_interactions(),
    )

    review = frame.loc[test_rows].reset_index(drop=True)
    models = {
        old_name: old_model,
        refresh_name: refresh_model,
        new_name: new_model,
    }
    prediction_columns = {}
    for index, (name, model) in enumerate(models.items(), start=1):
        column = f"prediction_{index}"
        review[column] = model.predict(review.loc[:, list(model.features)])
        prediction_columns[name] = column

    return build_underwriter_report(
        review,
        actual="ClaimRate",
        predictions=prediction_columns,
        sample_weight="Exposure",
        features=list(new_features),
        superglm_models=models,
        output_path=output_path,
        options=UnderwriterReportOptions(
            title="freMTPL frequency model vintage review",
            problem_type="frequency",
            top_k=12,
            double_lift_bins=10,
            curve_bins=160,
            distribution_bins=220,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "state/report_smoke/model_review.html",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="optional public-data row cap; the default uses every available row",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_demo(args.output.expanduser().resolve(), rows=args.rows, seed=args.seed)
    print(f"Report: {result.output_path}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
