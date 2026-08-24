from __future__ import annotations

import pandas as pd

from scripts.simulate_model_monitoring import run_simulation


def test_synthetic_stream_proves_fixed_paths_and_writes_figures(tmp_path):
    output_dir = tmp_path / "monitoring_simulation"
    result = run_simulation(
        output_dir,
        row_count=1000,
        seed=1729,
        max_reml_iter=2,
    )

    assert result["verified_monitoring_runs"] == 16
    assert result["total_monitoring_runs"] == 16
    assert result["frozen_lambda_max_absolute_delta"] == 0.0
    assert result["frozen_knot_max_absolute_delta"] == 0.0
    assert result["lambda_only_knot_max_absolute_delta"] == 0.0
    assert result["adaptive_knot_max_absolute_delta"] > 0.0

    summary = pd.read_csv(output_dir / "stream_summary.csv")
    assert (
        summary["scoring_model_trained_through_percent"] == summary["available_percent"] - 10
    ).all()
    first_arrival = summary[summary["available_percent"].eq(70)]
    assert first_arrival["new_window_mean_poisson_deviance"].nunique() == 1
    assert set(summary["invariant_status"]) == {"VERIFIED"}

    for filename in (
        "01_feature_drift.png",
        "02_lambda_paths.png",
        "03_knot_paths.png",
        "04_relativity_drift.png",
        "05_out_of_time_performance.png",
        "invariant_evidence.json",
        "simulation_report.md",
    ):
        assert (output_dir / filename).stat().st_size > 0
