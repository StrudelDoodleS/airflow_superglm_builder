from __future__ import annotations

import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from superglm import Categorical, Numeric, Spline, SuperGLM

from pricing_pipeline.modeling.scratch_diagnostics import weighted_quantile_bins
from pricing_pipeline.reporting import (
    ModelLikelihoodSpec,
    UnderwriterReportOptions,
    build_scored_model_report,
    build_underwriter_report,
)
from pricing_pipeline.reporting.adapters.superglm import SuperGLMReportAdapter
from pricing_pipeline.reporting.evidence import EvidenceRequest


class _PrintPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.pages: list[tuple[str, str, str]] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if page := values.get("data-print-page"):
            self.pages.append(
                (
                    page,
                    values.get("data-print-role", ""),
                    values.get("data-print-section", ""),
                )
            )


def _embedded_payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            "age": [20, 30, 25, 45, 35, 55, 65, 70, 50, 40],
            "actual": [0.2, 0.0, 0.3, 0.8, 0.5, 1.1, 1.4, 1.8, 0.9, 0.4],
            "weight": [1.0, 2.0, 0.0, 1.5, 2.5, 1.0, 0.5, 2.0, 1.0, 1.5],
            "model_a": [0.18, 0.22, 0.35, 0.7, 0.55, 1.0, 1.2, 1.6, 0.85, 0.45],
            "model_b": [0.3, 0.3, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 0.75, 0.5],
            "row_secret": [f"never-embed-{index}" for index in range(10)],
        }
    )


def _small_options(**values) -> UnderwriterReportOptions:
    return UnderwriterReportOptions(minimum_cell_size=2, **values)


def test_interaction_points_are_validated():
    assert UnderwriterReportOptions().interaction_points == 80
    with pytest.raises(ValueError, match="interaction_points"):
        UnderwriterReportOptions(interaction_points=1)


def test_rating_workbook_adapter_imports_without_superglm():
    script = """\
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("superglm"):
        raise AssertionError(f"forbidden import: {name}")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pricing_pipeline.reporting.evidence
import pricing_pipeline.reporting._core
import pricing_pipeline.reporting._underwriter_html
import pricing_pipeline.reporting.adapters.rating_workbook
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_prediction_movement_uses_tie_safe_weighted_rank_bins():
    from pricing_pipeline.reporting._underwriter_movement import (
        _tie_safe_weighted_bins,
    )

    bins = _tie_safe_weighted_bins(
        np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.0, 1.0, 2.0, 1.0, 3.0]),
        n_bins=3,
    )

    assert bins.tolist() == [0, 0, 0, 1, 1, 2]


def test_double_lift_privacy_bins_keep_tied_ratios_together_independent_of_row_order():
    from pricing_pipeline.reporting._core import _privacy_safe_bins

    ratios = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    weight = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    comparison_unit_codes = np.arange(len(ratios))
    permutation = np.array([7, 2, 4, 0, 5, 1, 6, 3])

    bins = _privacy_safe_bins(
        np.log(ratios),
        weight,
        comparison_unit_codes,
        n_bins=4,
        minimum_cell_size=2,
    )
    permuted_bins = _privacy_safe_bins(
        np.log(ratios[permutation]),
        weight[permutation],
        comparison_unit_codes[permutation],
        n_bins=4,
        minimum_cell_size=2,
    )
    restored_bins = np.empty_like(permuted_bins)
    restored_bins[permutation] = permuted_bins

    assert bins.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert restored_bins.tolist() == bins.tolist()
    assert all(np.unique(bins[ratios == ratio]).size == 1 for ratio in np.unique(ratios))
    assert all(
        np.unique(comparison_unit_codes[bins == label]).size >= 2 for label in np.unique(bins)
    )


def test_prediction_movement_suppresses_small_cells_and_uses_weighted_aggregates():
    from pricing_pipeline.reporting._underwriter_movement import (
        prediction_movement_payload,
    )

    result = prediction_movement_payload(
        {
            "Old": np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0]),
            "New": np.array([1.0, 1.0, 2.0, 2.0, 4.0, 8.0]),
        },
        np.array([1.0, 2.0, 1.0, 2.0, 1.0, 3.0]),
        np.arange(6),
        n_bins=3,
        minimum_cell_size=2,
    )["Old"]["New"]

    rank = result["rank"]
    assert rank["suppressed_weight_share"] == pytest.approx(0.4)
    assert sum(cell["weight_share"] for cell in rank["cells"]) == pytest.approx(0.6)
    assert all(cell["comparison_units"] >= 2 for cell in rank["cells"])
    first = next(cell for cell in rank["cells"] if (cell["x"], cell["y"]) == (1, 1))
    assert first == pytest.approx(
        {
            "x": 1,
            "y": 1,
            "rows": 2,
            "comparison_units": 2,
            "weight": 3.0,
            "weight_share": 0.3,
            "reference_prediction": 1.0,
            "comparison_prediction": 1.0,
            "prediction_ratio": 1.0,
        }
    )
    assert result["summary"]["weight_share_change_ge_10pct"] == pytest.approx(0.5)
    assert result["summary"]["weight_share_moved_ge_2_bins"] == pytest.approx(0.0)
    level = result["level"]
    assert all(cell["comparison_units"] >= 2 for cell in level["cells"])
    assert level["x_values"] == pytest.approx([1.0, 3.75])
    assert level["y_values"] == pytest.approx([1.0, 7.0])
    assert {(cell["x"], cell["y"]) for cell in level["cells"]} == {(1, 1), (2, 2)}


def test_likelihood_metadata_rejects_boolean_distribution_parameters():
    with pytest.raises(TypeError, match="tweedie_power"):
        ModelLikelihoodSpec(tweedie_power=True, dispersion=1.0)
    with pytest.raises(TypeError, match="dispersion"):
        ModelLikelihoodSpec(tweedie_power=1.5, dispersion=True)


@pytest.mark.parametrize("movement_bins", [True, 1, 2.5])
def test_report_options_reject_invalid_movement_bins(movement_bins):
    with pytest.raises((TypeError, ValueError), match="movement_bins"):
        UnderwriterReportOptions(movement_bins=movement_bins)


def test_unit_deviance_is_stable_for_large_nearly_equal_values():
    from pricing_pipeline.reporting.underwriter import _unit_tweedie_deviance

    actual = np.array([99_999_999.0, 100_000_001.0])
    prediction = np.full(2, 100_000_000.0)

    deviance = _unit_tweedie_deviance(actual, prediction, 1.0)

    assert np.all(deviance >= 0.0)
    assert deviance == pytest.approx([1.0e-8, 1.0e-8], rel=2e-8)


@pytest.mark.parametrize(
    ("predicted", "expected_signed", "expected_agreement"),
    [
        ([1.0, 2.0, 3.0], 1.0, 1.0),
        ([3.0, 2.0, 1.0], -1.0, 0.0),
        ([2.0, 2.0, 2.0], 0.0, 0.0),
    ],
)
def test_weighted_line_agreement_is_bounded(
    predicted: list[float],
    expected_signed: float,
    expected_agreement: float,
):
    from pricing_pipeline.reporting.underwriter import _weighted_line_agreement

    signed, agreement = _weighted_line_agreement(
        np.array([1.0, 2.0, 3.0]),
        np.asarray(predicted),
        np.ones(3),
    )

    assert signed == pytest.approx(expected_signed)
    assert agreement == pytest.approx(expected_agreement)


def test_weighted_line_agreement_handles_equal_constant_lines():
    from pricing_pipeline.reporting.underwriter import _weighted_line_agreement

    signed, agreement = _weighted_line_agreement(
        np.full(3, 2.0),
        np.full(3, 2.0),
        np.array([1.0, 2.0, 3.0]),
    )

    assert signed == pytest.approx(1.0)
    assert agreement == pytest.approx(1.0)


def test_prediction_only_report_is_self_contained_and_aggregate(tmp_path: Path):
    frame = _scored_frame()
    output = tmp_path / "review.html"

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        output_path=output,
        options=_small_options(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=2,
            movement_bins=2,
        ),
    )

    assert result.output_path == output.resolve()
    assert result.rows_used == 9
    assert result.zero_weight_rows_ignored == 1
    assert list(result.metrics["model"]) == ["Model A", "Model B"]
    assert np.isfinite(result.metrics["mean_deviance"]).all()
    html = output.read_text(encoding="utf-8")
    print_pages = _PrintPageParser()
    print_pages.feed(html)
    assert print_pages.pages == [
        ("overview", "summary", "Portfolio overview"),
        ("importance-figure", "figure", "Top features"),
        ("importance-evidence", "evidence", "Top-feature evidence"),
        ("relativities-figure", "figure", "Relativities"),
        ("relativities-evidence", "evidence", "Relativity evidence"),
        ("interactions-figure", "figure", "Interactions"),
        ("interactions-evidence", "evidence", "Interaction evidence"),
        ("distribution-figure", "figure", "Prediction distributions"),
        ("distribution-evidence", "evidence", "Distribution evidence"),
        ("movement-figure", "figure", "Prediction movement"),
        ("movement-evidence", "evidence", "Movement evidence"),
        ("curves-figure", "figure", "Lorenz and gains"),
        ("curves-evidence", "evidence", "Discrimination evidence"),
        ("double-lift-figure", "figure", "Double lift"),
        ("double-lift-evidence", "evidence", "Double-lift evidence"),
    ]
    assert "Top main effects" in html
    assert "Lorenz and cumulative gains" in html
    assert "Ratio numerator" in html
    assert "weighted Gaussian KDE" in html
    assert "Deselect all" in html
    assert 'id="distribution-model-picker"' in html
    assert 'id="distribution-model-options"' in html
    assert 'id="distribution-legend"' in html
    assert 'id="distribution-model"' not in html
    assert 'id="distribution-view"' in html
    assert 'data-view="movement"' in html
    assert 'id="movement-reference"' in html
    assert 'id="movement-comparison"' in html
    assert 'id="movement-view"' in html
    assert 'id="movement-chart"' in html
    assert 'id="movement-inspector"' in html
    assert 'id="movement-hover"' in html
    assert 'data-panel="interactions"' in html
    assert 'id="interactions"' in html
    assert 'id="interaction-model"' in html
    assert 'id="interaction-term"' in html
    assert 'id="interaction-view"' in html
    assert 'id="interaction-ci"' in html
    assert 'id="interaction-level-options"' in html
    assert 'id="interaction-chart"' in html
    assert 'id="interaction-print-chart"' in html
    assert 'id="interaction-inspector"' in html
    assert "function renderInteractions()" in html
    assert "const INTERACTION_THERMAL_SCALE = [" in html
    assert "function interactionSurfaceTraces" in html
    assert "function interactionHeatmapTraces" in html
    assert "function interactionCurveTraces" in html
    assert "function interactionBarTraces" in html
    assert "function renderInteractionInspector" in html
    assert 'frequency: "Poisson"' in html
    assert 'severity: "Gamma"' in html
    assert 'burn_cost: "Tweedie"' in html
    assert "function renderInteractionPrintChart" in html
    assert ".interaction-print-shell { display: none; }" in html
    assert "#interaction-chart { display: none !important; }" in html
    assert ".interaction-print-shell { display: block !important; }" in html
    print_lifecycle = html.split("function preparePrintPages()", 1)[1].split(
        "function restoreScreenPlots()", 1
    )[0]
    assert "interactionState.viewByTerm" not in print_lifecycle
    assert "%{x:.4f}" in html
    assert "%{z:.4f}" in html
    assert "function bindMovementTooltip" in html
    assert 'hoverinfo: "none"' in html
    assert 'id="curve-legend"' in html
    assert 'id="lift-legend"' in html
    assert 'id="lift-horizontal-scroll"' not in html
    assert "lift-scroll-control" not in html
    assert "syncLiftScrollControl" not in html
    assert 'type="range"' not in html
    assert 'tabindex="0" aria-label="Scrollable double-lift evidence table"' in html
    assert "overflow-x: scroll" in html
    assert "#lift-table.lift-table-scroll::-webkit-scrollbar-thumb" in html
    assert "zorder: 1" in html
    assert "zorder: 2" in html
    assert "zorder: 3" in html
    assert "const MOVEMENT_THERMAL_SCALE = [" in html
    assert '"#2b0a3d"' in html
    assert '"#fffdf5"' in html
    assert 'name: "unchanged rank"' not in html
    assert 'name: "equal prediction"' not in html
    assert 'dash: index === 0 ? null : "7 5"' not in html
    assert "%{y:.4f}" in html
    assert "fixedNumber(custom[2], 3)" in html
    assert "fixedPercent(custom[3], 3)" in html
    assert "break-inside: avoid-page" in html
    assert "page-break-inside: avoid" in html
    assert "break-after: page" in html
    assert "page-break-after: always" in html
    assert 'id="print-page-furniture-template"' in html
    assert 'class="print-page-header print-page-furniture"' in html
    assert 'class="print-page-footer print-page-furniture"' in html
    assert "function preparePrintPages()" in html
    assert 'document.querySelectorAll("[data-print-page]")' in html
    assert 'page.closest("[hidden]") === null' in html
    assert '[data-print-role="figure"]' in html
    assert "display: contents !important" in html
    assert "thead { display: table-header-group; }" in html
    assert "function preparePrintPlots" in html
    assert 'window.addEventListener("beforeprint", preparePrintPlots)' in html
    assert "const PRINT_CONTENT_WIDTH = 279 * 96 / 25.4" in html
    assert "const PRINT_CHART_HEIGHT = 500" in html
    assert "const PRINT_COMPARISON_HEIGHT = 462" in html
    assert "PRINT_GRID_GAP" not in html
    assert "Plotly.relayout(plot, {width: PRINT_CONTENT_WIDTH, height})" in html
    assert ".review-chart .svg-container" in html
    assert "#movement-content[hidden]" in html
    assert '{label: "Paired 95% score Δ"' not in html
    assert "Paired 95% score difference (denominator − numerator)" in html
    assert "No models selected" in html
    assert "superglm.editor.app" not in html
    assert "--yellow: #f4d35e" in html
    assert 'class="app-shell report-shell"' in html
    assert 'class="app-tab active"' in html
    assert 'class="inspector review-inspector"' in html
    assert "--yellow: #f4d35e" in html
    assert "never-embed" not in html
    assert re.search(r"<script[^>]+src=", html, flags=re.IGNORECASE) is None
    assert re.search(r"<link[^>]+href=", html, flags=re.IGNORECASE) is None
    assert "plotly.js v" in html.lower()
    assert "Plotly.react" in html

    payload = _embedded_payload(output)
    assert payload["models"] == ["Model A", "Model B"]
    assert payload["metadata"]["rows_used"] == 9
    assert payload["importance"] == {}
    distribution = payload["distributions"]["Model A"]
    assert len(distribution["x"]) >= 64
    assert np.trapezoid(distribution["density"], distribution["x"]) == pytest.approx(
        1.0,
        rel=2e-3,
    )
    assert distribution["bandwidth"] > 0
    assert set(payload["curves"]) == {"models", "benchmark"}
    movement = payload["movement"]["Model A"]["Model B"]
    assert movement["rank"]["cells"]
    assert movement["level"]["cells"]
    assert movement["summary"]["weight_share_change_ge_10pct"] >= 0.0
    assert payload["curves"]["benchmark"]["lorenz"]["x"][0] == 0.0
    assert payload["curves"]["benchmark"]["lorenz"]["x"][-1] == 1.0
    assert len(payload["double_lift"]["Model A"]["Model B"]["bins"]) >= 2
    lift_rows = payload["double_lift"]["Model A"]["Model B"]["bins"]
    positive = frame["weight"].gt(0).to_numpy()
    weight = frame.loc[positive, "weight"].to_numpy()
    model_a = frame.loc[positive, "model_a"].to_numpy()
    model_b = frame.loc[positive, "model_b"].to_numpy()
    actual = frame.loc[positive, "actual"].to_numpy()
    bins = weighted_quantile_bins(np.log(model_a / model_b), weight, n_bins=2)
    for row in lift_rows:
        assert set(row["predictions"]) == {"Model A", "Model B"}
        assert row["actual"] >= 0
        mask = bins == row["bin"] - 1
        assert row["rows"] == int(mask.sum())
        assert row["weight"] == pytest.approx(weight[mask].sum())
        assert row["actual"] == pytest.approx(
            np.sum(weight[mask] * actual[mask]) / weight[mask].sum()
        )
        assert row["predictions"]["Model A"] == pytest.approx(
            np.sum(weight[mask] * model_a[mask]) / weight[mask].sum()
        )
        assert row["aggregate_prediction_ratio"] == pytest.approx(
            np.sum(weight[mask] * model_a[mask]) / np.sum(weight[mask] * model_b[mask])
        )


def test_external_legend_preserves_each_reference_dash_pattern(tmp_path: Path):
    frame = _scored_frame()
    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        output_path=tmp_path / "legend.html",
        options=_small_options(problem_type="burn_cost", double_lift_bins=2),
    )

    html = result.output_path.read_text(encoding="utf-8")
    assert 'const swatch = svgElement("svg"' in html
    assert '"stroke-dasharray": item.dash || ""' in html


def test_double_lift_coarsens_cells_to_the_configured_minimum_size(tmp_path: Path):
    frame = _scored_frame()

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        output_path=tmp_path / "privacy-coarsened.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=10,
            minimum_cell_size=3,
        ),
    )

    payload = _embedded_payload(result.output_path)
    bins = payload["double_lift"]["Model A"]["Model B"]["bins"]
    html = result.output_path.read_text(encoding="utf-8")
    assert "distinct comparison units" in html
    assert "DATA.metadata.minimum_cell_size" in html
    assert payload["metadata"]["minimum_cell_size"] == 3
    assert sum(row["rows"] for row in bins) == result.rows_used
    assert all(row["comparison_units"] >= 3 for row in bins)
    assert len(bins) <= result.rows_used // 3


def test_report_rejects_too_few_comparison_units_for_aggregate_output(tmp_path: Path):
    frame = _scored_frame()

    with pytest.raises(ValueError, match="at least 10 distinct comparison units"):
        build_underwriter_report(
            frame,
            actual="actual",
            predictions={"Model A": "model_a", "Model B": "model_b"},
            sample_weight="weight",
            features=["segment", "age"],
            output_path=tmp_path / "too-small.html",
            options=UnderwriterReportOptions(
                problem_type="burn_cost",
                minimum_cell_size=10,
            ),
        )


def test_comparison_unit_identifier_cannot_also_be_a_report_feature(tmp_path: Path):
    frame = _scored_frame()
    frame["policy_id"] = [f"private-policy-{index}" for index in range(len(frame))]
    output = tmp_path / "identifier-feature.html"

    with pytest.raises(ValueError, match="comparison_unit.*report feature"):
        build_underwriter_report(
            frame,
            actual="actual",
            predictions={"Model A": "model_a", "Model B": "model_b"},
            sample_weight="weight",
            features=["segment", "policy_id"],
            comparison_unit="policy_id",
            output_path=output,
            options=_small_options(problem_type="burn_cost"),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("problem_type", "power", "expected"),
    [
        (
            "frequency",
            1.0,
            {
                "response": "Claim frequency",
                "prediction": "Predicted claim frequency",
                "volume": "Exposure",
                "curve_x": "Cumulative exposure share",
                "curve_y": "Cumulative claim-count share",
            },
        ),
        (
            "severity",
            2.0,
            {
                "response": "Claim severity",
                "prediction": "Predicted claim severity",
                "volume": "Claim count",
                "curve_x": "Cumulative claim-count share",
                "curve_y": "Cumulative claim-cost share",
            },
        ),
        (
            "burn_cost",
            1.5,
            {
                "response": "Burn cost",
                "prediction": "Predicted burn cost",
                "volume": "Exposure",
                "curve_x": "Cumulative exposure share",
                "curve_y": "Cumulative claim-cost share",
            },
        ),
    ],
)
def test_problem_type_drives_report_semantics(
    tmp_path: Path,
    problem_type: str,
    power: float,
    expected: dict[str, str],
):
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "actual": [0.5, 1.0, 1.5],
            "weight": [2.0, 3.0, 4.0],
            "prediction": [0.6, 0.9, 1.4],
        }
    )

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model": "prediction"},
        sample_weight="weight",
        features=["feature"],
        output_path=tmp_path / f"{problem_type}.html",
        options=_small_options(
            problem_type=problem_type,  # type: ignore[arg-type]
            tweedie_power=power,
        ),
    )

    assert _embedded_payload(result.output_path)["metadata"]["semantics"] == expected


def test_training_likelihood_metadata_enables_exact_tweedie_nll(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "actual": [0.0, 0.2, 1.1, 3.0, 8.5],
            "weight": [0.35, 1.0, 2.4, 0.8, 3.2],
            "with_metadata": [0.1, 0.4, 0.9, 2.2, 7.0],
            "prediction_only": [0.3, 0.25, 1.8, 3.7, 5.5],
        }
    )

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={
            "With metadata": "with_metadata",
            "Prediction only": "prediction_only",
        },
        sample_weight="weight",
        features=["feature"],
        model_likelihoods={
            "With metadata": {"tweedie_power": 1.5, "dispersion": 0.7},
        },
        output_path=tmp_path / "exact-nll.html",
        options=_small_options(problem_type="burn_cost", tweedie_power=1.5),
    )

    rows = {row["model"]: row for row in _embedded_payload(result.output_path)["metrics"]}
    exact = rows["With metadata"]
    assert exact["exact_mean_nll"] == pytest.approx(0.9035311063222131)
    assert exact["likelihood_power"] == 1.5
    assert exact["likelihood_dispersion"] == 0.7
    assert exact["likelihood_source"] == "supplied training metadata"
    assert rows["Prediction only"]["exact_mean_nll"] is None
    assert rows["Prediction only"]["likelihood_source"] is None


def test_facade_matches_likelihood_metadata_to_normalized_prediction_names(tmp_path: Path):
    frame = _scored_frame()

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={" Model A ": "model_a"},
        sample_weight="weight",
        features=["segment", "age"],
        model_likelihoods={"Model A": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.7)},
        output_path=tmp_path / "normalized-name.html",
        options=_small_options(problem_type="burn_cost", tweedie_power=1.5),
    )

    assert result.metrics.loc[0, "model"] == "Model A"
    assert result.metrics.loc[0, "likelihood_source"] == "supplied training metadata"


def test_facade_normalizes_direct_likelihood_metadata_names(tmp_path: Path):
    result = build_underwriter_report(
        _scored_frame(),
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["segment", "age"],
        model_likelihoods={" Model A ": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.7)},
        output_path=tmp_path / "normalized-likelihood-name.html",
        options=_small_options(problem_type="burn_cost", tweedie_power=1.5),
    )

    assert result.metrics.loc[0, "model"] == "Model A"
    assert result.metrics.loc[0, "likelihood_source"] == "supplied training metadata"


@pytest.mark.parametrize(
    "section",
    ["predictions", "superglm_models", "rating_workbooks", "model_likelihoods"],
)
def test_facade_rejects_duplicate_normalized_model_names(tmp_path: Path, section: str):
    predictions = {"Model A": "model_a"}
    optional: dict[str, object] = {}
    if section == "predictions":
        predictions[" Model A "] = "model_b"
    elif section == "superglm_models":
        optional[section] = {"Model A": object(), " Model A ": object()}
    elif section == "rating_workbooks":
        optional[section] = {
            "Model A": tmp_path / "first.xlsx",
            " Model A ": tmp_path / "second.xlsx",
        }
    else:
        optional[section] = {
            "Model A": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.7),
            " Model A ": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.8),
        }

    with pytest.raises(
        ValueError,
        match=rf"{section} contains duplicate normalized model name: 'Model A'",
    ):
        build_underwriter_report(
            _scored_frame(),
            actual="actual",
            predictions=predictions,
            sample_weight="weight",
            features=["segment", "age"],
            output_path=tmp_path / f"duplicate-{section}.html",
            options=_small_options(problem_type="burn_cost", tweedie_power=1.5),
            **optional,
        )


def test_double_lift_quantifies_pairwise_advantage_and_line_agreement(tmp_path: Path):
    actual = np.array([0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 3.7, 5.0])
    weight = np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.7, 1.8, 1.0])
    portfolio_mean = float(np.average(actual, weights=weight))
    frame = pd.DataFrame(
        {
            "feature": np.arange(len(actual)),
            "actual": actual,
            "weight": weight,
            "perfect": actual,
            "null": portfolio_mean,
        }
    )

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Perfect": "perfect", "Null": "null"},
        sample_weight="weight",
        features=["feature"],
        output_path=tmp_path / "comparison.html",
        options=_small_options(
            problem_type="severity",
            double_lift_bins=4,
        ),
    )

    payload = _embedded_payload(result.output_path)
    entry = payload["double_lift"]["Perfect"]["Null"]
    comparison = entry["comparison"]
    metric_deviance = {row["model"]: row["mean_deviance"] for row in payload["metrics"]}
    assert comparison["deviance_advantage"] == pytest.approx(
        metric_deviance["Null"] - metric_deviance["Perfect"]
    )
    assert comparison["deviance_advantage"] > 0
    assert sum(row["deviance_advantage_contribution"] for row in entry["bins"]) == pytest.approx(
        comparison["deviance_advantage"]
    )
    calibration = comparison["binned_calibration"]
    assert calibration["Perfect"]["d_squared"] == pytest.approx(1.0)
    assert calibration["Null"]["d_squared"] == pytest.approx(0.0)
    assert calibration["Perfect"]["signed_concordance"] == pytest.approx(1.0)
    assert calibration["Perfect"]["line_agreement"] == pytest.approx(1.0)
    assert calibration["Null"]["signed_concordance"] == pytest.approx(0.0)
    assert calibration["Null"]["line_agreement"] == pytest.approx(0.0)
    assert comparison["lower_score_model"] == "Perfect"
    assert comparison["higher_score_model"] == "Null"
    assert comparison["relative_score_reduction"] == pytest.approx(1.0)


def test_double_lift_uses_exact_nll_when_both_models_have_training_metadata(
    tmp_path: Path,
):
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "actual": [0.0, 0.2, 1.1, 3.0, 8.5],
            "weight": [0.35, 1.0, 2.4, 0.8, 3.2],
            "model_a": [0.1, 0.4, 0.9, 2.2, 7.0],
            "model_b": [0.3, 0.25, 1.8, 3.7, 5.5],
        }
    )

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["feature"],
        model_likelihoods={
            "Model A": {"tweedie_power": 1.5, "dispersion": 0.7},
            "Model B": {"tweedie_power": 1.5, "dispersion": 0.9},
        },
        output_path=tmp_path / "exact-comparison.html",
        options=_small_options(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=3,
        ),
    )

    entry = _embedded_payload(result.output_path)["double_lift"]["Model A"]["Model B"]
    comparison = entry["comparison"]
    assert comparison["primary_score"] == "exact_nll"
    assert comparison["mean_exact_nll"] == pytest.approx(
        {"Model A": 0.9035311063222131, "Model B": 1.2067311023942289}
    )
    assert comparison["exact_nll_advantage"] == pytest.approx(0.3031999960720156)
    assert comparison["lower_score_model"] == "Model A"
    assert comparison["higher_score_model"] == "Model B"
    assert comparison["relative_score_reduction"] == pytest.approx(
        0.3031999960720156 / 1.2067311023942289
    )
    assert sum(row["exact_nll_advantage_contribution"] for row in entry["bins"]) == pytest.approx(
        comparison["exact_nll_advantage"]
    )


def test_exact_tweedie_density_is_evaluated_once_per_model(tmp_path: Path, monkeypatch):
    import pricing_pipeline.reporting.adapters.superglm as superglm_adapter

    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "actual": [0.0, 0.2, 1.1, 3.0, 8.5],
            "weight": [0.35, 1.0, 2.4, 0.8, 3.2],
            "model_a": [0.1, 0.4, 0.9, 2.2, 7.0],
            "model_b": [0.3, 0.25, 1.8, 3.7, 5.5],
        }
    )
    original = superglm_adapter.tweedie_logpdf
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(superglm_adapter, "tweedie_logpdf", counted)
    build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["feature"],
        model_likelihoods={
            "Model A": {"tweedie_power": 1.5, "dispersion": 0.7},
            "Model B": {"tweedie_power": 1.5, "dispersion": 0.9},
        },
        output_path=tmp_path / "one-density-call.html",
        options=_small_options(
            problem_type="burn_cost",
            comparison_bootstrap_replicates=0,
        ),
    )

    assert calls == 2


def test_bootstrap_draws_are_shared_across_all_model_pairs(tmp_path: Path, monkeypatch):
    frame = _scored_frame()
    frame["model_c"] = np.sqrt(frame["model_a"] * frame["model_b"])
    original = np.random.default_rng
    calls = 0

    def counted(seed):
        nonlocal calls
        calls += 1
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", counted)
    build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b", "Model C": "model_c"},
        sample_weight="weight",
        features=["segment", "age"],
        output_path=tmp_path / "shared-bootstrap.html",
        options=_small_options(
            problem_type="burn_cost",
            double_lift_bins=2,
            comparison_bootstrap_replicates=100,
        ),
    )

    assert calls == 1


def test_double_lift_paired_cluster_interval_is_reproducible_and_aggregate_only(
    tmp_path: Path,
):
    actual = np.array([0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 3.7, 5.0])
    weight = np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.7, 1.8, 1.0])
    portfolio_mean = float(np.average(actual, weights=weight))
    frame = pd.DataFrame(
        {
            "feature": np.arange(len(actual)),
            "actual": actual,
            "weight": weight,
            "perfect": actual,
            "null": portfolio_mean,
            "policy": [f"private-policy-{index // 2}" for index in range(len(actual))],
        }
    )
    options = _small_options(
        problem_type="severity",
        double_lift_bins=4,
        comparison_bootstrap_replicates=200,
        comparison_bootstrap_seed=43,
    )

    comparisons = []
    for index in range(2):
        result = build_underwriter_report(
            frame,
            actual="actual",
            predictions={"Perfect": "perfect", "Null": "null"},
            sample_weight="weight",
            features=["feature"],
            comparison_unit="policy",
            output_path=tmp_path / f"cluster-{index}.html",
            options=options,
        )
        html = result.output_path.read_text(encoding="utf-8")
        assert "private-policy" not in html
        comparisons.append(
            _embedded_payload(result.output_path)["double_lift"]["Perfect"]["Null"]["comparison"]
        )

    comparison = comparisons[0]
    assert comparison["decision"] == "Perfect favoured"
    assert comparison["interval_lower"] > 0
    assert comparison["interval_upper"] >= comparison["interval_lower"]
    assert comparison["bootstrap_replicates"] == 200
    assert comparison["comparison_units"] == 4
    assert comparisons[0] == comparisons[1]


def test_superglm_object_takes_priority_and_supplies_native_evidence(tmp_path: Path):
    rng = np.random.default_rng(842)
    row_count = 180
    frame = pd.DataFrame(
        {
            "segment": np.resize(["A", "B", "C"], row_count),
            "age": rng.uniform(18, 80, row_count),
            "weight": rng.uniform(0.5, 2.0, row_count),
        }
    )
    eta = (
        -1.2
        + 0.3 * (frame["segment"] == "B")
        + 0.55 * (frame["segment"] == "C")
        + 0.012 * (frame["age"] - 45)
    )
    frame["actual"] = rng.poisson(np.exp(eta))
    model = SuperGLM(
        features={"segment": Categorical(base="first"), "age": Spline(k=6)},
        selection_penalty=0.0,
    ).fit(
        frame[["segment", "age"]],
        frame["actual"],
        sample_weight=frame["weight"],
    )
    frame["prediction"] = model.predict(frame[["segment", "age"]])

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Fitted GAM": "prediction"},
        sample_weight="weight",
        features=["segment", "age"],
        superglm_models={"Fitted GAM": model},
        rating_workbooks={"Fitted GAM": tmp_path / "must-not-be-read.xlsx"},
        model_likelihoods={"Fitted GAM": ModelLikelihoodSpec(tweedie_power=1.0, dispersion=1.0)},
        output_path=tmp_path / "superglm.html",
        options=_small_options(problem_type="frequency", top_k=2),
    )

    importance = result.importance["Fitted GAM"]
    assert set(importance["feature"]) == {"segment", "age"}
    assert importance["share"].sum() == pytest.approx(1.0)
    assert set(importance["method"]) == {"native_link_variance"}
    payload = _embedded_payload(result.output_path)
    metric = payload["metrics"][0]
    assert metric["exact_mean_nll"] is not None
    assert metric["likelihood_power"] == 1.0
    assert metric["likelihood_dispersion"] == 1.0
    assert metric["likelihood_source"] == "fitted SuperGLM object"
    rendered_importance = {row["feature"]: row for row in payload["importance"]["Fitted GAM"]}
    neutral_importance = importance.set_index("feature")
    for feature, row in rendered_importance.items():
        assert row["magnitude"] == pytest.approx(neutral_importance.loc[feature, "magnitude"])
        assert row["effective_df"] == pytest.approx(neutral_importance.loc[feature, "effective_df"])
    assert payload["relativities"]["segment"]["Fitted GAM"]["labels"] == ["A", "B", "C"]
    assert payload["relativities"]["age"]["Fitted GAM"]["source"] == "SuperGLM object"
    assert len(payload["relativities"]["age"]["Fitted GAM"]["relativity"]) == 200


def test_superglm_facade_binds_different_holdout_rows_with_report_time_offset(
    tmp_path: Path,
):
    train_rows = 72
    train = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, train_rows),
            "actual": np.resize([0.0, 1.0, 0.0, 2.0, 1.0, 3.0], train_rows),
            "weight": np.linspace(0.4, 2.1, train_rows),
            "fit_offset": np.log(np.linspace(0.7, 1.5, train_rows)),
        }
    )
    model = SuperGLM(
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(
        train[["x"]],
        train["actual"],
        sample_weight=train["weight"],
        offset=train["fit_offset"].to_numpy(),
    )

    holdout_rows = 19
    holdout = pd.DataFrame(
        {
            "x": np.linspace(-0.85, 1.15, holdout_rows),
            "actual": np.resize([0.0, 1.0, 2.0, 0.0, 1.0], holdout_rows),
            "weight": np.linspace(0.3, 1.9, holdout_rows),
            "report_offset": np.log(np.linspace(1.6, 0.8, holdout_rows)),
        }
    )
    holdout.loc[5, "weight"] = 0.0
    holdout["prediction"] = model.predict(
        holdout[["x"]],
        offset=holdout["report_offset"].to_numpy(),
    )

    result = build_underwriter_report(
        holdout,
        actual="actual",
        predictions={"Holdout GAM": "prediction"},
        sample_weight="weight",
        features=["x"],
        offset="report_offset",
        superglm_models={"Holdout GAM": model},
        output_path=tmp_path / "holdout-offset.html",
        options=_small_options(
            problem_type="frequency",
            comparison_bootstrap_replicates=0,
        ),
    )

    assert result.rows_used == holdout_rows - 1
    metric = result.metrics.iloc[0]
    assert metric["likelihood_source"] == "fitted SuperGLM object"
    assert np.isfinite(metric["exact_mean_nll"])


def test_superglm_facade_matches_model_neutral_builder(tmp_path: Path):
    rng = np.random.default_rng(117)
    row_count = 120
    frame = pd.DataFrame(
        {
            "segment": np.resize(["A", "B", "C"], row_count),
            "age": rng.uniform(18, 80, row_count),
            "weight": rng.uniform(0.5, 2.0, row_count),
        }
    )
    eta = (
        -1.2
        + 0.3 * (frame["segment"] == "B")
        + 0.55 * (frame["segment"] == "C")
        + 0.012 * (frame["age"] - 45)
    )
    frame["actual"] = rng.poisson(np.exp(eta))
    model = SuperGLM(
        features={"segment": Categorical(base="first"), "age": Spline(k=6)},
        selection_penalty=0.0,
    ).fit(
        frame[["segment", "age"]],
        frame["actual"],
        sample_weight=frame["weight"],
    )
    frame["prediction"] = model.predict(frame[["segment", "age"]])
    options = UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2)

    legacy = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"GAM": "prediction"},
        sample_weight="weight",
        features=["segment", "age"],
        superglm_models={"GAM": model},
        output_path=tmp_path / "legacy.html",
        options=options,
    )
    neutral = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"GAM": "prediction"},
        sample_weight="weight",
        features=["segment", "age"],
        evidence_requests=(EvidenceRequest("GAM", SuperGLMReportAdapter(), model),),
        output_path=tmp_path / "neutral.html",
        options=options,
    )

    pd.testing.assert_frame_equal(legacy.metrics, neutral.metrics)
    legacy_payload = _embedded_payload(legacy.output_path)
    neutral_payload = _embedded_payload(neutral.output_path)
    legacy_payload["metadata"].pop("generated_utc")
    neutral_payload["metadata"].pop("generated_utc")
    assert legacy_payload == neutral_payload


def test_continuous_relativity_exposure_matches_superglm_editor_payload(tmp_path: Path):
    from superglm.editor import EditorSession
    from superglm.editor.payloads import session_payload

    rng = np.random.default_rng(114)
    row_count = 120
    frame = pd.DataFrame(
        {
            "age": np.linspace(18.0, 82.0, row_count),
            "weight": rng.uniform(0.25, 2.5, row_count),
        }
    )
    frame["actual"] = rng.poisson(np.exp(-1.2 + 0.018 * (frame["age"] - 45.0)))
    model = SuperGLM(
        features={"age": Spline(k=6)},
        selection_penalty=0.0,
    ).fit(
        frame[["age"]],
        frame["actual"],
        sample_weight=frame["weight"],
    )
    frame["prediction"] = model.predict(frame[["age"]])

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Fitted GAM": "prediction"},
        sample_weight="weight",
        features=["age"],
        superglm_models={"Fitted GAM": model},
        output_path=tmp_path / "editor-exposure.html",
        options=UnderwriterReportOptions(problem_type="frequency"),
    )

    editor_session = EditorSession.from_model(
        model,
        n_points=200,
        centering="native",
        with_se=True,
        train_data=(frame, frame["actual"].to_numpy(), frame["weight"].to_numpy()),
    )
    expected = session_payload(editor_session)["age"]["exposure"]
    actual = _embedded_payload(result.output_path)["relativities"]["age"]["Fitted GAM"]
    assert actual["exposure"]["kind"] == expected["kind"]
    assert actual["exposure"]["x"] == expected["x"]
    assert actual["exposure"]["y"] == expected["y"]
    assert actual["exposure"]["kind"] == "density"
    assert actual["weight"] == expected["y"]


def test_supplied_likelihood_metadata_cannot_override_fitted_model(tmp_path: Path):
    frame = _scored_frame().loc[lambda value: value["weight"] > 0].reset_index(drop=True)
    frame["count"] = np.arange(len(frame)) % 3
    model = SuperGLM(
        features={"age": Spline(k=5)},
        selection_penalty=0.0,
    ).fit(
        frame[["age"]],
        frame["count"],
        sample_weight=frame["weight"],
    )

    with pytest.raises(ValueError, match="does not match the fitted SuperGLM object"):
        build_underwriter_report(
            frame,
            actual="actual",
            predictions={"Poisson model": "model_a"},
            sample_weight="weight",
            features=["age"],
            superglm_models={"Poisson model": model},
            model_likelihoods={
                "Poisson model": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.7)
            },
            output_path=tmp_path / "mismatched-family.html",
            options=UnderwriterReportOptions(
                problem_type="burn_cost",
                minimum_cell_size=2,
            ),
        )


def test_rating_workbook_supplies_relativity_and_labelled_proxy(tmp_path: Path):
    raw = pd.DataFrame([[None] * 3 for _ in range(11)])
    raw.iat[4, 0] = "segment"
    raw.iloc[6, 0:3] = ["segment", "Relativity", "Weight"]
    raw.iloc[7, 0:3] = ["A", 0.8, 10.0]
    raw.iloc[8, 0:3] = ["B", 1.0, 20.0]
    raw.iloc[9, 0:3] = ["C", 1.25, 30.0]
    workbook = tmp_path / "rating_tables.xlsx"
    raw.to_excel(workbook, sheet_name="Rating Tables", header=False, index=False)
    frame = _scored_frame()

    result = build_underwriter_report(
        frame,
        actual="actual",
        predictions={"Published GAM": "model_a"},
        sample_weight="weight",
        features=["segment", "age"],
        rating_workbooks={"Published GAM": workbook},
        output_path=tmp_path / "workbook.html",
        options=_small_options(),
    )

    importance = result.importance["Published GAM"]
    assert list(importance["feature"]) == ["segment"]
    assert importance.iloc[0]["method"] == "export_log_relativity_variance"
    payload = _embedded_payload(result.output_path)
    series = payload["relativities"]["segment"]["Published GAM"]
    assert series["labels"] == ["A", "B", "C"]
    assert series["relativity"] == [0.8, 1.0, 1.25]
    assert series["weight"] == [5.0, 3.5, 4.5]


def test_report_validates_problem_specific_response_domain(tmp_path: Path):
    frame = _scored_frame()
    with pytest.raises(ValueError, match="severity actuals must be strictly positive"):
        build_underwriter_report(
            frame,
            actual="actual",
            predictions={"Model A": "model_a"},
            sample_weight="weight",
            features=["segment"],
            output_path=tmp_path / "severity.html",
            options=_small_options(problem_type="severity"),
        )
