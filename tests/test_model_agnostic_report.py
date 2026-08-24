from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.modeling.scratch_diagnostics import weighted_quantile_bins
from pricing_pipeline.reporting import UnderwriterReportOptions, build_scored_model_report
from pricing_pipeline.reporting.evidence import (
    EvidenceRequest,
    ExactLossEvidence,
    FeatureImportanceEvidence,
    InteractionEvidence,
    MainEffectEvidence,
    ModelEvidence,
    SuppressionMetadata,
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


def test_direct_interaction_evidence_reaches_neutral_payload(tmp_path: Path):
    frame = _scored_frame().assign(density=np.linspace(1.0, 2.0, 10))
    effect = pd.DataFrame(
        {
            "x": [20.0, 40.0, 20.0, 40.0],
            "y": [1.0, 1.0, 2.0, 2.0],
            "value": [0.8, 1.0, 1.1, 1.3],
        }
    )
    evidence = {
        "Portable GBM": ModelEvidence(
            source="portable diagnostics",
            interactions={
                "age:density": InteractionEvidence(
                    name="age:density",
                    parents=("age", "density"),
                    semantic="partial_dependence",
                    plot_kind="surface",
                    effect=effect,
                    source="portable PDP",
                    grid_axes={
                        "x": np.array([20.0, 40.0]),
                        "y": np.array([1.0, 2.0]),
                    },
                )
            },
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Portable GBM": "model_a"},
        sample_weight="weight",
        features=["age", "density"],
        evidence=evidence,
        output_path=tmp_path / "interaction.html",
        options=UnderwriterReportOptions(problem_type="burn_cost", minimum_cell_size=2),
    )

    payload = _embedded_payload(result.output_path)
    assert payload["interactions"]["models"]["Portable GBM"]["age:density"] == {
        "name": "age:density",
        "parents": ["age", "density"],
        "semantic": "partial_dependence",
        "plot_kind": "surface",
        "source": "portable PDP",
        "effect": effect.to_dict("records"),
        "grid_axes": {"x": [20.0, 40.0], "y": [1.0, 2.0]},
        "density": None,
        "support": None,
        "default_levels": [],
        "level_diagnostics": None,
        "facts": [],
        "warnings": [],
    }
    serialized = json.dumps(payload["interactions"])
    assert "InteractionEvidence" not in serialized
    assert "SuperGLMReportAdapter" not in serialized
    assert "never-embed" not in serialized


def _headless_chromium() -> Path | None:
    for command in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "chrome-headless-shell",
    ):
        if executable := shutil.which(command):
            return Path(executable)
    cached = sorted(
        Path.home().glob(
            ".cache/puppeteer/chrome-headless-shell/*/chrome-headless-shell-linux64/"
            "chrome-headless-shell"
        ),
        reverse=True,
    )
    return cached[0] if cached else None


def test_prediction_only_core_never_imports_superglm_or_joblib(tmp_path):
    script = f"""\
import builtins
import pandas as pd
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "joblib" or name.startswith("superglm"):
        raise AssertionError(f"forbidden import: {{name}}")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from pricing_pipeline.reporting import UnderwriterReportOptions, build_scored_model_report
frame = pd.DataFrame({{
    "actual": [0.0, 1.0, 0.0, 2.0],
    "weight": [1.0, 1.0, 1.0, 1.0],
    "feature": ["A", "A", "B", "B"],
    "prediction": [0.2, 0.8, 0.3, 1.4],
}})
build_scored_model_report(
    frame,
    actual="actual",
    predictions={{"Model": "prediction"}},
    sample_weight="weight",
    features=["feature"],
    output_path={str(tmp_path / "agnostic.html")!r},
    options=UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2),
)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_html_owns_its_style_without_superglm_resources(tmp_path: Path):
    result = build_scored_model_report(
        _scored_frame(),
        actual="actual",
        predictions={"Model": "model_a"},
        sample_weight="weight",
        features=["segment"],
        output_path=tmp_path / "local-style.html",
        options=UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2),
    )

    html = result.output_path.read_text(encoding="utf-8")
    source = Path("pricing_pipeline/reporting/_underwriter_html.py").read_text(encoding="utf-8")

    assert "superglm.editor.app" not in html
    assert "--yellow: #f4d35e" in html
    assert 'class="app-shell report-shell"' in html
    assert 'class="inspector review-inspector"' in html
    assert "from importlib.resources import files" not in source
    assert "from importlib.metadata import distribution" not in source
    assert "def _superglm_editor_css" not in source


def test_report_css_has_one_owner_for_shared_components(tmp_path: Path):
    from pricing_pipeline.reporting import _underwriter_html
    from pricing_pipeline.reporting._underwriter_styles import REPORT_BASE_CSS

    result = build_scored_model_report(
        _scored_frame(),
        actual="actual",
        predictions={"Model": "model_a"},
        sample_weight="weight",
        features=["segment"],
        output_path=tmp_path / "component-style.html",
        options=UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2),
    )
    html = result.output_path.read_text(encoding="utf-8")
    document_css = _underwriter_html._DOCUMENT.split("</style>", 1)[1].split("</style>", 1)[0]

    for selector in (".mode-segments {", ".model-picker {", ".chart-legend-strip {"):
        assert REPORT_BASE_CSS.count(selector) == 1
        assert document_css.count(selector) == 0
        assert html.count(selector) == 1

    for selector in (
        ".review-empty {",
        ".review-tooltip {",
        ".movement-hover-tooltip {",
    ):
        assert selector in REPORT_BASE_CSS
        assert document_css.count(selector) == 0

    assert ".print-page-furniture { display: none; }" in REPORT_BASE_CSS
    assert ".print-page-furniture" not in document_css


def test_scored_report_preserves_aggregate_payload_with_neutral_evidence(
    tmp_path: Path,
):
    frame = _scored_frame()
    positive = frame["weight"].gt(0).to_numpy()
    exact_a = np.linspace(0.2, 1.0, int(positive.sum()))
    exact_b = np.linspace(0.4, 1.2, int(positive.sum()))
    evidence = {
        "Model A": ModelEvidence(
            source="direct diagnostics",
            importance=FeatureImportanceEvidence(
                pd.DataFrame(
                    {
                        "feature": ["segment", "age"],
                        "magnitude": [3.0, 1.0],
                        "effective_df": [2.0, 4.5],
                    }
                ),
                method="direct model importance",
                source="model-neutral adapter",
            ),
            main_effects={
                "segment": MainEffectEvidence(
                    feature="segment",
                    semantic="native_component",
                    effect=pd.DataFrame(
                        {
                            "label": ["A", "B", "C"],
                            "value": [0.8, 1.1, 1.3],
                            "lower": [0.7, 1.0, 1.2],
                            "upper": [0.9, 1.2, 1.4],
                        }
                    ),
                    source="model-neutral adapter",
                    effective_df=2.0,
                )
            },
            exact_loss=ExactLossEvidence(
                contributions=exact_a,
                size_basis="row_count",
                comparison_group="holdout log score",
                score_label="negative log likelihood",
                source="training artifact",
                family="Tweedie",
                tweedie_power=1.5,
                dispersion=0.7,
            ),
        ),
        "Model B": ModelEvidence(
            source="direct diagnostics",
            exact_loss=ExactLossEvidence(
                contributions=exact_b,
                size_basis="row_count",
                comparison_group="holdout log score",
                score_label="negative log likelihood",
                source="training artifact",
                family="Tweedie",
                tweedie_power=1.5,
                dispersion=0.9,
            ),
        ),
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        evidence=evidence,
        output_path=tmp_path / "neutral.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=2,
            movement_bins=2,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    payload = _embedded_payload(result.output_path)
    assert set(payload) == {
        "metadata",
        "models",
        "metrics",
        "importance",
        "relativities",
        "interactions",
        "distributions",
        "movement",
        "curves",
        "double_lift",
    }
    assert payload["interactions"] == {"models": {}, "unavailable": []}
    assert "never-embed" not in result.output_path.read_text(encoding="utf-8")

    importance = payload["importance"]["Model A"]
    assert [set(row) for row in importance] == [
        {"feature", "magnitude", "share", "effective_df", "method", "source"},
        {"feature", "magnitude", "share", "effective_df", "method", "source"},
    ]
    assert [row["share"] for row in importance] == pytest.approx([0.75, 0.25])
    assert all("sd_eta" not in row for row in importance)
    relativity = payload["relativities"]["segment"]["Model A"]
    assert relativity["labels"] == ["A", "B", "C"]
    assert relativity["x"] == [0, 1, 2]
    assert relativity["relativity"] == pytest.approx([0.8, 1.1, 1.3])
    assert relativity["ci_lower"] == pytest.approx([0.7, 1.0, 1.2])
    assert relativity["ci_upper"] == pytest.approx([0.9, 1.2, 1.4])
    assert relativity["semantic"] == "native_component"
    assert relativity["source"] == "model-neutral adapter"
    assert relativity["effective_df"] == pytest.approx(2.0)
    assert relativity["presentation"] == {
        "title": "segment",
        "axis_label": "relativity",
        "reference_value": 1.0,
        "kind_label": "Native fitted component",
        "value_label": "Fitted relativity",
        "note": (
            "Relativities are native fitted effects; exposure is descriptive context and "
            "uses the report sample for fitted objects."
        ),
    }
    assert relativity["density"] == [
        {"label": "A", "comparison_units": 4, "exposure": 5.0},
        {"label": "B", "comparison_units": 2, "exposure": 3.5},
        {"label": "C", "comparison_units": 3, "exposure": 4.5},
    ]

    rows = {row["model"]: row for row in payload["metrics"]}
    weight = frame.loc[positive, "weight"].to_numpy()
    actual = frame.loc[positive, "actual"].to_numpy()
    model_a = frame.loc[positive, "model_a"].to_numpy()
    assert rows["Model A"]["weighted_actual_mean"] == pytest.approx(
        np.sum(weight * actual) / np.sum(weight)
    )
    assert rows["Model A"]["weighted_prediction_mean"] == pytest.approx(
        np.sum(weight * model_a) / np.sum(weight)
    )
    assert rows["Model A"]["observed_to_predicted"] == pytest.approx(
        np.sum(weight * actual) / np.sum(weight * model_a)
    )
    assert rows["Model A"]["exact_mean_nll"] == pytest.approx(exact_a.sum() / len(weight))
    assert rows["Model A"]["likelihood_family"] == "Tweedie"
    assert rows["Model A"]["likelihood_power"] == pytest.approx(1.5)
    assert rows["Model A"]["likelihood_dispersion"] == pytest.approx(0.7)
    assert rows["Model A"]["likelihood_source"] == "training artifact"

    lift = payload["double_lift"]["Model A"]["Model B"]
    bins = weighted_quantile_bins(
        np.log(
            frame.loc[positive, "model_a"].to_numpy() / frame.loc[positive, "model_b"].to_numpy()
        ),
        weight,
        n_bins=2,
    )
    for row in lift["bins"]:
        mask = bins == row["bin"] - 1
        assert row["rows"] == int(mask.sum())
        assert row["weight"] == pytest.approx(weight[mask].sum())
        assert row["actual"] == pytest.approx(
            np.sum(weight[mask] * actual[mask]) / weight[mask].sum()
        )
        assert row["aggregate_prediction_ratio"] == pytest.approx(
            np.sum(weight[mask] * frame.loc[positive, "model_a"].to_numpy()[mask])
            / np.sum(weight[mask] * frame.loc[positive, "model_b"].to_numpy()[mask])
        )
    comparison = lift["comparison"]
    assert comparison["primary_score"] == "exact_nll"
    assert comparison["mean_exact_nll"] == pytest.approx(
        {"Model A": exact_a.sum() / len(weight), "Model B": exact_b.sum() / len(weight)}
    )


def test_incomparable_exact_losses_fall_back_to_common_power_deviance(tmp_path: Path):
    frame = _scored_frame()
    rows = int(frame["weight"].gt(0).sum())
    evidence = {
        name: ModelEvidence(
            source="direct diagnostics",
            exact_loss=ExactLossEvidence(
                contributions=np.full(rows, contribution),
                size_basis=size_basis,
                comparison_group="group-a" if name == "Model A" else "group-b",
                score_label="score",
                source="training artifact",
                family="Tweedie",
                tweedie_power=1.5,
                dispersion=0.8,
            ),
        )
        for name, contribution, size_basis in (
            ("Model A", 0.2, "row_count"),
            ("Model B", 0.1, "weight_sum"),
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        evidence=evidence,
        output_path=tmp_path / "incomparable.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=2,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    comparison = _embedded_payload(result.output_path)["double_lift"]["Model A"]["Model B"][
        "comparison"
    ]
    assert comparison["primary_score"] == "deviance"
    assert comparison["mean_exact_nll"] is None
    assert comparison["exact_nll_advantage"] is None


def test_burn_cost_exact_losses_retain_model_powers_and_compare_on_common_deviance(
    tmp_path: Path,
):
    frame = _scored_frame()
    rows = int(frame["weight"].gt(0).sum())
    evidence = {
        name: ModelEvidence(
            source="direct diagnostics",
            exact_loss=ExactLossEvidence(
                contributions=np.full(rows, contribution),
                size_basis="row_count",
                comparison_group=f"tweedie:{power}",
                score_label="Exact NLL",
                source="training artifact",
                family="Tweedie",
                tweedie_power=power,
                dispersion=dispersion,
            ),
        )
        for name, power, dispersion, contribution in (
            ("Model A", 1.3, 0.7, 0.2),
            ("Model B", 1.7, 0.9, 0.1),
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a", "Model B": "model_b"},
        sample_weight="weight",
        features=["segment", "age"],
        evidence=evidence,
        output_path=tmp_path / "different-exact-powers.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            double_lift_bins=2,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    payload = _embedded_payload(result.output_path)
    metrics = {row["model"]: row for row in payload["metrics"]}
    assert metrics["Model A"]["exact_mean_nll"] == pytest.approx(0.2)
    assert metrics["Model A"]["likelihood_power"] == pytest.approx(1.3)
    assert metrics["Model B"]["exact_mean_nll"] == pytest.approx(0.1)
    assert metrics["Model B"]["likelihood_power"] == pytest.approx(1.7)
    comparison = payload["double_lift"]["Model A"]["Model B"]["comparison"]
    assert comparison["primary_score"] == "deviance"
    assert comparison["mean_exact_nll"] is None
    assert comparison["exact_nll_advantage"] is None


def test_categorical_support_redacts_cells_below_the_privacy_threshold(tmp_path: Path):
    unsafe_label = "unsafe-rare-level"
    frame = _scored_frame()
    frame.loc[frame["segment"].eq("B"), "segment"] = unsafe_label
    evidence = {
        "Model A": ModelEvidence(
            source="direct diagnostics",
            main_effects={
                "segment": MainEffectEvidence(
                    feature="segment",
                    semantic="native_component",
                    effect=pd.DataFrame(
                        {
                            "label": ["A", unsafe_label, "C"],
                            "value": [0.8, 1.1, 1.3],
                            "lower": [0.7, 1.0, 1.2],
                            "upper": [0.9, 1.2, 1.4],
                        }
                    ),
                    source="model-neutral adapter",
                )
            },
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["segment"],
        evidence=evidence,
        output_path=tmp_path / "private-support.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=3,
        ),
    )

    series = _embedded_payload(result.output_path)["relativities"]["segment"]["Model A"]
    assert series["labels"] == ["A", "C"]
    assert series["x"] == [0, 1]
    assert series["relativity"] == pytest.approx([0.8, 1.3])
    assert series["ci_lower"] == pytest.approx([0.7, 1.2])
    assert series["ci_upper"] == pytest.approx([0.9, 1.4])
    assert series["density"] == [
        {"label": "A", "comparison_units": 4, "exposure": 5.0},
        {"label": "C", "comparison_units": 3, "exposure": 4.5},
    ]
    assert series["suppressed_levels"] == 1
    assert unsafe_label not in result.output_path.read_text(encoding="utf-8")


def _suppressed_numeric_report(tmp_path: Path, status: str):
    frame = _scored_frame()
    evidence = {
        "Model A": ModelEvidence(
            source="model-neutral adapter",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="native_component",
                    effect=pd.DataFrame({"x": [], "value": []}, dtype=float),
                    source="rating workbook",
                    suppression=SuppressionMetadata(
                        status=status,
                        reason="minimum_support",
                        presentation="curve_omitted",
                    ),
                )
            },
        )
    }
    return build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["age"],
        evidence=evidence,
        output_path=tmp_path / f"{status}-suppression.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )


@pytest.mark.parametrize("status", ["partial", "all"])
def test_numeric_curve_suppression_metadata_reaches_payload(
    tmp_path: Path,
    status: str,
):
    result = _suppressed_numeric_report(tmp_path, status)

    series = _embedded_payload(result.output_path)["relativities"]["age"]["Model A"]
    assert series["x"] == []
    assert series["relativity"] == []
    assert series["density"] is None
    assert series["suppression"] == {
        "status": status,
        "reason": "minimum_support",
        "presentation": "curve_omitted",
    }


@pytest.mark.parametrize(
    ("status", "expected_message"),
    [
        (
            "partial",
            (
                "Curve omitted for Model A because at least one interval did not meet minimum "
                "privacy support."
            ),
        ),
        (
            "all",
            "Curve omitted for Model A because no interval met minimum privacy support.",
        ),
    ],
)
def test_complete_html_displays_numeric_curve_suppression(
    tmp_path: Path,
    status: str,
    expected_message: str,
):
    chromium = _headless_chromium()
    if chromium is None:
        pytest.skip("headless Chromium is unavailable")
    result = _suppressed_numeric_report(tmp_path, status)

    completed = subprocess.run(
        [
            str(chromium),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--virtual-time-budget=5000",
            "--dump-dom",
            result.output_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(
        r'<div id="relativity-suppression-note"([^>]*)>(.*?)</div>',
        completed.stdout,
        flags=re.DOTALL,
    )

    assert match is not None
    assert "hidden" not in match.group(1)
    assert unescape(match.group(2)).strip() == expected_message


def test_complete_html_tracks_suppression_for_each_selected_model(tmp_path: Path):
    chromium = _headless_chromium()
    if chromium is None:
        pytest.skip("headless Chromium is unavailable")
    frame = _scored_frame()
    evidence = {
        "Model B": ModelEvidence(
            source="rating workbook",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="native_component",
                    effect=pd.DataFrame({"x": [], "value": []}, dtype=float),
                    source="rating workbook",
                    suppression=SuppressionMetadata(
                        status="partial",
                        reason="minimum_support",
                        presentation="curve_omitted",
                    ),
                )
            },
        ),
        "Model A": ModelEvidence(
            source="safe adapter",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="native_component",
                    effect=pd.DataFrame({"x": [20.0, 45.0, 70.0], "value": [0.9, 1.0, 1.2]}),
                    density=pd.DataFrame({"x": [20.0, 45.0, 70.0], "density": [1.0, 2.0, 3.0]}),
                    source="safe adapter",
                )
            },
        ),
    }
    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model B": "model_b", "Model A": "model_a"},
        sample_weight="weight",
        features=["age"],
        evidence=evidence,
        output_path=tmp_path / "mixed-suppression.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )
    probe = """
    <script>
      window.setTimeout(() => {
        const note = document.getElementById("relativity-suppression-note");
        const select = document.getElementById("relativity-model");
        const state = () => {
          const metrics = Object.fromEntries(
            [...document.querySelectorAll("#relativity-metrics .metric-item")].map(node => [
              node.querySelector(".metric-item-name").textContent,
              node.querySelector(".metric-item-value").textContent
            ])
          );
          const facts = Object.fromEntries(
            [...document.querySelectorAll("#relativity-inspector .summary-fact")].map(node => [
              node.querySelector("span").textContent,
              node.querySelector("strong").textContent
            ])
          );
          return {
            hidden: note.hidden,
            text: note.textContent,
            points: metrics.Points,
            source: facts.Source,
            exposure: facts["Exposure total"]
          };
        };
        const allModels = state();
        select.value = "Model A";
        select.dispatchEvent(new Event("change"));
        const safeModel = state();
        select.value = "Model B";
        select.dispatchEvent(new Event("change"));
        const suppressedModel = state();
        document.body.dataset.suppressionProbe = JSON.stringify({
          allModels, safeModel, suppressedModel
        });
      }, 1200);
    </script>
    """
    result.output_path.write_text(
        result.output_path.read_text(encoding="utf-8").replace(
            "</body>",
            f"{probe}</body>",
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(chromium),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--virtual-time-budget=5000",
            "--dump-dom",
            result.output_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r'data-suppression-probe="([^"]+)"', completed.stdout)

    assert match is not None
    observed = json.loads(unescape(match.group(1)))
    expected = {
        "allModels": {
            "hidden": False,
            "text": (
                "Curve omitted for Model B because at least one interval did not meet "
                "minimum privacy support."
            ),
            "points": "3",
            "source": "safe adapter",
            "exposure": "6",
        },
        "safeModel": {
            "hidden": True,
            "text": "",
            "points": "3",
            "source": "safe adapter",
            "exposure": "6",
        },
        "suppressedModel": {
            "hidden": False,
            "text": (
                "Curve omitted for Model B because at least one interval did not meet "
                "minimum privacy support."
            ),
            "points": "0",
            "source": "rating workbook",
            "exposure": "0",
        },
    }
    assert observed == expected


def test_categorical_main_effect_plots_weighted_business_exposure(tmp_path: Path):
    frame = _scored_frame()
    evidence = {
        "Model A": ModelEvidence(
            source="direct diagnostics",
            main_effects={
                "segment": MainEffectEvidence(
                    feature="segment",
                    semantic="native_component",
                    effect=pd.DataFrame(
                        {
                            "label": ["A", "B", "C"],
                            "value": [0.8, 1.1, 1.3],
                        }
                    ),
                    source="model-neutral adapter",
                )
            },
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["segment"],
        evidence=evidence,
        output_path=tmp_path / "weighted-exposure.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    series = _embedded_payload(result.output_path)["relativities"]["segment"]["Model A"]
    assert [row["comparison_units"] for row in series["density"]] == [4, 2, 3]
    assert [row["exposure"] for row in series["density"]] == pytest.approx([5.0, 3.5, 4.5])
    assert series["weight"] == pytest.approx([5.0, 3.5, 4.5])
    assert series["exposure"]["y"] == pytest.approx([5.0, 3.5, 4.5])
    assert series["support_basis"] == {
        "privacy": "distinct_comparison_units",
        "exposure": "sample_weight_sum",
    }


def test_relativity_exposure_axis_uses_lower_third_without_rescaling_values(
    tmp_path: Path,
):
    chromium = _headless_chromium()
    if chromium is None:
        pytest.skip("headless Chromium is unavailable")

    frame = _scored_frame()
    evidence = {
        "Model A": ModelEvidence(
            source="direct diagnostics",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="partial_dependence",
                    effect=pd.DataFrame({"x": [20.0, 45.0, 70.0], "value": [0.3, 0.5, 0.9]}),
                    source="model-neutral adapter",
                )
            },
        )
    }
    built = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["age"],
        evidence=evidence,
        output_path=tmp_path / "axis-source.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )
    payload = _embedded_payload(built.output_path)
    series = payload["relativities"]["age"]["Model A"]
    series["exposure"] = {
        "kind": "density",
        "x": [20.0, 45.0, 70.0],
        "y": [2.0, 8.0, 4.0],
    }

    from pricing_pipeline.reporting._underwriter_html import render_underwriter_html

    probe = """
    <script>
      window.setTimeout(() => {
        const layout = document.getElementById("relativity-chart")?._fullLayout;
        document.body.dataset.axisProbe = JSON.stringify(layout ? {
          primary_domain: layout.yaxis.domain,
          exposure_domain: layout.yaxis2.domain,
          exposure_range: layout.yaxis2.range
        } : {error: "relativity layout unavailable"});
      }, 1200);
    </script>
    """
    output = tmp_path / "axis-probe.html"
    output.write_text(
        render_underwriter_html(payload).replace("</body>", f"{probe}</body>"),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(chromium),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--virtual-time-budget=5000",
            "--dump-dom",
            output.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r'data-axis-probe="([^"]+)"', completed.stdout)
    assert match is not None
    resolved = json.loads(unescape(match.group(1)))

    assert resolved["primary_domain"] == pytest.approx([0.0, 1.0])
    assert resolved["exposure_domain"] == pytest.approx([0.0, 0.33])
    assert resolved["exposure_range"] == pytest.approx([0.0, 8.0])


def test_oversized_row_aligned_adapter_effect_is_rejected_before_html(tmp_path: Path):
    row_count = 513
    frame = pd.DataFrame(
        {
            "age": np.linspace(18.0, 80.0, row_count),
            "actual": np.linspace(0.0, 1.0, row_count),
            "weight": np.ones(row_count),
            "prediction": np.linspace(0.2, 1.2, row_count),
        }
    )

    class RowAlignedAdapter:
        def collect(self, *, model_name, source, context):
            del model_name, source
            return ModelEvidence(
                source="row-aligned adapter",
                main_effects={
                    "age": MainEffectEvidence(
                        feature="age",
                        semantic="partial_dependence",
                        effect=pd.DataFrame(
                            {
                                "x": context.frame["age"],
                                "value": context.predictions["Model"],
                                "row_marker": [
                                    f"private-row-{index}" for index in range(row_count)
                                ],
                            }
                        ),
                        source="row-aligned adapter",
                    )
                },
            )

    output_path = tmp_path / "oversized-effect.html"
    with pytest.raises(ValueError, match=r"at most 512 points"):
        build_scored_model_report(
            frame,
            actual="actual",
            predictions={"Model": "prediction"},
            sample_weight="weight",
            features=["age"],
            output_path=output_path,
            evidence_requests=(EvidenceRequest("Model", RowAlignedAdapter(), None),),
            options=UnderwriterReportOptions(
                problem_type="burn_cost",
                tweedie_power=1.5,
                comparison_bootstrap_replicates=0,
                minimum_cell_size=2,
            ),
        )
    assert not output_path.exists()


def test_direct_pdp_and_ale_html_contracts_are_semantic_aware(tmp_path: Path):
    frame = _scored_frame().assign(risk_score=np.linspace(-1.0, 1.0, 10))
    evidence = {
        "Model A": ModelEvidence(
            source="non-SuperGLM diagnostics",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="partial_dependence",
                    effect=pd.DataFrame({"x": [20.0, 45.0, 70.0], "value": [0.3, 0.5, 0.9]}),
                    source="generic PDP adapter",
                ),
                "risk_score": MainEffectEvidence(
                    feature="risk_score",
                    semantic="accumulated_local_effect",
                    effect=pd.DataFrame({"x": [-1.0, 0.0, 1.0], "value": [-0.2, 0.0, 0.3]}),
                    source="generic ALE adapter",
                ),
            },
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["age", "risk_score"],
        evidence=evidence,
        output_path=tmp_path / "generic-effects.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    payload = _embedded_payload(result.output_path)["relativities"]
    pdp = payload["age"]["Model A"]["presentation"]
    assert pdp == {
        "title": "age · partial dependence",
        "axis_label": "Response prediction",
        "reference_value": None,
        "kind_label": "Partial dependence",
        "value_label": "Response prediction",
        "note": (
            "Partial dependence is a model response prediction, not a fitted relativity; "
            "exposure is descriptive context."
        ),
    }
    ale = payload["risk_score"]["Model A"]["presentation"]
    assert ale == {
        "title": "risk_score · accumulated local effect",
        "axis_label": "Effect",
        "reference_value": 0.0,
        "kind_label": "Accumulated local effect",
        "value_label": "Effect",
        "note": (
            "Accumulated local effect is centered on zero and is not a fitted relativity; "
            "exposure is descriptive context."
        ),
    }
    html = result.output_path.read_text(encoding="utf-8")
    assert "title: presentation.title" in html
    assert "yLabel: presentation.axis_label" in html
    assert "baseline: presentation.reference_value" in html
    assert "], presentation.note);" in html


def test_mixed_effect_semantics_for_one_feature_are_rejected(tmp_path: Path):
    frame = _scored_frame()
    evidence = {
        "Model A": ModelEvidence(
            source="PDP adapter",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="partial_dependence",
                    effect=pd.DataFrame({"x": [20.0, 70.0], "value": [0.3, 0.9]}),
                    source="PDP adapter",
                )
            },
        ),
        "Model B": ModelEvidence(
            source="ALE adapter",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="accumulated_local_effect",
                    effect=pd.DataFrame({"x": [20.0, 70.0], "value": [-0.2, 0.3]}),
                    source="ALE adapter",
                )
            },
        ),
    }
    output_path = tmp_path / "mixed-semantics.html"

    with pytest.raises(ValueError, match=r"age.*incompatible semantics"):
        build_scored_model_report(
            frame,
            actual="actual",
            predictions={"Model A": "model_a", "Model B": "model_b"},
            sample_weight="weight",
            features=["age"],
            evidence=evidence,
            output_path=output_path,
            options=UnderwriterReportOptions(
                problem_type="burn_cost",
                tweedie_power=1.5,
                comparison_bootstrap_replicates=0,
                minimum_cell_size=2,
            ),
        )
    assert not output_path.exists()


def test_importance_top_k_sorts_by_magnitude_then_feature(tmp_path: Path):
    frame = _scored_frame().assign(small=1.0, tie_z=2.0, tie_a=3.0)
    evidence = {
        "Model A": ModelEvidence(
            source="direct diagnostics",
            importance=FeatureImportanceEvidence(
                pd.DataFrame(
                    {
                        "feature": ["small", "tie_z", "tie_a"],
                        "magnitude": [1.0, 5.0, 5.0],
                    }
                ),
                method="direct model importance",
                source="model-neutral adapter",
            ),
        )
    }

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Model A": "model_a"},
        sample_weight="weight",
        features=["small", "tie_z", "tie_a"],
        evidence=evidence,
        output_path=tmp_path / "sorted-importance.html",
        options=UnderwriterReportOptions(
            problem_type="burn_cost",
            tweedie_power=1.5,
            top_k=2,
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    assert result.importance["Model A"]["feature"].tolist() == [
        "tie_a",
        "tie_z",
        "small",
    ]
    payload = _embedded_payload(result.output_path)
    assert [row["feature"] for row in payload["importance"]["Model A"]] == [
        "tie_a",
        "tie_z",
    ]
