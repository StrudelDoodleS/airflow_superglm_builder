from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["A", "A", "B", "B", "C", "C"],
            "actual": [0.0, 0.2, 0.0, 0.4, 0.8, 1.0],
            "weight": [1.0] * 6,
            "current": [0.1, 0.2, 0.2, 0.4, 0.7, 0.9],
            "new": [0.1, 0.2, 0.1, 0.5, 0.8, 1.0],
        }
    )


def _portable_config_text(*, report_extra: str = "", trailing: str = "") -> str:
    return f"""
[report]
output_path = "review.html"
title = "Portable review"
model_type = "frequency"
minimum_cell_size = 2
{report_extra}

[data]
path = "scored.csv"

[columns]
actual = "actual"
sample_weight = "weight"
features = ["feature"]

[predictions]
"Current" = "current"
"New" = "new"
{trailing}
    """.strip()


def _embedded_payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_exporter_emits_copyable_prediction_report(tmp_path: Path):
    from scripts.export_portable_underwriter_report import render_portable_script

    portable = tmp_path / "portable_underwriter_report.py"
    portable.write_text(render_portable_script(), encoding="utf-8")
    driver = tmp_path / "run_report.py"
    driver.write_text(
        """\
import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import pyarrow

portable_path = Path(sys.argv[1]).resolve()
repository_root = Path(sys.argv[2]).resolve()
for entry in sys.path:
    resolved = Path(entry or ".").resolve()
    assert resolved != repository_root
    assert repository_root not in resolved.parents

original_import = builtins.__import__
forbidden = {
    "airflow",
    "joblib",
    "pricing_pipeline",
    "pyodbc",
    "sqlalchemy",
    "superglm",
}

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

spec = importlib.util.spec_from_file_location("portable_underwriter_report", portable_path)
assert spec is not None and spec.loader is not None
portable = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = portable
spec.loader.exec_module(portable)

frame = pd.DataFrame(
    {
        "feature": ["A", "A", "B", "B", "C", "C"],
        "actual": [0.0, 0.2, 0.0, 0.4, 0.8, 1.0],
        "weight": [1.0] * 6,
        "current": [0.1, 0.2, 0.2, 0.4, 0.7, 0.9],
        "new": [0.1, 0.2, 0.1, 0.5, 0.8, 1.0],
    }
)
result = portable.build_report(
    frame,
    actual="actual",
    predictions={"Current": "current", "New": "new"},
    sample_weight="weight",
    features=["feature"],
    model_type="frequency",
    output_path="portable.html",
    minimum_cell_size=2,
)
assert result.output_path == Path("portable.html").resolve()
html = result.output_path.read_text(encoding="utf-8")
assert "Pricing model review" in html
assert "Current" in html
assert "New" in html
""",
        encoding="utf-8",
    )

    uv = shutil.which("uv")
    assert uv is not None
    clean_environment = os.environ.copy()
    clean_environment.pop("VIRTUAL_ENV", None)
    clean_environment.pop("UV_PROJECT_ENVIRONMENT", None)
    completed = subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--offline",
            "--with",
            "numpy",
            "--with",
            "pandas",
            "--with",
            "plotly>=6.9",
            "--with",
            "pyarrow>=23.0.1",
            "python",
            "-I",
            str(driver),
            str(portable),
            str(Path.cwd()),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        env=clean_environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "portable.html").is_file()


def test_embedded_runtime_uses_advertised_python311_syntax():
    from scripts.export_portable_underwriter_report import _source_bundle

    _, _, sources = _source_bundle()

    for module_name, source in sources.items():
        ast.parse(source, filename=f"<embedded:{module_name}>", feature_version=(3, 11))


def test_exporter_writes_deterministic_artifact(tmp_path: Path):
    from scripts.export_portable_underwriter_report import (
        render_portable_script,
        write_portable_script,
    )

    output = tmp_path / "portable_underwriter_report.py"

    result = write_portable_script(output)
    first = output.read_text(encoding="utf-8")
    write_portable_script(output)

    assert result == output.resolve()
    assert first == render_portable_script()
    assert output.read_text(encoding="utf-8") == first
    assert first.startswith("# /// script\n")
    assert 'SOURCE_SHA256 = "' in first


def test_checked_in_portable_artifact_matches_canonical_sources():
    from scripts.export_portable_underwriter_report import PORTABLE_PATH, render_portable_script

    assert PORTABLE_PATH.read_text(encoding="utf-8") == render_portable_script()


def test_portable_direct_evidence_preserves_curve_suppression_metadata(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    frame = _scored_frame().assign(numeric_feature=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    evidence = {
        "Current": portable.ModelEvidence(
            source="portable adapter",
            main_effects={
                "numeric_feature": portable.MainEffectEvidence(
                    feature="numeric_feature",
                    semantic="native_component",
                    effect=pd.DataFrame({"x": [], "value": []}, dtype=float),
                    source="portable adapter",
                    suppression=portable.SuppressionMetadata(
                        status="all",
                        reason="minimum_support",
                        presentation="curve_omitted",
                    ),
                )
            },
        )
    }

    result = portable.build_report(
        frame,
        actual="actual",
        predictions={"Current": "current"},
        sample_weight="weight",
        features=["numeric_feature"],
        model_type="frequency",
        output_path=tmp_path / "portable-suppression.html",
        evidence=evidence,
        minimum_cell_size=2,
    )

    assert "SuppressionMetadata" in portable.__all__
    series = _embedded_payload(result.output_path)["relativities"]["numeric_feature"]["Current"]
    assert series["suppression"] == {
        "status": "all",
        "reason": "minimum_support",
        "presentation": "curve_omitted",
    }


def test_copied_script_builds_report_from_relative_csv_toml(tmp_path: Path):
    from scripts.export_portable_underwriter_report import PORTABLE_PATH

    portable = tmp_path / "portable_underwriter_report.py"
    portable.write_bytes(PORTABLE_PATH.read_bytes())
    _scored_frame().to_csv(tmp_path / "scored.csv", index=False)
    (tmp_path / "report.toml").write_text(
        """
[report]
output_path = "review.html"
title = "Portable review"
model_type = "frequency"
minimum_cell_size = 2

[data]
path = "scored.csv"

[columns]
actual = "actual"
sample_weight = "weight"
features = ["feature"]

[predictions]
"Current" = "current"
"New" = "new"
""".strip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(portable), "--config", "report.toml"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Report: {tmp_path / 'review.html'}" in completed.stdout
    assert (tmp_path / "review.html").is_file()


def test_portable_config_rejects_model_specific_sections(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    config_path = tmp_path / "report.toml"
    config_path.write_text(
        _portable_config_text(trailing='\n[superglm_objects]\n"Current" = "model.joblib"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown TOML sections: superglm_objects"):
        portable.load_config(config_path)


def test_portable_config_rejects_unknown_report_keys(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    config_path = tmp_path / "report.toml"
    config_path.write_text(
        _portable_config_text(report_extra='not_a_real_option = "quietly ignored"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown \[report\] keys: not_a_real_option"):
        portable.load_config(config_path)


def test_portable_config_requires_full_model_type_name(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    config_path = tmp_path / "report.toml"
    config_path.write_text(
        _portable_config_text().replace('model_type = "frequency"', 'model_type = "freq"'),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\[report\]\.model_type must be frequency, severity, or burn_cost",
    ):
        portable.load_config(config_path)


def test_portable_config_preserves_advanced_report_options(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    config_path = tmp_path / "report.toml"
    config_path.write_text(
        _portable_config_text(
            report_extra="""
top_k = 7
double_lift_bins = 6
curve_bins = 40
distribution_bins = 80
movement_bins = 5
comparison_bootstrap_replicates = 100
comparison_bootstrap_seed = 41
""".strip()
        ),
        encoding="utf-8",
    )

    config = portable.load_config(config_path)

    assert config.options.problem_type == "frequency"
    assert config.options.top_k == 7
    assert config.options.double_lift_bins == 6
    assert config.options.curve_bins == 40
    assert config.options.distribution_bins == 80
    assert config.options.movement_bins == 5
    assert config.options.comparison_bootstrap_replicates == 100
    assert config.options.comparison_bootstrap_seed == 41


@pytest.mark.parametrize(
    ("old", "new", "error_type", "message"),
    [
        (
            'title = "Portable review"',
            "title = false",
            TypeError,
            r"\[report\]\.title must be a string",
        ),
        (
            "minimum_cell_size = 2",
            "minimum_cell_size = true",
            ValueError,
            "minimum_cell_size must be an integer",
        ),
        (
            'features = ["feature"]',
            'features = ["feature", "feature"]',
            ValueError,
            r"\[columns\]\.features must not contain duplicates",
        ),
        (
            'features = ["feature"]',
            'features = "feature"',
            ValueError,
            r"\[columns\]\.features must be a non-empty string array",
        ),
        (
            '"Current" = "current"\n"New" = "new"',
            "",
            ValueError,
            r"\[predictions\] must be a non-empty table",
        ),
        (
            '"Current" = "current"',
            '"Current" = "   "',
            ValueError,
            r"\[predictions\] must map non-empty names to non-empty column names",
        ),
        (
            'path = "scored.csv"',
            'path = "scored.json"',
            ValueError,
            r"\[data\]\.path must end in .csv, .feather, or .parquet",
        ),
        (
            'output_path = "review.html"',
            'output_path = "review.txt"',
            ValueError,
            r"\[report\]\.output_path must end in .html or .htm",
        ),
        (
            'features = ["feature"]',
            'features = ["feature"]\ncomparison_unit = "feature"',
            ValueError,
            "comparison_unit must not also appear in features",
        ),
        (
            "minimum_cell_size = 2",
            "minimum_cell_size = 2\ntweedie_power = true",
            TypeError,
            r"\[report\]\.tweedie_power must be numeric, not boolean",
        ),
    ],
)
def test_portable_config_rejects_malformed_values(
    tmp_path: Path,
    old: str,
    new: str,
    error_type: type[Exception],
    message: str,
):
    from scripts import portable_underwriter_report as portable

    config_path = tmp_path / "report.toml"
    config_path.write_text(_portable_config_text().replace(old, new), encoding="utf-8")

    with pytest.raises(error_type, match=message):
        portable.load_config(config_path)


def test_portable_config_builds_from_parquet(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    _scored_frame().to_parquet(tmp_path / "scored.parquet", index=False)
    config_path = tmp_path / "report.toml"
    config_path.write_text(
        _portable_config_text().replace("scored.csv", "scored.parquet"),
        encoding="utf-8",
    )

    result = portable.build_report_from_config(portable.load_config(config_path))

    assert result.output_path == (tmp_path / "review.html").resolve()
    assert result.output_path.is_file()
    assert result.rows_used == 6


def test_exporter_check_detects_stale_artifact(tmp_path: Path):
    exporter = Path("scripts/export_portable_underwriter_report.py").resolve()
    output = tmp_path / "portable_underwriter_report.py"
    output.write_text("stale\n", encoding="utf-8")

    stale = subprocess.run(
        [sys.executable, str(exporter), "--check", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert stale.returncode == 1
    assert f"Portable artifact is stale: {output}" in stale.stderr

    subprocess.run(
        [sys.executable, str(exporter), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    current = subprocess.run(
        [sys.executable, str(exporter), "--check", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert current.returncode == 0, current.stderr


def test_portable_tutorial_executes_with_only_copied_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from scripts.export_portable_underwriter_report import PORTABLE_PATH

    notebook_path = Path("tutorials/01_portable_underwriter_report.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n\n".join("".join(cell["source"]) for cell in code_cells)
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    compile(source, str(notebook_path), "exec")

    (tmp_path / "portable_underwriter_report.py").write_bytes(PORTABLE_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace: dict[str, object] = {}

    exec(compile(source, str(notebook_path), "exec"), namespace)  # noqa: S102

    output = tmp_path / "portable_model_review.html"
    assert output.is_file()
    assert namespace["result"].output_path == output


def test_portable_payload_matches_canonical_generic_report(tmp_path: Path):
    from pricing_pipeline.reporting import UnderwriterReportOptions, build_scored_model_report
    from scripts import portable_underwriter_report as portable

    frame = _scored_frame()
    canonical = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Current": "current", "New": "new"},
        sample_weight="weight",
        features=["feature"],
        output_path=tmp_path / "canonical.html",
        options=UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2),
    )
    copied = portable.build_report(
        frame,
        actual="actual",
        predictions={"Current": "current", "New": "new"},
        sample_weight="weight",
        features=["feature"],
        model_type="frequency",
        output_path=tmp_path / "portable.html",
        minimum_cell_size=2,
    )

    canonical_payload = _embedded_payload(canonical.output_path)
    portable_payload = _embedded_payload(copied.output_path)
    canonical_payload["metadata"].pop("generated_utc")
    portable_payload["metadata"].pop("generated_utc")

    assert portable_payload == canonical_payload


def test_portable_report_never_serializes_comparison_unit_values(tmp_path: Path):
    from scripts import portable_underwriter_report as portable

    frame = _scored_frame().assign(policy_id=[f"private-policy-{index // 2}" for index in range(6)])

    result = portable.build_report(
        frame,
        actual="actual",
        predictions={"Current": "current", "New": "new"},
        sample_weight="weight",
        features=["feature"],
        model_type="frequency",
        output_path=tmp_path / "private.html",
        comparison_unit="policy_id",
        minimum_cell_size=2,
    )

    assert "private-policy-" not in result.output_path.read_text(encoding="utf-8")


def test_portable_script_runs_through_pep723_uv_metadata(tmp_path: Path):
    from scripts.export_portable_underwriter_report import PORTABLE_PATH

    uv = shutil.which("uv")
    assert uv is not None
    portable = tmp_path / "portable_underwriter_report.py"
    portable.write_bytes(PORTABLE_PATH.read_bytes())
    _scored_frame().to_csv(tmp_path / "scored.csv", index=False)
    (tmp_path / "report.toml").write_text(_portable_config_text(), encoding="utf-8")

    completed = subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--offline",
            str(portable),
            "--config",
            "report.toml",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Report: {tmp_path / 'review.html'}" in completed.stdout
