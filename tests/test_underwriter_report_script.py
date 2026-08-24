from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import build_underwriter_report as report_script
from scripts import build_underwriter_report_demo as demo_script


def _write_config(
    path: Path,
    data_path: Path,
    output_path: Path,
    extra: str = "",
    *,
    run_extra: str = "",
    columns_extra: str = "",
    title: str = '"Portfolio review"',
    tweedie_power: str = "1.5",
) -> Path:
    path.write_text(
        f"""
[run]
output_path = {str(output_path)!r}
title = {title}
problem_type = "burn_cost"
tweedie_power = {tweedie_power}
minimum_cell_size = 2
{run_extra}

[data]
path = {str(data_path)!r}

[columns]
actual = "actual"
sample_weight = "weight"
features = ["feature"]
{columns_extra}

[predictions]
"Model A" = "prediction_a"
"Model B" = "prediction_b"
{extra}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_config_loads_comparison_and_training_likelihood_metadata(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "report.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        run_extra=(
            "comparison_bootstrap_replicates = 300\n"
            "comparison_bootstrap_seed = 43\n"
            "movement_bins = 8\n"
            "interaction_points = 64"
        ),
        columns_extra='comparison_unit = "policy_id"',
        extra="""
[model_likelihoods."Model A"]
tweedie_power = 1.5
dispersion = 0.7

[model_likelihoods."Model B"]
tweedie_power = 1.4
dispersion = 0.9
""",
    )

    config = report_script.load_report_config(config_path)

    assert config.comparison_unit == "policy_id"
    assert config.options.comparison_bootstrap_replicates == 300
    assert config.options.comparison_bootstrap_seed == 43
    assert config.options.minimum_cell_size == 2
    assert config.options.movement_bins == 8
    assert config.options.interaction_points == 64
    assert config.model_likelihoods["Model A"].tweedie_power == 1.5
    assert config.model_likelihoods["Model A"].dispersion == 0.7
    assert config.model_likelihoods["Model B"].tweedie_power == 1.4


def test_public_demo_uses_every_available_row_by_default():
    assert demo_script._resolved_row_count(678_013, None) == 678_013
    assert demo_script._resolved_row_count(678_013, 12_000) == 12_000
    with pytest.raises(ValueError, match="at least 2,000"):
        demo_script._resolved_row_count(678_013, 1_999)


def test_configured_cli_runner_builds_html_from_parquet(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "actual": [0.0, 0.2, 0.4, 0.8, 1.0, 1.4],
            "weight": [1.0, 1.0, 2.0, 1.0, 1.5, 0.5],
            "prediction_a": [0.1, 0.2, 0.35, 0.7, 0.9, 1.2],
            "prediction_b": [0.2, 0.25, 0.4, 0.6, 0.8, 1.0],
            "policy_id": [
                "private-1",
                "private-1",
                "private-2",
                "private-2",
                "private-3",
                "private-3",
            ],
        }
    )
    data_path = tmp_path / "scored.parquet"
    frame.to_parquet(data_path, index=False)
    output_path = tmp_path / "review.html"
    config_path = _write_config(
        tmp_path / "report.toml",
        data_path,
        output_path,
        columns_extra='comparison_unit = "policy_id"',
        extra="""
[model_likelihoods."Model A"]
tweedie_power = 1.5
dispersion = 0.7

[model_likelihoods."Model B"]
tweedie_power = 1.5
dispersion = 0.8
""",
    )

    config = report_script.load_report_config(config_path)
    result = report_script.build_report_from_config(
        config,
        allow_trusted_model_load=False,
    )

    assert result.output_path == output_path
    assert output_path.is_file()
    assert list(result.metrics["model"]) == ["Model A", "Model B"]
    assert result.metrics["exact_mean_nll"].notna().all()
    assert "private-" not in output_path.read_text(encoding="utf-8")


def test_configured_runner_binds_holdout_superglm_with_report_time_offset(tmp_path: Path):
    import joblib
    import numpy as np
    from superglm import Numeric, SuperGLM

    train_rows = 40
    train = pd.DataFrame(
        {
            "feature": np.linspace(-1.0, 1.0, train_rows),
            "actual": np.resize([0.0, 1.0, 0.0, 2.0, 1.0], train_rows),
            "weight": np.linspace(0.5, 1.8, train_rows),
            "fit_offset": np.log(np.linspace(0.8, 1.4, train_rows)),
        }
    )
    model = SuperGLM(
        features={"feature": Numeric()},
        selection_penalty=0.0,
    ).fit(
        train[["feature"]],
        train["actual"],
        sample_weight=train["weight"],
        offset=train["fit_offset"].to_numpy(),
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    holdout_rows = 9
    frame = pd.DataFrame(
        {
            "feature": np.linspace(-0.8, 1.2, holdout_rows),
            "actual": np.resize([0.0, 1.0, 2.0], holdout_rows),
            "weight": np.linspace(0.4, 1.6, holdout_rows),
            "report_offset": np.log(np.linspace(1.5, 0.9, holdout_rows)),
        }
    )
    frame["prediction_a"] = model.predict(
        frame[["feature"]],
        offset=frame["report_offset"].to_numpy(),
    )
    data_path = tmp_path / "holdout.parquet"
    frame.to_parquet(data_path, index=False)
    output_path = tmp_path / "holdout.html"
    config_path = tmp_path / "holdout.toml"
    config_path.write_text(
        f"""
[run]
output_path = {str(output_path)!r}
problem_type = "frequency"
minimum_cell_size = 2
comparison_bootstrap_replicates = 0

[data]
path = {str(data_path)!r}

[columns]
actual = "actual"
sample_weight = "weight"
features = ["feature"]
offset = "report_offset"

[predictions]
"Model A" = "prediction_a"

[superglm_objects]
"Model A" = {str(model_path)!r}
""".strip(),
        encoding="utf-8",
    )

    config = report_script.load_report_config(config_path)
    assert config.offset == "report_offset"
    result = report_script.build_report_from_config(
        config,
        allow_trusted_model_load=True,
    )

    assert result.output_path == output_path
    assert result.metrics.loc[0, "likelihood_source"] == "fitted SuperGLM object"


def test_prediction_only_cli_run_does_not_import_joblib_or_superglm(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "actual": [0.0, 0.2, 0.4, 0.8, 1.0, 1.4],
            "weight": [1.0, 1.0, 2.0, 1.0, 1.5, 0.5],
            "prediction_a": [0.1, 0.2, 0.35, 0.7, 0.9, 1.2],
            "prediction_b": [0.2, 0.25, 0.4, 0.6, 0.8, 1.0],
        }
    )
    scored_path = tmp_path / "scored.parquet"
    output_path = tmp_path / "guarded.html"
    config_path = _write_config(tmp_path / "guarded.toml", scored_path, output_path)
    frame.to_parquet(scored_path, index=False)
    script = """\
import builtins
import sys
from pathlib import Path

original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "joblib" or name.startswith("superglm"):
        raise AssertionError(f"forbidden import: {name}")
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
from scripts.build_underwriter_report import build_report_from_config, load_report_config

config = load_report_config(Path(sys.argv[1]))
build_report_from_config(config, allow_trusted_model_load=False)
"""

    subprocess.run([sys.executable, "-c", script, str(config_path)], check=True)

    html = output_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["metrics"]
    assert payload["distributions"]
    assert payload["movement"]
    assert payload["curves"]
    assert payload["double_lift"]


def test_config_requires_explicit_trust_before_joblib_load(tmp_path: Path):
    data_path = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "feature": [1, 2],
            "actual": [0.2, 0.4],
            "weight": [1.0, 1.0],
            "prediction_a": [0.2, 0.3],
            "prediction_b": [0.25, 0.35],
        }
    ).to_parquet(data_path, index=False)
    config_path = _write_config(
        tmp_path / "report.toml",
        data_path,
        tmp_path / "review.html",
        extra='\n[superglm_objects]\n"Model A" = "model.joblib"\n',
    )
    config = report_script.load_report_config(config_path)

    with pytest.raises(RuntimeError, match="--allow-trusted-model-load"):
        report_script.build_report_from_config(config, allow_trusted_model_load=False)


def test_config_rejects_unknown_keys(tmp_path: Path):
    config_path = tmp_path / "report.toml"
    config_path.write_text(
        """
[run]
output_path = "report.html"
surprise = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown \[run\] keys: surprise"):
        report_script.load_report_config(config_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"title": "true"}, r"\[run\]\.title must be a string"),
        (
            {"tweedie_power": "true"},
            r"\[run\]\.tweedie_power must be numeric, not boolean",
        ),
        ({"tweedie_power": '"1.5"'}, r"\[run\]\.tweedie_power must be numeric"),
    ],
)
def test_config_rejects_coerced_run_value_types(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
):
    config_path = _write_config(
        tmp_path / "strict.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        **overrides,
    )

    with pytest.raises(TypeError, match=message):
        report_script.load_report_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "tweedie_power",
            "true",
            r'\[model_likelihoods\."Model A"\]\.tweedie_power must be numeric, not boolean',
        ),
        (
            "tweedie_power",
            '"1.5"',
            r'\[model_likelihoods\."Model A"\]\.tweedie_power must be numeric',
        ),
        (
            "dispersion",
            "true",
            r'\[model_likelihoods\."Model A"\]\.dispersion must be numeric, not boolean',
        ),
        (
            "dispersion",
            '"0.7"',
            r'\[model_likelihoods\."Model A"\]\.dispersion must be numeric',
        ),
    ],
)
def test_config_rejects_coerced_likelihood_value_types(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    power = value if field == "tweedie_power" else "1.5"
    dispersion = value if field == "dispersion" else "0.7"
    config_path = _write_config(
        tmp_path / "strict-likelihood.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        extra=f"""
[model_likelihoods."Model A"]
tweedie_power = {power}
dispersion = {dispersion}
""",
    )

    with pytest.raises(TypeError, match=message):
        report_script.load_report_config(config_path)


@pytest.mark.parametrize(
    ("section", "extra"),
    [
        ("predictions", '" Model A " = "prediction_b"'),
        (
            "superglm_objects",
            """
[superglm_objects]
"Model A" = "first.joblib"
" Model A " = "second.joblib"
""",
        ),
        (
            "rating_workbooks",
            """
[rating_workbooks]
"Model A" = "first.xlsx"
" Model A " = "second.xlsx"
""",
        ),
        (
            "model_likelihoods",
            """
[model_likelihoods."Model A"]
tweedie_power = 1.5
dispersion = 0.7
[model_likelihoods." Model A "]
tweedie_power = 1.5
dispersion = 0.8
""",
        ),
    ],
)
def test_config_rejects_duplicate_normalized_model_names(
    tmp_path: Path,
    section: str,
    extra: str,
):
    config_path = _write_config(
        tmp_path / f"duplicate-{section}.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        extra=extra,
    )

    with pytest.raises(
        ValueError,
        match=rf"\[{section}\] contains duplicate normalized model name: 'Model A'",
    ):
        report_script.load_report_config(config_path)


def test_config_rejects_noncanonical_likelihood_model_name(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "noncanonical-likelihood.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        extra="""
[model_likelihoods." Model A "]
tweedie_power = 1.5
dispersion = 0.7
""",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"\[model_likelihoods\] model name must be canonical: ' Model A '; "
            r"use 'Model A'"
        ),
    ):
        report_script.load_report_config(config_path)


@pytest.mark.parametrize("section", ["superglm_objects", "rating_workbooks"])
def test_unknown_artifact_name_is_rejected_before_any_io(
    tmp_path: Path,
    monkeypatch,
    section: str,
):
    artifact_suffix = ".joblib" if section == "superglm_objects" else ".xlsx"
    config_path = _write_config(
        tmp_path / f"unknown-{section}.toml",
        tmp_path / "scored.parquet",
        tmp_path / "review.html",
        extra=f'\n[{section}]\n"Unknown model" = "unknown{artifact_suffix}"\n',
    )
    opened: list[object] = []

    def forbidden_io(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("artifact or data I/O occurred before name validation")

    monkeypatch.setattr(report_script, "_read_frame", forbidden_io)
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=forbidden_io))
    monkeypatch.setattr(pd, "read_excel", forbidden_io)

    with pytest.raises(
        ValueError,
        match=rf"\[{section}\] contains models without predictions: Unknown model",
    ):
        config = report_script.load_report_config(config_path)
        report_script.build_report_from_config(config, allow_trusted_model_load=True)

    assert opened == []
