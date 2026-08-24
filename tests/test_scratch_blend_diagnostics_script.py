from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from superglm import Tweedie

from scripts import run_scratch_blend_diagnostics as diagnostic_script


def _config(*, data_path: Path, output_dir: Path) -> diagnostic_script.DiagnosticRunConfig:
    return diagnostic_script.DiagnosticRunConfig(
        data_path=data_path,
        output_dir=output_dir,
        random_seed=1729,
        sample_rows=None,
        target="response",
        offset_source="term",
        offset_divisor=12.0,
        sample_weight="credibility",
        features=("x", "category", "period"),
        categorical_features=("category",),
        linear_features=(),
        tweedie_power=1.5,
        spline_kind="ps",
        spline_k=6,
        knot_strategy="quantile_tempered",
        knot_alpha=0.2,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        thread_count=1,
        cv_folds=3,
        max_reml_iter=2,
        simplified_gam=None,
        train_fraction=0.6,
        validation_fraction=0.2,
        governed_gam_weights=(0.4, 0.5),
        double_lift_bins=5,
        risk_bins=6,
        lorenz_bins=20,
        feature_bins=4,
        interaction_bins=3,
        max_categorical_levels=6,
        min_cell_rows=1,
        min_cell_weight_fraction=0.0,
        top_interactions=2,
    )


def test_strict_external_toml_loads_without_reading_input(tmp_path):
    config_path = tmp_path / "example.toml"
    config_path.write_text(
        """
[run]
output_dir = "state/diagnostic_test"
random_seed = 17

[data]
path = "/absolute/local/path/model_input.parquet"

[columns]
target = "response"
offset_source = "term"
offset_divisor = 12
sample_weight = "credibility"
features = ["x", "category", "period"]
categorical_features = ["category"]

[model]
tweedie_power = 1.5

[diagnostics]
governed_gam_weights = [0.4, 0.5]
""".strip(),
        encoding="utf-8",
    )

    config = diagnostic_script.load_diagnostic_config(config_path)

    assert config.data_path == Path("/absolute/local/path/model_input.parquet")
    assert config.tweedie_power == 1.5
    assert config.offset_divisor == 12.0
    assert config.categorical_features == ("category",)
    assert config.output_dir.is_relative_to((diagnostic_script.ROOT / "state").resolve())


def test_external_toml_rejects_unknown_or_unsafe_output(tmp_path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[run]
output_dir = "/tmp/not-ignored"
unknown = true

[data]
path = "/private/local/model_input.parquet"

[columns]
target = "response"
offset_source = "term"
sample_weight = "credibility"
features = ["x"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown \[run\] keys"):
        diagnostic_script.load_diagnostic_config(config_path)


def test_cli_requires_explicit_local_input_acknowledgement(tmp_path):
    with pytest.raises(SystemExit, match="Refusing to read configured input"):
        diagnostic_script.main(["--config", str(tmp_path / "missing.toml")])


def test_fit_superglm_binds_levels_from_training_partition_only(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class RecordingGam:
        def bind_levels(self, frame, *, sample_weight):
            captured["frame"] = frame.copy(deep=True)
            captured["weight"] = np.asarray(sample_weight).copy()
            return self

        def fit_reml(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(diagnostic_script, "SuperGLM", lambda **_kwargs: RecordingGam())
    frame = pd.DataFrame(
        {
            "category": ["train-a", "holdout-only", "train-b", "test-only"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    target = np.array([0.2, 0.4, 0.6, 0.8])
    exposure = np.ones(4)
    weight = np.array([1.0, 10.0, 2.0, 20.0])

    diagnostic_script._fit_superglm(
        frame,
        target,
        exposure,
        weight,
        np.array([0, 2]),
        features={},
        config=_config(data_path=tmp_path / "unused.parquet", output_dir=tmp_path),
    )

    assert captured["frame"]["category"].tolist() == ["train-a", "train-b"]
    assert np.asarray(captured["weight"]).tolist() == [1.0, 2.0]


def test_runner_rejects_unseen_validation_levels_before_model_fit(tmp_path, monkeypatch):
    row_count = 30
    frame = pd.DataFrame(
        {
            "response": np.resize([0.0, 1.0], row_count),
            "term": np.full(row_count, 12.0),
            "credibility": np.ones(row_count),
            "x": np.arange(row_count, dtype=float),
            "category": ["A"] * 18 + ["B"] * 12,
            "period": np.resize([2023, 2024, 2025], row_count),
        }
    )
    data_path = tmp_path / "unseen.parquet"
    frame.to_parquet(data_path, index=False)
    splits = iter(
        [
            (np.arange(18), np.arange(18, 30)),
            (np.arange(18, 24), np.arange(24, 30)),
        ]
    )
    monkeypatch.setattr(
        diagnostic_script,
        "train_test_split",
        lambda *_args, **_kwargs: next(splits),
    )

    def forbidden_model_fit(**_kwargs):
        raise AssertionError("holdout levels must be checked before model construction")

    monkeypatch.setattr(diagnostic_script, "SuperGLM", forbidden_model_fit)

    with pytest.raises(ValueError, match=r"validation.*category.*B"):
        diagnostic_script.run_diagnostics(
            _config(data_path=data_path, output_dir=tmp_path / "diagnostics")
        )


def test_aggregate_only_runner_writes_failure_evidence_without_row_predictions(
    tmp_path,
    monkeypatch,
):
    row_count = 300
    rng = np.random.default_rng(1729)
    x = np.linspace(0.0, 1.0, row_count)
    category = rng.choice(["A", "B", "C"], size=row_count)
    period = rng.choice([2023, 2024, 2025], size=row_count)
    term = np.linspace(3.0, 12.0, row_count)
    exposure = term / 12.0
    credibility = np.linspace(1.0, 2.0, row_count)
    gam_rate = np.exp(-0.2 + 0.4 * x + 0.05 * (period - 2023))
    frame = pd.DataFrame(
        {
            "response": exposure * gam_rate,
            "term": term,
            "credibility": credibility,
            "x": x,
            "category": category,
            "period": period,
        }
    )
    data_path = tmp_path / "local.parquet"
    frame.to_parquet(data_path, index=False)

    class FakeGam:
        family = Tweedie(p=1.5)

        def bind_levels(self, *_args, **_kwargs):
            return self

        def fit_reml(self, *_args, **_kwargs):
            return self

        def predict(self, features, *, offset):
            rate = np.exp(
                -0.2
                + 0.4 * features["x"].to_numpy()
                + 0.05 * (features["period"].to_numpy() - 2023)
            )
            return np.exp(offset) * rate

    class FakeGBM:
        def __init__(self):
            self.weights = {"catboost": 0.5, "lightgbm": 0.25, "xgboost": 0.25}
            self.metrics = pd.DataFrame(
                {
                    "model": ["catboost", "lightgbm", "xgboost", "blend"],
                    "mean_unit_deviance": [1.2, 1.1, 1.3, 1.0],
                    "blend_weight": [0.5, 0.25, 0.25, 1.0],
                }
            )

        def predict_rate(self, features):
            base = np.exp(
                -0.2
                + 0.4 * features["x"].to_numpy()
                + 0.05 * (features["period"].to_numpy() - 2023)
            )
            bad_interaction = (features["x"].to_numpy() > 0.65) & features["category"].eq(
                "B"
            ).to_numpy()
            return base * np.exp(0.6 * bad_interaction)

        def predict_components(self, features):
            prediction = self.predict_rate(features)
            return pd.DataFrame(
                {
                    "catboost": prediction,
                    "lightgbm": prediction,
                    "xgboost": prediction,
                },
                index=features.index,
            )

    monkeypatch.setattr(diagnostic_script, "SuperGLM", lambda **_kwargs: FakeGam())
    monkeypatch.setattr(
        diagnostic_script,
        "unconstrained_superglm_features",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic_script,
        "fit_boosted_blend",
        lambda *_args, **_kwargs: FakeGBM(),
    )
    monkeypatch.setattr(
        diagnostic_script,
        "_convex_tweedie_weights",
        lambda *_args, **_kwargs: np.array([0.45, 0.55]),
    )
    output_dir = tmp_path / "diagnostics"

    result = diagnostic_script.run_diagnostics(_config(data_path=data_path, output_dir=output_dir))

    assert result["rows"] == row_count
    assert result["technical_gam_weight"] == pytest.approx(0.45)
    assert set(result["top_failure_pair"]) == {"x", "category"}
    expected = {
        "blend_summary.csv",
        "gbm_component_summary.csv",
        "all_model_summary.csv",
        "double_lift.csv",
        "risk_calibration.csv",
        "lorenz_curve.csv",
        "feature_calibration.csv",
        "interaction_ranking.csv",
        "interaction_failure_cells.csv",
        "01_blend_weight_curve.png",
        "02_double_lift.png",
        "03_risk_calibration.png",
        "04_lorenz.png",
        "diagnostic_report.md",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert not any("prediction" in path.name for path in output_dir.iterdir())
    report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
    assert "Row-level data and predictions: not retained" in report
    failures = pd.read_csv(output_dir / "interaction_failure_cells.csv")
    assert failures.iloc[0]["gbm_minus_gam_deviance_numerator"] > 0
