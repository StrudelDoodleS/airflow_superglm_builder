from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline import mlflow_tracking
from pricing_pipeline import training


def raw_training_frame(**overrides) -> pd.DataFrame:
    data = {
        "IDpol": [1, 2, 3],
        "ClaimNb": [0, 2, 1],
        "Exposure": [0.5, 0.0, 2.0],
        "Area": ["A", "B", "C"],
        "VehPower": [6, 7, 8],
        "VehAge": [3, 4, 5],
        "DrivAge": [45, 50, 55],
        "BonusMalus": [50, 60, 70],
        "VehBrand": ["B1", "B2", "B3"],
        "VehGas": ["Regular", "Diesel", "Regular"],
        "Density": [0.2, 50.0, 123.0],
        "Region": ["R1", "R2", "R3"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class PickleableFakeModel:
    family = "poisson"

    def __init__(self, fit_calls: list[dict[str, object]] | None = None):
        self.fit_calls = fit_calls if fit_calls is not None else []
        self.result = SimpleNamespace(deviance=12.5)

    def fit_reml(self, X, y, sample_weight=None, offset=None):
        self.fit_calls.append(
            {
                "X": X.copy(),
                "y": y.copy(),
                "sample_weight": None if sample_weight is None else sample_weight.copy(),
                "offset": offset.copy(),
            }
        )
        return self


def test_build_training_frame_filters_exposure_derives_log_density_and_offset():
    raw = raw_training_frame()

    X, y, exposure, offset = training.build_training_frame(raw)

    assert list(X.columns) == training.FEATURE_COLUMNS
    assert X["LogDensity"].tolist() == [0.0, np.log(123.0)]
    assert X["Area"].tolist() == ["A", "C"]
    np.testing.assert_array_equal(y, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(exposure, np.array([0.5, 2.0]))
    np.testing.assert_allclose(offset, np.log(exposure))


def test_build_training_frame_validates_missing_required_columns():
    raw = raw_training_frame().drop(columns=["ClaimNb", "Density", "VehBrand"])

    with pytest.raises(ValueError) as exc:
        training.build_training_frame(raw)

    message = str(exc.value)
    assert "missing columns" in message
    assert "ClaimNb" in message
    assert "Density" in message
    assert "VehBrand" in message


def test_build_model_constructs_poisson_discrete_superglm_with_expected_features():
    model = training.build_model()

    assert model.family == "poisson"
    assert model._discrete is True
    assert model._n_bins == 256
    assert list(model.features) == training.FEATURE_COLUMNS
    assert type(model.features["VehAge"]).__name__.lower().endswith("spline")
    assert type(model.features["DrivAge"]).__name__.lower().endswith("spline")
    assert type(model.features["BonusMalus"]).__name__.lower().endswith("spline")
    assert type(model.features["LogDensity"]).__name__ == "Numeric"
    assert type(model.features["Area"]).__name__ == "Categorical"
    assert type(model.features["VehPower"]).__name__ == "Categorical"
    assert type(model.features["VehBrand"]).__name__ == "Categorical"
    assert type(model.features["VehGas"]).__name__ == "Categorical"
    assert type(model.features["Region"]).__name__ == "Categorical"


def test_configure_mlflow_sets_tracking_uri(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mlflow_tracking.mlflow,
        "set_tracking_uri",
        lambda tracking_uri: calls.append(tracking_uri),
    )

    mlflow_tracking.configure_mlflow("http://mlflow:5000")

    assert calls == ["http://mlflow:5000"]


def test_train_superglm_reads_data_fits_logs_model_artifact_and_metric(
    monkeypatch, tmp_path
):
    raw = raw_training_frame(Exposure=[0.5, 1.0, 2.0])
    fit_calls = []
    mlflow_calls = []

    class FakeRun:
        info = SimpleNamespace(run_id="run-123")

    class FakeStartRun:
        def __enter__(self):
            mlflow_calls.append(("start_run_enter",))
            return FakeRun()

        def __exit__(self, exc_type, exc, tb):
            mlflow_calls.append(("start_run_exit", exc_type))
            return False

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: mlflow_calls.append(
            ("set_experiment", experiment)
        ),
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: mlflow_calls.append(("log_param", key, value)),
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            ("log_artifact", path, artifact_path)
        ),
        log_metric=lambda key, value: mlflow_calls.append(("log_metric", key, value)),
    )

    def fake_read_sql_query(sql, engine):
        mlflow_calls.append(("read_sql_query", sql, engine))
        return raw

    monkeypatch.setattr(training.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(training, "build_model", lambda: PickleableFakeModel(fit_calls))
    monkeypatch.setattr(training, "mlflow", fake_mlflow)

    engine = object()
    model_dir = tmp_path / "model"
    result = training.train_superglm(
        engine,
        model_dir=model_dir,
        mlflow_experiment="pricing-mtpl-frequency",
    )

    assert result["mlflow_run_id"] == "run-123"
    model_path = Path(result["model_path"])
    assert model_path == model_dir / "superglm_model.pkl"
    with model_path.open("rb") as f:
        assert isinstance(pickle.load(f), PickleableFakeModel)

    assert ("read_sql_query", training.TRAINING_SQL, engine) in mlflow_calls
    assert ("set_experiment", "pricing-mtpl-frequency") in mlflow_calls
    assert ("log_param", "family", "poisson") in mlflow_calls
    assert ("log_param", "target", "ClaimNb") in mlflow_calls
    assert ("log_param", "offset", "log(Exposure)") in mlflow_calls
    assert ("log_param", "row_count", 3) in mlflow_calls
    assert ("log_param", "feature_columns", ",".join(training.FEATURE_COLUMNS)) in mlflow_calls
    assert (
        "log_artifact",
        str(model_dir / "superglm_model.pkl"),
        "model",
    ) in mlflow_calls
    assert ("log_metric", "deviance", 12.5) in mlflow_calls

    assert len(fit_calls) == 1
    fit_call = fit_calls[0]
    assert list(fit_call["X"].columns) == training.FEATURE_COLUMNS
    np.testing.assert_array_equal(fit_call["y"], raw["ClaimNb"].to_numpy(dtype=float))
    assert fit_call["sample_weight"] is None
    np.testing.assert_allclose(fit_call["offset"], np.log(raw["Exposure"]))


def test_train_superglm_script_help_runs_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/train_superglm.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--model-dir" in result.stdout
    assert "--experiment" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
