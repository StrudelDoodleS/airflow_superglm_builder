from __future__ import annotations

import os
import pickle
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra import mlflow_tracking
from pricing_models.mtpl_frequency import training


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


class LoggingFakeModel(PickleableFakeModel):
    def fit_reml(self, X, y, sample_weight=None, offset=None):
        logging.info("superglm iteration=0 deviance=18.25")
        logging.info("superglm iteration=1 deviance=12.5")
        return super().fit_reml(X, y, sample_weight=sample_weight, offset=offset)


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
        mlflow_tracking,
        "mlflow",
        SimpleNamespace(set_tracking_uri=lambda tracking_uri: calls.append(tracking_uri)),
    )

    mlflow_tracking.configure_mlflow("http://mlflow:5000")

    assert calls == ["http://mlflow:5000"]


def test_configure_mlflow_returns_noop_client_when_mlflow_is_missing(monkeypatch):
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)

    client = mlflow_tracking.configure_mlflow("http://mlflow:5000")

    client.set_experiment("pricing")
    client.log_param("model", "MTPL_FREQ")
    client.log_metric("deviance", 1.25)
    client.log_artifact("/tmp/model.pkl", artifact_path="model")
    with client.start_run() as run:
        assert run.info.run_id == ""
    with client.start_span("fit_reml") as span:
        span.set_inputs({"rows": 3})
        span.set_outputs({"loss": 1.0})


def test_optional_mlflow_client_swallows_tracking_backend_failures(monkeypatch):
    class FailingMlflow:
        def set_tracking_uri(self, tracking_uri):
            raise RuntimeError("mlflow server down")

        def set_experiment(self, experiment_name):
            raise RuntimeError("mlflow server down")

        def start_run(self):
            raise RuntimeError("mlflow server down")

        def log_param(self, key, value):
            raise RuntimeError("mlflow server down")

        def log_artifact(self, local_path, artifact_path=None):
            raise RuntimeError("mlflow server down")

        def log_metric(self, key, value, **kwargs):
            raise RuntimeError("mlflow server down")

        def start_span(self, name, span_type=None, attributes=None):
            raise RuntimeError("mlflow server down")

    monkeypatch.setattr(mlflow_tracking, "mlflow", FailingMlflow())

    client = mlflow_tracking.configure_mlflow("http://mlflow:5000")
    client.set_experiment("pricing")
    client.log_param("model", "MTPL_FREQ")
    client.log_metric("deviance", 1.25)
    client.log_artifact("/tmp/model.pkl", artifact_path="model")
    with client.start_run() as run:
        assert run.info.run_id == ""
    with client.start_span("fit_reml") as span:
        span.set_inputs({"rows": 3})
        span.set_outputs({"loss": 1.0})


def test_optional_mlflow_client_delegates_start_span():
    calls = []

    class FakeSpan:
        def set_inputs(self, value):
            calls.append(("set_inputs", value))

        def set_outputs(self, value):
            calls.append(("set_outputs", value))

        def set_attributes(self, value):
            calls.append(("set_attributes", value))

    class FakeSpanContext:
        def __enter__(self):
            calls.append(("span_enter",))
            return FakeSpan()

        def __exit__(self, exc_type, exc, tb):
            calls.append(("span_exit", exc_type))
            return False

    class FakeMlflow:
        def start_span(self, name, span_type=None, attributes=None):
            calls.append(("start_span", name, span_type, attributes))
            return FakeSpanContext()

    client = mlflow_tracking.optional_mlflow_client(FakeMlflow())

    with client.start_span(
        "superglm.fit_reml",
        span_type="TRAINING",
        attributes={"row_count": 4},
    ) as span:
        span.set_inputs({"rows": 4})
        span.set_outputs({"loss": 10.5})
        span.set_attributes({"iteration_count": 2})

    assert ("start_span", "superglm.fit_reml", "TRAINING", {"row_count": 4}) in calls
    assert ("set_inputs", {"rows": 4}) in calls
    assert ("set_outputs", {"loss": 10.5}) in calls
    assert ("set_attributes", {"iteration_count": 2}) in calls
    assert ("span_exit", None) in calls


def test_train_superglm_reads_data_fits_logs_model_artifact_and_metric(monkeypatch, tmp_path):
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

    def fake_log_metric(key, value, **kwargs):
        mlflow_calls.append(("log_metric", key, value, kwargs))

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: mlflow_calls.append(("set_experiment", experiment)),
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: mlflow_calls.append(("log_param", key, value)),
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            ("log_artifact", path, artifact_path)
        ),
        log_metric=fake_log_metric,
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
    assert (
        "log_artifact",
        str(model_dir / "superglm_fit.log"),
        "training_diagnostics",
    ) in mlflow_calls
    assert ("log_metric", "deviance", 12.5, {}) in mlflow_calls

    assert len(fit_calls) == 1
    fit_call = fit_calls[0]
    assert list(fit_call["X"].columns) == training.FEATURE_COLUMNS
    np.testing.assert_array_equal(fit_call["y"], raw["ClaimNb"].to_numpy(dtype=float))
    assert fit_call["sample_weight"] is None
    np.testing.assert_allclose(fit_call["offset"], np.log(raw["Exposure"]))


def test_train_superglm_logs_superglm_fit_diagnostics_to_mlflow(monkeypatch, tmp_path):
    raw = raw_training_frame(Exposure=[0.5, 1.0, 2.0])
    fit_calls = []
    mlflow_calls = []

    class FakeRun:
        info = SimpleNamespace(run_id="run-123")

    class FakeStartRun:
        def __enter__(self):
            return FakeRun()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_log_metric(key, value, **kwargs):
        mlflow_calls.append(("log_metric", key, value, kwargs))

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: None,
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: None,
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            ("log_artifact", path, artifact_path)
        ),
        log_metric=fake_log_metric,
    )

    monkeypatch.setattr(training.pd, "read_sql_query", lambda sql, engine: raw)
    monkeypatch.setattr(training, "build_model", lambda: LoggingFakeModel(fit_calls))
    monkeypatch.setattr(training, "mlflow", fake_mlflow)

    model_dir = tmp_path / "model"
    training.train_superglm(
        object(),
        model_dir=model_dir,
        mlflow_experiment="pricing-mtpl-frequency",
    )

    fit_log = model_dir / "superglm_fit.log"
    assert fit_log.exists()
    log_text = fit_log.read_text(encoding="utf-8")
    assert "superglm iteration=0 deviance=18.25" in log_text
    assert "superglm iteration=1 deviance=12.5" in log_text
    assert (
        "log_artifact",
        str(fit_log),
        "training_diagnostics",
    ) in mlflow_calls
    assert ("log_metric", "fit_iteration_deviance", 18.25, {"step": 0}) in mlflow_calls
    assert ("log_metric", "fit_iteration_deviance", 12.5, {"step": 1}) in mlflow_calls


def test_train_superglm_continues_when_mlflow_logging_fails(monkeypatch, tmp_path):
    raw = raw_training_frame(Exposure=[0.5, 1.0, 2.0])
    fit_calls = []

    class FailingStartRun:
        def __enter__(self):
            raise RuntimeError("mlflow unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    class FailingMlflow:
        def set_experiment(self, experiment):
            raise RuntimeError("mlflow unavailable")

        def start_run(self):
            return FailingStartRun()

        def log_param(self, key, value):
            raise RuntimeError("mlflow unavailable")

        def log_artifact(self, path, artifact_path=None):
            raise RuntimeError("mlflow unavailable")

        def log_metric(self, key, value, **kwargs):
            raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr(training.pd, "read_sql_query", lambda sql, engine: raw)
    monkeypatch.setattr(training, "build_model", lambda: PickleableFakeModel(fit_calls))
    monkeypatch.setattr(training, "mlflow", FailingMlflow())

    model_dir = tmp_path / "model"
    result = training.train_superglm(
        object(),
        model_dir=model_dir,
        mlflow_experiment="pricing-mtpl-frequency",
    )

    assert result["mlflow_run_id"] == ""
    assert Path(result["model_path"]).exists()
    assert (model_dir / "superglm_fit.log").exists()
    assert len(fit_calls) == 1


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
