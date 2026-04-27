from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from superglm import Categorical, Numeric, Spline, SuperGLM

try:
    import mlflow
except ModuleNotFoundError:

    class _MissingMLflow:
        def set_experiment(self, experiment_name: str) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def start_run(self):
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_param(self, key: str, value) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_metric(self, key: str, value: float) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

    mlflow = _MissingMLflow()


FEATURE_COLUMNS = [
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "LogDensity",
    "Area",
    "VehPower",
    "VehBrand",
    "VehGas",
    "Region",
]

FEATURE_SOURCE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "LogDensity"]
REQUIRED_RAW_COLUMNS = ["ClaimNb", "Exposure", "Density", *FEATURE_SOURCE_COLUMNS]
TRAINING_SQL = "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol"


def build_training_frame(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    training = raw.loc[raw["Exposure"] > 0, REQUIRED_RAW_COLUMNS].copy()
    training["LogDensity"] = np.log(training["Density"].astype(float).clip(lower=1.0))

    X = training.loc[:, FEATURE_COLUMNS].copy()
    y = training["ClaimNb"].to_numpy(dtype=float)
    exposure = training["Exposure"].to_numpy(dtype=float)
    offset = np.log(exposure)
    return X, y, exposure, offset


def build_model() -> SuperGLM:
    features = {
        "VehAge": Spline(),
        "DrivAge": Spline(),
        "BonusMalus": Spline(),
        "LogDensity": Numeric(),
        "Area": Categorical(),
        "VehPower": Categorical(),
        "VehBrand": Categorical(),
        "VehGas": Categorical(),
        "Region": Categorical(),
    }
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=features,
    )


def train_superglm(
    engine,
    *,
    model_dir: Path,
    mlflow_experiment: str,
) -> dict[str, str]:
    raw = pd.read_sql_query(TRAINING_SQL, engine)
    X, y, exposure, offset = build_training_frame(raw)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run() as run:
        model = build_model()
        mlflow.log_param("family", getattr(model, "family", "poisson"))
        mlflow.log_param("target", "ClaimNb")
        mlflow.log_param("offset", "log(Exposure)")
        mlflow.log_param("row_count", len(X))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        fitted_model = model.fit_reml(X, y, sample_weight=exposure, offset=offset)
        if fitted_model is None:
            fitted_model = model

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "superglm_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(fitted_model, f)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        deviance = getattr(getattr(fitted_model, "result", None), "deviance", None)
        if deviance is not None:
            mlflow.log_metric("deviance", float(deviance))

        return {
            "mlflow_run_id": run.info.run_id,
            "model_path": str(model_path),
        }
