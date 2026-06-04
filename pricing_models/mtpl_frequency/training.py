from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from superglm import Categorical, Numeric, Spline, SuperGLM

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

from pricing_pipeline.infra.mlflow_tracking import optional_mlflow_client
from pricing_pipeline.models.superglm_diagnostics import (
    fit_reml_with_diagnostics,
    parse_deviance_log_metrics,  # noqa: F401
)


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

    mlflow_client = optional_mlflow_client(mlflow)
    mlflow_client.set_experiment(mlflow_experiment)
    with mlflow_client.start_run() as run:
        model = build_model()
        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow_client.log_param("family", getattr(model, "family", "poisson"))
        mlflow_client.log_param("target", "ClaimNb")
        mlflow_client.log_param("offset", "log(Exposure)")
        mlflow_client.log_param("row_count", len(X))
        mlflow_client.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        fitted_model = fit_reml_with_diagnostics(
            model,
            X,
            y,
            offset=offset,
            diagnostics_path=model_dir / "superglm_fit.log",
            mlflow_client=mlflow_client,
        )

        model_path = model_dir / "superglm_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(fitted_model, f)
        mlflow_client.log_artifact(str(model_path), artifact_path="model")

        deviance = getattr(getattr(fitted_model, "result", None), "deviance", None)
        if deviance is not None:
            mlflow_client.log_metric("deviance", float(deviance))

        return {
            "mlflow_run_id": str(getattr(run.info, "run_id", "")),
            "model_path": str(model_path),
        }
