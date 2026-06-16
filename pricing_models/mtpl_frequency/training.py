"""Compatibility helpers for the older MTPL factory/scripts path.

New custom builds should use pricing_models.mtpl_frequency.modeling directly.
This module re-exports the model primitives so legacy ModelSpec-based utilities
do not carry a second implementation of the same SuperGLM model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

from pricing_models.mtpl_frequency.modeling import (
    FEATURE_COLUMNS,
    build_model,
    build_training_frame,
)
from pricing_pipeline.infra.mlflow_tracking import optional_mlflow_client
from pricing_pipeline.models.superglm_diagnostics import (
    fit_reml_with_diagnostics,
    parse_deviance_log_metrics,  # noqa: F401
)


__all__ = [
    "FEATURE_COLUMNS",
    "TRAINING_SQL",
    "build_model",
    "build_training_frame",
    "parse_deviance_log_metrics",
    "train_superglm",
]

TRAINING_SQL = "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol"


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
