from __future__ import annotations

import io
import logging
import pickle
import re
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
_DEVIANCE_LOG_PATTERN = re.compile(
    r"(?i)(?:iter(?:ation)?\s*[=: ]+\s*(?P<iteration>\d+).*?)?"
    r"\bdeviance\b\s*[=: ]+\s*(?P<deviance>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


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


def parse_deviance_log_metrics(log_text: str) -> list[tuple[int, float]]:
    metrics: list[tuple[int, float]] = []
    for match in _DEVIANCE_LOG_PATTERN.finditer(log_text):
        iteration = match.group("iteration")
        step = int(iteration) if iteration is not None else len(metrics)
        metrics.append((step, float(match.group("deviance"))))
    return metrics


def fit_reml_with_diagnostics(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    offset: np.ndarray,
    diagnostics_path: Path,
    mlflow_client=mlflow,
    sample_weight: np.ndarray | None = None,
):
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    root_logger.addHandler(handler)
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)

    fit_kwargs = {"offset": offset}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    try:
        fitted_model = model.fit_reml(X, y, **fit_kwargs)
    finally:
        handler.flush()
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_root_level)

    if fitted_model is None:
        fitted_model = model

    log_text = log_buffer.getvalue()
    if not log_text:
        log_text = "No Python logging records were captured during superglm fit_reml.\n"
    diagnostics_path.write_text(log_text, encoding="utf-8")
    mlflow_client.log_artifact(str(diagnostics_path), artifact_path="training_diagnostics")

    for step, deviance in parse_deviance_log_metrics(log_text):
        mlflow_client.log_metric("fit_iteration_deviance", deviance, step=step)

    return fitted_model


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
        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_param("family", getattr(model, "family", "poisson"))
        mlflow.log_param("target", "ClaimNb")
        mlflow.log_param("offset", "log(Exposure)")
        mlflow.log_param("row_count", len(X))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        fitted_model = fit_reml_with_diagnostics(
            model,
            X,
            y,
            offset=offset,
            diagnostics_path=model_dir / "superglm_fit.log",
            mlflow_client=mlflow,
        )

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
