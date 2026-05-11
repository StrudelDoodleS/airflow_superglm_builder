from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:

    class _MissingMLflow:
        def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_metric(self, key: str, value: float, **kwargs) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

    mlflow = _MissingMLflow()


_DEVIANCE_LOG_PATTERN = re.compile(
    r"(?i)(?:iter(?:ation)?\s*[=: ]+\s*(?P<iteration>\d+).*?)?"
    r"\bdeviance\b\s*[=: ]+\s*(?P<deviance>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
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
