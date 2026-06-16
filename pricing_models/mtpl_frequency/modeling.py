"""Custom build logic for the MTPL frequency SuperGLM model.

Edit the model-owned sections below for the freMTPL tutorial. The final recipe
at the bottom wires those pieces into the shared manifest and publish contract.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from superglm import Categorical, Numeric, Spline, SuperGLM

# Model-owned frame contract and model config.
from pricing_models.mtpl_frequency.config import MODEL_CONFIG
from pricing_models.mtpl_frequency.data import MODEL_FRAME

# Shared lifecycle helpers; usually leave these imports alone.
from pricing_pipeline.data.manifest import (
    create_model_frame_manifest_with_split,
    validation_split_indices,
)
from pricing_pipeline.infra.mlflow_tracking import configure_mlflow, optional_mlflow_client
from pricing_pipeline.orchestration.completed_build_helpers import (
    completed_model_build_payload,
    effective_from_for_run,
    required_payload_text,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export
from pricing_pipeline.publishing.rating_export import build_export_id, export_rating_tables


# ---------------------------------------------------------------------------
# Edit These Model-Specific Functions
# ---------------------------------------------------------------------------

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
MLFLOW_EXPERIMENT = "pricing-mtpl-frequency"
_POI_ITER_PATTERN = re.compile(
    r"POI iter\s+(?P<iteration>\d+)\s+"
    r"obj=(?P<objective>[^\s]+)\s+"
    r"\|grad\|=(?P<gradient>[^\s]+)\s+"
    r"delta_obj=(?P<delta>[^\s]+)"
)


def read_prepared_source(engine, prepared: Mapping[str, Any]) -> pd.DataFrame:
    """Read the source data identified by data.py's prepared payload."""
    source_sql = required_payload_text(prepared, "source_sql")
    return pd.read_sql_query(source_sql, engine)


def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the final frame used for validation, training, export, and manifesting."""
    if MODEL_FRAME.weight_column is None:
        raise ValueError("MTPL frequency model requires an exposure weight column")

    frame = raw.copy()
    frame = frame.loc[frame[MODEL_FRAME.weight_column].astype(float) > 0].copy()
    frame["LogDensity"] = np.log(frame["Density"].astype(float).clip(lower=1.0))
    return frame.loc[
        :,
        [
            *MODEL_FRAME.pk_columns,
            MODEL_FRAME.target_column,
            MODEL_FRAME.weight_column,
            *FEATURE_COLUMNS,
        ],
    ].copy()


def build_training_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Convert the final model frame into SuperGLM inputs."""
    if MODEL_FRAME.weight_column is None:
        raise ValueError("MTPL frequency model requires an exposure weight column")

    required = [*FEATURE_COLUMNS, MODEL_FRAME.target_column, MODEL_FRAME.weight_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        if "Density" in frame.columns:
            frame = build_final_model_frame(frame)
        else:
            raise ValueError(f"missing columns: {', '.join(missing)}")

    X = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame[MODEL_FRAME.target_column].to_numpy(dtype=float)
    exposure = frame[MODEL_FRAME.weight_column].to_numpy(dtype=float)
    offset = np.log(exposure)
    return X, y, exposure, offset


def build_model() -> SuperGLM:
    """Build the SuperGLM model object used by the custom MTPL recipe."""
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


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except TypeError, ValueError:
            pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _write_json_artifact(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _finite_metric(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        metric = float(value)
    except TypeError, ValueError:
        return None
    return metric if np.isfinite(metric) else None


def _superglm_trace_rows(
    *,
    fit_log_text: str,
    reml_diagnostics: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows_by_step: dict[int, dict[str, object]] = {}

    for match in _POI_ITER_PATTERN.finditer(fit_log_text):
        step = int(match.group("iteration")) - 1
        row = rows_by_step.setdefault(step, {"step": step})
        objective = _finite_metric(match.group("objective"))
        gradient = _finite_metric(match.group("gradient"))
        delta = _finite_metric(match.group("delta"))
        if objective is not None:
            row["training_objective"] = objective
        if gradient is not None:
            row["reml_gradient_norm"] = gradient
        if delta is not None:
            row["reml_delta_objective"] = delta

    if not any("training_objective" in row for row in rows_by_step.values()):
        objective_history = reml_diagnostics.get("objective_history")
        if isinstance(objective_history, list):
            for step, value in enumerate(objective_history):
                objective = _finite_metric(value)
                if objective is not None:
                    row = rows_by_step.setdefault(step, {"step": step})
                    row["training_objective"] = objective

    lambda_history = reml_diagnostics.get("lambda_history")
    if isinstance(lambda_history, list):
        for step, lambdas in enumerate(lambda_history):
            if not isinstance(lambdas, Mapping):
                continue
            row = rows_by_step.setdefault(step, {"step": step})
            for term, value in sorted(lambdas.items()):
                metric = _finite_metric(value)
                if metric is not None:
                    safe_term = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(term)).strip("_")
                    if safe_term:
                        row[f"lambda_{safe_term}"] = metric

    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _write_training_trace_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    pd.DataFrame(columns=["step", "training_objective", "reml_gradient_norm"]).to_csv(
        path,
        index=False,
    )


def _log_superglm_trace_metrics(mlflow_client, rows: list[dict[str, object]]) -> None:
    for row in rows:
        step = int(row["step"])
        objective = _finite_metric(row.get("training_objective"))
        gradient = _finite_metric(row.get("reml_gradient_norm"))
        if objective is not None:
            mlflow_client.log_metric("superglm_training_objective", objective, step=step)
        if gradient is not None:
            mlflow_client.log_metric("superglm_reml_gradient_norm", gradient, step=step)


def fit_validate_export_rating_tables(
    frame: pd.DataFrame,
    *,
    split_indices: list[tuple[Any, Any]],
    output_dir: str | Path,
    model_version: str,
    effective_from: str,
    mlflow_client=None,
) -> tuple[str | Path, str | Path | None, dict[str, float]]:
    """Fit on the full frame and visibly log SuperGLM diagnostics to MLflow."""
    X, y, exposure, offset = build_training_frame(frame)
    output_path = Path(output_dir)
    artifact_dir = output_path / f"{model_version}_{effective_from}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow_client = optional_mlflow_client(mlflow_client)

    workbook_path = artifact_dir / f"rating_tables_{model_version}_{effective_from}.xlsx"
    model_path = artifact_dir / "superglm_model.pkl"
    summary_path = artifact_dir / "model_summary.txt"
    fit_log_path = artifact_dir / "superglm_fit.log"
    diagnostics_path = artifact_dir / "superglm_diagnostics.json"
    reml_diagnostics_path = artifact_dir / "superglm_reml_diagnostics.json"
    training_trace_path = artifact_dir / "superglm_training_trace.csv"

    model = build_model()
    mlflow_client.log_param("family", getattr(model, "family", "poisson"))
    mlflow_client.log_param("target", MODEL_FRAME.target_column)
    mlflow_client.log_param("offset", "log(Exposure)")
    mlflow_client.log_param("row_count", len(X))
    mlflow_client.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
    mlflow_client.log_param("model_version", model_version)
    mlflow_client.log_param("effective_from", effective_from)
    mlflow_client.log_param("validation_fold_count", len(split_indices))

    with mlflow_client.start_span(
        "superglm.fit_reml",
        span_type="TRAINING",
        attributes={"row_count": len(X), "feature_count": len(FEATURE_COLUMNS)},
    ) as fit_span:
        fit_span.set_inputs({"rows": len(X), "features": FEATURE_COLUMNS})
        log_buffer = io.StringIO()
        logging.info("Starting SuperGLM fit_reml for %s rows", len(X))
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            fitted = model.fit_reml(X, y, offset=offset, verbose=True)
        if fitted is None:
            fitted = model

        fit_log_text = log_buffer.getvalue()
        if not fit_log_text:
            fit_log_text = "No SuperGLM stdout/stderr diagnostics were captured during fit_reml.\n"

        diagnostics = fitted.diagnostics() if callable(getattr(fitted, "diagnostics", None)) else {}
        reml_diagnostics = (
            fitted.reml_diagnostics() if callable(getattr(fitted, "reml_diagnostics", None)) else {}
        )
        trace_rows = _superglm_trace_rows(
            fit_log_text=fit_log_text,
            reml_diagnostics=reml_diagnostics,
        )
        objective_values = [
            value
            for row in trace_rows
            if (value := _finite_metric(row.get("training_objective"))) is not None
        ]
        fit_outputs = {"iteration_count": len(objective_values)}
        if objective_values:
            fit_outputs["final_training_objective"] = objective_values[-1]
        fit_span.set_outputs(fit_outputs)

    fit_log_path.write_text(fit_log_text, encoding="utf-8")
    mlflow_client.log_artifact(str(fit_log_path), artifact_path="training_diagnostics")
    _write_json_artifact(diagnostics_path, diagnostics)
    _write_json_artifact(reml_diagnostics_path, reml_diagnostics)
    mlflow_client.log_artifact(str(diagnostics_path), artifact_path="training_diagnostics")
    mlflow_client.log_artifact(str(reml_diagnostics_path), artifact_path="training_diagnostics")
    _write_training_trace_artifact(training_trace_path, trace_rows)
    mlflow_client.log_artifact(str(training_trace_path), artifact_path="training_diagnostics")
    _log_superglm_trace_metrics(mlflow_client, trace_rows)

    export_rating_tables(
        fitted,
        X,
        y,
        exposure,
        output_path=workbook_path,
        mlflow_client=mlflow_client,
    )
    with model_path.open("wb") as handle:
        pickle.dump(fitted, handle)
    summary_path.write_text(str(fitted.summary(detail="compact")), encoding="utf-8")
    mlflow_client.log_artifact(str(model_path), artifact_path="model")
    mlflow_client.log_artifact(str(summary_path), artifact_path="model")

    result = getattr(fitted, "result", None)
    metrics = {
        "row_count": float(len(X)),
        "exposure_sum": float(np.sum(exposure)),
        "claim_count_sum": float(np.sum(y)),
        "validation_fold_count": float(len(split_indices)),
    }
    if getattr(result, "deviance", None) is not None:
        metrics["deviance"] = float(result.deviance)
    if getattr(result, "n_iter", None) is not None:
        metrics["n_iter"] = float(result.n_iter)
    if getattr(result, "converged", None) is not None:
        metrics["converged"] = 1.0 if bool(result.converged) else 0.0

    for metric_name, metric_value in sorted(metrics.items()):
        mlflow_client.log_metric(metric_name, metric_value)

    return workbook_path, model_path, metrics


def validation_split_indices_for_model(frame: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Return validation folds used for both metrics and manifest metadata."""
    return validation_split_indices(frame, MODEL_CONFIG.validation_split)


# ---------------------------------------------------------------------------
# Standard Build Recipe - Usually Leave This Alone
# ---------------------------------------------------------------------------


def train_validate_export_model(
    prepared: Mapping[str, Any],
    *,
    engine,
    settings,
    created_by: str = "airflow",
) -> dict[str, Any]:
    run_key = str(prepared.get("run_key") or "manual")
    export_id = build_export_id(MODEL_CONFIG.model_key, run_key)
    model_version = resolve_model_version_for_export(
        engine,
        model_key=MODEL_CONFIG.model_key,
        export_id=export_id,
    )
    effective_from = effective_from_for_run(required_payload_text(prepared, "effective_from"))
    data_as_of_date = required_payload_text(prepared, "data_as_of_date")

    raw = read_prepared_source(engine, prepared)
    frame = build_final_model_frame(raw)
    frame = frame.sort_values(list(MODEL_FRAME.pk_columns)).reset_index(drop=True)
    split_indices = validation_split_indices_for_model(frame)
    mlflow_client = configure_mlflow(
        settings.mlflow_tracking_uri,
        enabled=settings.mlflow_enabled,
    )
    mlflow_client.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow_client.start_run() as mlflow_run:
        rating_workbook_path, model_artifact_path, metrics = fit_validate_export_rating_tables(
            frame,
            split_indices=split_indices,
            output_dir=prepared.get("output_dir") or Path("state") / run_key,
            model_version=model_version,
            effective_from=effective_from,
            mlflow_client=mlflow_client,
        )
        mlflow_run_id = str(getattr(getattr(mlflow_run, "info", None), "run_id", "") or "")
    manifest = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=MODEL_FRAME.manifest_spec(data_as_of_date),
        validation_split=MODEL_CONFIG.validation_split,
        validation_split_artifact_root=settings.validation_split_artifact_root,
        split_indices=split_indices,
        created_by=created_by,
    )

    return completed_model_build_payload(
        rating_workbook_path=rating_workbook_path,
        model_version=model_version,
        effective_from=effective_from,
        export_id=export_id,
        created_by=created_by,
        manifest_id=manifest.manifest_id,
        split_set_id=manifest.split_set_id,
        mlflow_run_id=mlflow_run_id or None,
        model_artifact_path=model_artifact_path,
        metrics=metrics,
    )
