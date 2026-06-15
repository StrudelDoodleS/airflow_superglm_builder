"""Custom build logic for the MTPL frequency SuperGLM model.

Edit the model-owned sections below for the freMTPL tutorial. The final recipe
at the bottom wires those pieces into the shared manifest and publish contract.
"""

from __future__ import annotations

import pickle
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
from pricing_pipeline.models.superglm_diagnostics import fit_reml_with_diagnostics
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


def fit_validate_export_rating_tables(
    frame: pd.DataFrame,
    *,
    split_indices: list[tuple[Any, Any]],
    output_dir: str | Path,
    model_version: str,
    effective_from: str,
) -> tuple[str | Path, str | Path | None, dict[str, float]]:
    """Fit on the full frame and export the rating workbook/model artifact."""
    X, y, exposure, offset = build_training_frame(frame)
    output_path = Path(output_dir)
    artifact_dir = output_path / f"{model_version}_{effective_from}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fitted = fit_reml_with_diagnostics(
        build_model(),
        X,
        y,
        offset=offset,
        diagnostics_path=artifact_dir / "superglm_fit.log",
    )

    workbook_path = artifact_dir / f"rating_tables_{model_version}_{effective_from}.xlsx"
    model_path = artifact_dir / "superglm_model.pkl"
    summary_path = artifact_dir / "model_summary.txt"

    export_rating_tables(
        fitted,
        X,
        y,
        exposure,
        output_path=workbook_path,
    )
    with model_path.open("wb") as handle:
        pickle.dump(fitted, handle)
    summary_path.write_text(str(fitted.summary(detail="compact")), encoding="utf-8")

    result = getattr(fitted, "result", None)
    metrics = {
        "row_count": float(len(X)),
        "exposure_sum": float(np.sum(exposure)),
        "claim_count_sum": float(np.sum(y)),
        "validation_fold_count": float(len(split_indices)),
    }
    if split_indices:
        first_train, first_test = split_indices[0]
        metrics["first_fold_train_rows"] = float(len(first_train))
        metrics["first_fold_test_rows"] = float(len(first_test))
    if getattr(result, "deviance", None) is not None:
        metrics["deviance"] = float(result.deviance)
    if getattr(result, "n_iter", None) is not None:
        metrics["n_iter"] = float(result.n_iter)

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
    rating_workbook_path, model_artifact_path, metrics = fit_validate_export_rating_tables(
        frame,
        split_indices=split_indices,
        output_dir=prepared.get("output_dir") or Path("state") / run_key,
        model_version=model_version,
        effective_from=effective_from,
    )
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
        model_artifact_path=model_artifact_path,
        metrics=metrics,
    )
