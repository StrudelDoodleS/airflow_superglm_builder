from __future__ import annotations

import pickle
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text
from superglm import Categorical, Numeric, SuperGLM

from pricing_models.demo_custom_publish.data import (
    DATASET_NAME,
    FEATURE_COLUMNS,
    PK_COLUMNS,
    SOURCE_SYSTEM,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)
from pricing_pipeline.data.manifest import (
    DatasetManifestResult,
    ModelFrameManifestSpec,
    create_model_frame_manifest_with_split,
)
from pricing_pipeline.models.config import ValidationSplitConfig
from pricing_pipeline.models.spec import TrainingFrame
from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild
from pricing_pipeline.orchestration.run_context import run_key_for_value


MODEL_KEY = "DEMO_CUSTOM_FREQ"
_MODEL_VERSION_PATTERN = re.compile(r"^v(\d+)$")


def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = [*PK_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    frame = raw.loc[raw[WEIGHT_COLUMN].astype(float) > 0, required].copy()
    return frame.sort_values(list(PK_COLUMNS)).reset_index(drop=True)


def build_training_frame(raw: pd.DataFrame) -> TrainingFrame:
    frame = build_final_model_frame(raw)
    exposure = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    return TrainingFrame(
        X=frame.loc[:, FEATURE_COLUMNS].copy(),
        y=frame[TARGET_COLUMN].to_numpy(dtype=float),
        exposure=exposure,
        offset=np.log(exposure),
    )


def build_model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "territory": Categorical(),
            "vehicle_age_band": Categorical(),
            "driver_age": Numeric(),
        },
    )


def next_model_version_from_existing(existing_versions: Iterable[str]) -> str:
    version_numbers = []
    for value in existing_versions:
        match = _MODEL_VERSION_PATTERN.match(str(value).strip())
        if match:
            version_numbers.append(int(match.group(1)))
    return f"v{max(version_numbers, default=0) + 1}"


def next_trained_model_version(engine, *, model_key: str = MODEL_KEY) -> str:
    with engine.begin() as con:
        versions = list(
            con.execute(
                text(
                    """
                    SELECT rp.model_version
                    FROM pricing.PRICING_RATE_PACKAGE AS rp
                    JOIN pricing.PRICING_MODEL AS pm
                      ON pm.model_id = rp.model_id
                    WHERE pm.model_key = :model_key
                      AND rp.parent_rate_package_id IS NULL
                    """
                ),
                {"model_key": model_key},
            ).scalars()
        )
    return next_model_version_from_existing(versions)


def existing_model_version_for_export(
    engine,
    *,
    model_key: str = MODEL_KEY,
    export_id: str,
) -> str | None:
    with engine.begin() as con:
        version = con.execute(
            text(
                """
                SELECT TOP (1) rp.model_version
                FROM pricing.PRICING_RATE_PACKAGE AS rp
                JOIN pricing.PRICING_MODEL AS pm
                  ON pm.model_id = rp.model_id
                WHERE pm.model_key = :model_key
                  AND rp.source_export_id = :export_id
                ORDER BY rp.rate_package_id DESC
                """
            ),
            {"model_key": model_key, "export_id": export_id},
        ).scalar()
    return None if version is None else str(version)


def trained_model_version_for_export(
    engine,
    *,
    model_key: str = MODEL_KEY,
    export_id: str,
) -> str:
    existing = existing_model_version_for_export(
        engine,
        model_key=model_key,
        export_id=export_id,
    )
    if existing is not None:
        return existing
    return next_trained_model_version(engine, model_key=model_key)


def effective_from_for_run(value: date | datetime | str | None = None) -> str:
    if value is None:
        return date.today().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    cleaned = str(value).strip()
    if not cleaned:
        return date.today().isoformat()
    return cleaned[:10]


def export_superglm_completed_build(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    model_version: str,
    effective_from: str,
    created_by: str,
    export_id: str | None = None,
) -> dict[str, object]:
    training = build_training_frame(frame)
    model = build_model()
    fitted = model.fit(
        training.X,
        training.y,
        sample_weight=training.exposure,
        offset=training.offset,
    )

    output_path = Path(output_dir)
    artifact_key = (
        str(export_id).strip()
        if export_id
        else run_key_for_value(f"{model_version}_{effective_from}")
    )
    artifact_dir = output_path / artifact_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = artifact_dir / f"rating_tables_{model_version}_{effective_from}.xlsx"
    model_path = artifact_dir / "superglm_model.pkl"
    summary_path = artifact_dir / "model_summary.txt"

    fitted.export_rating_tables(
        workbook_path,
        training.X,
        training.y,
        sample_weight=training.exposure,
        n_bins=50,
    )
    with model_path.open("wb") as handle:
        pickle.dump(fitted, handle)
    summary_path.write_text(str(fitted.summary(detail="compact")), encoding="utf-8")

    result = getattr(fitted, "result", None)
    metrics = {
        "row_count": float(len(training.X)),
        "exposure_sum": float(np.sum(training.exposure)),
        "claim_count_sum": float(np.sum(training.y)),
    }
    if getattr(result, "deviance", None) is not None:
        metrics["deviance"] = float(result.deviance)
    if getattr(result, "n_iter", None) is not None:
        metrics["n_iter"] = float(result.n_iter)

    return CompletedModelBuild(
        rating_workbook_path=str(workbook_path),
        model_version=model_version,
        effective_from=effective_from,
        created_by=created_by,
        export_id=export_id,
        mlflow_run_id=None,
        model_artifact_path=str(model_path),
        metrics=metrics,
    ).to_dict()


def create_demo_model_frame_manifest(
    engine,
    *,
    frame: pd.DataFrame,
    data_as_of_date: str,
    validation_split: ValidationSplitConfig,
    validation_split_artifact_root: Path | None,
    created_by: str,
) -> DatasetManifestResult:
    final_frame = build_final_model_frame(frame)
    return create_model_frame_manifest_with_split(
        engine,
        frame=final_frame,
        spec=ModelFrameManifestSpec(
            dataset_name=DATASET_NAME,
            source_system=SOURCE_SYSTEM,
            data_as_of_date=data_as_of_date,
            pk_columns=PK_COLUMNS,
            target_column=TARGET_COLUMN,
            weight_column=WEIGHT_COLUMN,
        ),
        validation_split=validation_split,
        validation_split_artifact_root=validation_split_artifact_root,
        created_by=created_by,
    )
