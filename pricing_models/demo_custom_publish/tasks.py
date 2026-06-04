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

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.spec import TrainingFrame
from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild


MODEL_KEY = "DEMO_CUSTOM_FREQ"
TRAINING_TABLE = "DEMO_CUSTOM_PUBLISH_TRAINING"
TRAINING_SQL = f"""
SELECT
    policy_id,
    territory,
    vehicle_age_band,
    driver_age,
    exposure,
    claim_count
FROM pricing_stg.{TRAINING_TABLE}
ORDER BY policy_id
"""
FEATURE_COLUMNS = ("territory", "vehicle_age_band", "driver_age")
DEFAULT_OUTPUT_DIR = Path("state/demo_custom_publish")
DEFAULT_ROW_COUNT = 240
DEFAULT_SEED = 20260604

_MODEL_VERSION_PATTERN = re.compile(r"^v(\d+)$")


def build_demo_training_frame(
    *,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    territory = rng.choice(
        ["A", "B", "C", "D"],
        row_count,
        p=[0.25, 0.35, 0.25, 0.15],
    )
    vehicle_age_band = rng.choice(
        ["new", "mid", "old"],
        row_count,
        p=[0.20, 0.55, 0.25],
    )
    driver_age = rng.integers(18, 78, row_count)
    exposure = rng.uniform(0.15, 1.0, row_count)

    territory_factor = {"A": 0.80, "B": 1.00, "C": 1.20, "D": 1.45}
    vehicle_factor = {"new": 1.30, "mid": 1.00, "old": 0.90}
    linear_predictor = (
        -1.45
        + np.array([np.log(territory_factor[value]) for value in territory])
        + np.array([np.log(vehicle_factor[value]) for value in vehicle_age_band])
        + 0.006 * (driver_age - 45.0)
    )
    claim_count = rng.poisson(exposure * np.exp(linear_predictor))

    return pd.DataFrame(
        {
            "policy_id": np.arange(1, row_count + 1, dtype=np.int64),
            "territory": territory,
            "vehicle_age_band": vehicle_age_band,
            "driver_age": driver_age,
            "exposure": exposure,
            "claim_count": claim_count,
        }
    )


def write_training_frame(frame: pd.DataFrame, output_dir: str | Path) -> str:
    path = Path(output_dir) / "training_frame.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def read_training_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def materialize_training_source(engine, frame: pd.DataFrame) -> int:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        frame.to_sql(
            TRAINING_TABLE,
            con,
            schema=schemas.pricing_staging,
            if_exists="replace",
            index=False,
        )
    return int(len(frame))


def build_training_frame(raw: pd.DataFrame) -> TrainingFrame:
    required = [*FEATURE_COLUMNS, "claim_count", "exposure"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    training = raw.loc[raw["exposure"].astype(float) > 0, required].copy()
    exposure = training["exposure"].to_numpy(dtype=float)
    return TrainingFrame(
        X=training.loc[:, FEATURE_COLUMNS].copy(),
        y=training["claim_count"].to_numpy(dtype=float),
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
    output_path.mkdir(parents=True, exist_ok=True)
    workbook_path = output_path / f"rating_tables_{model_version}_{effective_from}.xlsx"
    model_path = output_path / "superglm_model.pkl"
    summary_path = output_path / "model_summary.txt"

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


def prepare_training_data_task(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
    runtime_module: str | None = None,
    task_id: str = "prepare_training_data",
):
    from airflow.sdk import task
    from scripts.pricing_db import get_runtime

    @task(task_id=task_id)
    def _prepare_training_data() -> str:
        runtime = get_runtime(runtime_module)
        frame = build_demo_training_frame(row_count=row_count, seed=seed)
        materialize_training_source(runtime.get_engine(), frame)
        return write_training_frame(frame, output_dir)

    return _prepare_training_data


def _context_logical_date(context: dict[str, object]) -> object | None:
    value = context.get("logical_date")
    if value is not None:
        return value
    dag_run = context.get("dag_run")
    return (
        getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "run_after", None)
        or getattr(dag_run, "execution_date", None)
    )


def train_validate_export_task(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "train_validate_export",
):
    from airflow.sdk import get_current_context, task
    from scripts.pricing_db import get_runtime

    @task(task_id=task_id)
    def _train_validate_export(training_frame_path: str) -> dict[str, object]:
        runtime = get_runtime(runtime_module)
        engine = runtime.get_engine()
        context = get_current_context()
        model_version = next_trained_model_version(engine)
        effective_from = effective_from_for_run(_context_logical_date(context))
        return export_superglm_completed_build(
            read_training_frame(training_frame_path),
            output_dir=output_dir,
            model_version=model_version,
            effective_from=effective_from,
            created_by=created_by,
        )

    return _train_validate_export
