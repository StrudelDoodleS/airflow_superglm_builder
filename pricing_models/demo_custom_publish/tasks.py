from __future__ import annotations

import hashlib
import pickle
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text
from superglm import Categorical, Numeric, SuperGLM

from pricing_pipeline.data.manifest import (
    DatasetManifestResult,
    create_dataset_manifest_with_split,
    new_manifest_id,
)
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec
from pricing_pipeline.models.spec import TrainingFrame
from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild
from pricing_pipeline.publishing.model_registry import ensure_pricing_model
from pricing_pipeline.publishing.rating_export import build_export_id


MODEL_KEY = "DEMO_CUSTOM_FREQ"
DATASET_NAME = "demo_custom_frequency_training"
SOURCE_SYSTEM = "demo_sql_server_staging"
TARGET_COLUMN = "claim_count"
WEIGHT_COLUMN = "exposure"
TRAINING_TABLE = "DEMO_CUSTOM_PUBLISH_TRAINING"
SQL_SERVER_IDENTIFIER_MAX_LENGTH = 128
RUN_KEY_MAX_LENGTH = SQL_SERVER_IDENTIFIER_MAX_LENGTH - len(TRAINING_TABLE) - 1
RUN_KEY_DIGEST_LENGTH = 10
TRAINING_SQL_TEMPLATE = """
SELECT
    policy_id,
    territory,
    vehicle_age_band,
    driver_age,
    exposure,
    claim_count
FROM pricing_stg.{table_name}
ORDER BY policy_id
"""
TRAINING_SQL = TRAINING_SQL_TEMPLATE.format(table_name=TRAINING_TABLE)
FEATURE_COLUMNS = ("territory", "vehicle_age_band", "driver_age")
DEFAULT_OUTPUT_DIR = Path("state/demo_custom_publish")
DEFAULT_ROW_COUNT = 240
DEFAULT_SEED = 20260604

_MODEL_VERSION_PATTERN = re.compile(r"^v(\d+)$")
_SAFE_RUN_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def run_key_for_value(value: object | None) -> str:
    raw = "manual" if value is None else str(value).strip()
    compact = raw.replace("-", "").replace(":", "").replace("+", "")
    safe = _SAFE_RUN_KEY_PATTERN.sub("_", compact).strip("_").lower()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:RUN_KEY_DIGEST_LENGTH]
    suffix = f"_{digest}"
    prefix_length = max(RUN_KEY_MAX_LENGTH - len(suffix), 1)
    prefix = (safe or "manual")[:prefix_length].rstrip("_") or "manual"
    return f"{prefix}{suffix}"


def training_table_for_run(run_key: str) -> str:
    return f"{TRAINING_TABLE}_{run_key_for_value(run_key)}"


def training_sql_for_table(table_name: str) -> str:
    return TRAINING_SQL_TEMPLATE.format(table_name=table_name)


def dataset_spec_for_training_table(table_name: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_name=DATASET_NAME,
        source_system=SOURCE_SYSTEM,
        manifest_sql=training_sql_for_table(table_name),
        pk_columns=("policy_id",),
        target_column=TARGET_COLUMN,
        weight_column=WEIGHT_COLUMN,
        raw_loader=None,
    )


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


def materialize_training_source(
    engine,
    frame: pd.DataFrame,
    *,
    table_name: str = TRAINING_TABLE,
) -> int:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        frame.to_sql(
            table_name,
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


def create_manifest_for_training_table(
    engine,
    *,
    table_name: str,
    model_config: ModelBuildConfig,
    validation_split_artifact_root: Path | None,
    created_by: str,
) -> DatasetManifestResult:
    dataset = dataset_spec_for_training_table(table_name)
    return create_dataset_manifest_with_split(
        engine,
        dataset=dataset,
        manifest_id=new_manifest_id(dataset.dataset_name),
        validation_split=model_config.validation_split,
        validation_split_artifact_root=validation_split_artifact_root,
        created_by=created_by,
    )


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
    artifact_key = str(export_id).strip() if export_id else run_key_for_value(
        f"{model_version}_{effective_from}"
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

    # This is the handoff payload consumed by publish_completed_model_build_task.
    # Real DAGs should derive these values from the task outputs, Airflow run
    # context, SQL package history, and optional manifest/split tasks.
    return CompletedModelBuild(
        rating_workbook_path=str(workbook_path),  # required, produced by export_rating_tables
        model_version=model_version,  # required, normally next vN from SQL history
        effective_from=effective_from,  # required, Airflow/business as-of date
        created_by=created_by,  # optional in Airflow; wrapper can fill default actor
        export_id=export_id,  # optional but recommended for idempotent reruns
        mlflow_run_id=None,  # optional; keep None when MLflow is not used
        model_artifact_path=str(model_path),  # optional fitted-model artifact
        metrics=metrics,  # optional small numeric metrics for lineage/review
    ).to_dict()


def prepare_training_data_task(
    *,
    model_config: ModelBuildConfig,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "prepare_training_data",
):
    from airflow.sdk import get_current_context, task
    from scripts.pricing_db import get_runtime

    @task(task_id=task_id)
    def _prepare_training_data() -> dict[str, str | None]:
        runtime = get_runtime(runtime_module)
        context = get_current_context()
        run_value = context.get("run_id") or _context_logical_date(context)
        run_key = run_key_for_value(run_value)
        table_name = training_table_for_run(run_value)
        run_output_dir = Path(output_dir) / run_key
        frame = build_demo_training_frame(row_count=row_count, seed=seed)
        engine = runtime.get_engine()
        materialize_training_source(engine, frame, table_name=table_name)
        manifest = create_manifest_for_training_table(
            engine,
            table_name=table_name,
            model_config=model_config,
            validation_split_artifact_root=runtime.settings.validation_split_artifact_root,
            created_by=created_by,
        )
        return {
            "training_frame_path": write_training_frame(frame, run_output_dir),
            "training_table": table_name,
            "manifest_id": manifest.manifest_id,
            "split_set_id": manifest.split_set_id,
            "output_dir": str(run_output_dir),
            "run_key": run_key,
        }

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
    def _train_validate_export(prepared_training: dict[str, str | None]) -> dict[str, object]:
        runtime = get_runtime(runtime_module)
        engine = runtime.get_engine()
        context = get_current_context()
        model_version = next_trained_model_version(engine)
        effective_from = effective_from_for_run(_context_logical_date(context))
        export_run_key = prepared_training.get("run_key") or run_key_for_value(
            context.get("run_id") or model_version
        )
        export_id = build_export_id(
            MODEL_KEY,
            str(export_run_key),
        )
        completed = export_superglm_completed_build(
            read_training_frame(str(prepared_training["training_frame_path"])),
            output_dir=prepared_training.get("output_dir") or output_dir,
            model_version=model_version,
            effective_from=effective_from,
            created_by=created_by,
            export_id=export_id,
        )
        completed["manifest_id"] = prepared_training.get("manifest_id")
        completed["split_set_id"] = prepared_training.get("split_set_id")
        return completed

    return _train_validate_export


def register_demo_model_task(
    *,
    model_config: ModelBuildConfig,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "register_demo_model",
):
    from airflow.sdk import task
    from scripts.pricing_db import get_runtime

    @task(task_id=task_id)
    def _register_demo_model() -> int:
        runtime = get_runtime(runtime_module)
        return ensure_pricing_model(
            runtime.get_engine(),
            model_key=model_config.model_key,
            model_label=model_config.model_label,
            target_name=model_config.target_name,
            model_type=model_config.model_type,
            created_by=created_by,
        )

    return _register_demo_model
