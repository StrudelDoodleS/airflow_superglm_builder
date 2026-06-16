from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_models.mtpl_frequency.data import MODEL_FRAME, SOURCE_SQL  # noqa: E402
from pricing_models.mtpl_frequency.modeling import (  # noqa: E402
    MLFLOW_EXPERIMENT,
    build_final_model_frame,
    fit_validate_export_rating_tables,
    read_prepared_source,
    validation_split_indices_for_model,
)
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG  # noqa: E402
from pricing_pipeline.data.fremtpl import (  # noqa: E402
    FREMTPL_COLUMNS,
    fetch_fremtpl,
    prepare_fremtpl_raw_frame,
)
from pricing_pipeline.data.manifest import create_model_frame_manifest_with_split  # noqa: E402
from pricing_pipeline.infra.config import Settings  # noqa: E402
from pricing_pipeline.infra.mlflow_tracking import configure_mlflow  # noqa: E402
from pricing_pipeline.orchestration.completed_build_helpers import (  # noqa: E402
    completed_model_build_payload,
    effective_from_for_run,
)
from pricing_pipeline.orchestration.run_context import run_key_for_value  # noqa: E402
from pricing_pipeline.publishing.rating_export import build_export_id  # noqa: E402


DEFAULT_DB_ROOT = Path("state/offline/mtpl_frequency")
OFFLINE_DDL_DIR = ROOT / "db" / "offline_sqlite"
SCHEMA_DB_FILES = {
    "pricing": "pricing.sqlite",
    "pricing_stg": "pricing_stg.sqlite",
    "mlops": "mlops.sqlite",
}
INSPECTABLE_TABLES = {
    "pricing": [
        "FREMTPL_RAW",
        "DATASET_MANIFEST",
        "DATASET_COLUMN",
        "CV_SPLIT_SET",
        "CV_FOLD",
        "CV_FOLD_METRIC",
        "PRICING_MODEL",
        "MODEL_RUN",
        "PRICING_RATE_PACKAGE",
    ],
    "pricing_stg": [
        "STG_RATING_EXPORT",
        "STG_RATE_CELL",
        "STG_CELL_LEVEL",
    ],
    "mlops": [
        "MODEL_RUN_DATASET",
        "MODEL_RUN_SPLIT_SET",
        "MODEL_RUN_METRIC",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the freMTPL custom model path into offline SQLite databases. "
            "This is a local smoke run for inspecting source, manifest, split, "
            "model-run, metric, and package rows without SQL Server or Airflow."
        )
    )
    parser.add_argument(
        "--db-root",
        default=str(DEFAULT_DB_ROOT),
        help="Directory that will contain pricing.sqlite and run artifacts.",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=None,
        help=("Optional row limit after loading freMTPL. Omit this for the full freMTPL dataset."),
    )
    parser.add_argument(
        "--synthetic-source",
        action="store_true",
        help=(
            "Use deterministic freMTPL-like generated rows instead of fetching "
            "the full OpenML freMTPL dataset. Intended for quick script tests."
        ),
    )
    parser.add_argument(
        "--effective-from",
        default=None,
        help="Effective date. Defaults to today for direct manual runs.",
    )
    parser.add_argument("--created-by", default="offline_sqlite")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing offline SQLite files and artifacts before running.",
    )
    return parser.parse_args()


def sqlite_engine_with_offline_schemas(db_paths: dict[str, Path]):
    for db_path in db_paths.values():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _attach_pricing_schema(dbapi_connection, _connection_record):
        for schema, db_path in db_paths.items():
            dbapi_connection.execute(f"ATTACH DATABASE '{db_path.as_posix()}' AS {schema}")

    return engine


def apply_offline_ddl(engine) -> None:
    connection = engine.raw_connection()
    try:
        for schema in SCHEMA_DB_FILES:
            ddl_path = OFFLINE_DDL_DIR / f"{schema}.sql"
            connection.executescript(ddl_path.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def fre_mtpl_like_raw_frame(row_count: int) -> pd.DataFrame:
    if row_count < 5:
        raise ValueError("row_count must be at least 5 for the configured 5-fold split")

    rng = np.random.default_rng(20260616)
    index = np.arange(row_count)
    exposure = rng.uniform(0.1, 1.0, size=row_count).round(6)
    veh_age = rng.integers(0, 25, size=row_count)
    driv_age = rng.integers(18, 85, size=row_count)
    bonus_malus = rng.integers(50, 180, size=row_count)
    density = rng.lognormal(mean=5.5, sigma=1.0, size=row_count).round(6)
    area = np.array(["A", "B", "C", "D", "E", "F"])[index % 6]
    veh_power = rng.integers(4, 12, size=row_count)
    veh_brand = np.array(["B1", "B2", "B3", "B4", "B5"])[index % 5]
    veh_gas = np.where(index % 3 == 0, "Diesel", "Regular")
    region = np.array(["R11", "R21", "R22", "R23", "R24"])[index % 5]
    log_rate = (
        -3.3
        + 0.006 * (bonus_malus - 100)
        + 0.012 * np.maximum(driv_age - 65, 0)
        + 0.015 * np.maximum(veh_age - 10, 0)
        + 0.08 * (area == "F")
        + 0.04 * (veh_gas == "Diesel")
    )
    claim_nb = rng.poisson(exposure * np.exp(log_rate))

    return pd.DataFrame(
        {
            "IDpol": (100000 + index).astype("int64"),
            "ClaimNb": claim_nb.astype("int64"),
            "Exposure": exposure.astype(float),
            "Area": area,
            "VehPower": veh_power.astype("int64"),
            "VehAge": veh_age.astype("int64"),
            "DrivAge": driv_age.astype("int64"),
            "BonusMalus": bonus_malus.astype("int64"),
            "VehBrand": veh_brand,
            "VehGas": veh_gas,
            "Density": density.astype(float),
            "Region": region,
        },
        columns=FREMTPL_COLUMNS,
    )


def offline_source_frame(
    *,
    row_count: int | None = None,
    synthetic_source: bool = False,
) -> pd.DataFrame:
    if synthetic_source:
        return fre_mtpl_like_raw_frame(row_count or 120)

    frame = prepare_fremtpl_raw_frame(fetch_fremtpl())
    if row_count is not None:
        return frame.head(row_count).copy()
    return frame


def seed_fremtpl_raw(
    engine,
    *,
    row_count: int | None = None,
    synthetic_source: bool = False,
) -> int:
    frame = offline_source_frame(
        row_count=row_count,
        synthetic_source=synthetic_source,
    )
    with engine.begin() as con:
        frame.to_sql(
            "FREMTPL_RAW",
            con,
            schema="pricing",
            if_exists="replace",
            index=False,
        )
    return int(len(frame))


def insert_offline_lifecycle_rows(
    engine,
    *,
    completed_build: dict[str, Any],
    created_by: str,
) -> dict[str, str]:
    model_id = MODEL_CONFIG.model_key
    export_id = str(completed_build["export_id"])
    model_run_id = f"{export_id}__run"
    rate_package_id = f"{export_id}__package"
    metrics = completed_build.get("metrics") or {}
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT OR REPLACE INTO pricing.PRICING_MODEL (
                    model_id, model_key, model_label, target_name, model_type,
                    model_status, created_by
                ) VALUES (
                    :model_id, :model_key, :model_label, :target_name, :model_type,
                    :model_status, :created_by
                )
                """
            ),
            {
                "model_id": model_id,
                "model_key": MODEL_CONFIG.model_key,
                "model_label": MODEL_CONFIG.model_label,
                "target_name": MODEL_CONFIG.target_name,
                "model_type": MODEL_CONFIG.model_type,
                "model_status": "ACTIVE",
                "created_by": created_by,
            },
        )
        con.execute(
            text(
                """
                INSERT OR REPLACE INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_version, package_status,
                    source_export_id, effective_from, manifest_id, split_set_id,
                    rating_workbook_path, model_artifact_path, created_by
                ) VALUES (
                    :rate_package_id, :model_id, :model_version, :package_status,
                    :source_export_id, :effective_from, :manifest_id, :split_set_id,
                    :rating_workbook_path, :model_artifact_path, :created_by
                )
                """
            ),
            {
                "rate_package_id": rate_package_id,
                "model_id": model_id,
                "model_version": completed_build["model_version"],
                "package_status": MODEL_CONFIG.default_package_status,
                "source_export_id": export_id,
                "effective_from": completed_build["effective_from"],
                "manifest_id": completed_build["manifest_id"],
                "split_set_id": completed_build.get("split_set_id"),
                "rating_workbook_path": completed_build["rating_workbook_path"],
                "model_artifact_path": completed_build.get("model_artifact_path"),
                "created_by": created_by,
            },
        )
        con.execute(
            text(
                """
                INSERT OR REPLACE INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id, manifest_id,
                    split_set_id, rate_package_id, rating_workbook_path,
                    model_artifact_path, effective_from, created_by
                ) VALUES (
                    :model_run_id, :model_id, :model_version, :export_id, :manifest_id,
                    :split_set_id, :rate_package_id, :rating_workbook_path,
                    :model_artifact_path, :effective_from, :created_by
                )
                """
            ),
            {
                "model_run_id": model_run_id,
                "model_id": model_id,
                "model_version": completed_build["model_version"],
                "export_id": export_id,
                "manifest_id": completed_build["manifest_id"],
                "split_set_id": completed_build.get("split_set_id"),
                "rate_package_id": rate_package_id,
                "rating_workbook_path": completed_build["rating_workbook_path"],
                "model_artifact_path": completed_build.get("model_artifact_path"),
                "effective_from": completed_build["effective_from"],
                "created_by": created_by,
            },
        )
        for metric_name, metric_value in sorted(metrics.items()):
            con.execute(
                text(
                    """
                    INSERT OR REPLACE INTO mlops.MODEL_RUN_METRIC (
                        model_run_id, metric_name, metric_value, metric_scope
                    ) VALUES (
                        :model_run_id, :metric_name, :metric_value, :metric_scope
                    )
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "metric_scope": "model_run",
                },
            )
    return {"model_run_id": model_run_id, "rate_package_id": rate_package_id}


def table_counts(engine) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    with engine.begin() as con:
        for schema, table_names in INSPECTABLE_TABLES.items():
            counts[schema] = {}
            for table_name in table_names:
                counts[schema][table_name] = int(
                    con.execute(text(f"SELECT COUNT(*) FROM {schema}.{table_name}")).scalar_one()
                )
    return counts


def run_mtpl_frequency_offline_sqlite(
    *,
    db_root: str | Path = DEFAULT_DB_ROOT,
    row_count: int | None = None,
    synthetic_source: bool = False,
    effective_from: str | None = None,
    created_by: str = "offline_sqlite",
    reset: bool = False,
) -> dict[str, Any]:
    root = Path(db_root)
    db_paths = {schema: root / file_name for schema, file_name in SCHEMA_DB_FILES.items()}
    artifact_root = root / "artifacts"
    output_root = root / "runs"
    if reset:
        for db_path in db_paths.values():
            if db_path.exists():
                db_path.unlink()
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        if output_root.exists():
            shutil.rmtree(output_root)

    engine = sqlite_engine_with_offline_schemas(db_paths)
    apply_offline_ddl(engine)
    seeded_rows = seed_fremtpl_raw(
        engine,
        row_count=row_count,
        synthetic_source=synthetic_source,
    )

    effective = effective_from_for_run(effective_from)
    run_key = run_key_for_value(f"offline_sqlite__{effective}")
    output_dir = output_root / run_key
    env_settings = Settings.from_env(os.environ)
    settings = replace(
        env_settings,
        rating_export_root=output_root,
        validation_split_artifact_root=artifact_root / "validation_splits",
    )
    prepared = {
        "run_key": run_key,
        "output_dir": str(output_dir),
        "source_sql": SOURCE_SQL,
        "source_row_count": seeded_rows,
        "effective_from": effective,
        "data_as_of_date": effective,
    }
    export_id = build_export_id(MODEL_CONFIG.model_key, run_key)
    model_version = "v1"

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
            output_dir=output_dir,
            model_version=model_version,
            effective_from=effective,
            mlflow_client=mlflow_client,
        )
        mlflow_run_id = str(getattr(getattr(mlflow_run, "info", None), "run_id", "") or "")
    manifest = create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=MODEL_FRAME.manifest_spec(effective),
        validation_split=MODEL_CONFIG.validation_split,
        validation_split_artifact_root=settings.validation_split_artifact_root,
        split_indices=split_indices,
        created_by=created_by,
    )
    completed_build = completed_model_build_payload(
        rating_workbook_path=rating_workbook_path,
        model_version=model_version,
        effective_from=effective,
        export_id=export_id,
        created_by=created_by,
        manifest_id=manifest.manifest_id,
        split_set_id=manifest.split_set_id,
        mlflow_run_id=mlflow_run_id or None,
        model_artifact_path=model_artifact_path,
        metrics=metrics,
    )
    lifecycle = insert_offline_lifecycle_rows(
        engine,
        completed_build=completed_build,
        created_by=created_by,
    )

    return {
        "db_paths": {schema: str(path) for schema, path in db_paths.items()},
        "artifact_root": str(artifact_root),
        "run_key": run_key,
        "export_id": export_id,
        "model_version": model_version,
        "manifest_id": manifest.manifest_id,
        "split_set_id": manifest.split_set_id,
        "split_artifact_uri": manifest.split_artifact_uri,
        **lifecycle,
        "tables": table_counts(engine),
    }


def main() -> None:
    args = parse_args()
    result = run_mtpl_frequency_offline_sqlite(
        db_root=args.db_root,
        row_count=args.row_count,
        synthetic_source=args.synthetic_source,
        effective_from=args.effective_from,
        created_by=args.created_by,
        reset=args.reset,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
