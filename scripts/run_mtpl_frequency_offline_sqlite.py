from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

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
    PUBLICATION_RECEIPT_FILENAME,
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
from pricing_pipeline.publishing.staging import stage_rating_export  # noqa: E402


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
        "PRICING_FEATURE",
        "PRICING_FEATURE_LEVEL_SET",
        "PRICING_FEATURE_LEVEL",
        "PRICING_TERM",
        "PRICING_TERM_FEATURE",
        "PRICING_RATE_CELL",
        "PRICING_RATE_CELL_LEVEL",
        "PRICING_COMPILED_RATE_CELL",
        "PRICING_COMPILED_1D_RATE_BAND",
        "MODEL_RUN",
        "PRICING_RATE_PACKAGE",
    ],
    "pricing_stg": [
        "STG_RATING_EXPORT",
        "STG_TERM_METADATA",
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
        con.execute(text("DELETE FROM pricing.FREMTPL_RAW"))
        frame.to_sql(
            "FREMTPL_RAW",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
        )
    return int(len(frame))


def register_offline_model(engine, *, created_by: str) -> int:
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_name, model_label, target_name, model_type,
                    model_status, created_by
                ) VALUES (
                    :model_name, :model_label, :target_name, :model_type,
                    :model_status, :created_by
                )
                ON CONFLICT(model_name) DO UPDATE SET
                    model_label = excluded.model_label,
                    target_name = excluded.target_name,
                    model_type = excluded.model_type,
                    model_status = excluded.model_status
                """
            ),
            {
                "model_name": MODEL_CONFIG.model_name,
                "model_label": MODEL_CONFIG.model_label,
                "target_name": MODEL_CONFIG.target_name,
                "model_type": MODEL_CONFIG.model_type,
                "model_status": "ACTIVE",
                "created_by": created_by,
            },
        )
        return int(
            con.execute(
                text(
                    """
                    SELECT model_id
                    FROM pricing.PRICING_MODEL
                    WHERE model_name = :model_name
                    """
                ),
                {"model_name": MODEL_CONFIG.model_name},
            ).scalar_one()
        )


def _scalar_int(value: object) -> int:
    if value is None:
        raise ValueError("expected SQLite insert row id")
    return int(value)


def _insert_feature(con, row: Mapping[str, Any]) -> int:
    feature_name = str(row["feature_name"])
    existing = con.execute(
        text("SELECT feature_id FROM pricing.PRICING_FEATURE WHERE feature_name = :name"),
        {"name": feature_name},
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    result = con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_FEATURE (
                feature_name, feature_value_type, is_ordered
            ) VALUES (
                :feature_name, :feature_value_type, :is_ordered
            )
            """
        ),
        {
            "feature_name": feature_name,
            "feature_value_type": row["feature_value_type"],
            "is_ordered": 1 if row["level_set_type"] in {"NUMERIC_BAND", "SPLINE_GRID_1D"} else 0,
        },
    )
    return _scalar_int(result.lastrowid)


def _insert_level_set(con, *, model_id: int, feature_id: int, row: Mapping[str, Any]) -> int:
    params = {
        "model_id": model_id,
        "feature_id": feature_id,
        "level_set_name": row["level_set_name"],
    }
    existing = con.execute(
        text(
            """
            SELECT level_set_id
            FROM pricing.PRICING_FEATURE_LEVEL_SET
            WHERE model_id = :model_id
              AND feature_id = :feature_id
              AND level_set_name = :level_set_name
            """
        ),
        params,
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    level_set_type = str(row["level_set_type"])
    if level_set_type == "SPLINE_GRID_1D":
        binning_strategy = "SPLINE_EVAL_GRID"
    elif level_set_type == "NUMERIC_BAND":
        binning_strategy = "EXPLICIT_BANDS"
    else:
        binning_strategy = "EXPLICIT_LEVELS"

    result = con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_FEATURE_LEVEL_SET (
                model_id, feature_id, level_set_name, level_set_type,
                binning_strategy, grid_width
            ) VALUES (
                :model_id, :feature_id, :level_set_name, :level_set_type,
                :binning_strategy, NULL
            )
            """
        ),
        {
            **params,
            "level_set_type": level_set_type,
            "binning_strategy": binning_strategy,
        },
    )
    return _scalar_int(result.lastrowid)


def _insert_feature_level(con, *, level_set_id: int, row: Mapping[str, Any]) -> int:
    params = {"level_set_id": level_set_id, "level_code": row["level_code"]}
    existing = con.execute(
        text(
            """
            SELECT feature_level_id
            FROM pricing.PRICING_FEATURE_LEVEL
            WHERE level_set_id = :level_set_id
              AND level_code = :level_code
            """
        ),
        params,
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    result = con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_FEATURE_LEVEL (
                level_set_id, level_code, level_label, order_index, lower_bound,
                upper_bound, representative_value, is_missing, is_other
            ) VALUES (
                :level_set_id, :level_code, :level_label, :order_index, :lower_bound,
                :upper_bound, :representative_value, :is_missing, :is_other
            )
            """
        ),
        {
            "level_set_id": level_set_id,
            "level_code": row["level_code"],
            "level_label": row["level_label"],
            "order_index": row["order_index"],
            "lower_bound": row["lower_bound"],
            "upper_bound": row["upper_bound"],
            "representative_value": row["representative_value"],
            "is_missing": int(row["is_missing"] or 0),
            "is_other": int(row["is_other"] or 0),
        },
    )
    return _scalar_int(result.lastrowid)


def publish_offline_rating_package(
    engine,
    *,
    completed_build: dict[str, Any],
    model_id: int,
    created_by: str,
) -> dict[str, int | bool]:
    export_id = str(completed_build["export_id"])
    with engine.begin() as con:
        export_row = (
            con.execute(
                text(
                    """
                    SELECT
                        export_id,
                        model_id,
                        model_name,
                        model_version,
                        base_rate,
                        effective_from_date,
                        effective_to_date,
                        source_file,
                        publication_receipt_json,
                        publication_receipt_sha256,
                        package_metadata_json,
                        offset_handling,
                        offset_factor_name,
                        offset_source_name,
                        offset_label,
                        metadata_origin
                    FROM pricing_stg.STG_RATING_EXPORT
                    WHERE export_id = :export_id
                    """
                ),
                {"export_id": export_id},
            )
            .mappings()
            .one()
        )
        existing = (
            con.execute(
                text(
                    """
                    SELECT rate_package_id, package_version, publication_receipt_sha256
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE model_id = :model_id
                      AND source_export_id = :export_id
                    """
                ),
                {"model_id": model_id, "export_id": export_id},
            )
            .mappings()
            .one_or_none()
        )
        if existing is not None:
            if existing["publication_receipt_sha256"] != export_row["publication_receipt_sha256"]:
                raise ValueError(
                    f"export_id {export_id!r} is already published with a different "
                    "publication_receipt_sha256"
                )
            return {
                "rate_package_id": int(existing["rate_package_id"]),
                "package_version": int(existing["package_version"]),
                "was_existing": True,
            }

        offset_handling = export_row["offset_handling"] or "UNKNOWN"

        package_version = (
            int(
                con.execute(
                    text(
                        """
                        SELECT COALESCE(MAX(package_version), 0) + 1
                        FROM pricing.PRICING_RATE_PACKAGE
                        WHERE model_id = :model_id
                        """
                    ),
                    {"model_id": model_id},
                ).scalar_one()
            )
            or 1
        )
        package_result = con.execute(
            text(
                """
                INSERT OR REPLACE INTO pricing.PRICING_RATE_PACKAGE (
                    parent_rate_package_id,
                    model_id,
                    model_name,
                    model_version,
                    package_version,
                    base_rate,
                    effective_from_date,
                    effective_to_date,
                    package_status,
                    source_export_id,
                    source_file,
                    publication_receipt_json,
                    publication_receipt_sha256,
                    package_metadata_json,
                    revision_metadata_json,
                    offset_handling,
                    offset_factor_name,
                    offset_source_name,
                    offset_label,
                    metadata_origin,
                    manifest_id,
                    split_set_id,
                    rating_workbook_path,
                    model_artifact_path,
                    created_by
                ) VALUES (
                    NULL,
                    :model_id,
                    :model_name,
                    :model_version,
                    :package_version,
                    :base_rate,
                    :effective_from_date,
                    :effective_to_date,
                    :package_status,
                    :source_export_id,
                    :source_file,
                    :publication_receipt_json,
                    :publication_receipt_sha256,
                    :package_metadata_json,
                    NULL,
                    :offset_handling,
                    :offset_factor_name,
                    :offset_source_name,
                    :offset_label,
                    :metadata_origin,
                    :manifest_id,
                    :split_set_id,
                    :rating_workbook_path,
                    :model_artifact_path,
                    :created_by
                )
                """
            ),
            {
                "model_id": model_id,
                "model_name": export_row["model_name"],
                "model_version": export_row["model_version"],
                "package_version": package_version,
                "base_rate": export_row["base_rate"],
                "effective_from_date": export_row["effective_from_date"],
                "effective_to_date": export_row["effective_to_date"],
                "package_status": MODEL_CONFIG.default_package_status,
                "source_export_id": export_id,
                "source_file": export_row["source_file"],
                "publication_receipt_json": export_row["publication_receipt_json"],
                "publication_receipt_sha256": export_row["publication_receipt_sha256"],
                "package_metadata_json": export_row["package_metadata_json"],
                "offset_handling": offset_handling,
                "offset_factor_name": export_row["offset_factor_name"],
                "offset_source_name": export_row["offset_source_name"],
                "offset_label": export_row["offset_label"],
                "metadata_origin": export_row["metadata_origin"],
                "manifest_id": completed_build["manifest_id"],
                "split_set_id": completed_build.get("split_set_id"),
                "rating_workbook_path": completed_build["rating_workbook_path"],
                "model_artifact_path": completed_build.get("model_artifact_path"),
                "created_by": created_by,
            },
        )
        rate_package_id = _scalar_int(package_result.lastrowid)

        rate_rows = (
            con.execute(
                text(
                    """
                    SELECT *
                    FROM pricing_stg.STG_RATE_CELL
                    WHERE export_id = :export_id
                    ORDER BY row_id
                    """
                ),
                {"export_id": export_id},
            )
            .mappings()
            .all()
        )
        level_rows = (
            con.execute(
                text(
                    """
                    SELECT *
                    FROM pricing_stg.STG_CELL_LEVEL
                    WHERE export_id = :export_id
                    ORDER BY row_id, position_no
                    """
                ),
                {"export_id": export_id},
            )
            .mappings()
            .all()
        )
        if not rate_rows or not level_rows:
            raise ValueError(f"staged export {export_id!r} has no rating cells")
        if offset_handling == "EXPORTED_FACTOR":
            offset_factor_name = export_row["offset_factor_name"]
            if not offset_factor_name or not any(
                row["term_name"] == offset_factor_name and row["term_type"] == "OFFSET_FACTOR"
                for row in rate_rows
            ):
                raise ValueError(
                    "staged export declares EXPORTED_FACTOR offset handling but "
                    f"has no OFFSET_FACTOR term named {offset_factor_name!r}"
                )
        elif offset_handling == "ALREADY_APPLIED_SQL_EXPOSURE" and any(
            row["term_type"] == "OFFSET_FACTOR" for row in rate_rows
        ):
            raise ValueError(
                "staged export declares ALREADY_APPLIED_SQL_EXPOSURE but also "
                "contains an OFFSET_FACTOR term"
            )

        rate_by_row_id = {int(row["row_id"]): row for row in rate_rows}
        levels_by_row: dict[int, list[Mapping[str, Any]]] = {}
        for row in level_rows:
            levels_by_row.setdefault(int(row["row_id"]), []).append(row)
        term_metadata = {
            str(row["term_name"]): row["term_metadata_json"]
            for row in con.execute(
                text(
                    """
                    SELECT term_name, term_metadata_json
                    FROM pricing_stg.STG_TERM_METADATA
                    WHERE export_id = :export_id
                    """
                ),
                {"export_id": export_id},
            )
            .mappings()
            .all()
        }

        feature_ids: dict[str, int] = {}
        level_set_ids: dict[tuple[str, str], int] = {}
        feature_level_ids: dict[tuple[int, str], int] = {}
        for row in level_rows:
            feature_name = str(row["feature_name"])
            feature_id = feature_ids.get(feature_name)
            if feature_id is None:
                feature_id = _insert_feature(con, row)
                feature_ids[feature_name] = feature_id

            level_set_key = (feature_name, str(row["level_set_name"]))
            level_set_id = level_set_ids.get(level_set_key)
            if level_set_id is None:
                level_set_id = _insert_level_set(
                    con,
                    model_id=model_id,
                    feature_id=feature_id,
                    row=row,
                )
                level_set_ids[level_set_key] = level_set_id

            feature_level_key = (level_set_id, str(row["level_code"]))
            if feature_level_key not in feature_level_ids:
                feature_level_ids[feature_level_key] = _insert_feature_level(
                    con,
                    level_set_id=level_set_id,
                    row=row,
                )

        term_ids: dict[str, int] = {}
        for row in rate_rows:
            term_name = str(row["term_name"])
            if term_name in term_ids:
                continue
            result = con.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_TERM (
                        rate_package_id, term_name, term_type, sequence_no,
                        term_metadata_json
                    ) VALUES (
                        :rate_package_id, :term_name, :term_type, :sequence_no,
                        :term_metadata_json
                    )
                    """
                ),
                {
                    "rate_package_id": rate_package_id,
                    "term_name": term_name,
                    "term_type": row["term_type"],
                    "sequence_no": row["sequence_no"],
                    "term_metadata_json": term_metadata.get(term_name),
                },
            )
            term_ids[term_name] = _scalar_int(result.lastrowid)

        inserted_term_features: set[tuple[int, int]] = set()
        for row in level_rows:
            rate_row = rate_by_row_id[int(row["row_id"])]
            term_id = term_ids[str(rate_row["term_name"])]
            position_no = int(row["position_no"])
            key = (term_id, position_no)
            if key in inserted_term_features:
                continue
            feature_name = str(row["feature_name"])
            feature_id = feature_ids[feature_name]
            level_set_id = level_set_ids[(feature_name, str(row["level_set_name"]))]
            con.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_TERM_FEATURE (
                        term_id, position_no, feature_id, level_set_id, input_column_name
                    ) VALUES (
                        :term_id, :position_no, :feature_id, :level_set_id,
                        :input_column_name
                    )
                    """
                ),
                {
                    "term_id": term_id,
                    "position_no": position_no,
                    "feature_id": feature_id,
                    "level_set_id": level_set_id,
                    "input_column_name": feature_name,
                },
            )
            inserted_term_features.add(key)

        cell_ids: dict[int, int] = {}
        for row in rate_rows:
            row_id = int(row["row_id"])
            term_id = term_ids[str(row["term_name"])]
            cell_key_text = str(row["cell_key_text"])
            cell_key_digest = hashlib.sha256(cell_key_text.encode("utf-8")).hexdigest()
            result = con.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_RATE_CELL (
                        term_id, cell_key_text, cell_key_digest, multiplier,
                        log_coefficient, exposure_weight, record_count,
                        is_reference, is_default
                    ) VALUES (
                        :term_id, :cell_key_text, :cell_key_digest, :multiplier,
                        :log_coefficient, :exposure_weight, :record_count,
                        :is_reference, :is_default
                    )
                    """
                ),
                {
                    "term_id": term_id,
                    "cell_key_text": cell_key_text,
                    "cell_key_digest": cell_key_digest,
                    "multiplier": row["multiplier"],
                    "log_coefficient": row["log_coefficient"],
                    "exposure_weight": row["exposure_weight"],
                    "record_count": row["record_count"],
                    "is_reference": int(row["is_reference"] or 0),
                    "is_default": int(row["is_default"] or 0),
                },
            )
            cell_id = _scalar_int(result.lastrowid)
            cell_ids[row_id] = cell_id

            for level_row in levels_by_row[row_id]:
                feature_name = str(level_row["feature_name"])
                level_set_id = level_set_ids[(feature_name, str(level_row["level_set_name"]))]
                feature_level_id = feature_level_ids[(level_set_id, str(level_row["level_code"]))]
                con.execute(
                    text(
                        """
                        INSERT INTO pricing.PRICING_RATE_CELL_LEVEL (
                            cell_id, position_no, feature_level_id
                        ) VALUES (
                            :cell_id, :position_no, :feature_level_id
                        )
                        """
                    ),
                    {
                        "cell_id": cell_id,
                        "position_no": int(level_row["position_no"]),
                        "feature_level_id": feature_level_id,
                    },
                )

        for row in rate_rows:
            row_id = int(row["row_id"])
            term_id = term_ids[str(row["term_name"])]
            cell_key_text = str(row["cell_key_text"])
            cell_key_digest = hashlib.sha256(cell_key_text.encode("utf-8")).hexdigest()
            con.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_COMPILED_RATE_CELL (
                        rate_package_id, term_id, cell_key_digest, term_name,
                        term_type, sequence_no, cell_key_text, multiplier,
                        log_coefficient, exposure_weight, record_count,
                        is_default, is_reference
                    ) VALUES (
                        :rate_package_id, :term_id, :cell_key_digest, :term_name,
                        :term_type, :sequence_no, :cell_key_text, :multiplier,
                        :log_coefficient, :exposure_weight, :record_count,
                        :is_default, :is_reference
                    )
                    """
                ),
                {
                    "rate_package_id": rate_package_id,
                    "term_id": term_id,
                    "cell_key_digest": cell_key_digest,
                    "term_name": row["term_name"],
                    "term_type": row["term_type"],
                    "sequence_no": row["sequence_no"],
                    "cell_key_text": cell_key_text,
                    "multiplier": row["multiplier"],
                    "log_coefficient": row["log_coefficient"],
                    "exposure_weight": row["exposure_weight"],
                    "record_count": row["record_count"],
                    "is_default": int(row["is_default"] or 0),
                    "is_reference": int(row["is_reference"] or 0),
                },
            )

            if row["term_type"] not in {"DISCRETIZED_SPLINE_1D", "NUMERIC_BANDED_1D"}:
                continue
            level_row = levels_by_row[row_id][0]
            feature_name = str(level_row["feature_name"])
            level_set_id = level_set_ids[(feature_name, str(level_row["level_set_name"]))]
            feature_level_id = feature_level_ids[(level_set_id, str(level_row["level_code"]))]
            con.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_COMPILED_1D_RATE_BAND (
                        rate_package_id, term_id, feature_level_id, term_name,
                        feature_name, level_code, sort_order, lower_bound,
                        upper_bound, representative_value, multiplier, log_coefficient
                    ) VALUES (
                        :rate_package_id, :term_id, :feature_level_id, :term_name,
                        :feature_name, :level_code, :sort_order, :lower_bound,
                        :upper_bound, :representative_value, :multiplier,
                        :log_coefficient
                    )
                    """
                ),
                {
                    "rate_package_id": rate_package_id,
                    "term_id": term_id,
                    "feature_level_id": feature_level_id,
                    "term_name": row["term_name"],
                    "feature_name": feature_name,
                    "level_code": level_row["level_code"],
                    "sort_order": int(level_row["order_index"] or 0),
                    "lower_bound": level_row["lower_bound"],
                    "upper_bound": level_row["upper_bound"],
                    "representative_value": level_row["representative_value"],
                    "multiplier": row["multiplier"],
                    "log_coefficient": row["log_coefficient"],
                },
            )

    return {
        "rate_package_id": rate_package_id,
        "package_version": package_version,
        "was_existing": False,
    }


def insert_offline_model_run_rows(
    engine,
    *,
    completed_build: dict[str, Any],
    model_id: int,
    rate_package_id: int,
    created_by: str,
) -> dict[str, str]:
    export_id = str(completed_build["export_id"])
    model_run_id = f"{export_id}__run"
    metrics = completed_build.get("metrics") or {}
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT OR REPLACE INTO pricing.MODEL_RUN (
                    model_run_id, model_id, dag_id, airflow_run_id, mlflow_run_id,
                    model_version, export_id, manifest_id, split_set_id, rate_package_id,
                    model_name, rating_workbook_path, model_artifact_path, effective_from,
                    publication_receipt_path, publication_receipt_sha256,
                    run_status, completed_ts, created_by
                ) VALUES (
                    :model_run_id, :model_id, :dag_id, :airflow_run_id, :mlflow_run_id,
                    :model_version, :export_id, :manifest_id, :split_set_id, :rate_package_id,
                    :model_name, :rating_workbook_path, :model_artifact_path, :effective_from,
                    :publication_receipt_path, :publication_receipt_sha256,
                    :run_status, CURRENT_TIMESTAMP, :created_by
                )
                """
            ),
            {
                "model_run_id": model_run_id,
                "model_id": model_id,
                "dag_id": "offline_sqlite",
                "airflow_run_id": export_id,
                "mlflow_run_id": completed_build.get("mlflow_run_id"),
                "model_version": completed_build["model_version"],
                "export_id": export_id,
                "manifest_id": completed_build["manifest_id"],
                "split_set_id": completed_build.get("split_set_id"),
                "rate_package_id": rate_package_id,
                "model_name": MODEL_CONFIG.model_name,
                "rating_workbook_path": completed_build["rating_workbook_path"],
                "model_artifact_path": completed_build.get("model_artifact_path"),
                "effective_from": completed_build["effective_from"],
                "publication_receipt_path": completed_build.get("publication_receipt_path"),
                "publication_receipt_sha256": completed_build.get("publication_receipt_sha256"),
                "run_status": "SUCCEEDED",
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
    return {"model_run_id": model_run_id}


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
    model_id = register_offline_model(engine, created_by=created_by)
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
    export_id = build_export_id(MODEL_CONFIG.model_name, run_key)
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
    publication_receipt_path = Path(model_artifact_path).parent / PUBLICATION_RECEIPT_FILENAME
    publication_receipt_sha256 = hashlib.sha256(publication_receipt_path.read_bytes()).hexdigest()
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
        publication_receipt_path=publication_receipt_path,
        publication_receipt_sha256=publication_receipt_sha256,
        metrics=metrics,
    )
    stage_rating_export(
        engine,
        workbook_path=Path(rating_workbook_path),
        export_id=export_id,
        model_name=MODEL_CONFIG.model_name,
        model_version=model_version,
        effective_from=effective,
        target_name=MODEL_CONFIG.target_name,
        model_type=MODEL_CONFIG.model_type,
        created_by=created_by,
        replace=True,
        model_id=model_id,
        publication_receipt_path=publication_receipt_path,
        publication_receipt_sha256=publication_receipt_sha256,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
    )
    package = publish_offline_rating_package(
        engine,
        completed_build=completed_build,
        model_id=model_id,
        created_by=created_by,
    )
    lifecycle = insert_offline_model_run_rows(
        engine,
        completed_build=completed_build,
        model_id=model_id,
        rate_package_id=int(package["rate_package_id"]),
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
        "rate_package_id": package["rate_package_id"],
        "package_version": package["package_version"],
        "was_existing": package["was_existing"],
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
