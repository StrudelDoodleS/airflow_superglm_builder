from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from superglm import Categorical, SuperGLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.publishing.rating_export import export_rating_tables  # noqa: E402
from pricing_pipeline.publishing.staging import stage_rating_export  # noqa: E402
from pricing_pipeline.publishing.superglm_metadata import (  # noqa: E402
    build_superglm_publication_receipt,
)
from pricing_pipeline.publishing.superglm_publication_receipt import (  # noqa: E402
    OffsetExportContract,
    write_publication_receipt,
)
from scripts.run_mtpl_frequency_offline_sqlite import (  # noqa: E402
    SCHEMA_DB_FILES,
    apply_offline_ddl,
    publish_offline_rating_package,
    sqlite_engine_with_offline_schemas,
    table_counts,
)

DEFAULT_DB_ROOT = Path("state/offline/offset_export_smoke")
MODEL_NAME = "offset_export_smoke"
MODEL_VERSION = "v1"
TARGET_NAME = "ClaimCount"
MODEL_TYPE = "superglm_poisson"
OFFSET_FACTOR_NAME = "TermMonths"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny SuperGLM with offset=log(TermMonths / 12), export a "
            "source-aware offset factor, and publish it into offline SQLite tables."
        )
    )
    parser.add_argument(
        "--db-root",
        default=str(DEFAULT_DB_ROOT),
        help="Directory that will contain pricing.sqlite, staging SQLite, and artifacts.",
    )
    parser.add_argument("--effective-from", default="2026-06-19")
    parser.add_argument("--created-by", default="offset_export_smoke")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def offset_smoke_frame(row_count: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(20260619)
    row_no = np.arange(row_count)
    term_months = np.where(row_no % 2 == 0, 12, 36)
    region = np.where(row_no % 3 == 0, "North", "South")
    exposure = np.ones(row_count, dtype=float)
    offset = np.log(term_months / 12.0)
    region_effect = np.where(region == "North", 0.25, -0.15)
    expected_claim_count = np.exp(-1.9 + region_effect + offset)

    return pd.DataFrame(
        {
            "PolicyID": row_no + 1,
            "Region": region,
            "TermMonths": term_months,
            "Exposure": exposure,
            "ClaimCount": rng.poisson(expected_claim_count).astype(float),
        }
    )


def register_smoke_model(engine, *, created_by: str) -> int:
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_name, model_label, target_name, model_type,
                    model_status, created_by
                ) VALUES (
                    :model_name, :model_label, :target_name, :model_type,
                    'ACTIVE', :created_by
                )
                ON CONFLICT(model_name) DO UPDATE SET
                    model_label = excluded.model_label,
                    target_name = excluded.target_name,
                    model_type = excluded.model_type,
                    model_status = excluded.model_status
                """
            ),
            {
                "model_name": MODEL_NAME,
                "model_label": "Offset export smoke",
                "target_name": TARGET_NAME,
                "model_type": MODEL_TYPE,
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
                {"model_name": MODEL_NAME},
            ).scalar_one()
        )


def fit_and_export_offset_workbook(
    frame: pd.DataFrame,
    *,
    output_path: Path,
) -> tuple[Path, Path, str]:
    X = frame[["Region"]].copy()
    y = frame[TARGET_NAME].to_numpy(dtype=float)
    exposure = frame["Exposure"].to_numpy(dtype=float)
    offset = np.log(frame["TermMonths"].to_numpy(dtype=float) / 12.0)

    model = SuperGLM(
        family="poisson",
        features={"Region": Categorical()},
        selection_penalty=0.0,
        discrete=True,
        n_bins=64,
        retain_fit_state=False,
    )
    fitted = model.fit(
        X,
        y,
        sample_weight=exposure,
        offset=offset,
    )
    if fitted is None:
        fitted = model

    workbook_path = export_rating_tables(
        fitted,
        X,
        y,
        exposure,
        output_path=output_path,
        offset=offset,
        offset_source=frame["TermMonths"].rename(OFFSET_FACTOR_NAME),
        offset_name=OFFSET_FACTOR_NAME,
        offset_kind="auto",
        n_bins=64,
    )
    receipt = build_superglm_publication_receipt(
        fitted,
        offset_contract=OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name=OFFSET_FACTOR_NAME,
            published_factor_name=OFFSET_FACTOR_NAME,
            source_name=OFFSET_FACTOR_NAME,
            label="Policy term months",
        ),
    )
    receipt_path = output_path.with_name("superglm_publication_receipt.json")
    receipt_sha256 = write_publication_receipt(receipt, receipt_path)
    return workbook_path, receipt_path, receipt_sha256


def query_offset_rows(
    engine,
    *,
    export_id: str,
    rate_package_id: int,
) -> dict[str, list[dict[str, Any]]]:
    with engine.begin() as con:
        staging_rows = (
            con.execute(
                text(
                    """
                    SELECT
                        c.term_name,
                        c.term_type,
                        c.cell_key_text,
                        c.multiplier,
                        c.log_coefficient,
                        l.feature_name,
                        l.level_code
                    FROM pricing_stg.STG_RATE_CELL AS c
                    JOIN pricing_stg.STG_CELL_LEVEL AS l
                      ON l.export_id = c.export_id
                     AND l.row_id = c.row_id
                    WHERE c.export_id = :export_id
                      AND c.term_name = :term_name
                    ORDER BY CAST(c.cell_key_text AS INTEGER)
                    """
                ),
                {"export_id": export_id, "term_name": OFFSET_FACTOR_NAME},
            )
            .mappings()
            .all()
        )
        final_rows = (
            con.execute(
                text(
                    """
                    SELECT
                        t.term_name,
                        t.term_type,
                        c.cell_key_text,
                        c.multiplier,
                        c.log_coefficient,
                        f.feature_name,
                        fl.level_code
                    FROM pricing.PRICING_TERM AS t
                    JOIN pricing.PRICING_RATE_CELL AS c
                      ON c.term_id = t.term_id
                    JOIN pricing.PRICING_RATE_CELL_LEVEL AS cl
                      ON cl.cell_id = c.cell_id
                    JOIN pricing.PRICING_FEATURE_LEVEL AS fl
                      ON fl.feature_level_id = cl.feature_level_id
                    JOIN pricing.PRICING_FEATURE_LEVEL_SET AS fls
                      ON fls.level_set_id = fl.level_set_id
                    JOIN pricing.PRICING_FEATURE AS f
                      ON f.feature_id = fls.feature_id
                    WHERE t.term_name = :term_name
                      AND t.rate_package_id = :rate_package_id
                    ORDER BY CAST(c.cell_key_text AS INTEGER)
                    """
                ),
                {"term_name": OFFSET_FACTOR_NAME, "rate_package_id": rate_package_id},
            )
            .mappings()
            .all()
        )
        compiled_rows = (
            con.execute(
                text(
                    """
                    SELECT
                        term_name,
                        term_type,
                        cell_key_text,
                        multiplier,
                        log_coefficient
                    FROM pricing.PRICING_COMPILED_RATE_CELL
                    WHERE term_name = :term_name
                      AND rate_package_id = :rate_package_id
                    ORDER BY CAST(cell_key_text AS INTEGER)
                    """
                ),
                {"term_name": OFFSET_FACTOR_NAME, "rate_package_id": rate_package_id},
            )
            .mappings()
            .all()
        )

    return {
        "staging_offset_rows": [dict(row) for row in staging_rows],
        "final_offset_rows": [dict(row) for row in final_rows],
        "compiled_offset_rows": [dict(row) for row in compiled_rows],
    }


def run_offset_export_smoke(
    *,
    db_root: str | Path = DEFAULT_DB_ROOT,
    effective_from: str = "2026-06-19",
    created_by: str = "offset_export_smoke",
    reset: bool = False,
) -> dict[str, Any]:
    root = Path(db_root)
    db_paths = {schema: root / file_name for schema, file_name in SCHEMA_DB_FILES.items()}
    artifact_root = root / "artifacts"
    if reset:
        for db_path in db_paths.values():
            if db_path.exists():
                db_path.unlink()
        if artifact_root.exists():
            shutil.rmtree(artifact_root)

    engine = sqlite_engine_with_offline_schemas(db_paths)
    apply_offline_ddl(engine)
    model_id = register_smoke_model(engine, created_by=created_by)

    export_id = f"{MODEL_NAME}__{effective_from.replace('-', '')}"
    workbook_path = artifact_root / "rating_tables" / f"{export_id}.xlsx"
    frame = offset_smoke_frame()
    workbook_path, receipt_path, receipt_sha256 = fit_and_export_offset_workbook(
        frame,
        output_path=workbook_path,
    )

    stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id=export_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        effective_from=effective_from,
        target_name=TARGET_NAME,
        model_type=MODEL_TYPE,
        created_by=created_by,
        replace=True,
        model_id=model_id,
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=receipt_sha256,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
    )
    completed_build = {
        "export_id": export_id,
        "model_version": MODEL_VERSION,
        "effective_from": effective_from,
        "rating_workbook_path": str(workbook_path),
        "model_artifact_path": None,
        "manifest_id": f"{export_id}__manifest",
        "split_set_id": None,
    }
    package = publish_offline_rating_package(
        engine,
        completed_build=completed_build,
        model_id=model_id,
        created_by=created_by,
    )

    rows = query_offset_rows(
        engine,
        export_id=export_id,
        rate_package_id=package["rate_package_id"],
    )
    return {
        "db_paths": {schema: str(path) for schema, path in db_paths.items()},
        "artifact_root": str(artifact_root),
        "workbook_path": str(workbook_path),
        "publication_receipt_path": str(receipt_path),
        "publication_receipt_sha256": receipt_sha256,
        "export_id": export_id,
        "model_id": model_id,
        "rate_package_id": package["rate_package_id"],
        "package_version": package["package_version"],
        "tables": table_counts(engine),
        **rows,
    }


def main() -> None:
    args = parse_args()
    result = run_offset_export_smoke(
        db_root=args.db_root,
        effective_from=args.effective_from,
        created_by=args.created_by,
        reset=args.reset,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
