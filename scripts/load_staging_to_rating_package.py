"""Convert staged SuperGLM rating export into normalized pricing tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_engine  # noqa: E402
from pricing_pipeline.model_registry import ensure_pricing_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--export-id", required=True)
    p.add_argument("--created-by", default="python")
    p.add_argument("--package-status", default="DRAFT")
    p.add_argument("--set-pointer", default=None)
    return p.parse_args()


def load_staging_to_rating_package(engine, args: argparse.Namespace) -> int:
    with engine.begin() as con:
        meta = con.execute(text("""
            SELECT
                export_id,
                model_id,
                model_name,
                model_version,
                base_rate,
                effective_from_date,
                effective_to_date
            FROM pricing_stg.STG_RATING_EXPORT
            WHERE export_id = :export_id
        """), {"export_id": args.export_id}).mappings().one()

        model_id = meta["model_id"]
        if model_id is None:
            model_id = ensure_pricing_model(
                con,
                model_key=meta["model_name"],
                target_name="ClaimNb",
                model_type="superglm_poisson",
                created_by=args.created_by,
            )
            con.execute(text("""
                UPDATE pricing_stg.STG_RATING_EXPORT
                SET model_id = :model_id
                WHERE export_id = :export_id
            """), {"model_id": model_id, "export_id": args.export_id})

        package_version = con.execute(text("""
            SELECT ISNULL(MAX(package_version), 0) + 1
            FROM pricing.PRICING_RATE_PACKAGE
            WHERE model_id = :model_id
        """), {"model_id": model_id}).scalar_one()

        rate_package_id = con.execute(text("""
            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                parent_rate_package_id,
                model_id,
                model_name,
                model_version,
                package_version,
                base_rate,
                effective_from_date,
                effective_to_date,
                package_status,
                created_by
            )
            OUTPUT INSERTED.rate_package_id
            VALUES (
                NULL,
                :model_id,
                :model_name,
                :model_version,
                :package_version,
                :base_rate,
                :effective_from_date,
                :effective_to_date,
                :package_status,
                :created_by
            )
        """), {
            "model_id": model_id,
            "model_name": meta["model_name"],
            "model_version": meta["model_version"],
            "package_version": package_version,
            "base_rate": meta["base_rate"],
            "effective_from_date": meta["effective_from_date"],
            "effective_to_date": meta["effective_to_date"],
            "package_status": args.package_status,
            "created_by": args.created_by,
        }).scalar_one()

        # Features
        con.execute(text("""
            INSERT INTO pricing.PRICING_FEATURE (
                feature_name,
                feature_value_type,
                is_ordered
            )
            SELECT DISTINCT
                s.feature_name,
                s.feature_value_type,
                CASE WHEN s.level_set_type IN ('NUMERIC_BAND', 'SPLINE_GRID_1D') THEN 1 ELSE 0 END
            FROM pricing_stg.STG_CELL_LEVEL s
            WHERE s.export_id = :export_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_FEATURE f
                  WHERE f.feature_name = s.feature_name
              );
        """), {"export_id": args.export_id})

        # Level sets
        con.execute(text("""
            INSERT INTO pricing.PRICING_FEATURE_LEVEL_SET (
                feature_id,
                model_id,
                level_set_name,
                level_set_type,
                binning_strategy,
                grid_width
            )
            SELECT DISTINCT
                f.feature_id,
                :model_id,
                s.level_set_name,
                s.level_set_type,
                CASE
                    WHEN s.level_set_type = 'SPLINE_GRID_1D' THEN 'SPLINE_EVAL_GRID'
                    WHEN s.level_set_type = 'NUMERIC_BAND' THEN 'EXPLICIT_BANDS'
                    ELSE 'EXPLICIT_LEVELS'
                END,
                NULL
            FROM pricing_stg.STG_CELL_LEVEL s
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            WHERE s.export_id = :export_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_FEATURE_LEVEL_SET ls
                  WHERE ls.model_id = :model_id
                    AND ls.feature_id = f.feature_id
                    AND ls.level_set_name = s.level_set_name
              );
        """), {"export_id": args.export_id, "model_id": model_id})

        # Levels
        con.execute(text("""
            INSERT INTO pricing.PRICING_FEATURE_LEVEL (
                level_set_id,
                level_code,
                level_label,
                order_index,
                lower_bound,
                upper_bound,
                representative_value,
                is_missing,
                is_other
            )
            SELECT DISTINCT
                ls.level_set_id,
                s.level_code,
                s.level_label,
                s.order_index,
                s.lower_bound,
                s.upper_bound,
                s.representative_value,
                s.is_missing,
                s.is_other
            FROM pricing_stg.STG_CELL_LEVEL s
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.model_id = :model_id
             AND ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            WHERE s.export_id = :export_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_FEATURE_LEVEL fl
                  WHERE fl.level_set_id = ls.level_set_id
                    AND fl.level_code = s.level_code
              )
            ORDER BY
                ls.level_set_id,
                s.order_index,
                s.lower_bound,
                s.upper_bound,
                s.level_code;
        """), {"export_id": args.export_id, "model_id": model_id})

        # Terms
        con.execute(text("""
            INSERT INTO pricing.PRICING_TERM (
                rate_package_id,
                term_name,
                term_type,
                sequence_no
            )
            SELECT DISTINCT
                :rate_package_id,
                c.term_name,
                c.term_type,
                c.sequence_no
            FROM pricing_stg.STG_RATE_CELL c
            WHERE c.export_id = :export_id;
        """), {"export_id": args.export_id, "rate_package_id": rate_package_id})

        # Term features
        con.execute(text("""
            INSERT INTO pricing.PRICING_TERM_FEATURE (
                term_id,
                position_no,
                feature_id,
                level_set_id,
                input_column_name
            )
            SELECT DISTINCT
                t.term_id,
                s.position_no,
                f.feature_id,
                ls.level_set_id,
                s.feature_name
            FROM pricing_stg.STG_CELL_LEVEL s
            JOIN pricing_stg.STG_RATE_CELL c
              ON c.export_id = s.export_id
             AND c.row_id = s.row_id
            JOIN pricing.PRICING_TERM t
              ON t.rate_package_id = :rate_package_id
             AND t.term_name = c.term_name
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.model_id = :model_id
             AND ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            WHERE s.export_id = :export_id;
        """), {
            "export_id": args.export_id,
            "rate_package_id": rate_package_id,
            "model_id": model_id,
        })

        # Cells
        con.execute(text("""
            INSERT INTO pricing.PRICING_RATE_CELL (
                term_id,
                cell_key_text,
                cell_key_digest,
                multiplier,
                log_coefficient,
                exposure_weight,
                record_count,
                is_reference,
                is_default
            )
            SELECT
                t.term_id,
                c.cell_key_text,
                HASHBYTES('SHA2_256', c.cell_key_text),
                c.multiplier,
                c.log_coefficient,
                c.exposure_weight,
                c.record_count,
                c.is_reference,
                c.is_default
            FROM pricing_stg.STG_RATE_CELL c
            JOIN pricing.PRICING_TERM t
              ON t.rate_package_id = :rate_package_id
             AND t.term_name = c.term_name
            WHERE c.export_id = :export_id;
        """), {"export_id": args.export_id, "rate_package_id": rate_package_id})

        # Cell-level mapping
        con.execute(text("""
            INSERT INTO pricing.PRICING_RATE_CELL_LEVEL (
                cell_id,
                position_no,
                feature_level_id
            )
            SELECT
                rc.cell_id,
                s.position_no,
                fl.feature_level_id
            FROM pricing_stg.STG_CELL_LEVEL s
            JOIN pricing_stg.STG_RATE_CELL c
              ON c.export_id = s.export_id
             AND c.row_id = s.row_id
            JOIN pricing.PRICING_TERM t
              ON t.rate_package_id = :rate_package_id
             AND t.term_name = c.term_name
            JOIN pricing.PRICING_RATE_CELL rc
              ON rc.term_id = t.term_id
             AND rc.cell_key_text = c.cell_key_text
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.model_id = :model_id
             AND ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            JOIN pricing.PRICING_FEATURE_LEVEL fl
              ON fl.level_set_id = ls.level_set_id
             AND fl.level_code = s.level_code
            WHERE s.export_id = :export_id;
        """), {
            "export_id": args.export_id,
            "rate_package_id": rate_package_id,
            "model_id": model_id,
        })

        # Minimal compile step: flat rate cells
        con.execute(text("""
            INSERT INTO pricing.PRICING_COMPILED_RATE_CELL (
                rate_package_id,
                term_id,
                cell_key_digest,
                term_name,
                term_type,
                sequence_no,
                cell_key_text,
                multiplier,
                log_coefficient,
                exposure_weight,
                record_count,
                is_default,
                is_reference
            )
            SELECT
                :rate_package_id,
                t.term_id,
                c.cell_key_digest,
                t.term_name,
                t.term_type,
                t.sequence_no,
                c.cell_key_text,
                c.multiplier,
                c.log_coefficient,
                c.exposure_weight,
                c.record_count,
                c.is_default,
                c.is_reference
            FROM pricing.PRICING_TERM t
            JOIN pricing.PRICING_RATE_CELL c
              ON c.term_id = t.term_id
            WHERE t.rate_package_id = :rate_package_id
              AND c.is_deleted = 0;
        """), {"rate_package_id": rate_package_id})

        # Compile 1D bands for spline/numeric-band terms
        con.execute(text("""
            INSERT INTO pricing.PRICING_COMPILED_1D_RATE_BAND (
                rate_package_id,
                term_id,
                feature_level_id,
                term_name,
                feature_name,
                level_code,
                sort_order,
                lower_bound,
                upper_bound,
                representative_value,
                multiplier,
                log_coefficient
            )
            SELECT
                :rate_package_id,
                t.term_id,
                fl.feature_level_id,
                t.term_name,
                f.feature_name,
                fl.level_code,
                COALESCE(fl.order_index, 0),
                fl.lower_bound,
                fl.upper_bound,
                fl.representative_value,
                rc.multiplier,
                rc.log_coefficient
            FROM pricing.PRICING_TERM t
            JOIN pricing.PRICING_RATE_CELL rc
              ON rc.term_id = t.term_id
            JOIN pricing.PRICING_RATE_CELL_LEVEL rcl
              ON rcl.cell_id = rc.cell_id
             AND rcl.position_no = 1
            JOIN pricing.PRICING_FEATURE_LEVEL fl
              ON fl.feature_level_id = rcl.feature_level_id
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.level_set_id = fl.level_set_id
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_id = ls.feature_id
            WHERE t.rate_package_id = :rate_package_id
              AND t.term_type IN ('DISCRETIZED_SPLINE_1D', 'NUMERIC_BANDED_1D')
              AND rc.is_deleted = 0
            ORDER BY
                t.sequence_no,
                COALESCE(fl.order_index, 0),
                fl.lower_bound,
                fl.upper_bound,
                fl.level_code;
        """), {"rate_package_id": rate_package_id})

        if args.set_pointer:
            con.execute(text("""
                MERGE pricing.PRICING_PACKAGE_POINTER AS tgt
                USING (
                    SELECT
                        :model_id AS model_id,
                        :pointer_name AS pointer_name,
                        :rate_package_id AS rate_package_id,
                        :updated_by AS updated_by
                ) AS src
                ON tgt.model_id = src.model_id
                   AND tgt.pointer_name = src.pointer_name
                WHEN MATCHED THEN
                    UPDATE SET
                        rate_package_id = src.rate_package_id,
                        updated_ts = SYSUTCDATETIME(),
                        updated_by = src.updated_by
                WHEN NOT MATCHED THEN
                    INSERT (model_id, pointer_name, rate_package_id, updated_by)
                    VALUES (src.model_id, src.pointer_name, src.rate_package_id, src.updated_by);
            """), {
                "model_id": model_id,
                "pointer_name": args.set_pointer,
                "rate_package_id": rate_package_id,
                "updated_by": args.created_by,
            })
            con.execute(text("""
                UPDATE pricing.PRICING_MODEL_DEPLOYMENT
                SET effective_to_ts = SYSUTCDATETIME()
                WHERE model_id = :model_id
                  AND deployment_slot = :deployment_slot
                  AND effective_to_ts IS NULL;
            """), {
                "model_id": model_id,
                "deployment_slot": args.set_pointer,
            })
            con.execute(text("""
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    model_id,
                    rate_package_id,
                    deployment_slot,
                    deployed_by
                )
                VALUES (
                    :model_id,
                    :rate_package_id,
                    :deployment_slot,
                    :deployed_by
                );
            """), {
                "model_id": model_id,
                "rate_package_id": rate_package_id,
                "deployment_slot": args.set_pointer,
                "deployed_by": args.created_by,
            })

        args.package_version = package_version

    return int(rate_package_id)


def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> int:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=pointer_name,
    )
    return load_staging_to_rating_package(engine, args)


def main() -> None:
    args = parse_args()
    engine = get_engine()
    rate_package_id = load_staging_to_rating_package(engine, args)

    print(f"rate_package_id={rate_package_id}")
    print(f"package_version={args.package_version}")
    if args.set_pointer:
        print(f"pointer={args.set_pointer}")


if __name__ == "__main__":
    main()
