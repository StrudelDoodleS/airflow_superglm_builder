"""Convert staged SuperGLM rating export into normalized pricing tables."""
from __future__ import annotations

import argparse

from sqlalchemy import text

from pricing_db import get_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--export-id", required=True)
    p.add_argument("--created-by", default="python")
    p.add_argument("--package-status", default="DRAFT")
    p.add_argument("--set-pointer", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = get_engine()

    with engine.begin() as con:
        meta = con.execute(text("""
            SELECT
                export_id,
                model_name,
                model_version,
                base_rate,
                effective_from_date,
                effective_to_date
            FROM pricing.STG_RATING_EXPORT
            WHERE export_id = :export_id
        """), {"export_id": args.export_id}).mappings().one()

        package_version = con.execute(text("""
            SELECT ISNULL(MAX(package_version), 0) + 1
            FROM pricing.PRICING_RATE_PACKAGE
            WHERE model_name = :model_name
        """), {"model_name": meta["model_name"]}).scalar_one()

        rate_package_id = con.execute(text("""
            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                parent_rate_package_id,
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
            FROM pricing.STG_CELL_LEVEL s
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
                level_set_name,
                level_set_type,
                binning_strategy,
                grid_width
            )
            SELECT DISTINCT
                f.feature_id,
                s.level_set_name,
                s.level_set_type,
                CASE
                    WHEN s.level_set_type = 'SPLINE_GRID_1D' THEN 'SPLINE_EVAL_GRID'
                    WHEN s.level_set_type = 'NUMERIC_BAND' THEN 'EXPLICIT_BANDS'
                    ELSE 'EXPLICIT_LEVELS'
                END,
                NULL
            FROM pricing.STG_CELL_LEVEL s
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            WHERE s.export_id = :export_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_FEATURE_LEVEL_SET ls
                  WHERE ls.feature_id = f.feature_id
                    AND ls.level_set_name = s.level_set_name
              );
        """), {"export_id": args.export_id})

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
            FROM pricing.STG_CELL_LEVEL s
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            WHERE s.export_id = :export_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM pricing.PRICING_FEATURE_LEVEL fl
                  WHERE fl.level_set_id = ls.level_set_id
                    AND fl.level_code = s.level_code
              );
        """), {"export_id": args.export_id})

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
            FROM pricing.STG_RATE_CELL c
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
            FROM pricing.STG_CELL_LEVEL s
            JOIN pricing.STG_RATE_CELL c
              ON c.export_id = s.export_id
             AND c.row_id = s.row_id
            JOIN pricing.PRICING_TERM t
              ON t.rate_package_id = :rate_package_id
             AND t.term_name = c.term_name
            JOIN pricing.PRICING_FEATURE f
              ON f.feature_name = s.feature_name
            JOIN pricing.PRICING_FEATURE_LEVEL_SET ls
              ON ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            WHERE s.export_id = :export_id;
        """), {"export_id": args.export_id, "rate_package_id": rate_package_id})

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
            FROM pricing.STG_RATE_CELL c
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
            FROM pricing.STG_CELL_LEVEL s
            JOIN pricing.STG_RATE_CELL c
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
              ON ls.feature_id = f.feature_id
             AND ls.level_set_name = s.level_set_name
            JOIN pricing.PRICING_FEATURE_LEVEL fl
              ON fl.level_set_id = ls.level_set_id
             AND fl.level_code = s.level_code
            WHERE s.export_id = :export_id;
        """), {"export_id": args.export_id, "rate_package_id": rate_package_id})

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
              AND rc.is_deleted = 0;
        """), {"rate_package_id": rate_package_id})

        if args.set_pointer:
            con.execute(text("""
                MERGE pricing.PRICING_PACKAGE_POINTER AS tgt
                USING (
                    SELECT
                        :pointer_name AS pointer_name,
                        :rate_package_id AS rate_package_id,
                        :updated_by AS updated_by
                ) AS src
                ON tgt.pointer_name = src.pointer_name
                WHEN MATCHED THEN
                    UPDATE SET
                        rate_package_id = src.rate_package_id,
                        updated_ts = SYSUTCDATETIME(),
                        updated_by = src.updated_by
                WHEN NOT MATCHED THEN
                    INSERT (pointer_name, rate_package_id, updated_by)
                    VALUES (src.pointer_name, src.rate_package_id, src.updated_by);
            """), {
                "pointer_name": args.set_pointer,
                "rate_package_id": rate_package_id,
                "updated_by": args.created_by,
            })

    print(f"rate_package_id={rate_package_id}")
    print(f"package_version={package_version}")
    if args.set_pointer:
        print(f"pointer={args.set_pointer}")


if __name__ == "__main__":
    main()
