"""Convert staged SuperGLM rating export into normalized pricing tables."""

from __future__ import annotations

import argparse

from sqlalchemy import text

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.model_registry import ModelRegistryError


def _identity_text(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _existing_export_conflicts(existing_package, meta) -> list[str]:
    conflicts: list[str] = []
    for field_name in (
        "model_version",
        "effective_from_date",
        "effective_to_date",
        "source_file",
        "publication_receipt_sha256",
    ):
        existing_value = _identity_text(existing_package[field_name])
        staged_value = _identity_text(meta[field_name])
        if field_name == "source_file" and (existing_value is None or staged_value is None):
            continue
        if existing_value != staged_value:
            conflicts.append(
                f"{field_name} existing={existing_package[field_name]!r} "
                f"staged={meta[field_name]!r}"
            )
    return conflicts


def load_staging_to_rating_package(engine, args: argparse.Namespace) -> int:
    if getattr(args, "set_pointer", None):
        raise ValueError(
            "package publish no longer deploys rate packages; publish the package "
            "first, then deploy it with ModelPublisher.deploy or the deploy DAG"
        )

    with engine.begin() as con:
        args.was_existing = False
        requested_package_status = args.package_status
        meta = (
            con.execute(
                text("""
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
        """),
                {"export_id": args.export_id},
            )
            .mappings()
            .one()
        )

        model_id = meta["model_id"]
        if model_id is None:
            raise ModelRegistryError(
                "staged rating export is missing model_id; validate/register the "
                f"model before staging export_id={args.export_id!r}"
            )

        existing_package = (
            con.execute(
                text("""
            SELECT
                rate_package_id,
                package_version,
                model_id,
                model_name,
                model_version,
                effective_from_date,
                effective_to_date,
                package_status,
                source_export_id,
                source_file,
                publication_receipt_sha256
            FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
            WHERE model_id = :model_id
              AND source_export_id = :export_id
        """),
                {
                    "model_id": model_id,
                    "export_id": args.export_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if existing_package is not None:
            conflicts = _existing_export_conflicts(existing_package, meta)
            if conflicts:
                raise ValueError(
                    f"export_id {args.export_id!r} is already published with "
                    "incompatible metadata: " + "; ".join(conflicts)
                )
            args.package_version = int(existing_package["package_version"])
            args.package_status = str(existing_package["package_status"])
            args.was_existing = True
            return int(existing_package["rate_package_id"])

        offset_handling = meta["offset_handling"] or "UNKNOWN"
        offset_factor_name = meta["offset_factor_name"]
        if offset_handling == "EXPORTED_FACTOR":
            if not offset_factor_name:
                raise ValueError(
                    "staged export declares EXPORTED_FACTOR offset handling but "
                    "does not include offset_factor_name"
                )
            offset_factor_exists = con.execute(
                text("""
                SELECT TOP 1 1
                FROM pricing_stg.STG_RATE_CELL
                WHERE export_id = :export_id
                  AND term_name = :offset_factor_name
                  AND term_type = 'OFFSET_FACTOR'
            """),
                {
                    "export_id": args.export_id,
                    "offset_factor_name": offset_factor_name,
                },
            ).scalar_one_or_none()
            if offset_factor_exists is None:
                raise ValueError(
                    "staged export declares EXPORTED_FACTOR offset handling but "
                    f"has no OFFSET_FACTOR term named {offset_factor_name!r}"
                )
        elif offset_handling == "ALREADY_APPLIED_SQL_EXPOSURE":
            staged_offset_factor = con.execute(
                text("""
                SELECT TOP 1 1
                FROM pricing_stg.STG_RATE_CELL
                WHERE export_id = :export_id
                  AND term_type = 'OFFSET_FACTOR'
            """),
                {"export_id": args.export_id},
            ).scalar_one_or_none()
            if staged_offset_factor is not None:
                raise ValueError(
                    "staged export declares ALREADY_APPLIED_SQL_EXPOSURE but "
                    "also contains an OFFSET_FACTOR term"
                )

        package_version = con.execute(
            text("""
            SELECT ISNULL(MAX(package_version), 0) + 1
            FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
            WHERE model_id = :model_id
        """),
            {"model_id": model_id},
        ).scalar_one()

        rate_package_id = con.execute(
            text("""
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
                :created_by
            )
        """),
            {
                "model_id": model_id,
                "model_name": meta["model_name"],
                "model_version": meta["model_version"],
                "package_version": package_version,
                "base_rate": meta["base_rate"],
                "effective_from_date": meta["effective_from_date"],
                "effective_to_date": meta["effective_to_date"],
                "package_status": "DRAFT",
                "source_export_id": args.export_id,
                "source_file": meta["source_file"],
                "publication_receipt_json": meta["publication_receipt_json"],
                "publication_receipt_sha256": meta["publication_receipt_sha256"],
                "package_metadata_json": meta["package_metadata_json"],
                "offset_handling": offset_handling,
                "offset_factor_name": offset_factor_name,
                "offset_source_name": meta["offset_source_name"],
                "offset_label": meta["offset_label"],
                "metadata_origin": meta["metadata_origin"],
                "created_by": args.created_by,
            },
        ).scalar_one()

        # Features
        con.execute(
            text("""
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
        """),
            {"export_id": args.export_id},
        )

        # Level sets
        con.execute(
            text("""
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
        """),
            {"export_id": args.export_id, "model_id": model_id},
        )

        # Levels
        con.execute(
            text("""
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
        """),
            {"export_id": args.export_id, "model_id": model_id},
        )

        # Terms
        con.execute(
            text("""
            INSERT INTO pricing.PRICING_TERM (
                rate_package_id,
                term_name,
                term_type,
                sequence_no,
                term_metadata_json
            )
            SELECT DISTINCT
                :rate_package_id,
                c.term_name,
                c.term_type,
                c.sequence_no,
                tm.term_metadata_json
            FROM pricing_stg.STG_RATE_CELL c
            LEFT JOIN pricing_stg.STG_TERM_METADATA tm
              ON tm.export_id = c.export_id
             AND tm.term_name = c.term_name
            WHERE c.export_id = :export_id;
        """),
            {"export_id": args.export_id, "rate_package_id": rate_package_id},
        )

        # Term features
        con.execute(
            text("""
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
        """),
            {
                "export_id": args.export_id,
                "rate_package_id": rate_package_id,
                "model_id": model_id,
            },
        )

        # Cells
        con.execute(
            text("""
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
        """),
            {"export_id": args.export_id, "rate_package_id": rate_package_id},
        )

        # Cell-level mapping
        con.execute(
            text("""
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
        """),
            {
                "export_id": args.export_id,
                "rate_package_id": rate_package_id,
                "model_id": model_id,
            },
        )

        # Minimal compile step: flat rate cells
        con.execute(
            text("""
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
        """),
            {"rate_package_id": rate_package_id},
        )

        # Compile 1D bands for spline/numeric-band terms
        con.execute(
            text("""
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
        """),
            {"rate_package_id": rate_package_id},
        )

        con.execute(
            text("""
            UPDATE pricing.PRICING_RATE_PACKAGE
            SET package_status = :package_status
            WHERE rate_package_id = :rate_package_id;
        """),
            {
                "package_status": requested_package_status,
                "rate_package_id": rate_package_id,
            },
        )

        args.package_version = package_version

    return int(rate_package_id)


def publish_rating_package(
    engine,
    *,
    export_id: str,
    created_by: str = "python",
    package_status: str = "PUBLISHED",
) -> PublishResult:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=None,
    )
    rate_package_id = load_staging_to_rating_package(engine, args)
    return PublishResult(
        mlflow_run_id="",
        export_id=export_id,
        rate_package_id=int(rate_package_id),
        package_version=int(args.package_version),
        rating_workbook_path="",
        package_status=str(args.package_status),
        was_existing=bool(getattr(args, "was_existing", False)),
    )
