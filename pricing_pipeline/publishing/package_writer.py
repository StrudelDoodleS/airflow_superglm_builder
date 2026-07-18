"""Convert staged SuperGLM rating export into normalized pricing tables."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.model_registry import ModelRegistryError
from pricing_pipeline.publishing.staging_lock import acquire_staging_export_lock


_STAGED_IDENTITY_FIELDS = (
    "export_id",
    "model_id",
    "model_name",
    "model_version",
    "effective_from_date",
    "effective_to_date",
    "source_file",
    "publication_receipt_sha256",
    "staging_content_sha256",
)


def _identity_text(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _existing_export_conflicts(
    existing_package,
    meta,
    *,
    parent_rate_package_id: int | None,
    revision_metadata_json: str | None,
    build_fingerprint_sha256: str | None,
) -> list[str]:
    conflicts: list[str] = []
    for field_name in (
        "model_version",
        "effective_from_date",
        "effective_to_date",
        "source_file",
        "publication_receipt_sha256",
        "staging_content_sha256",
    ):
        existing_value = _identity_text(existing_package[field_name])
        staged_value = _identity_text(meta[field_name])
        if field_name == "source_file" and (existing_value is None or staged_value is None):
            continue
        if field_name == "staging_content_sha256" and existing_value is None:
            # V028 intentionally left pre-migration packages without a digest.
            # Their remaining immutable metadata and model-run lineage still
            # provide the compatibility check available when they were written.
            continue
        if existing_value != staged_value:
            conflicts.append(
                f"{field_name} existing={existing_package[field_name]!r} "
                f"staged={meta[field_name]!r}"
            )
    requested_identity = {
        "parent_rate_package_id": parent_rate_package_id,
        "revision_metadata_json": revision_metadata_json,
        "build_fingerprint_sha256": build_fingerprint_sha256,
    }
    for field_name, requested_value in requested_identity.items():
        existing_value = existing_package.get(field_name)
        if _identity_text(existing_value) != _identity_text(requested_value):
            conflicts.append(
                f"{field_name} existing={existing_value!r} requested={requested_value!r}"
            )
    return conflicts


def _staged_export_conflicts(
    meta,
    expected: Mapping[str, object] | None,
) -> list[str]:
    if expected is None:
        return []
    unknown_fields = set(expected) - set(_STAGED_IDENTITY_FIELDS)
    if unknown_fields:
        raise ValueError(
            "expected_staged_metadata contains unsupported fields: "
            + ", ".join(sorted(unknown_fields))
        )

    conflicts: list[str] = []
    for field_name in _STAGED_IDENTITY_FIELDS:
        if field_name not in expected:
            continue
        staged_value = meta[field_name]
        expected_value = expected[field_name]
        if _identity_text(staged_value) != _identity_text(expected_value):
            conflicts.append(f"{field_name} expected={expected_value!r} staged={staged_value!r}")
    return conflicts


def _canonical_revision_metadata(value: Mapping[str, object] | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("revision_metadata must be a mapping")

    def normalise(item: object) -> object:
        if isinstance(item, Mapping):
            normalised: dict[str, object] = {}
            for key, nested_value in item.items():
                if not isinstance(key, str):
                    raise ValueError("revision_metadata keys must be strings")
                normalised[key] = normalise(nested_value)
            return normalised
        if isinstance(item, list | tuple):
            return [normalise(nested_value) for nested_value in item]
        return item

    normalised_value = normalise(value)
    try:
        return json.dumps(
            normalised_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("revision_metadata must contain only finite numbers") from exc
    except TypeError as exc:
        raise ValueError("revision_metadata must contain only JSON-serializable values") from exc


def publish_rating_package(
    engine,
    *,
    export_id: str,
    expected_database: str,
    created_by: str = "python",
    parent_rate_package_id: int | None = None,
    revision_metadata: Mapping[str, object] | None = None,
    draft_validator=None,
    package_lineage_writer: Callable[[Connection, int], int | None] | None = None,
    expected_staged_metadata: Mapping[str, object] | None = None,
    build_fingerprint_sha256: str | None = None,
) -> PublishResult:
    revision_metadata_json = _canonical_revision_metadata(revision_metadata)
    if draft_validator is not None and not callable(draft_validator):
        raise TypeError("draft_validator must be callable")
    if package_lineage_writer is not None and not callable(package_lineage_writer):
        raise TypeError("package_lineage_writer must be callable")
    model_run_id: int | None = None
    with engine.begin() as con:
        _verify_expected_database(con, expected_database)
        acquire_staging_export_lock(con, export_id)
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
                metadata_origin,
                staging_content_sha256
            FROM pricing_stg.STG_RATING_EXPORT
            WHERE export_id = :export_id
        """),
                {"export_id": export_id},
            )
            .mappings()
            .one()
        )

        staged_conflicts = _staged_export_conflicts(
            meta,
            expected_staged_metadata,
        )
        if staged_conflicts:
            raise ValueError(
                f"staged export changed before package publication for "
                f"export_id={export_id!r}: " + "; ".join(staged_conflicts)
            )

        model_id = meta["model_id"]
        if model_id is None:
            raise ModelRegistryError(
                "staged rating export is missing model_id; validate/register the "
                f"model before staging export_id={export_id!r}"
            )

        locked_model_id = con.execute(
            text(
                """
                SELECT pm.model_id
                FROM pricing.PRICING_MODEL AS pm WITH (UPDLOCK, HOLDLOCK)
                WHERE pm.model_id = :model_id
                """
            ),
            {"model_id": model_id},
        ).scalar_one_or_none()
        if locked_model_id is None:
            raise ModelRegistryError(
                f"staged rating export refers to unregistered model_id={model_id!r}"
            )

        if parent_rate_package_id is None:
            fingerprint = str(build_fingerprint_sha256 or "").strip()
            if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise ValueError(
                    "root package publication requires build_fingerprint_sha256 "
                    "as a lowercase SHA-256 digest"
                )
            build_fingerprint_sha256 = fingerprint
            canonical_root = (
                con.execute(
                    text(
                        """
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
                            publication_receipt_sha256,
                            staging_content_sha256,
                            parent_rate_package_id,
                            revision_metadata_json,
                            build_fingerprint_sha256
                        FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
                        WHERE model_id = :model_id
                          AND parent_rate_package_id IS NULL
                          AND build_fingerprint_sha256 = :build_fingerprint_sha256
                        """
                    ),
                    {
                        "model_id": model_id,
                        "build_fingerprint_sha256": build_fingerprint_sha256,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if canonical_root is not None:
                if _identity_text(canonical_root["model_version"]) != _identity_text(
                    meta["model_version"]
                ):
                    raise ValueError(
                        "canonical root model_version does not match staged model_version: "
                        f"canonical={canonical_root['model_version']!r}, "
                        f"staged={meta['model_version']!r}"
                    )
                return PublishResult(
                    mlflow_run_id="",
                    export_id=str(canonical_root["source_export_id"]),
                    rate_package_id=int(canonical_root["rate_package_id"]),
                    package_version=int(canonical_root["package_version"]),
                    rating_workbook_path="",
                    package_status=str(canonical_root["package_status"]),
                    was_existing=True,
                )
        elif build_fingerprint_sha256 is not None:
            raise ValueError("edited child packages cannot carry a root build fingerprint")

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
                publication_receipt_sha256,
                staging_content_sha256,
                parent_rate_package_id,
                revision_metadata_json,
                build_fingerprint_sha256
            FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
            WHERE model_id = :model_id
              AND source_export_id = :export_id
        """),
                {
                    "model_id": model_id,
                    "export_id": export_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if existing_package is not None:
            conflicts = _existing_export_conflicts(
                existing_package,
                meta,
                parent_rate_package_id=parent_rate_package_id,
                revision_metadata_json=revision_metadata_json,
                build_fingerprint_sha256=build_fingerprint_sha256,
            )
            if conflicts:
                raise ValueError(
                    f"export_id {export_id!r} is already published with "
                    "incompatible metadata: " + "; ".join(conflicts)
                )
            return PublishResult(
                mlflow_run_id="",
                export_id=export_id,
                rate_package_id=int(existing_package["rate_package_id"]),
                package_version=int(existing_package["package_version"]),
                rating_workbook_path="",
                package_status=str(existing_package["package_status"]),
                was_existing=True,
            )

        if parent_rate_package_id is not None:
            parent = (
                con.execute(
                    text("""
                    SELECT
                        rate_package_id,
                        model_id,
                        model_version,
                        effective_from_date,
                        effective_to_date,
                        package_status
                    FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
                    WHERE rate_package_id = :parent_rate_package_id
                    """),
                    {"parent_rate_package_id": parent_rate_package_id},
                )
                .mappings()
                .one_or_none()
            )
            if parent is None:
                raise ValueError(f"parent rate package {parent_rate_package_id} does not exist")
            if int(parent["model_id"]) != int(model_id):
                raise ValueError("parent rate package belongs to a different model")
            if str(parent["package_status"]) != "PUBLISHED":
                raise ValueError("parent rate package must have PUBLISHED status")
            for field_name in (
                "model_version",
                "effective_from_date",
                "effective_to_date",
            ):
                if _identity_text(parent[field_name]) != _identity_text(meta[field_name]):
                    raise ValueError(
                        f"parent {field_name}={parent[field_name]!r} does not match "
                        f"staged {field_name}={meta[field_name]!r}"
                    )

        if parent_rate_package_id is None:
            staged_version = _identity_text(meta["model_version"])
            if staged_version is None:
                raise ValueError("root package publication requires model_version")
            reservation = (
                con.execute(
                    text("""
                    SELECT
                        model_id,
                        export_id,
                        model_version
                    FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE model_id = :model_id
                      AND export_id = :export_id
                    """),
                    {"model_id": model_id, "export_id": export_id},
                )
                .mappings()
                .one_or_none()
            )
            if reservation is None:
                con.execute(
                    text("""
                    INSERT INTO pricing.PRICING_MODEL_VERSION_RESERVATION (
                        model_id,
                        export_id,
                        model_version
                    ) VALUES (
                        :model_id,
                        :export_id,
                        :model_version
                    )
                    """),
                    {
                        "model_id": model_id,
                        "export_id": export_id,
                        "model_version": staged_version,
                    },
                )
            else:
                reserved_version = _identity_text(reservation["model_version"])
                if reserved_version != staged_version:
                    raise ValueError(
                        f"reserved model_version {reserved_version!r} does not match "
                        f"staged model_version {staged_version!r} for "
                        f"export_id={export_id!r}"
                    )

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
                    "export_id": export_id,
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
                {"export_id": export_id},
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
                staging_content_sha256,
                package_metadata_json,
                revision_metadata_json,
                offset_handling,
                offset_factor_name,
                offset_source_name,
                offset_label,
                metadata_origin,
                build_fingerprint_sha256,
                created_by
            )
            OUTPUT INSERTED.rate_package_id
            VALUES (
                :parent_rate_package_id,
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
                :staging_content_sha256,
                :package_metadata_json,
                :revision_metadata_json,
                :offset_handling,
                :offset_factor_name,
                :offset_source_name,
                :offset_label,
                :metadata_origin,
                :build_fingerprint_sha256,
                :created_by
            )
        """),
            {
                "model_id": model_id,
                "parent_rate_package_id": parent_rate_package_id,
                "model_name": meta["model_name"],
                "model_version": meta["model_version"],
                "package_version": package_version,
                "base_rate": meta["base_rate"],
                "effective_from_date": meta["effective_from_date"],
                "effective_to_date": meta["effective_to_date"],
                "package_status": "DRAFT",
                "source_export_id": export_id,
                "source_file": meta["source_file"],
                "publication_receipt_json": meta["publication_receipt_json"],
                "publication_receipt_sha256": meta["publication_receipt_sha256"],
                "staging_content_sha256": meta["staging_content_sha256"],
                "package_metadata_json": meta["package_metadata_json"],
                "revision_metadata_json": revision_metadata_json,
                "offset_handling": offset_handling,
                "offset_factor_name": offset_factor_name,
                "offset_source_name": meta["offset_source_name"],
                "offset_label": meta["offset_label"],
                "metadata_origin": meta["metadata_origin"],
                "build_fingerprint_sha256": build_fingerprint_sha256,
                "created_by": created_by,
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
            {"export_id": export_id},
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
            {"export_id": export_id, "model_id": model_id},
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
            {"export_id": export_id, "model_id": model_id},
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
            {"export_id": export_id, "rate_package_id": rate_package_id},
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
                "export_id": export_id,
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
            {"export_id": export_id, "rate_package_id": rate_package_id},
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
                "export_id": export_id,
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
                CASE
                    WHEN ROW_NUMBER() OVER (
                        PARTITION BY t.term_id
                        ORDER BY
                            CASE WHEN fl.lower_bound IS NULL THEN 1 ELSE 0 END,
                            fl.lower_bound DESC,
                            COALESCE(fl.order_index, 0) DESC,
                            fl.feature_level_id DESC
                    ) = 1 THEN NULL
                    ELSE fl.upper_bound
                END,
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
              AND (
                  t.term_type IN ('DISCRETIZED_SPLINE_1D', 'NUMERIC_BANDED_1D')
                  OR (
                      t.term_type = 'OFFSET_FACTOR'
                      AND ls.level_set_type IN ('NUMERIC_BAND', 'SPLINE_GRID_1D')
                  )
              )
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

        if draft_validator is not None:
            draft_validator(con, int(rate_package_id))
        if package_lineage_writer is not None:
            model_run_id = package_lineage_writer(con, int(rate_package_id))

        con.execute(
            text("""
            UPDATE pricing.PRICING_RATE_PACKAGE
            SET package_status = :package_status
            WHERE rate_package_id = :rate_package_id;
        """),
            {
                "package_status": "PUBLISHED",
                "rate_package_id": rate_package_id,
            },
        )

    return PublishResult(
        mlflow_run_id="",
        export_id=export_id,
        rate_package_id=int(rate_package_id),
        package_version=int(package_version),
        rating_workbook_path="",
        package_status="PUBLISHED",
        was_existing=False,
        model_run_id=model_run_id,
    )


def _verify_expected_database(connection: Connection, expected_database: str) -> None:
    expected = str(expected_database).strip()
    if not expected:
        raise ValueError("expected_database must be a non-empty database name")
    actual_value = connection.execute(text("SELECT DB_NAME()")).scalar_one()
    actual = str(actual_value or "").strip()
    if not actual:
        raise RuntimeError("Remote connection did not report a database name")
    if actual.casefold() != expected.casefold():
        raise RuntimeError(
            "Remote database mismatch: "
            f"expected {expected!r}, connected to {actual!r}; "
            "package transaction aborted before writes"
        )
