from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import (
    RatePackageRevisionResult,
    RatePackageSelector,
    RatePackageSnapshot,
)


class ManualRevisionError(RuntimeError):
    """Raised when a manual rate package revision cannot be loaded or validated."""


_IDENTITY_RATE_CELL_COLUMNS = (
    "term_id",
    "cell_key_text",
    "cell_key_digest",
    "is_reference",
    "is_default",
)
_MULTIPLIER_RTOL = 1e-12
_MULTIPLIER_ATOL = 1e-12
_REQUIRED_REVISION_PARENT_METADATA_KEYS = (
    "rate_package_id",
    "model_id",
    "model_name",
    "model_version",
    "base_rate",
    "effective_from_date",
    "effective_to_date",
    "package_status",
    "package_version",
)


def _require_columns(frame_name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ManualRevisionError(
            f"{frame_name} rate cell frame is missing required column(s): {', '.join(missing)}",
        )


def _validate_cell_id_key(frame_name: str, frame: pd.DataFrame) -> None:
    _require_columns(frame_name, frame, ("cell_id",))
    if frame["cell_id"].isna().any():
        raise ManualRevisionError(f"{frame_name} rate cell frame contains null cell_id values")

    duplicates = frame.loc[frame["cell_id"].duplicated(keep=False), "cell_id"]
    if not duplicates.empty:
        duplicate_values = ", ".join(str(value) for value in pd.unique(duplicates))
        raise ManualRevisionError(
            f"duplicate cell_id values in {frame_name} rate cell frame: {duplicate_values}",
        )


def _align_rate_cells_by_cell_id(
    original: pd.DataFrame,
    edited: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_cell_id_key("original", original)
    _validate_cell_id_key("edited", edited)

    original_cell_ids = set(original["cell_id"])
    edited_cell_ids = set(edited["cell_id"])
    if original_cell_ids != edited_cell_ids:
        raise ManualRevisionError("edited frame must contain the same cell_id values as original")

    original_aligned = original.reset_index(drop=True).copy()
    edited_aligned = (
        edited.set_index("cell_id", drop=False)
        .loc[original_aligned["cell_id"].to_list()]
        .reset_index(drop=True)
    )
    return original_aligned, edited_aligned


def _equal_with_matching_nulls(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def _numeric_rate_cell_column(
    frame_name: str,
    column_name: str,
    values: pd.Series,
    *,
    error_message: str | None = None,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
    finite_values = np.isfinite(numeric_values.to_numpy())
    if not finite_values.all():
        raise ManualRevisionError(
            error_message or f"{frame_name} {column_name} values must be numeric finite numbers",
        )
    return numeric_values


def diff_rate_cell_edits(original: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    """Return changed manual multiplier edits aligned one-to-one by cell_id."""
    _require_columns("original", original, ("cell_id", "multiplier", "log_coefficient"))
    _require_columns("edited", edited, ("cell_id", "multiplier"))
    original_aligned, edited_aligned = _align_rate_cells_by_cell_id(original, edited)
    original_multiplier = _numeric_rate_cell_column(
        "original",
        "multiplier",
        original_aligned["multiplier"],
    )
    edited_multiplier = _numeric_rate_cell_column(
        "edited",
        "multiplier",
        edited_aligned["multiplier"],
    )
    original_log_coefficient = _numeric_rate_cell_column(
        "original",
        "log_coefficient",
        original_aligned["log_coefficient"],
    )

    changed = ~np.isclose(
        original_multiplier.to_numpy(),
        edited_multiplier.to_numpy(),
        rtol=_MULTIPLIER_RTOL,
        atol=_MULTIPLIER_ATOL,
    )

    return pd.DataFrame(
        {
            "cell_id": original_aligned.loc[changed, "cell_id"].to_numpy(),
            "old_multiplier": original_multiplier.loc[changed].to_numpy(),
            "new_multiplier": edited_multiplier.loc[changed].to_numpy(),
            "old_log_coefficient": original_log_coefficient.loc[changed].to_numpy(),
        },
    )


def validate_rate_cell_edits(original: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    """Validate a manually edited rate cell frame and return changed multipliers."""
    _require_columns("original", original, ("cell_id", "multiplier", "log_coefficient"))
    _require_columns("edited", edited, ("cell_id", "multiplier"))
    original_aligned, edited_aligned = _align_rate_cells_by_cell_id(original, edited)

    for column in _IDENTITY_RATE_CELL_COLUMNS:
        if column not in original_aligned.columns or column not in edited_aligned.columns:
            continue
        changed_identity = ~_equal_with_matching_nulls(
            original_aligned[column],
            edited_aligned[column],
        )
        if changed_identity.any():
            raise ManualRevisionError(
                f"manual rate cell edits cannot change identity column {column}",
            )

    edited_multiplier = _numeric_rate_cell_column(
        "edited",
        "multiplier",
        edited_aligned["multiplier"],
        error_message=(
            "edited multiplier values must be positive finite numbers for every rate cell"
        ),
    )
    if not edited_multiplier.gt(0).all():
        raise ManualRevisionError(
            "edited multiplier values must be positive finite numbers for every rate cell",
        )

    diff = diff_rate_cell_edits(original_aligned, edited_aligned)
    if diff.empty:
        raise ManualRevisionError("no manual rate cell changes found")
    return diff


def _required_text(value: str | None, field_name: str) -> str:
    if value is None:
        raise ManualRevisionError(f"{field_name} is required")
    cleaned = value.strip()
    if not cleaned:
        raise ManualRevisionError(f"{field_name} is required")
    return cleaned


def _validate_revision_parent_metadata(parent: RatePackageSnapshot) -> None:
    missing = [key for key in _REQUIRED_REVISION_PARENT_METADATA_KEYS if key not in parent.metadata]
    if missing:
        raise ManualRevisionError(
            "parent metadata is missing required key(s): " + ", ".join(missing),
        )


def _validate_revision_parent_model(
    config: ModelBuildConfig,
    parent: RatePackageSnapshot,
) -> None:
    configured_model_name = config.model_name
    for metadata_key in ("model_name", "registry_model_name", "package_model_name"):
        if metadata_key not in parent.metadata:
            continue
        parent_model = str(parent.metadata[metadata_key]).strip()
        if parent_model != configured_model_name:
            raise ManualRevisionError(
                "parent model/configured model mismatch: "
                f"parent {metadata_key}={parent_model!r} does not match "
                f"configured model_name={configured_model_name!r}",
            )


def _manual_revision_metadata(parent_rate_package_id: int, reason: str) -> str:
    return json.dumps(
        {
            "parent_rate_package_id": parent_rate_package_id,
            "reason": reason,
            "revision_kind": "MANUAL",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _offset_factor_terms_for_diff(
    parent: RatePackageSnapshot,
    diff: pd.DataFrame,
) -> list[str]:
    if parent.terms.empty or "term_type" not in parent.terms.columns:
        return []
    if "term_id" not in parent.terms.columns or "term_id" not in parent.rate_cells.columns:
        return []

    edited_cell_ids = set(pd.to_numeric(diff["cell_id"], errors="coerce").dropna().astype("int64"))
    edited_cells = parent.rate_cells.loc[parent.rate_cells["cell_id"].isin(edited_cell_ids)]
    if edited_cells.empty:
        return []

    offset_terms = parent.terms.loc[
        parent.terms["term_type"].astype(str).eq("OFFSET_FACTOR"),
        ["term_id", "term_name"],
    ]
    if offset_terms.empty:
        return []

    offset_term_ids = set(
        pd.to_numeric(offset_terms["term_id"], errors="coerce").dropna().astype("int64")
    )
    edited_offset_term_ids = (
        set(pd.to_numeric(edited_cells["term_id"], errors="coerce").dropna().astype("int64"))
        & offset_term_ids
    )
    if not edited_offset_term_ids:
        return []

    return sorted(
        str(row.term_name)
        for row in offset_terms.itertuples(index=False)
        if int(row.term_id) in edited_offset_term_ids
    )


def _reject_offset_factor_edits(parent: RatePackageSnapshot, diff: pd.DataFrame) -> None:
    term_names = _offset_factor_terms_for_diff(parent, diff)
    if term_names:
        raise ManualRevisionError(
            "manual revisions cannot edit OFFSET_FACTOR cells: " + ", ".join(term_names)
        )


def _write_manual_revision(
    engine,
    config: ModelBuildConfig,
    *,
    parent: RatePackageSnapshot,
    edited_rate_cells: pd.DataFrame,
    diff: pd.DataFrame,
    reason: str,
    created_by: str,
) -> tuple[int, int]:
    manual_edit_rows = []
    for row in diff.itertuples(index=False):
        multiplier = float(row.new_multiplier)
        manual_edit_rows.append(
            {
                "cell_id": int(row.cell_id),
                "multiplier": multiplier,
                "log_coefficient": float(np.log(multiplier)),
            },
        )

    parent_metadata = parent.metadata
    parent_rate_package_id = int(parent_metadata["rate_package_id"])
    model_id = int(parent_metadata["model_id"])
    revision_metadata_json = _manual_revision_metadata(parent_rate_package_id, reason)

    with engine.begin() as con:
        con.execute(
            text("""
            DROP TABLE IF EXISTS #manual_rate_cell_edits;
            DROP TABLE IF EXISTS #term_map;
            DROP TABLE IF EXISTS #cell_map;
        """)
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
                :parent_rate_package_id,
                :model_id,
                :model_name,
                :model_version,
                :package_version,
                :base_rate,
                :effective_from_date,
                :effective_to_date,
                :package_status,
                :publication_receipt_json,
                :publication_receipt_sha256,
                :package_metadata_json,
                :revision_metadata_json,
                :offset_handling,
                :offset_factor_name,
                :offset_source_name,
                :offset_label,
                :metadata_origin,
                :created_by
            );
        """),
            {
                "parent_rate_package_id": parent_rate_package_id,
                "model_id": model_id,
                "model_name": parent_metadata["model_name"],
                "model_version": parent_metadata["model_version"],
                "package_version": package_version,
                "base_rate": parent_metadata["base_rate"],
                "effective_from_date": parent_metadata["effective_from_date"],
                "effective_to_date": parent_metadata["effective_to_date"],
                "package_status": "DRAFT",
                "publication_receipt_json": parent_metadata.get("publication_receipt_json"),
                "publication_receipt_sha256": parent_metadata.get("publication_receipt_sha256"),
                "package_metadata_json": parent_metadata.get("package_metadata_json"),
                "revision_metadata_json": revision_metadata_json,
                "offset_handling": parent_metadata.get("offset_handling") or "UNKNOWN",
                "offset_factor_name": parent_metadata.get("offset_factor_name"),
                "offset_source_name": parent_metadata.get("offset_source_name"),
                "offset_label": parent_metadata.get("offset_label"),
                "metadata_origin": parent_metadata.get("metadata_origin"),
                "created_by": created_by,
            },
        ).scalar_one()

        con.execute(
            text("""
            CREATE TABLE #manual_rate_cell_edits (
                cell_id BIGINT NOT NULL PRIMARY KEY,
                multiplier DECIMAL(19,10) NOT NULL,
                log_coefficient DECIMAL(19,12) NOT NULL
            );
        """)
        )
        if manual_edit_rows:
            con.execute(
                text("""
                INSERT INTO #manual_rate_cell_edits (
                    cell_id,
                    multiplier,
                    log_coefficient
                )
                VALUES (
                    :cell_id,
                    :multiplier,
                    :log_coefficient
                );
            """),
                manual_edit_rows,
            )

        con.execute(
            text("""
            CREATE TABLE #term_map (
                old_term_id BIGINT NOT NULL PRIMARY KEY,
                new_term_id BIGINT NOT NULL UNIQUE
            );

            CREATE TABLE #cell_map (
                old_cell_id BIGINT NOT NULL PRIMARY KEY,
                new_cell_id BIGINT NOT NULL UNIQUE
            );

            MERGE pricing.PRICING_TERM AS tgt
            USING (
                SELECT
                    term_id AS old_term_id,
                    term_name,
                    term_type,
                    sequence_no,
                    default_multiplier,
                    default_log_coefficient,
                    active_flag,
                    term_metadata_json
                FROM pricing.PRICING_TERM
                WHERE rate_package_id = :parent_rate_package_id
            ) AS src
            ON 1 = 0
            WHEN NOT MATCHED THEN
                INSERT (
                    rate_package_id,
                    term_name,
                    term_type,
                    sequence_no,
                    default_multiplier,
                    default_log_coefficient,
                    active_flag,
                    term_metadata_json
                )
                VALUES (
                    :rate_package_id,
                    src.term_name,
                    src.term_type,
                    src.sequence_no,
                    src.default_multiplier,
                    src.default_log_coefficient,
                    src.active_flag,
                    src.term_metadata_json
                )
            OUTPUT
                src.old_term_id,
                INSERTED.term_id
            INTO #term_map (
                old_term_id,
                new_term_id
            );

            INSERT INTO pricing.PRICING_TERM_FEATURE (
                term_id,
                position_no,
                feature_id,
                level_set_id,
                input_column_name
            )
            SELECT
                tm.new_term_id,
                tf.position_no,
                tf.feature_id,
                tf.level_set_id,
                tf.input_column_name
            FROM pricing.PRICING_TERM_FEATURE AS tf
            JOIN #term_map AS tm
              ON tm.old_term_id = tf.term_id;

            MERGE pricing.PRICING_RATE_CELL AS tgt
            USING (
                SELECT
                    rc.cell_id AS old_cell_id,
                    tm.new_term_id,
                    rc.cell_key_text,
                    rc.cell_key_digest,
                    COALESCE(edit.multiplier, rc.multiplier) AS multiplier,
                    COALESCE(edit.log_coefficient, rc.log_coefficient) AS log_coefficient,
                    rc.exposure_weight,
                    rc.record_count,
                    rc.is_reference,
                    rc.is_default,
                    rc.is_deleted
                FROM pricing.PRICING_RATE_CELL AS rc
                JOIN #term_map AS tm
                  ON tm.old_term_id = rc.term_id
                LEFT JOIN #manual_rate_cell_edits AS edit
                  ON edit.cell_id = rc.cell_id
            ) AS src
            ON 1 = 0
            WHEN NOT MATCHED THEN
                INSERT (
                    term_id,
                    cell_key_text,
                    cell_key_digest,
                    multiplier,
                    log_coefficient,
                    exposure_weight,
                    record_count,
                    is_reference,
                    is_default,
                    is_deleted
                )
                VALUES (
                    src.new_term_id,
                    src.cell_key_text,
                    src.cell_key_digest,
                    src.multiplier,
                    src.log_coefficient,
                    src.exposure_weight,
                    src.record_count,
                    src.is_reference,
                    src.is_default,
                    src.is_deleted
                )
            OUTPUT
                src.old_cell_id,
                INSERTED.cell_id
            INTO #cell_map (
                old_cell_id,
                new_cell_id
            );

            INSERT INTO pricing.PRICING_RATE_CELL_LEVEL (
                cell_id,
                position_no,
                feature_level_id
            )
            SELECT
                cm.new_cell_id,
                rcl.position_no,
                rcl.feature_level_id
            FROM pricing.PRICING_RATE_CELL_LEVEL AS rcl
            JOIN #cell_map AS cm
              ON cm.old_cell_id = rcl.cell_id;

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
                tm.new_term_id,
                crc.cell_key_digest,
                crc.term_name,
                crc.term_type,
                crc.sequence_no,
                crc.cell_key_text,
                COALESCE(edit.multiplier, crc.multiplier),
                COALESCE(edit.log_coefficient, crc.log_coefficient),
                crc.exposure_weight,
                crc.record_count,
                crc.is_default,
                crc.is_reference
            FROM pricing.PRICING_COMPILED_RATE_CELL AS crc
            JOIN pricing.PRICING_TERM AS src_term
              ON src_term.term_id = crc.term_id
             AND src_term.rate_package_id = :parent_rate_package_id
            JOIN #term_map AS tm
              ON tm.old_term_id = src_term.term_id
            LEFT JOIN pricing.PRICING_RATE_CELL AS src_rc
              ON src_rc.term_id = src_term.term_id
             AND src_rc.cell_key_digest = crc.cell_key_digest
             AND src_rc.cell_key_text = crc.cell_key_text
            LEFT JOIN #manual_rate_cell_edits AS edit
              ON edit.cell_id = src_rc.cell_id
            WHERE crc.rate_package_id = :parent_rate_package_id;

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
                tm.new_term_id,
                band.feature_level_id,
                band.term_name,
                band.feature_name,
                band.level_code,
                band.sort_order,
                band.lower_bound,
                band.upper_bound,
                band.representative_value,
                COALESCE(edit.multiplier, band.multiplier),
                COALESCE(edit.log_coefficient, band.log_coefficient)
            FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS band
            JOIN pricing.PRICING_TERM AS src_term
              ON src_term.term_id = band.term_id
             AND src_term.rate_package_id = :parent_rate_package_id
            JOIN #term_map AS tm
              ON tm.old_term_id = src_term.term_id
            LEFT JOIN (
                SELECT
                    rc.cell_id,
                    rc.term_id,
                    rcl.feature_level_id
                FROM pricing.PRICING_RATE_CELL AS rc
                JOIN pricing.PRICING_RATE_CELL_LEVEL AS rcl
                  ON rcl.cell_id = rc.cell_id
                 AND rcl.position_no = 1
                WHERE rc.is_deleted = 0
            ) AS src_band_cell
              ON src_band_cell.term_id = src_term.term_id
             AND src_band_cell.feature_level_id = band.feature_level_id
            LEFT JOIN #manual_rate_cell_edits AS edit
              ON edit.cell_id = src_band_cell.cell_id
            WHERE band.rate_package_id = :parent_rate_package_id;
        """),
            {
                "parent_rate_package_id": parent_rate_package_id,
                "rate_package_id": rate_package_id,
            },
        )

        con.execute(
            text("""
            UPDATE pricing.PRICING_RATE_PACKAGE
            SET package_status = 'PUBLISHED'
            WHERE rate_package_id = :rate_package_id;
        """),
            {"rate_package_id": rate_package_id},
        )

        con.execute(
            text("""
            DROP TABLE IF EXISTS #cell_map;
            DROP TABLE IF EXISTS #term_map;
            DROP TABLE IF EXISTS #manual_rate_cell_edits;
        """)
        )

    return int(rate_package_id), int(package_version)


def create_manual_revision(
    engine,
    config: ModelBuildConfig,
    *,
    parent: RatePackageSnapshot,
    edited_rate_cells: pd.DataFrame,
    reason: str,
    created_by: str,
) -> RatePackageRevisionResult:
    reason = _required_text(reason, "reason")
    created_by = _required_text(created_by, "created_by")
    _validate_revision_parent_metadata(parent)
    _validate_revision_parent_model(config, parent)

    if parent.metadata.get("package_status") != "PUBLISHED":
        raise ManualRevisionError("manual revisions require a PUBLISHED parent package")

    diff = validate_rate_cell_edits(parent.rate_cells, edited_rate_cells)
    _reject_offset_factor_edits(parent, diff)
    rate_package_id, package_version = _write_manual_revision(
        engine,
        config,
        parent=parent,
        edited_rate_cells=edited_rate_cells,
        diff=diff,
        reason=reason,
        created_by=created_by,
    )

    return RatePackageRevisionResult(
        rate_package_id=int(rate_package_id),
        package_version=int(package_version),
        parent_rate_package_id=int(parent.metadata["rate_package_id"]),
        changed_rate_cell_count=len(diff),
        base_rate_changed=False,
        diff_summary=diff,
    )


def _metadata_query(selector: RatePackageSelector):
    if selector.rate_package_id is not None:
        return text("""
            SELECT
                rp.rate_package_id,
                rp.parent_rate_package_id,
                rp.model_id,
                m.model_name AS model_name,
                m.model_name AS registry_model_name,
                m.model_label,
                rp.model_name AS package_model_name,
                rp.model_version,
                rp.package_version,
                rp.base_rate,
                rp.effective_from_date,
                rp.effective_to_date,
                rp.package_status,
                rp.publication_receipt_json,
                rp.publication_receipt_sha256,
                rp.package_metadata_json,
                rp.revision_metadata_json,
                rp.offset_handling,
                rp.offset_factor_name,
                rp.offset_source_name,
                rp.offset_label,
                rp.metadata_origin,
                rp.created_ts,
                rp.created_by
            FROM pricing.PRICING_RATE_PACKAGE AS rp
            JOIN pricing.PRICING_MODEL AS m
              ON m.model_id = rp.model_id
            WHERE m.model_name = :model_name
              AND rp.rate_package_id = :rate_package_id
        """)
    return text("""
        SELECT
            rp.rate_package_id,
            rp.parent_rate_package_id,
            rp.model_id,
            m.model_name AS model_name,
            m.model_name AS registry_model_name,
            m.model_label,
            rp.model_name AS package_model_name,
            rp.model_version,
            rp.package_version,
            rp.base_rate,
            rp.effective_from_date,
            rp.effective_to_date,
            rp.package_status,
            rp.publication_receipt_json,
            rp.publication_receipt_sha256,
            rp.package_metadata_json,
            rp.revision_metadata_json,
            rp.offset_handling,
            rp.offset_factor_name,
            rp.offset_source_name,
            rp.offset_label,
            rp.metadata_origin,
            rp.created_ts,
            rp.created_by
        FROM pricing.PRICING_RATE_PACKAGE AS rp
        JOIN pricing.PRICING_MODEL AS m
          ON m.model_id = rp.model_id
        WHERE m.model_name = :model_name
          AND rp.package_version = :package_version
    """)


def _metadata_params(
    config: ModelBuildConfig,
    selector: RatePackageSelector,
) -> dict[str, int | str]:
    params: dict[str, int | str] = {"model_name": config.model_name}
    if selector.rate_package_id is not None:
        params["rate_package_id"] = selector.rate_package_id
    else:
        params["package_version"] = selector.package_version
    return params


def load_rate_package_snapshot(
    engine,
    config: ModelBuildConfig,
    selector: RatePackageSelector,
) -> RatePackageSnapshot:
    metadata = pd.read_sql_query(
        _metadata_query(selector),
        engine,
        params=_metadata_params(config, selector),
    )
    if len(metadata) != 1:
        raise ManualRevisionError("rate package selector must resolve to exactly one package")

    metadata_row = metadata.iloc[0].to_dict()
    rate_package_id = int(metadata_row["rate_package_id"])
    package_params = {"rate_package_id": rate_package_id}

    terms = pd.read_sql_query(
        text("""
        SELECT
            t.*
        FROM pricing.PRICING_TERM AS t
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY
            t.sequence_no,
            t.term_id
    """),
        engine,
        params=package_params,
    )

    rate_cells = pd.read_sql_query(
        text("""
        SELECT
            rc.*
        FROM pricing.PRICING_RATE_CELL AS rc
        JOIN pricing.PRICING_TERM AS t
          ON t.term_id = rc.term_id
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY
            t.sequence_no,
            rc.cell_id
    """),
        engine,
        params=package_params,
    )

    cell_levels = pd.read_sql_query(
        text("""
        SELECT
            rcl.*
        FROM pricing.PRICING_RATE_CELL_LEVEL AS rcl
        JOIN pricing.PRICING_RATE_CELL AS rc
          ON rc.cell_id = rcl.cell_id
        JOIN pricing.PRICING_TERM AS t
          ON t.term_id = rc.term_id
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY
            t.sequence_no,
            rc.cell_id,
            rcl.position_no
    """),
        engine,
        params=package_params,
    )

    compiled_rate_cells = pd.read_sql_query(
        text("""
        SELECT
            crc.*
        FROM pricing.PRICING_COMPILED_RATE_CELL AS crc
        WHERE crc.rate_package_id = :rate_package_id
        ORDER BY
            crc.sequence_no,
            crc.term_id,
            crc.cell_key_text
    """),
        engine,
        params=package_params,
    )

    compiled_1d_bands = pd.read_sql_query(
        text("""
        SELECT
            b.*
        FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS b
        WHERE b.rate_package_id = :rate_package_id
        ORDER BY
            b.term_id,
            b.sort_order,
            b.feature_level_id
    """),
        engine,
        params=package_params,
    )

    return RatePackageSnapshot(
        metadata=metadata_row,
        terms=terms,
        rate_cells=rate_cells,
        cell_levels=cell_levels,
        compiled_rate_cells=compiled_rate_cells,
        compiled_1d_bands=compiled_1d_bands,
    )
