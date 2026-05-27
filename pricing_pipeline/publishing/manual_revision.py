from __future__ import annotations

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


def _require_columns(frame_name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ManualRevisionError(
            f"{frame_name} rate cell frame is missing required column(s): "
            f"{', '.join(missing)}",
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
            error_message
            or f"{frame_name} {column_name} values must be numeric finite numbers",
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


def _write_manual_revision(
    engine,
    config: ModelBuildConfig,
    parent: RatePackageSnapshot,
    edited_rate_cells: pd.DataFrame,
    diff: pd.DataFrame,
    reason: str,
    created_by: str,
) -> tuple[int, int]:
    raise NotImplementedError("manual revision SQL writer is implemented in the next step")


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

    if parent.metadata.get("package_status") != "PUBLISHED":
        raise ManualRevisionError("manual revisions require a PUBLISHED parent package")

    diff = validate_rate_cell_edits(parent.rate_cells, edited_rate_cells)
    rate_package_id, package_version = _write_manual_revision(
        engine,
        config,
        parent,
        edited_rate_cells,
        diff,
        reason,
        created_by,
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
                m.model_key,
                m.model_label,
                rp.model_name,
                rp.model_version,
                rp.package_version,
                rp.base_rate,
                rp.effective_from_date,
                rp.effective_to_date,
                rp.package_status,
                rp.created_ts,
                rp.created_by
            FROM pricing.PRICING_RATE_PACKAGE AS rp
            JOIN pricing.PRICING_MODEL AS m
              ON m.model_id = rp.model_id
            WHERE m.model_key = :model_key
              AND rp.rate_package_id = :rate_package_id
        """)
    return text("""
        SELECT
            rp.rate_package_id,
            rp.parent_rate_package_id,
            rp.model_id,
            m.model_key,
            m.model_label,
            rp.model_name,
            rp.model_version,
            rp.package_version,
            rp.base_rate,
            rp.effective_from_date,
            rp.effective_to_date,
            rp.package_status,
            rp.created_ts,
            rp.created_by
        FROM pricing.PRICING_RATE_PACKAGE AS rp
        JOIN pricing.PRICING_MODEL AS m
          ON m.model_id = rp.model_id
        WHERE m.model_key = :model_key
          AND rp.package_version = :package_version
    """)


def _metadata_params(
    config: ModelBuildConfig,
    selector: RatePackageSelector,
) -> dict[str, int | str]:
    params: dict[str, int | str] = {"model_key": config.model_key}
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

    terms = pd.read_sql_query(text("""
        SELECT
            t.*
        FROM pricing.PRICING_TERM AS t
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY
            t.sequence_no,
            t.term_id
    """), engine, params=package_params)

    rate_cells = pd.read_sql_query(text("""
        SELECT
            rc.*
        FROM pricing.PRICING_RATE_CELL AS rc
        JOIN pricing.PRICING_TERM AS t
          ON t.term_id = rc.term_id
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY
            t.sequence_no,
            rc.cell_id
    """), engine, params=package_params)

    cell_levels = pd.read_sql_query(text("""
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
    """), engine, params=package_params)

    compiled_rate_cells = pd.read_sql_query(text("""
        SELECT
            crc.*
        FROM pricing.PRICING_COMPILED_RATE_CELL AS crc
        WHERE crc.rate_package_id = :rate_package_id
        ORDER BY
            crc.sequence_no,
            crc.term_id,
            crc.cell_key_text
    """), engine, params=package_params)

    compiled_1d_bands = pd.read_sql_query(text("""
        SELECT
            b.*
        FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS b
        WHERE b.rate_package_id = :rate_package_id
        ORDER BY
            b.term_id,
            b.sort_order,
            b.feature_level_id
    """), engine, params=package_params)

    return RatePackageSnapshot(
        metadata=metadata_row,
        terms=terms,
        rate_cells=rate_cells,
        cell_levels=cell_levels,
        compiled_rate_cells=compiled_rate_cells,
        compiled_1d_bands=compiled_1d_bands,
    )
