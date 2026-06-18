"""Stage deployable fixed offsets from a SuperGLM rating workbook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable

FIXED_OFFSET_SHEET_NAME = "Fixed Offsets"
FIXED_OFFSET_WORKBOOK_COLUMNS = (
    "Term",
    "Term Type",
    "Source Feature",
    "Transform",
    "Reference Value",
    "Coefficient",
    "Sequence",
)
FIXED_OFFSET_STAGING_COLUMNS = (
    "export_id",
    "term_name",
    "source_feature_name",
    "transform_type",
    "reference_value",
    "coefficient",
    "sequence_no",
)


def _empty_staging_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FIXED_OFFSET_STAGING_COLUMNS)


def build_fixed_offset_staging_frame(
    workbook_path: str | Path,
    *,
    export_id: str,
    sheet_name: str = FIXED_OFFSET_SHEET_NAME,
) -> pd.DataFrame:
    """Read and validate fixed-offset metadata from a rating workbook."""
    workbook_path = Path(workbook_path)
    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        if sheet_name not in workbook.sheet_names:
            return _empty_staging_frame()
        raw = pd.read_excel(workbook, sheet_name=sheet_name)

    if raw.empty:
        return _empty_staging_frame()

    missing = [column for column in FIXED_OFFSET_WORKBOOK_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{sheet_name!r} is missing required column(s): {', '.join(missing)}"
        )

    frame = raw.loc[:, FIXED_OFFSET_WORKBOOK_COLUMNS].copy()
    required_text = ("Term", "Term Type", "Source Feature", "Transform")
    for column in required_text:
        if frame[column].isna().any():
            raise ValueError(f"{sheet_name!r} column {column!r} cannot contain null values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{sheet_name!r} column {column!r} cannot contain blank values")

    if not frame["Term Type"].eq("FIXED_OFFSET").all():
        values = sorted(frame.loc[~frame["Term Type"].eq("FIXED_OFFSET"), "Term Type"].unique())
        raise ValueError(f"Unsupported fixed offset term type(s): {values}")
    if not frame["Transform"].eq("LOG_RATIO").all():
        values = sorted(frame.loc[~frame["Transform"].eq("LOG_RATIO"), "Transform"].unique())
        raise ValueError(f"Unsupported fixed offset transform(s): {values}")

    for column in ("Reference Value", "Coefficient", "Sequence"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{sheet_name!r} column {column!r} must be finite numeric values")

    if not frame["Reference Value"].gt(0).all():
        raise ValueError("Fixed offset reference values must be strictly positive")
    if not np.equal(frame["Sequence"], np.floor(frame["Sequence"])).all():
        raise ValueError("Fixed offset sequence values must be integers")
    if not frame["Sequence"].gt(0).all():
        raise ValueError("Fixed offset sequence values must be positive")
    if frame["Term"].duplicated().any():
        duplicates = sorted(frame.loc[frame["Term"].duplicated(keep=False), "Term"].unique())
        raise ValueError(f"Fixed offset term names must be unique: {duplicates}")
    if frame["Sequence"].duplicated().any():
        duplicates = sorted(
            int(value)
            for value in frame.loc[frame["Sequence"].duplicated(keep=False), "Sequence"].unique()
        )
        raise ValueError(f"Fixed offset sequence values must be unique: {duplicates}")

    staging = pd.DataFrame(
        {
            "export_id": export_id,
            "term_name": frame["Term"],
            "source_feature_name": frame["Source Feature"],
            "transform_type": frame["Transform"],
            "reference_value": frame["Reference Value"].astype(float),
            "coefficient": frame["Coefficient"].astype(float),
            "sequence_no": frame["Sequence"].astype(int),
        }
    )
    return staging.loc[:, FIXED_OFFSET_STAGING_COLUMNS]


def stage_fixed_offsets(
    engine,
    *,
    workbook_path: str | Path,
    export_id: str,
    sheet_name: str = FIXED_OFFSET_SHEET_NAME,
) -> int:
    """Replace staged fixed offsets for one already-staged rating export."""
    frame = build_fixed_offset_staging_frame(
        workbook_path,
        export_id=export_id,
        sheet_name=sheet_name,
    )
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM pricing_stg.STG_FIXED_OFFSET WHERE export_id = :export_id"),
            {"export_id": export_id},
        )
        if not frame.empty:
            frame.to_sql(
                "STG_FIXED_OFFSET",
                connection,
                schema=schemas.pricing_staging,
                if_exists="append",
                index=False,
            )
    return int(len(frame))


__all__ = [
    "FIXED_OFFSET_SHEET_NAME",
    "FIXED_OFFSET_STAGING_COLUMNS",
    "FIXED_OFFSET_WORKBOOK_COLUMNS",
    "build_fixed_offset_staging_frame",
    "stage_fixed_offsets",
]
