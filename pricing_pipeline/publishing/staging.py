from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.model_registry import ModelRegistryError, get_pricing_model
from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    SuperGLMPublicationReceipt,
    canonical_receipt_bytes,
    load_publication_receipt,
)

INTERVAL_RE = re.compile(
    r"^\s*[\[\(]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+|inf|Inf|INF)\s*[\]\)]\s*$"
)
RANGE_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[-:]\s*([-+]?\d*\.?\d+)\s*$")


def cell_to_zero_index(cell: str) -> tuple[int, int]:
    m = re.match(r"^([A-Za-z]+)(\d+)$", cell.strip())
    if not m:
        raise ValueError(f"Bad Excel cell reference: {cell}")
    letters, row = m.groups()
    col = 0
    for ch in letters.upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return int(row) - 1, col - 1


def clean_text(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def parse_interval(level: str) -> tuple[float | None, float | None, float | None]:
    s = str(level).strip()
    m = INTERVAL_RE.match(s)
    if not m:
        m = RANGE_RE.match(s)
    if not m:
        return None, None, None

    lo_s, hi_s = m.groups()
    lo = float(lo_s)
    hi = math.inf if hi_s.lower() == "inf" else float(hi_s)
    rep = None if not math.isfinite(hi) else (lo + hi) / 2.0
    return lo, hi if math.isfinite(hi) else None, rep


def find_blocks(raw: pd.DataFrame, term_row: int, header_row: int) -> list[dict[str, Any]]:
    tr = term_row - 1
    hr = header_row - 1
    blocks: list[dict[str, Any]] = []

    for c in range(0, raw.shape[1] - 2):
        term_name = clean_text(raw.iat[tr, c])
        h0 = clean_text(raw.iat[hr, c])
        h1 = clean_text(raw.iat[hr, c + 1])
        h2 = clean_text(raw.iat[hr, c + 2])
        if not term_name or not h0 or not h1 or not h2:
            continue
        headers = [h0.lower(), h1.lower(), h2.lower()]
        level_header = "level" in headers[0] or clean_identifier(h0) == clean_identifier(term_name)
        if level_header and "relativity" in headers[1] and "weight" in headers[2]:
            blocks.append(
                {
                    "term_name": clean_identifier(term_name),
                    "level_col": c,
                    "mult_col": c + 1,
                    "weight_col": c + 2,
                }
            )

    return blocks


def infer_term_type(term_name: str, levels: pd.Series, term_type_map: dict[str, str]) -> str:
    if term_name in term_type_map:
        return term_type_map[term_name]

    non_null = levels.dropna().astype(str)
    if len(non_null) and non_null.map(lambda x: parse_interval(x)[0] is not None).mean() > 0.8:
        return "DISCRETIZED_SPLINE_1D"

    return "CATEGORICAL_MAIN"


def split_interaction_level(level_code: str, features: list[str]) -> list[tuple[str, str]]:
    parts = [p.strip() for p in str(level_code).split("|")]
    out: list[tuple[str, str]] = []

    for i, feature in enumerate(features):
        token = parts[i] if i < len(parts) else ""
        if "=" in token:
            _, lv = token.split("=", 1)
        else:
            lv = token
        out.append((clean_identifier(feature), lv.strip()))

    return out


def build_staging_frames(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(args.xlsx, sheet_name=args.sheet, header=None, engine="openpyxl")

    if args.base_rate is not None:
        base_rate = args.base_rate
    else:
        r, c = cell_to_zero_index(args.base_rate_cell)
        base_rate = float(raw.iat[r, c])

    term_type_map = json.loads(args.term_type_map_json)
    interaction_features = json.loads(args.interaction_features_json)

    blocks = find_blocks(raw, args.term_row, args.header_row)
    if not blocks:
        raise RuntimeError("No rating table blocks found. Check --term-row/--header-row/--sheet.")

    export_df = pd.DataFrame(
        [
            {
                "export_id": args.export_id,
                "model_name": args.model_name,
                "model_version": args.model_version,
                "base_rate": base_rate,
                "effective_from_date": args.effective_from,
                "effective_to_date": args.effective_to,
                "source_file": str(Path(args.xlsx).resolve()),
                "created_by": args.created_by,
            }
        ]
    )

    rate_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    row_id = 0
    sequence_no = 0
    start = args.data_start_row - 1

    for block in blocks:
        sequence_no += 1
        term_name = block["term_name"]
        level_col = block["level_col"]
        mult_col = block["mult_col"]
        weight_col = block["weight_col"]

        block_df = raw.iloc[start:, [level_col, mult_col, weight_col]].copy()
        block_df.columns = ["level_code", "multiplier", "exposure_weight"]
        block_df = block_df.dropna(subset=["level_code", "multiplier"], how="any")
        if block_df.empty:
            continue

        term_type = infer_term_type(term_name, block_df["level_code"], term_type_map)
        is_band = term_type in {
            "DISCRETIZED_SPLINE_1D",
            "NUMERIC_BANDED_1D",
            "ORDERED_CATEGORICAL_MAIN",
        }

        features = interaction_features.get(term_name)
        if features:
            term_type = term_type_map.get(term_name, "CATEGORICAL_INTERACTION")

        for order_index, rec in enumerate(block_df.to_dict("records"), start=1):
            row_id += 1
            level_code = str(rec["level_code"]).strip()
            multiplier = float(rec["multiplier"])
            exposure_weight = (
                None if pd.isna(rec.get("exposure_weight")) else float(rec["exposure_weight"])
            )
            cell_key = f"{term_name}={level_code}"

            rate_rows.append(
                {
                    "export_id": args.export_id,
                    "row_id": row_id,
                    "term_name": term_name,
                    "term_type": term_type,
                    "sequence_no": sequence_no,
                    "cell_key_text": cell_key,
                    "multiplier": multiplier,
                    "log_coefficient": float(np.log(multiplier)),
                    "exposure_weight": exposure_weight,
                    "record_count": None,
                    "is_reference": 1 if np.isclose(multiplier, 1.0) else 0,
                    "is_default": 0,
                }
            )

            if features:
                pairs = split_interaction_level(level_code, features)
            else:
                pairs = [(term_name, level_code)]

            for position_no, (feature_name, lv_code) in enumerate(pairs, start=1):
                lo, hi, rep = parse_interval(lv_code)
                level_set_type = "NUMERIC_BAND" if lo is not None else "CATEGORICAL"
                if len(pairs) == 1 and term_type == "DISCRETIZED_SPLINE_1D":
                    level_set_type = "SPLINE_GRID_1D"

                level_rows.append(
                    {
                        "export_id": args.export_id,
                        "row_id": row_id,
                        "position_no": position_no,
                        "feature_name": feature_name,
                        "feature_value_type": "NUMERIC"
                        if lo is not None or is_band
                        else "CATEGORICAL",
                        "level_set_name": f"{feature_name}__{args.export_id}",
                        "level_set_type": level_set_type,
                        "level_code": lv_code,
                        "level_label": lv_code,
                        "order_index": order_index,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "representative_value": rep,
                        "is_missing": 1
                        if lv_code.lower() in {"missing", "na", "nan", "null"}
                        else 0,
                        "is_other": 1 if lv_code.lower() in {"other", "else"} else 0,
                    }
                )

    rate_df = pd.DataFrame(rate_rows)
    level_df = pd.DataFrame(level_rows)
    return export_df, rate_df, level_df


def _resolve_registered_model_id(con, args: argparse.Namespace) -> int:
    model_id = getattr(args, "model_id", None)
    if model_id is not None:
        return int(model_id)

    record = get_pricing_model(con, args.model_name)
    if record is None:
        raise ModelRegistryError(
            f"model_name {args.model_name!r} is not registered; "
            "run explicit model registration first"
        )

    mismatches: list[str] = []
    if getattr(args, "model_label", None) is not None and record.model_label != args.model_label:
        mismatches.append(f"model_label db={record.model_label!r} staging={args.model_label!r}")
    if record.target_name != args.target_name:
        mismatches.append(f"target_name db={record.target_name!r} staging={args.target_name!r}")
    if record.model_type != args.model_type:
        mismatches.append(f"model_type db={record.model_type!r} staging={args.model_type!r}")
    if record.model_status != "ACTIVE":
        mismatches.append(f"model_status db={record.model_status!r} expected='ACTIVE'")

    if mismatches:
        raise ModelRegistryError(
            f"registered model {args.model_name!r} does not match staged export: "
            + "; ".join(mismatches)
        )
    return record.model_id


def _deterministic_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _empty_term_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["export_id", "term_name", "term_metadata_json"])


def _term_metadata_frame(
    export_id: str,
    receipt: SuperGLMPublicationReceipt,
) -> pd.DataFrame:
    receipt_data = receipt.model_dump(mode="json")
    term_metadata = receipt_data["term_metadata"]
    rows = [
        {
            "export_id": export_id,
            "term_name": term_name,
            "term_metadata_json": _deterministic_json(term_metadata[term_name]),
        }
        for term_name in sorted(term_metadata)
    ]
    return pd.DataFrame(rows, columns=["export_id", "term_name", "term_metadata_json"])


def _apply_publication_receipt_metadata(
    *,
    args: argparse.Namespace,
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    receipt: SuperGLMPublicationReceipt | None,
    receipt_sha256: str | None,
) -> pd.DataFrame:
    if receipt is None:
        return _empty_term_metadata_frame()

    offset_contract = receipt.offset_contract
    if offset_contract.handling == "EXPORTED_FACTOR":
        offset_factor_name = offset_contract.published_factor_name
        matching_term = rate_df["term_name"] == offset_factor_name
        if not matching_term.any():
            raise ValueError(
                "publication receipt declares exported offset factor "
                f"{offset_factor_name!r}, but no staged workbook term matches"
            )
        rate_df.loc[matching_term, "term_type"] = "OFFSET_FACTOR"
    elif offset_contract.handling == "ALREADY_APPLIED_SQL_EXPOSURE" and (
        rate_df["term_type"] == "OFFSET_FACTOR"
    ).any():
        raise ValueError(
            "publication receipt offset handling ALREADY_APPLIED_SQL_EXPOSURE "
            "cannot stage OFFSET_FACTOR terms"
        )

    receipt_data = receipt.model_dump(mode="json")
    export_df["publication_receipt_json"] = canonical_receipt_bytes(receipt).decode("utf-8")
    export_df["publication_receipt_sha256"] = receipt_sha256
    export_df["package_metadata_json"] = _deterministic_json(receipt_data["package_metadata"])
    export_df["offset_handling"] = offset_contract.handling
    export_df["offset_factor_name"] = offset_contract.published_factor_name
    export_df["offset_source_name"] = offset_contract.source_name
    export_df["offset_label"] = offset_contract.label
    export_df["metadata_origin"] = receipt.metadata_origin

    return _term_metadata_frame(args.export_id, receipt)


def insert_staging_frames(
    engine,
    args: argparse.Namespace,
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
    term_metadata_df: pd.DataFrame | None = None,
) -> None:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        model_id = _resolve_registered_model_id(con, args)

        if args.replace:
            con.execute(
                text("DELETE FROM pricing_stg.STG_TERM_METADATA WHERE export_id = :export_id"),
                {"export_id": args.export_id},
            )
            con.execute(
                text("DELETE FROM pricing_stg.STG_CELL_LEVEL WHERE export_id = :export_id"),
                {"export_id": args.export_id},
            )
            con.execute(
                text("DELETE FROM pricing_stg.STG_RATE_CELL WHERE export_id = :export_id"),
                {"export_id": args.export_id},
            )
            con.execute(
                text("DELETE FROM pricing_stg.STG_RATING_EXPORT WHERE export_id = :export_id"),
                {"export_id": args.export_id},
            )

        export_df.to_sql(
            "STG_RATING_EXPORT",
            con,
            schema=schemas.pricing_staging,
            if_exists="append",
            index=False,
        )
        con.execute(
            text(
                "UPDATE pricing_stg.STG_RATING_EXPORT "
                "SET model_id = :model_id WHERE export_id = :export_id"
            ),
            {"export_id": args.export_id, "model_id": model_id},
        )
        rate_df.to_sql(
            "STG_RATE_CELL",
            con,
            schema=schemas.pricing_staging,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        level_df.to_sql(
            "STG_CELL_LEVEL",
            con,
            schema=schemas.pricing_staging,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        if term_metadata_df is not None and not term_metadata_df.empty:
            term_metadata_df.to_sql(
                "STG_TERM_METADATA",
                con,
                schema=schemas.pricing_staging,
                if_exists="append",
                index=False,
                chunksize=5000,
            )


def stage_rating_export(
    engine,
    *,
    workbook_path: Path,
    export_id: str,
    model_name: str,
    model_version: str | None,
    effective_from: str,
    target_name: str = "ClaimNb",
    model_type: str = "superglm_poisson",
    effective_to: str | None = None,
    created_by: str = "python",
    replace: bool = False,
    model_id: int | None = None,
    publication_receipt_path: str | Path | None = None,
    publication_receipt_sha256: str | None = None,
    metadata_mode: Literal[
        "REQUIRE_SUPERGLM_RECEIPT", "ALLOW_WORKBOOK_ONLY"
    ] = "REQUIRE_SUPERGLM_RECEIPT",
) -> None:
    if metadata_mode not in {"REQUIRE_SUPERGLM_RECEIPT", "ALLOW_WORKBOOK_ONLY"}:
        raise ValueError(f"unknown metadata_mode: {metadata_mode}")

    receipt: SuperGLMPublicationReceipt | None = None
    if publication_receipt_path is not None:
        if publication_receipt_sha256 is None:
            raise ValueError(
                "publication_receipt_sha256 is required when publication_receipt_path is supplied"
            )
        receipt = load_publication_receipt(
            publication_receipt_path,
            expected_sha256=publication_receipt_sha256,
        )
    elif publication_receipt_sha256 is not None:
        raise ValueError(
            "publication_receipt_path is required when publication_receipt_sha256 is supplied"
        )

    if metadata_mode == "REQUIRE_SUPERGLM_RECEIPT" and receipt is None:
        raise ValueError("publication receipt is required")

    args = argparse.Namespace(
        xlsx=workbook_path,
        sheet="Rating Tables",
        export_id=export_id,
        model_name=model_name,
        model_label=None,
        target_name=target_name,
        model_type=model_type,
        model_status="ACTIVE",
        model_version=model_version,
        effective_from=effective_from,
        effective_to=effective_to,
        base_rate=None,
        base_rate_cell="C2",
        term_row=5,
        header_row=7,
        data_start_row=8,
        term_type_map_json="{}",
        interaction_features_json="{}",
        created_by=created_by,
        replace=replace,
        model_id=model_id,
    )
    export_df, rate_df, level_df = build_staging_frames(args)
    term_metadata_df = _apply_publication_receipt_metadata(
        args=args,
        export_df=export_df,
        rate_df=rate_df,
        receipt=receipt,
        receipt_sha256=publication_receipt_sha256,
    )
    insert_staging_frames(engine, args, export_df, rate_df, level_df, term_metadata_df)
