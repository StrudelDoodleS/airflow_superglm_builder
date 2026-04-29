"""Parse a SuperGLM-style Excel rating-table output into staging tables.

Expected block format, repeated across columns:

    row term_row:      <term name>
    row header_row:    Level | Relativity | Weight
    row data_start:    <level> | <relativity> | <weight>

This script intentionally stages data only. Run load_staging_to_rating_package.py next.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_engine  # noqa: E402
from pricing_pipeline.model_registry import ensure_pricing_model  # noqa: E402

INTERVAL_RE = re.compile(
    r"^\s*[\[\(]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+|inf|Inf|INF)\s*[\]\)]\s*$"
)
RANGE_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[-:]\s*([-+]?\d*\.?\d+)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--sheet", default="Rating Tables")
    p.add_argument("--export-id", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-label", default=None)
    p.add_argument("--target-name", default="ClaimNb")
    p.add_argument("--model-type", default="superglm_poisson")
    p.add_argument("--model-status", default="ACTIVE")
    p.add_argument("--model-version", default=None)
    p.add_argument("--effective-from", required=True)
    p.add_argument("--effective-to", default=None)
    p.add_argument("--base-rate", type=float, default=None)
    p.add_argument("--base-rate-cell", default="C2")
    p.add_argument("--term-row", type=int, default=5)
    p.add_argument("--header-row", type=int, default=7)
    p.add_argument("--data-start-row", type=int, default=8)
    p.add_argument("--term-type-map-json", default="{}")
    p.add_argument("--interaction-features-json", default="{}")
    p.add_argument("--created-by", default="python")
    p.add_argument("--replace", action="store_true")
    return p.parse_args()


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


def clean_identifier(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


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
            blocks.append({
                "term_name": clean_identifier(term_name),
                "level_col": c,
                "mult_col": c + 1,
                "weight_col": c + 2,
            })

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


def build_staging_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    export_df = pd.DataFrame([{
        "export_id": args.export_id,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "base_rate": base_rate,
        "effective_from_date": args.effective_from,
        "effective_to_date": args.effective_to,
        "source_file": str(Path(args.xlsx).resolve()),
        "created_by": args.created_by,
    }])

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
        is_band = term_type in {"DISCRETIZED_SPLINE_1D", "NUMERIC_BANDED_1D", "ORDERED_CATEGORICAL_MAIN"}

        features = interaction_features.get(term_name)
        if features:
            term_type = term_type_map.get(term_name, "CATEGORICAL_INTERACTION")

        for order_index, rec in enumerate(block_df.to_dict("records"), start=1):
            row_id += 1
            level_code = str(rec["level_code"]).strip()
            multiplier = float(rec["multiplier"])
            exposure_weight = None if pd.isna(rec.get("exposure_weight")) else float(rec["exposure_weight"])
            cell_key = f"{term_name}={level_code}"

            rate_rows.append({
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
            })

            if features:
                pairs = split_interaction_level(level_code, features)
            else:
                pairs = [(term_name, level_code)]

            for position_no, (feature_name, lv_code) in enumerate(pairs, start=1):
                lo, hi, rep = parse_interval(lv_code)
                level_set_type = "NUMERIC_BAND" if lo is not None else "CATEGORICAL"
                if len(pairs) == 1 and term_type == "DISCRETIZED_SPLINE_1D":
                    level_set_type = "SPLINE_GRID_1D"

                level_rows.append({
                    "export_id": args.export_id,
                    "row_id": row_id,
                    "position_no": position_no,
                    "feature_name": feature_name,
                    "feature_value_type": "NUMERIC" if lo is not None or is_band else "CATEGORICAL",
                    "level_set_name": f"{feature_name}__{args.export_id}",
                    "level_set_type": level_set_type,
                    "level_code": lv_code,
                    "level_label": lv_code,
                    "order_index": order_index,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "representative_value": rep,
                    "is_missing": 1 if lv_code.lower() in {"missing", "na", "nan", "null"} else 0,
                    "is_other": 1 if lv_code.lower() in {"other", "else"} else 0,
                })

    rate_df = pd.DataFrame(rate_rows)
    level_df = pd.DataFrame(level_rows)
    return export_df, rate_df, level_df


def insert_staging_frames(
    engine,
    args: argparse.Namespace,
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
) -> None:
    with engine.begin() as con:
        model_id = ensure_pricing_model(
            con,
            model_key=args.model_name,
            model_label=args.model_label,
            target_name=args.target_name,
            model_type=args.model_type,
            model_status=args.model_status,
            created_by=args.created_by,
        )

        if args.replace:
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
            schema="pricing_stg",
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
            schema="pricing_stg",
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        level_df.to_sql(
            "STG_CELL_LEVEL",
            con,
            schema="pricing_stg",
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
) -> None:
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
    )
    export_df, rate_df, level_df = build_staging_frames(args)
    insert_staging_frames(engine, args, export_df, rate_df, level_df)


def main() -> None:
    args = parse_args()
    engine = get_engine()
    export_df, rate_df, level_df = build_staging_frames(args)
    insert_staging_frames(engine, args, export_df, rate_df, level_df)

    print(f"export_id={args.export_id}")
    print(f"terms={rate_df['term_name'].nunique()}")
    print(f"rate_cells={len(rate_df):,}")
    print(f"cell_levels={len(level_df):,}")


if __name__ == "__main__":
    main()
