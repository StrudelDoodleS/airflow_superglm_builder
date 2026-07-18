from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.model_registry import ModelRegistryError, get_pricing_model
from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.staging_lock import acquire_staging_export_lock
from pricing_pipeline.publishing.superglm_publication_receipt import (
    SuperGLMPublicationReceipt,
    canonical_receipt_bytes,
    load_publication_receipt,
)

INTERVAL_RE = re.compile(
    r"^\s*[\[\(]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+|inf|Inf|INF)\s*[\]\)]\s*$"
)
RANGE_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[-:]\s*([-+]?\d*\.?\d+)\s*$")
RATING_SHEET = "Rating Tables"
BASE_RATE_CELL = "C2"
TERM_ROW = 5
HEADER_ROW = 7
DATA_START_ROW = 8


@dataclass(frozen=True)
class StagingExport:
    workbook_path: Path
    export_id: str
    model_name: str
    target_name: str
    model_type: str
    model_version: str | None
    effective_from: str | None
    effective_to: str | None
    interaction_features: Mapping[str, Any]
    created_by: str
    replace: bool
    model_id: int | None


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


def infer_term_type(term_name: str, levels: pd.Series) -> str:
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


def _normalise_interaction_specs(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("interaction feature metadata must be a JSON object")
    specs: dict[str, dict[str, Any]] = {}
    for term_name, metadata in value.items():
        if isinstance(metadata, list):
            continue
        if not isinstance(metadata, Mapping):
            raise ValueError(f"interaction metadata for {term_name!r} must be an object")
        source_term_name = str(metadata.get("source_term_name") or "").strip()
        parent_names = metadata.get("parent_names")
        input_column_names = metadata.get("input_column_names")
        if (
            not source_term_name
            or not isinstance(parent_names, list)
            or not isinstance(input_column_names, list)
            or len(parent_names) != 2
            or len(input_column_names) != 2
        ):
            raise ValueError(
                f"interaction metadata for {term_name!r} must declare two ordered parents"
            )
        cleaned_parents = [str(name).strip() for name in parent_names]
        cleaned_inputs = [clean_identifier(str(name)) for name in input_column_names]
        if any(not name for name in cleaned_parents + cleaned_inputs):
            raise ValueError(
                f"interaction metadata for {term_name!r} contains an empty parent name"
            )
        specs[clean_identifier(str(term_name))] = {
            "source_term_name": source_term_name,
            "parent_names": cleaned_parents,
            "input_column_names": cleaned_inputs,
        }
    return specs


def _interaction_title_positions(
    raw: pd.DataFrame,
    interaction_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    for term_name, spec in interaction_specs.items():
        expected_titles = {str(spec["source_term_name"]), term_name}
        matches: list[tuple[int, int]] = []
        for row_index in range(raw.shape[0]):
            for column_index in range(raw.shape[1]):
                value = clean_text(raw.iat[row_index, column_index])
                if value in expected_titles:
                    matches.append((row_index, column_index))
        if len(matches) != 1:
            raise ValueError(
                f"interaction {term_name!r} must have exactly one workbook title; "
                f"found {len(matches)}"
            )
        positions[term_name] = matches[0]
    return positions


def _positive_multiplier(value: Any, *, term_name: str, location: str) -> float:
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"term {term_name!r} has a non-numeric relativity at {location}: {value!r}"
        ) from exc
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(
            f"term {term_name!r} has a non-positive or non-finite relativity "
            f"at {location}: {value!r}"
        )
    return multiplier


def _append_interaction_matrix_rows(
    *,
    raw: pd.DataFrame,
    export_id: str,
    interaction_specs: Mapping[str, Mapping[str, Any]],
    title_positions: Mapping[str, tuple[int, int]],
    rate_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
    row_id: int,
    sequence_no: int,
) -> tuple[int, int]:
    ordered_terms = sorted(interaction_specs, key=lambda name: title_positions[name][0])
    level_order = {
        (str(row["feature_name"]), str(row["level_code"])): int(row["order_index"])
        for row in level_rows
    }
    for term_position, term_name in enumerate(ordered_terms):
        sequence_no += 1
        spec = interaction_specs[term_name]
        parent_names = list(spec["parent_names"])
        input_columns = list(spec["input_column_names"])
        title_row, _title_column = title_positions[term_name]
        boundary_row = (
            title_positions[ordered_terms[term_position + 1]][0]
            if term_position + 1 < len(ordered_terms)
            else raw.shape[0]
        )

        header: tuple[int, int, list[str]] | None = None
        expected_left_headers = {parent_names[0], input_columns[0]}
        for header_row in range(title_row + 1, boundary_row):
            for left_column in range(raw.shape[1]):
                if clean_text(raw.iat[header_row, left_column]) not in expected_left_headers:
                    continue
                top_levels: list[str] = []
                for column_index in range(left_column + 1, raw.shape[1]):
                    top_level = clean_text(raw.iat[header_row, column_index])
                    if top_level is None:
                        break
                    top_levels.append(top_level)
                if top_levels:
                    header = (header_row, left_column, top_levels)
                    break
            if header is not None:
                break
        if header is None:
            raise ValueError(
                f"interaction {term_name!r} has no matrix header for {parent_names[0]!r}"
            )
        header_row, left_column, top_levels = header
        if len(set(top_levels)) != len(top_levels):
            raise ValueError(f"interaction {term_name!r} has duplicate column levels")

        left_levels_seen: set[str] = set()
        matrix_cell_count = 0
        for matrix_row in range(header_row + 1, boundary_row):
            left_level = clean_text(raw.iat[matrix_row, left_column])
            if left_level is None:
                break
            if left_level in left_levels_seen:
                raise ValueError(
                    f"interaction {term_name!r} has duplicate row level {left_level!r}"
                )
            left_levels_seen.add(left_level)
            left_order = level_order.get(
                (input_columns[0], left_level),
                len(left_levels_seen),
            )
            level_order[(input_columns[0], left_level)] = left_order

            for top_index, top_level in enumerate(top_levels, start=1):
                value_column = left_column + top_index
                if value_column >= raw.shape[1] or pd.isna(raw.iat[matrix_row, value_column]):
                    raise ValueError(
                        f"interaction {term_name!r} has a ragged matrix at "
                        f"row {matrix_row + 1}, column {value_column + 1}"
                    )
                multiplier = _positive_multiplier(
                    raw.iat[matrix_row, value_column],
                    term_name=term_name,
                    location=f"row {matrix_row + 1}, column {value_column + 1}",
                )
                top_order = level_order.get(
                    (input_columns[1], top_level),
                    top_index,
                )
                level_order[(input_columns[1], top_level)] = top_order
                row_id += 1
                matrix_cell_count += 1
                cell_key = (
                    f"{term_name}={input_columns[0]}={left_level}|{input_columns[1]}={top_level}"
                )
                rate_rows.append(
                    {
                        "export_id": export_id,
                        "row_id": row_id,
                        "term_name": term_name,
                        "term_type": "CATEGORICAL_INTERACTION",
                        "sequence_no": sequence_no,
                        "cell_key_text": cell_key,
                        "multiplier": multiplier,
                        "log_coefficient": float(np.log(multiplier)),
                        "exposure_weight": None,
                        "record_count": None,
                        "is_reference": 1 if np.isclose(multiplier, 1.0) else 0,
                        "is_default": 0,
                    }
                )
                for position_no, (feature_name, level_code, order_index) in enumerate(
                    (
                        (input_columns[0], left_level, left_order),
                        (input_columns[1], top_level, top_order),
                    ),
                    start=1,
                ):
                    lower_value = level_code.lower()
                    level_rows.append(
                        {
                            "export_id": export_id,
                            "row_id": row_id,
                            "position_no": position_no,
                            "feature_name": feature_name,
                            "feature_value_type": "CATEGORICAL",
                            "level_set_name": f"{feature_name}__{export_id}",
                            "level_set_type": "CATEGORICAL",
                            "level_code": level_code,
                            "level_label": level_code,
                            "order_index": order_index,
                            "lower_bound": None,
                            "upper_bound": None,
                            "representative_value": None,
                            "is_missing": 1
                            if lower_value in {"missing", "na", "nan", "null"}
                            else 0,
                            "is_other": 1 if lower_value in {"other", "else"} else 0,
                        }
                    )
        if not left_levels_seen or matrix_cell_count == 0:
            raise ValueError(f"interaction {term_name!r} has an empty matrix")
    return row_id, sequence_no


def build_staging_frames(
    args: StagingExport,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(
        args.workbook_path,
        sheet_name=RATING_SHEET,
        header=None,
        engine="openpyxl",
    )

    r, c = cell_to_zero_index(BASE_RATE_CELL)
    base_rate = float(raw.iat[r, c])

    interaction_features = args.interaction_features
    interaction_specs = _normalise_interaction_specs(interaction_features)
    interaction_titles = _interaction_title_positions(raw, interaction_specs)
    main_effect_stop = (
        min(row for row, _column in interaction_titles.values())
        if interaction_titles
        else raw.shape[0]
    )

    blocks = find_blocks(raw, TERM_ROW, HEADER_ROW)
    if not blocks:
        raise RuntimeError("No rating table blocks found in the standard rating-table layout.")

    export_df = pd.DataFrame(
        [
            {
                "export_id": args.export_id,
                "model_name": args.model_name,
                "model_version": args.model_version,
                "base_rate": base_rate,
                "effective_from_date": args.effective_from,
                "effective_to_date": args.effective_to,
                "source_file": str(Path(args.workbook_path).resolve()),
                "created_by": args.created_by,
            }
        ]
    )

    rate_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    row_id = 0
    sequence_no = 0
    start = DATA_START_ROW - 1

    for block in blocks:
        sequence_no += 1
        term_name = block["term_name"]
        level_col = block["level_col"]
        mult_col = block["mult_col"]
        weight_col = block["weight_col"]

        block_df = raw.iloc[start:main_effect_stop, [level_col, mult_col, weight_col]].copy()
        block_df.columns = ["level_code", "multiplier", "exposure_weight"]
        block_df = block_df.dropna(subset=["level_code", "multiplier"], how="any")
        if block_df.empty:
            continue

        term_type = infer_term_type(term_name, block_df["level_code"])
        is_band = term_type in {
            "DISCRETIZED_SPLINE_1D",
            "NUMERIC_BANDED_1D",
            "ORDERED_CATEGORICAL_MAIN",
        }

        features = interaction_features.get(term_name)
        if features:
            term_type = "CATEGORICAL_INTERACTION"

        for order_index, rec in enumerate(block_df.to_dict("records"), start=1):
            row_id += 1
            level_code = str(rec["level_code"]).strip()
            multiplier = _positive_multiplier(
                rec["multiplier"],
                term_name=term_name,
                location=f"main-effect row {start + order_index + 1}",
            )
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

    row_id, sequence_no = _append_interaction_matrix_rows(
        raw=raw,
        export_id=args.export_id,
        interaction_specs=interaction_specs,
        title_positions=interaction_titles,
        rate_rows=rate_rows,
        level_rows=level_rows,
        row_id=row_id,
        sequence_no=sequence_no,
    )
    rate_df = pd.DataFrame(rate_rows)
    level_df = pd.DataFrame(level_rows)
    return export_df, rate_df, level_df


def _resolve_registered_model_id(con, args: StagingExport) -> int:
    model_id = args.model_id
    if model_id is not None:
        return int(model_id)

    record = get_pricing_model(con, args.model_name)
    if record is None:
        raise ModelRegistryError(
            f"model_name {args.model_name!r} is not registered; "
            "run explicit model registration first"
        )

    mismatches: list[str] = []
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


def _canonical_staging_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_) and missing:
        return None
    if isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return value.hex()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _canonical_staging_frame(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    content = frame.drop(columns=["staging_content_sha256"], errors="ignore")
    columns = sorted(str(column) for column in content.columns)
    rows = [
        [_canonical_staging_value(row[column]) for column in columns]
        for _, row in content.loc[:, columns].iterrows()
    ]
    rows.sort(
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return {"name": name, "columns": columns, "rows": rows}


def staging_content_sha256(
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
    term_metadata_df: pd.DataFrame | None = None,
) -> str:
    """Return a canonical digest binding every row staged for one export."""

    term_frame = _empty_term_metadata_frame() if term_metadata_df is None else term_metadata_df
    payload = [
        _canonical_staging_frame("rating_export", export_df),
        _canonical_staging_frame("rate_cell", rate_df),
        _canonical_staging_frame("cell_level", level_df),
        _canonical_staging_frame("term_metadata", term_frame),
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _term_metadata_frame(
    export_id: str,
    receipt: SuperGLMPublicationReceipt,
    *,
    staged_terms: set[str] | None = None,
) -> pd.DataFrame:
    receipt_data = receipt.model_dump(mode="json")
    term_metadata = receipt_data["term_metadata"]
    term_names = sorted(staged_terms if staged_terms is not None else term_metadata)
    rows = [
        {
            "export_id": export_id,
            "term_name": term_name,
            "term_metadata_json": _deterministic_json(term_metadata[term_name]),
        }
        for term_name in term_names
    ]
    return pd.DataFrame(rows, columns=["export_id", "term_name", "term_metadata_json"])


def _metadata_feature_kind(metadata: Mapping[str, Any]) -> str | None:
    feature_kind = metadata.get("feature_kind")
    if feature_kind is None:
        return None
    return str(feature_kind)


def _receipt_interaction_features(
    receipt: SuperGLMPublicationReceipt | None,
) -> dict[str, dict[str, Any]]:
    if receipt is None:
        return {}
    interaction_features: dict[str, dict[str, Any]] = {}
    for term_name, metadata in receipt.model_dump(mode="json")["term_metadata"].items():
        if _metadata_feature_kind(metadata) != "categorical_interaction":
            continue
        source_term_name = metadata.get("source_term_name")
        parent_names = metadata.get("parent_names")
        input_column_names = metadata.get("input_column_names")
        if (
            not isinstance(source_term_name, str)
            or not source_term_name.strip()
            or not isinstance(parent_names, list)
            or not isinstance(input_column_names, list)
            or len(parent_names) != 2
            or len(input_column_names) != 2
        ):
            raise ValueError(
                f"publication receipt interaction {term_name!r} has invalid parent metadata"
            )
        interaction_features[term_name] = {
            "source_term_name": source_term_name,
            "parent_names": parent_names,
            "input_column_names": input_column_names,
        }
    return interaction_features


def _receipt_term_type(
    *,
    term_name: str,
    existing_term_type: str,
    levels: pd.Series,
    metadata: Mapping[str, Any],
) -> str:
    feature_kind = _metadata_feature_kind(metadata)
    if feature_kind == "offset":
        return "OFFSET_FACTOR"
    if feature_kind == "numeric":
        return "NUMERIC_MAIN"
    if feature_kind == "categorical":
        return "CATEGORICAL_MAIN"
    if feature_kind == "categorical_interaction":
        return "CATEGORICAL_INTERACTION"
    if feature_kind == "ordered_categorical":
        return "ORDERED_CATEGORICAL_MAIN"
    if feature_kind == "spline":
        return "DISCRETIZED_SPLINE_1D"
    if feature_kind == "polynomial":
        non_null = levels.dropna().astype(str)
        if len(non_null) and non_null.map(lambda x: parse_interval(x)[0] is not None).mean() > 0.8:
            return "NUMERIC_BANDED_1D"
        return "POLYNOMIAL_MAIN"
    if feature_kind in {None, "unknown"}:
        return existing_term_type
    raise ValueError(
        f"publication receipt for term {term_name!r} has unsupported feature_kind {feature_kind!r}"
    )


def _validate_numeric_main_staging(
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
) -> None:
    numeric_terms = sorted(
        rate_df.loc[rate_df["term_type"] == "NUMERIC_MAIN", "term_name"]
        .dropna()
        .astype(str)
        .unique()
    )
    for term_name in numeric_terms:
        term_rows = rate_df[
            (rate_df["term_name"] == term_name) & (rate_df["term_type"] == "NUMERIC_MAIN")
        ]
        if len(term_rows) != 1:
            raise ValueError(
                f"numeric main term {term_name!r} must contain exactly one per_unit cell"
            )
        term_row = term_rows.iloc[0]
        log_coefficient = float(term_row["log_coefficient"])
        if not math.isfinite(log_coefficient):
            raise ValueError(f"numeric main term {term_name!r} has a non-finite log coefficient")
        matching_levels = level_df[level_df["row_id"] == term_row["row_id"]]
        if len(matching_levels) != 1:
            raise ValueError(f"numeric main term {term_name!r} must map exactly one feature level")
        level_index = matching_levels.index[0]
        level = matching_levels.iloc[0]
        if int(level["position_no"]) != 1 or str(level["level_code"]).lower() != "per_unit":
            raise ValueError(
                f"numeric main term {term_name!r} must use a position-1 per_unit level"
            )
        level_df.loc[level_index, "feature_value_type"] = "NUMERIC"


def _apply_publication_receipt_metadata(
    *,
    args: StagingExport,
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
    receipt: SuperGLMPublicationReceipt | None,
    receipt_sha256: str | None,
) -> pd.DataFrame:
    if receipt is None:
        return _empty_term_metadata_frame()

    receipt_data = receipt.model_dump(mode="json")
    term_metadata: dict[str, dict[str, Any]] = dict(receipt_data["term_metadata"])
    staged_terms = set(rate_df["term_name"].dropna().astype(str).unique())
    receipt_terms = set(term_metadata)
    missing_metadata = sorted(staged_terms - receipt_terms)
    if missing_metadata:
        raise ValueError(
            "publication receipt metadata is missing for staged workbook term(s): "
            + ", ".join(missing_metadata)
        )
    missing_workbook_terms = sorted(receipt_terms - staged_terms)
    if missing_workbook_terms:
        raise ValueError(
            "publication receipt term metadata is not present in staged workbook term(s): "
            + ", ".join(missing_workbook_terms)
        )

    for term_name in sorted(staged_terms):
        matching_term = rate_df["term_name"] == term_name
        levels = rate_df.loc[matching_term, "cell_key_text"].astype(str).str.split("=", n=1).str[-1]
        rate_df.loc[matching_term, "term_type"] = _receipt_term_type(
            term_name=term_name,
            existing_term_type=str(rate_df.loc[matching_term, "term_type"].iloc[0]),
            levels=levels,
            metadata=term_metadata[term_name],
        )

    _validate_numeric_main_staging(rate_df, level_df)

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
    elif (
        offset_contract.handling == "ALREADY_APPLIED_SQL_EXPOSURE"
        and (rate_df["term_type"] == "OFFSET_FACTOR").any()
    ):
        raise ValueError(
            "publication receipt offset handling ALREADY_APPLIED_SQL_EXPOSURE "
            "cannot stage OFFSET_FACTOR terms"
        )

    export_df["publication_receipt_json"] = canonical_receipt_bytes(receipt).decode("utf-8")
    export_df["publication_receipt_sha256"] = receipt_sha256
    export_df["package_metadata_json"] = _deterministic_json(receipt_data["package_metadata"])
    export_df["offset_handling"] = offset_contract.handling
    export_df["offset_factor_name"] = offset_contract.published_factor_name
    export_df["offset_source_name"] = offset_contract.source_name
    export_df["offset_label"] = offset_contract.label
    export_df["metadata_origin"] = receipt.metadata_origin

    return _term_metadata_frame(args.export_id, receipt, staged_terms=staged_terms)


def insert_staging_frames(
    engine,
    args: StagingExport,
    export_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_df: pd.DataFrame,
    term_metadata_df: pd.DataFrame | None = None,
    staging_content_sha256: str | None = None,
    *,
    expected_database: str | None,
) -> None:
    if (
        staging_content_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", staging_content_sha256) is None
    ):
        raise ValueError("staging_content_sha256 must be a lowercase SHA-256 digest")
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        _verify_expected_database(con, expected_database)
        acquire_staging_export_lock(con, args.export_id)
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
                "SET model_id = :model_id, "
                "staging_content_sha256 = :staging_content_sha256 "
                "WHERE export_id = :export_id"
            ),
            {
                "export_id": args.export_id,
                "model_id": model_id,
                "staging_content_sha256": staging_content_sha256,
            },
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
    expected_database: str | None,
    model_name: str,
    model_version: str | None,
    effective_from: str | None,
    target_name: str = "ClaimNb",
    model_type: str = "superglm_poisson",
    effective_to: str | None = None,
    created_by: str = "python",
    replace: bool = False,
    model_id: int | None = None,
    publication_receipt_path: str | Path,
    publication_receipt_sha256: str,
) -> str:
    receipt = load_publication_receipt(
        publication_receipt_path,
        expected_sha256=publication_receipt_sha256,
    )

    args = StagingExport(
        workbook_path=workbook_path,
        export_id=export_id,
        model_name=model_name,
        target_name=target_name,
        model_type=model_type,
        model_version=model_version,
        effective_from=effective_from,
        effective_to=effective_to,
        interaction_features=_receipt_interaction_features(receipt),
        created_by=created_by,
        replace=replace,
        model_id=model_id,
    )
    export_df, rate_df, level_df = build_staging_frames(args)
    term_metadata_df = _apply_publication_receipt_metadata(
        args=args,
        export_df=export_df,
        rate_df=rate_df,
        level_df=level_df,
        receipt=receipt,
        receipt_sha256=publication_receipt_sha256,
    )
    content_sha256 = staging_content_sha256(
        export_df,
        rate_df,
        level_df,
        term_metadata_df,
    )
    insert_staging_frames(
        engine,
        args,
        export_df,
        rate_df,
        level_df,
        term_metadata_df,
        staging_content_sha256=content_sha256,
        expected_database=expected_database,
    )
    return content_sha256


def _verify_expected_database(connection, expected_database: str | None) -> None:
    if expected_database is None:
        dialect_name = str(getattr(getattr(connection, "dialect", None), "name", "")).casefold()
        if dialect_name == "sqlite":
            return
        raise ValueError("expected_database is required for non-SQLite staging writes")
    expected = str(expected_database).strip()
    if not expected:
        raise ValueError("expected_database must be a non-empty database name or None")
    actual_value = connection.execute(text("SELECT DB_NAME()")).scalar_one()
    actual = str(actual_value or "").strip()
    if not actual:
        raise RuntimeError("Remote connection did not report a database name")
    if actual.casefold() != expected.casefold():
        raise RuntimeError(
            "Remote database mismatch: "
            f"expected {expected!r}, connected to {actual!r}; "
            "staging transaction aborted before writes"
        )
