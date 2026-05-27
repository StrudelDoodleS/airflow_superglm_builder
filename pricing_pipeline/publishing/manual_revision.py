from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot


class ManualRevisionError(RuntimeError):
    """Raised when a manual rate package revision cannot be loaded."""


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
