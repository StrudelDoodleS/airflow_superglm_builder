from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_engine  # noqa: I001


RESET_SQL = """
DELETE FROM mlops.MODEL_RUN_SPLIT_SET;
DELETE FROM mlops.MODEL_RUN_DATASET;
DELETE FROM mlops.MODEL_RUN_METRIC;
DELETE FROM pricing.CV_FOLD_METRIC;
DELETE FROM pricing.MODEL_RUN;
DELETE FROM pricing.PRICING_MODEL_DEPLOYMENT;
DELETE FROM pricing.PRICING_PACKAGE_POINTER;
DELETE FROM pricing.PRICING_COMPILED_1D_RATE_BAND;
DELETE FROM pricing.PRICING_COMPILED_RATE_CELL;
DELETE FROM pricing.PRICING_RATE_CELL_LEVEL;
DELETE FROM pricing.PRICING_RATE_CELL;
DELETE FROM pricing.PRICING_TERM_FEATURE;
DELETE FROM pricing.PRICING_TERM;
DELETE FROM pricing.PRICING_RATE_PACKAGE;
DELETE FROM pricing.PRICING_FEATURE_LEVEL;
DELETE FROM pricing.PRICING_FEATURE_LEVEL_SET;
DELETE FROM pricing.PRICING_FEATURE;
DELETE FROM pricing.CV_FOLD;
DELETE FROM pricing.CV_SPLIT_SET;
DELETE FROM pricing.DATASET_COLUMN;
DELETE FROM pricing.DATASET_MANIFEST;
DELETE FROM pricing_stg.STG_CELL_LEVEL;
DELETE FROM pricing_stg.STG_RATE_CELL;
DELETE FROM pricing_stg.STG_RATING_EXPORT;
DELETE FROM pricing.PRICING_MODEL;
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of pricing experiment history in the local database.",
    )
    return parser


def _monitoring_history_exists(con, *, sqlite: bool) -> bool:
    table_probe = (
        "SELECT 1 FROM pricing.sqlite_master "
        "WHERE type = 'table' AND name = 'MODEL_FIT_CONTRACT' LIMIT 1"
        if sqlite
        else "SELECT 1 WHERE OBJECT_ID('mlops.MODEL_FIT_CONTRACT', 'U') IS NOT NULL"
    )
    if con.execute(text(table_probe)).scalar_one_or_none() is None:
        return False
    sql = (
        "SELECT 1 FROM pricing.MODEL_FIT_CONTRACT LIMIT 1"
        if sqlite
        else "SELECT TOP (1) 1 FROM mlops.MODEL_FIT_CONTRACT"
    )
    return con.execute(text(sql)).scalar_one_or_none() is not None


def reset_pricing_experiments() -> None:
    engine = get_engine()
    with engine.begin() as con:
        if _monitoring_history_exists(con, sqlite=engine.dialect.name == "sqlite"):
            raise SystemExit(
                "Refusing to reset pricing experiments while immutable monitoring "
                "history exists. Run: uv run python "
                "scripts/reset_remote_pricing_schema.py --execute "
                "--i-understand-this-drops-pricing-objects"
            )
        for statement in RESET_SQL.strip().split(";"):
            sql = statement.strip()
            if sql:
                con.execute(text(sql))


def main() -> None:
    args = build_parser().parse_args()
    if not args.yes:
        raise SystemExit("Refusing to reset pricing experiment tables without --yes")
    reset_pricing_experiments()
    print("reset_pricing_experiments=ok")


if __name__ == "__main__":
    main()
