"""Print a quick summary of a rating package."""
from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from pricing_db import get_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rate-package-id", type=int)
    g.add_argument("--pointer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = get_engine()

    with engine.begin() as con:
        if args.pointer:
            rate_package_id = con.execute(text("""
                SELECT rate_package_id
                FROM pricing.PRICING_PACKAGE_POINTER
                WHERE pointer_name = :pointer
            """), {"pointer": args.pointer}).scalar_one()
        else:
            rate_package_id = args.rate_package_id

    pkg = pd.read_sql_query(text("""
        SELECT *
        FROM pricing.PRICING_RATE_PACKAGE
        WHERE rate_package_id = :rate_package_id
    """), engine, params={"rate_package_id": rate_package_id})

    terms = pd.read_sql_query(text("""
        SELECT
            t.term_id,
            t.term_name,
            t.term_type,
            t.sequence_no,
            COUNT(c.cell_id) AS cell_count,
            SUM(COALESCE(c.exposure_weight, 0)) AS exposure_weight_sum
        FROM pricing.PRICING_TERM t
        LEFT JOIN pricing.PRICING_RATE_CELL c
          ON c.term_id = t.term_id
        WHERE t.rate_package_id = :rate_package_id
        GROUP BY t.term_id, t.term_name, t.term_type, t.sequence_no
        ORDER BY t.sequence_no, t.term_name
    """), engine, params={"rate_package_id": rate_package_id})

    sample = pd.read_sql_query(text("""
        SELECT TOP 20
            t.term_name,
            t.term_type,
            c.cell_key_text,
            c.multiplier,
            c.exposure_weight
        FROM pricing.PRICING_TERM t
        JOIN pricing.PRICING_RATE_CELL c
          ON c.term_id = t.term_id
        WHERE t.rate_package_id = :rate_package_id
        ORDER BY t.sequence_no, c.cell_id
    """), engine, params={"rate_package_id": rate_package_id})

    print("\nPACKAGE")
    print(pkg.to_string(index=False))
    print("\nTERMS")
    print(terms.to_string(index=False))
    print("\nSAMPLE CELLS")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
