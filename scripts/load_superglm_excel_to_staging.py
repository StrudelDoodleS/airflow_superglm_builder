"""Parse a SuperGLM-style Excel rating-table output into staging tables.

Expected block format, repeated across columns:

    row term_row:      <term name>
    row header_row:    Level | Relativity | Weight
    row data_start:    <level> | <relativity> | <weight>

This script intentionally stages data only. Run load_staging_to_rating_package.py next.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.publishing.staging import (  # noqa: E402
    build_staging_frames,
    insert_staging_frames,
    stage_rating_export,
)
from scripts.pricing_db import get_engine  # noqa: E402

__all__ = ["build_staging_frames", "insert_staging_frames", "stage_rating_export"]


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
