from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra.config import Settings  # noqa: E402
from pricing_pipeline.infra.db import get_engine  # noqa: E402
from pricing_pipeline.tools.db_diagrams import (  # noqa: E402
    load_schema_metadata,
    prepare_display_metadata,
    write_diagram_site,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static SQL Server ERD site from catalog metadata.",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=["pricing"],
        help="SQL Server schema names to include. Defaults to pricing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/pricing/state/db_diagrams"),
        help="Directory that receives index.html, schema.mmd, and metadata.json.",
    )
    parser.add_argument(
        "--include-staging",
        action="store_true",
        help="Include STG_* import tables in the generated diagrams.",
    )
    parser.add_argument(
        "--include-row-keys",
        action="store_true",
        help="Include row-key materialization tables in the generated diagrams.",
    )
    return parser.parse_args()


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    args = parse_args()
    settings = Settings.from_env(os.environ)
    engine = get_engine(settings, database=settings.pricing_database)
    metadata = load_schema_metadata(engine, args.schemas)
    display_metadata = prepare_display_metadata(
        metadata,
        include_staging=args.include_staging,
        include_row_keys=args.include_row_keys,
    )
    write_diagram_site(
        metadata,
        output_dir=args.output_dir,
        database_name=settings.pricing_database,
        schema_names=args.schemas,
        include_staging=args.include_staging,
        include_row_keys=args.include_row_keys,
    )
    print(
        "generated_db_diagrams="
        f"{args.output_dir} source_tables={len(metadata.tables)} "
        f"visible_tables={len(display_metadata.tables)} "
        f"visible_fk_columns={len(display_metadata.foreign_keys)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
