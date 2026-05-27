"""Convert staged SuperGLM rating export into normalized pricing tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.publishing.package_writer import (  # noqa: E402
    load_staging_to_rating_package,
)
from scripts.pricing_db import get_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--export-id", required=True)
    p.add_argument("--created-by", default="python")
    p.add_argument("--package-status", default="DRAFT")
    p.add_argument("--set-pointer", default=None)
    return p.parse_args()


def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> int:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=pointer_name,
    )
    return load_staging_to_rating_package(engine, args)


def main() -> None:
    args = parse_args()
    engine = get_engine()
    rate_package_id = load_staging_to_rating_package(engine, args)

    print(f"rate_package_id={rate_package_id}")
    print(f"package_version={args.package_version}")
    if args.set_pointer:
        print(f"pointer={args.set_pointer}")


if __name__ == "__main__":
    main()
