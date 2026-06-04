from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_db import get_engine  # noqa: E402
from pricing_pipeline.data.manifest import (  # noqa: E402
    FREMTPL_DATASET_NAME,
    create_fremtpl_manifest,
    new_manifest_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--created-by", default="airflow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_id = args.manifest_id or new_manifest_id(FREMTPL_DATASET_NAME)
    created_manifest_id = create_fremtpl_manifest(
        get_engine(),
        manifest_id=manifest_id,
        n_splits=args.n_splits,
        random_state=args.random_state,
        created_by=args.created_by,
    )
    print(f"manifest_id={created_manifest_id}")


if __name__ == "__main__":
    main()
