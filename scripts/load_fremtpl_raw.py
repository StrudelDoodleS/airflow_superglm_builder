from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_engine  # noqa: E402
from pricing_pipeline.fremtpl import load_fremtpl_raw  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_fremtpl_raw(get_engine(), replace=args.replace)
    print(f"fremtpl_raw_rows={rows}")


if __name__ == "__main__":
    main()
