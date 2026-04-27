from __future__ import annotations

import argparse
import os

from pricing_pipeline.config import Settings
from pricing_pipeline.db import get_engine
from pricing_pipeline.fremtpl import load_fremtpl_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env(os.environ)
    rows = load_fremtpl_raw(get_engine(settings), replace=args.replace)
    print(f"fremtpl_raw_rows={rows}")


if __name__ == "__main__":
    main()
