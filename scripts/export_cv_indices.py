from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.data.cv_splits import load_cv_folds  # noqa: E402
from pricing_pipeline.data.cv_splits import materialize_cv_folds  # noqa: E402
from pricing_pipeline.data.cv_splits import fetch_split_set  # noqa: E402
from pricing_pipeline.data.manifest import FREMTPL_RAW_SELECT_SQL  # noqa: E402
from scripts.pricing_db import get_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-set-id", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Write all fold indices and mark the split set MATERIALIZED.",
    )
    return parser.parse_args()


def write_fold_npz(
    folds: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    fold_no: int,
    output_path: Path,
) -> None:
    if fold_no not in folds:
        raise ValueError(f"fold {fold_no} not found; available folds: {sorted(folds)}")
    train_idx, test_idx = folds[fold_no]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, train_idx=train_idx, test_idx=test_idx)


def write_all_folds_npz(
    folds: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for fold_no, (train_idx, test_idx) in folds.items():
        arrays[f"fold_{fold_no}_train_idx"] = train_idx
        arrays[f"fold_{fold_no}_test_idx"] = test_idx
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    args = parse_args()
    engine = get_engine()

    def load_fremtpl_dataset(_manifest_id: str) -> pd.DataFrame:
        return pd.read_sql_query(text(FREMTPL_RAW_SELECT_SQL), engine)

    output_path = Path(args.out)
    if args.materialize:
        split_set = fetch_split_set(engine, args.split_set_id)
        materialize_cv_folds(
            engine,
            split_set,
            load_fremtpl_dataset(split_set.manifest_id),
            output_path=output_path,
        )
        print(f"split_set_id={args.split_set_id}")
        print("mode=MATERIALIZED")
        print(f"out={output_path}")
        return

    if args.fold is None:
        raise ValueError("--fold is required unless --materialize is set")

    folds = load_cv_folds(
        engine,
        args.split_set_id,
        dataset_loader=load_fremtpl_dataset,
    )
    write_fold_npz(folds, fold_no=args.fold, output_path=output_path)
    train_idx, test_idx = folds[args.fold]
    print(f"split_set_id={args.split_set_id}")
    print(f"fold={args.fold}")
    print(f"train_rows={len(train_idx)}")
    print(f"test_rows={len(test_idx)}")
    print(f"out={output_path}")


if __name__ == "__main__":
    main()
