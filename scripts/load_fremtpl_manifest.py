"""Load freMTPL/OpenML row-key metadata and 5-fold CV assignments.

Stores only row keys, hashes, column metadata, and fold assignment.
Does not store the full dataset.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import date

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sqlalchemy import text

from pricing_db import get_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-id", type=int, default=41214)
    p.add_argument("--dataset-name", default="freMTPL2freq")
    p.add_argument("--source-system", default="openml_41214")
    p.add_argument("--pk-col", default="IDpol")
    p.add_argument("--target-col", default="ClaimNb")
    p.add_argument("--weight-col", default="Exposure")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--manifest-id", default=None)
    p.add_argument("--created-by", default="python")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = get_engine()

    df = fetch_openml(data_id=args.data_id, as_frame=True).frame.reset_index(drop=True).copy()

    if args.pk_col not in df.columns:
        df.insert(0, args.pk_col, np.arange(1, len(df) + 1))

    manifest_id = args.manifest_id or f"{args.dataset_name}_{uuid.uuid4().hex[:10]}"

    # one small loop over folds only; all row-level work is vectorized/SQL
    fold_no = np.empty(len(df), dtype=np.int16)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    for k, (_, test_idx) in enumerate(kf.split(df), start=1):
        fold_no[test_idx] = k
    df["cv_fold_no"] = fold_no

    manifest_df = pd.DataFrame([{
        "manifest_id": manifest_id,
        "dataset_name": args.dataset_name,
        "source_system": args.source_system,
        "data_as_of_date": date.today(),
        "row_count": int(len(df)),
        "pk_columns_json": json.dumps([args.pk_col]),
        "target_column": args.target_col,
        "weight_column": args.weight_col,
        "created_by": args.created_by,
    }])

    manifest_df.to_sql(
        "DATASET_MANIFEST",
        engine,
        schema="pricing",
        if_exists="append",
        index=False,
    )

    column_df = pd.DataFrame({
        "manifest_id": manifest_id,
        "ordinal_no": np.arange(1, len(df.columns) + 1),
        "column_name": df.columns,
        "pandas_dtype": df.dtypes.astype(str).values,
        "null_count": df.isna().sum().astype(int).values,
        "distinct_count": df.nunique(dropna=True).astype(int).values,
    })

    column_df["column_role"] = "FEATURE"
    column_df.loc[column_df["column_name"].eq(args.pk_col), "column_role"] = "KEY"
    column_df.loc[column_df["column_name"].eq(args.target_col), "column_role"] = "TARGET"
    column_df.loc[column_df["column_name"].eq(args.weight_col), "column_role"] = "WEIGHT"
    column_df.loc[column_df["column_name"].eq("cv_fold_no"), "column_role"] = "CV_FOLD"

    column_df[[
        "manifest_id",
        "ordinal_no",
        "column_name",
        "column_role",
        "pandas_dtype",
        "null_count",
        "distinct_count",
    ]].to_sql(
        "DATASET_COLUMN",
        engine,
        schema="pricing",
        if_exists="append",
        index=False,
        chunksize=5000,
    )

    row_key_df = pd.DataFrame({
        "manifest_id": manifest_id,
        "source_pk_text": args.pk_col + "=" + df[args.pk_col].astype(str),
        "row_ordinal": np.arange(1, len(df) + 1, dtype=np.int64),
        "cv_fold_no": df["cv_fold_no"].astype(int),
    })

    with engine.begin() as con:
        con.execute(text("TRUNCATE TABLE pricing.STG_DATASET_ROW_KEY"))

    row_key_df.to_sql(
        "STG_DATASET_ROW_KEY",
        engine,
        schema="pricing",
        if_exists="append",
        index=False,
        chunksize=20000,
    )

    with engine.begin() as con:
        con.execute(text("""
        INSERT INTO pricing.DATASET_ROW_KEY (
            manifest_id,
            row_key_hash,
            source_pk_text,
            row_ordinal,
            cv_fold_no
        )
        SELECT
            manifest_id,
            HASHBYTES('SHA2_256', source_pk_text),
            source_pk_text,
            row_ordinal,
            cv_fold_no
        FROM pricing.STG_DATASET_ROW_KEY
        WHERE manifest_id = :manifest_id;
        """), {"manifest_id": manifest_id})

    cv_split_df = pd.DataFrame({
        "manifest_id": manifest_id,
        "split_no": np.arange(1, args.n_splits + 1),
        "test_fold_no": np.arange(1, args.n_splits + 1),
    })
    cv_split_df["train_folds_json"] = cv_split_df["test_fold_no"].map(
        lambda test_fold: json.dumps([f for f in range(1, args.n_splits + 1) if f != test_fold])
    )

    cv_split_df[["manifest_id", "split_no", "train_folds_json", "test_fold_no"]].to_sql(
        "CV_SPLIT",
        engine,
        schema="pricing",
        if_exists="append",
        index=False,
    )

    print(f"manifest_id={manifest_id}")
    print(f"rows={len(df):,}")


if __name__ == "__main__":
    main()
