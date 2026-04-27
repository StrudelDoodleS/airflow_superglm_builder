from __future__ import annotations

import json
import re
import uuid
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sqlalchemy import text
from sqlalchemy.engine import Engine


FREMTPL_DATASET_NAME = "freMTPL2freq"
FREMTPL_SOURCE_SYSTEM = "openml_41214"
FREMTPL_RAW_SELECT_SQL = "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol"
TRUNCATE_STAGING_SQL = "TRUNCATE TABLE pricing.STG_DATASET_ROW_KEY"
INSERT_ROW_KEYS_SQL = """
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
"""


def new_manifest_id(dataset_name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_]+", "_", dataset_name).strip("_") or "dataset"
    return f"{prefix}_{date.today():%Y%m%d}_{uuid.uuid4().hex[:10]}"


def build_column_metadata(frame: pd.DataFrame, *, manifest_id: str) -> pd.DataFrame:
    column_df = pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "ordinal_no": np.arange(1, len(frame.columns) + 1, dtype=np.int32),
            "column_name": frame.columns,
            "column_role": "FEATURE",
            "pandas_dtype": frame.dtypes.astype(str).to_numpy(),
            "null_count": frame.isna().sum().astype("int64").to_numpy(),
            "distinct_count": frame.nunique(dropna=True).astype("int64").to_numpy(),
        }
    )

    column_df.loc[column_df["column_name"].eq("IDpol"), "column_role"] = "KEY"
    column_df.loc[column_df["column_name"].eq("ClaimNb"), "column_role"] = "TARGET"
    column_df.loc[column_df["column_name"].eq("Exposure"), "column_role"] = "WEIGHT"
    return column_df


def build_row_keys(
    frame: pd.DataFrame,
    *,
    manifest_id: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    fold_no = np.empty(len(frame), dtype=np.int16)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for split_no, (_, test_idx) in enumerate(kf.split(frame), start=1):
        fold_no[test_idx] = split_no

    return pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "source_pk_text": "IDpol=" + frame["IDpol"].astype(str),
            "row_ordinal": np.arange(1, len(frame) + 1, dtype=np.int64),
            "cv_fold_no": fold_no.astype(np.int32),
        }
    )


def build_cv_splits(manifest_id: str, *, n_splits: int) -> pd.DataFrame:
    split_numbers = np.arange(1, n_splits + 1, dtype=np.int32)
    cv_split_df = pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "split_no": split_numbers,
            "test_fold_no": split_numbers,
        }
    )
    cv_split_df["train_folds_json"] = cv_split_df["test_fold_no"].map(
        lambda test_fold: json.dumps(
            [fold for fold in range(1, n_splits + 1) if fold != test_fold]
        )
    )
    return cv_split_df[
        ["manifest_id", "split_no", "train_folds_json", "test_fold_no"]
    ]


def create_fremtpl_manifest(
    engine: Engine,
    *,
    manifest_id: str,
    n_splits: int = 5,
    random_state: int = 42,
    created_by: str = "airflow",
) -> str:
    frame = pd.read_sql_query(text(FREMTPL_RAW_SELECT_SQL), engine)

    manifest_df = pd.DataFrame(
        [
            {
                "manifest_id": manifest_id,
                "dataset_name": FREMTPL_DATASET_NAME,
                "source_system": FREMTPL_SOURCE_SYSTEM,
                "data_as_of_date": date.today(),
                "row_count": int(len(frame)),
                "pk_columns_json": json.dumps(["IDpol"]),
                "target_column": "ClaimNb",
                "weight_column": "Exposure",
                "created_by": created_by,
            }
        ]
    )
    column_df = build_column_metadata(frame, manifest_id=manifest_id)
    row_key_df = build_row_keys(
        frame,
        manifest_id=manifest_id,
        n_splits=n_splits,
        random_state=random_state,
    )
    cv_split_df = build_cv_splits(manifest_id, n_splits=n_splits)

    with engine.begin() as con:
        manifest_df.to_sql(
            "DATASET_MANIFEST",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
        )
        column_df.to_sql(
            "DATASET_COLUMN",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        con.execute(text(TRUNCATE_STAGING_SQL))
        row_key_df.to_sql(
            "STG_DATASET_ROW_KEY",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
            chunksize=20000,
        )
        con.execute(text(INSERT_ROW_KEYS_SQL), {"manifest_id": manifest_id})
        cv_split_df.to_sql(
            "CV_SPLIT",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
        )

    return manifest_id
