from __future__ import annotations

import json
import re
import hashlib
import platform
import sys
import uuid
from datetime import date
from importlib import metadata

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.models.spec import DatasetSpec


FREMTPL_DATASET_NAME = "freMTPL2freq"
FREMTPL_SOURCE_SYSTEM = "openml_41214"
FREMTPL_RAW_SELECT_SQL = "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol"


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def runtime_dependency_metadata() -> str:
    payload = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": _package_version("scikit-learn"),
            "superglm": _package_version("superglm"),
        },
    }
    return json.dumps(payload, sort_keys=True)


def new_manifest_id(dataset_name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_]+", "_", dataset_name).strip("_") or "dataset"
    return f"{prefix}_{date.today():%Y%m%d}_{uuid.uuid4().hex[:10]}"


def build_column_metadata(
    frame: pd.DataFrame,
    *,
    manifest_id: str,
    pk_columns: tuple[str, ...] = ("IDpol",),
    target_column: str | None = "ClaimNb",
    weight_column: str | None = "Exposure",
) -> pd.DataFrame:
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

    column_df.loc[column_df["column_name"].isin(pk_columns), "column_role"] = "KEY"
    if target_column is not None:
        column_df.loc[column_df["column_name"].eq(target_column), "column_role"] = "TARGET"
    if weight_column is not None:
        column_df.loc[column_df["column_name"].eq(weight_column), "column_role"] = "WEIGHT"
    return column_df


def compute_row_order_sha256(
    frame: pd.DataFrame,
    *,
    pk_column: str | None = None,
    pk_columns: tuple[str, ...] | None = None,
) -> str:
    if pk_columns is None:
        if pk_column is None:
            raise ValueError("pk_column or pk_columns is required")
        pk_columns = (pk_column,)

    digest = hashlib.sha256()
    for row in frame.loc[:, list(pk_columns)].itertuples(index=False, name=None):
        digest.update(json.dumps(row, default=str, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_set_id_for_manifest(
    manifest_id: str,
    *,
    n_splits: int,
    random_state: int,
) -> str:
    return f"{manifest_id}__kfold_{n_splits}_seed_{random_state}"


def build_cv_split_set(
    frame: pd.DataFrame,
    *,
    manifest_id: str,
    n_splits: int,
    random_state: int,
    pk_columns: tuple[str, ...] = ("IDpol",),
    created_by: str = "airflow",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_set_id": split_set_id_for_manifest(
                    manifest_id,
                    n_splits=n_splits,
                    random_state=random_state,
                ),
                "manifest_id": manifest_id,
                "split_mode": "REPLAYABLE",
                "splitter_class": "sklearn.model_selection.KFold",
                "splitter_params_json": json.dumps(
                    {
                        "n_splits": n_splits,
                        "shuffle": True,
                        "random_state": random_state,
                    },
                    sort_keys=True,
                ),
                "row_order_sha256": compute_row_order_sha256(frame, pk_columns=pk_columns),
                "row_count": int(len(frame)),
                "fold_count": n_splits,
                "groups_column": None,
                "stratify_column": None,
                "artifact_uri": None,
                "artifact_sha256": None,
                "runtime_metadata_json": runtime_dependency_metadata(),
                "created_by": created_by,
            }
        ]
    )


def build_cv_folds(
    frame: pd.DataFrame,
    *,
    split_set_id: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_no, (train_idx, test_idx) in enumerate(kf.split(frame), start=1):
        rows.append(
            {
                "split_set_id": split_set_id,
                "fold_no": fold_no,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return pd.DataFrame(rows)


def create_dataset_manifest(
    engine: Engine,
    *,
    dataset: DatasetSpec,
    manifest_id: str,
    n_splits: int = 5,
    random_state: int = 42,
    created_by: str = "airflow",
) -> str:
    frame = pd.read_sql_query(text(dataset.manifest_sql), engine)

    manifest_df = pd.DataFrame(
        [
            {
                "manifest_id": manifest_id,
                "dataset_name": dataset.dataset_name,
                "source_system": dataset.source_system,
                "data_as_of_date": date.today(),
                "row_count": int(len(frame)),
                "pk_columns_json": json.dumps(list(dataset.pk_columns)),
                "target_column": dataset.target_column,
                "weight_column": dataset.weight_column,
                "created_by": created_by,
            }
        ]
    )
    column_df = build_column_metadata(
        frame,
        manifest_id=manifest_id,
        pk_columns=dataset.pk_columns,
        target_column=dataset.target_column,
        weight_column=dataset.weight_column,
    )
    split_set_df = build_cv_split_set(
        frame,
        manifest_id=manifest_id,
        n_splits=n_splits,
        random_state=random_state,
        pk_columns=dataset.pk_columns,
        created_by=created_by,
    )
    split_set_id = split_set_df.loc[0, "split_set_id"]
    cv_fold_df = build_cv_folds(
        frame,
        split_set_id=split_set_id,
        n_splits=n_splits,
        random_state=random_state,
    )
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
        split_set_df.to_sql(
            "CV_SPLIT_SET",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
        )
        cv_fold_df.to_sql(
            "CV_FOLD",
            con,
            schema="pricing",
            if_exists="append",
            index=False,
        )

    return manifest_id


def create_fremtpl_manifest(
    engine: Engine,
    *,
    manifest_id: str,
    n_splits: int = 5,
    random_state: int = 42,
    created_by: str = "airflow",
) -> str:
    dataset = DatasetSpec(
        dataset_name=FREMTPL_DATASET_NAME,
        source_system=FREMTPL_SOURCE_SYSTEM,
        manifest_sql=FREMTPL_RAW_SELECT_SQL,
        pk_columns=("IDpol",),
        target_column="ClaimNb",
        weight_column="Exposure",
    )
    return create_dataset_manifest(
        engine,
        dataset=dataset,
        manifest_id=manifest_id,
        n_splits=n_splits,
        random_state=random_state,
        created_by=created_by,
    )
