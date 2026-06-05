from __future__ import annotations

import json
import re
import hashlib
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ValidationSplitConfig
from pricing_pipeline.models.spec import DatasetSpec


FREMTPL_DATASET_NAME = "freMTPL2freq"
FREMTPL_SOURCE_SYSTEM = "openml_41214"
FREMTPL_RAW_SELECT_SQL = "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol"


@dataclass(frozen=True)
class DatasetManifestResult:
    manifest_id: str
    split_set_id: str | None = None
    split_artifact_uri: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "manifest_id": self.manifest_id,
            "split_set_id": self.split_set_id,
            "split_artifact_uri": self.split_artifact_uri,
        }


def _normalise_date(value: date | datetime | str, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a date, datetime, or ISO date string")

    cleaned = value.strip()
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        try:
            return date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a date, datetime, or ISO date string") from exc


def _required_text(value: str | None, *, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return str(value).strip()


def _optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string or None")
    return cleaned


def _normalise_pk_columns(value: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        raise ValueError("pk_columns must contain at least one column")
    cleaned = tuple(_required_text(column, field_name="pk_columns") for column in value)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("pk_columns must not contain duplicates")
    return cleaned


@dataclass(frozen=True)
class ModelFrameManifestSpec:
    dataset_name: str
    source_system: str
    data_as_of_date: date | datetime | str
    pk_columns: tuple[str, ...]
    target_column: str | None
    weight_column: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_name",
            _required_text(self.dataset_name, field_name="dataset_name"),
        )
        object.__setattr__(
            self,
            "source_system",
            _required_text(self.source_system, field_name="source_system"),
        )
        object.__setattr__(
            self,
            "data_as_of_date",
            _normalise_date(self.data_as_of_date, field_name="data_as_of_date"),
        )
        object.__setattr__(self, "pk_columns", _normalise_pk_columns(self.pk_columns))
        object.__setattr__(
            self,
            "target_column",
            _optional_text(self.target_column, field_name="target_column"),
        )
        object.__setattr__(
            self,
            "weight_column",
            _optional_text(self.weight_column, field_name="weight_column"),
        )


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
    split_column: str | None = None,
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
    if split_column is not None:
        column_df.loc[column_df["column_name"].eq(split_column), "column_role"] = "SPLIT"
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


def _format_split_float(value: float) -> str:
    formatted = f"{value:.12g}"
    return formatted.replace(".", "_").replace("-", "neg_")


def split_set_id_for_validation_split(
    manifest_id: str,
    validation_split: ValidationSplitConfig,
) -> str | None:
    if validation_split.method == "none":
        return None
    if validation_split.method == "kfold":
        return split_set_id_for_manifest(
            manifest_id,
            n_splits=int(validation_split.n_splits or 5),
            random_state=int(validation_split.random_state or 0),
        )
    if validation_split.method == "train_test_split":
        return (
            f"{manifest_id}__train_test_split_test_"
            f"{_format_split_float(float(validation_split.test_size or 0.2))}"
            f"_seed_{validation_split.random_state}"
        )
    if validation_split.method in {"column_kfold", "column_holdout"}:
        column_token = re.sub(r"[^A-Za-z0-9_]+", "_", str(validation_split.column)).strip("_")
        column_token = column_token or "source_column"
        return f"{manifest_id}__{validation_split.method}_{column_token}"
    raise ValueError(f"Unsupported validation split method: {validation_split.method}")


def validation_split_source_column(validation_split: ValidationSplitConfig) -> str | None:
    if validation_split.method in {"column_kfold", "column_holdout"}:
        return _required_text(validation_split.column, field_name="validation_split.column")
    return None


def _json_clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


def _ordered_unique_non_null_values(series: pd.Series, *, column: str) -> list:
    if series.isna().any():
        raise ValueError(f"validation split column {column!r} contains null values")
    values = list(pd.unique(series))
    return sorted(values, key=lambda value: str(_json_clean_value(value)))


def _source_column_splitter_params(
    frame: pd.DataFrame,
    validation_split: ValidationSplitConfig,
) -> dict[str, object]:
    column = validation_split_source_column(validation_split)
    if column is None:
        raise ValueError("validation split is not source-column based")
    if column not in frame.columns:
        raise ValueError(f"validation split column is missing from model frame: {column}")

    if validation_split.method == "column_kfold":
        return {
            "method": validation_split.method,
            "column": column,
            "fold_values": [
                _json_clean_value(value)
                for value in _ordered_unique_non_null_values(frame[column], column=column)
            ],
        }
    if validation_split.method == "column_holdout":
        return {
            "method": validation_split.method,
            "column": column,
            "train_values": [_json_clean_value(value) for value in validation_split.train_values],
            "test_values": [_json_clean_value(value) for value in validation_split.test_values],
            "unexpected_values": "error",
        }
    raise ValueError(
        f"Unsupported source-column validation split method: {validation_split.method}"
    )


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


def build_validation_split_set(
    frame: pd.DataFrame,
    *,
    manifest_id: str,
    validation_split: ValidationSplitConfig,
    pk_columns: tuple[str, ...] = ("IDpol",),
    created_by: str = "airflow",
    artifact_uri: str | None = None,
    artifact_sha256: str | None = None,
) -> pd.DataFrame:
    split_set_id = split_set_id_for_validation_split(manifest_id, validation_split)
    if split_set_id is None:
        return pd.DataFrame()

    if validation_split.method == "kfold":
        splitter_class = "sklearn.model_selection.KFold"
        params = {
            "n_splits": int(validation_split.n_splits or 5),
            "shuffle": bool(validation_split.shuffle),
            "random_state": validation_split.random_state,
        }
        fold_count = int(validation_split.n_splits or 5)
    elif validation_split.method == "train_test_split":
        splitter_class = "sklearn.model_selection.train_test_split"
        params = {
            "test_size": float(validation_split.test_size or 0.2),
            "random_state": validation_split.random_state,
            "shuffle": bool(validation_split.shuffle),
        }
        if validation_split.stratify_column is not None:
            params["stratify_column"] = validation_split.stratify_column
        fold_count = 1
    elif validation_split.method in {"column_kfold", "column_holdout"}:
        splitter_class = "source_column"
        params = _source_column_splitter_params(frame, validation_split)
        fold_count = len(params["fold_values"]) if validation_split.method == "column_kfold" else 1
    else:
        raise ValueError(f"Unsupported validation split method: {validation_split.method}")

    return pd.DataFrame(
        [
            {
                "split_set_id": split_set_id,
                "manifest_id": manifest_id,
                "split_mode": "MATERIALIZED" if artifact_uri is not None else "REPLAYABLE",
                "splitter_class": splitter_class,
                "splitter_params_json": json.dumps(params, sort_keys=True),
                "row_order_sha256": compute_row_order_sha256(frame, pk_columns=pk_columns),
                "row_count": int(len(frame)),
                "fold_count": fold_count,
                "groups_column": None,
                "stratify_column": (
                    validation_split.stratify_column
                    if validation_split.method == "train_test_split"
                    else None
                ),
                "artifact_uri": artifact_uri,
                "artifact_sha256": artifact_sha256,
                "runtime_metadata_json": runtime_dependency_metadata(),
                "created_by": created_by,
            }
        ]
    )


def build_validation_folds(
    frame: pd.DataFrame,
    *,
    split_set_id: str,
    validation_split: ValidationSplitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for fold_no, (train_idx, test_idx) in enumerate(
        validation_split_indices(frame, validation_split),
        start=1,
    ):
        rows.append(
            {
                "split_set_id": split_set_id,
                "fold_no": fold_no,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return pd.DataFrame(rows)


def build_cv_folds(
    frame: pd.DataFrame,
    *,
    split_set_id: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    return build_validation_folds(
        frame,
        split_set_id=split_set_id,
        validation_split=ValidationSplitConfig.kfold(
            n_splits=n_splits,
            random_state=random_state,
        ),
    )


def validation_split_indices(
    frame: pd.DataFrame,
    validation_split: ValidationSplitConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if validation_split.method == "none":
        return []
    if validation_split.method == "kfold":
        kf = KFold(
            n_splits=int(validation_split.n_splits or 5),
            shuffle=bool(validation_split.shuffle),
            random_state=validation_split.random_state,
        )
        return [
            (np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in kf.split(frame)
        ]
    if validation_split.method == "train_test_split":
        indices = np.arange(len(frame), dtype=np.int64)
        stratify = (
            frame[validation_split.stratify_column]
            if validation_split.stratify_column is not None
            else None
        )
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(validation_split.test_size or 0.2),
            random_state=validation_split.random_state,
            shuffle=bool(validation_split.shuffle),
            stratify=stratify,
        )
        return [(np.asarray(train_idx), np.asarray(test_idx))]
    if validation_split.method == "column_kfold":
        column = validation_split_source_column(validation_split)
        if column not in frame.columns:
            raise ValueError(f"validation split column is missing from model frame: {column}")
        fold_values = _ordered_unique_non_null_values(frame[column], column=column)
        if len(fold_values) < 2:
            raise ValueError("validation split column must contain at least two fold values")

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_value in fold_values:
            test_mask = frame[column].eq(fold_value)
            train_mask = ~test_mask
            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError("validation split column produced an empty train or test fold")
            folds.append((train_idx, test_idx))
        return folds
    if validation_split.method == "column_holdout":
        column = validation_split_source_column(validation_split)
        if column not in frame.columns:
            raise ValueError(f"validation split column is missing from model frame: {column}")
        if not validation_split.train_values:
            raise ValueError("validation_split.train_values must not be empty")
        if not validation_split.test_values:
            raise ValueError("validation_split.test_values must not be empty")
        if any(
            train_value == test_value
            for train_value in validation_split.train_values
            for test_value in validation_split.test_values
        ):
            raise ValueError("validation_split.train_values and test_values must not overlap")
        series = frame[column]
        if series.isna().any():
            raise ValueError(f"validation split column {column!r} contains null values")

        train_mask = series.isin(validation_split.train_values)
        test_mask = series.isin(validation_split.test_values)
        unexpected_values = _ordered_unique_non_null_values(
            series.loc[~(train_mask | test_mask)],
            column=column,
        )
        if unexpected_values:
            raise ValueError(
                "validation split column contains unexpected values: "
                + ", ".join(str(value) for value in unexpected_values)
            )

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0:
            raise ValueError("validation split column produced no train rows")
        if len(test_idx) == 0:
            raise ValueError("validation split column produced no test rows")
        return [(train_idx, test_idx)]
    raise ValueError(f"Unsupported validation split method: {validation_split.method}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_validation_split_npz(
    frame: pd.DataFrame,
    *,
    validation_split: ValidationSplitConfig,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for fold_no, (train_idx, test_idx) in enumerate(
        validation_split_indices(frame, validation_split),
        start=1,
    ):
        arrays[f"fold_{fold_no}_train_idx"] = train_idx.astype(np.int64, copy=False)
        arrays[f"fold_{fold_no}_test_idx"] = test_idx.astype(np.int64, copy=False)
    np.savez_compressed(output_path, **arrays)
    return file_sha256(output_path)


def _validate_model_frame(
    frame: pd.DataFrame,
    *,
    spec: ModelFrameManifestSpec,
    validation_split: ValidationSplitConfig,
) -> None:
    if frame.empty:
        raise ValueError("model frame must not be empty")

    column_names = [str(column).strip() for column in frame.columns]
    if any(not column for column in column_names):
        raise ValueError("model frame contains a blank column name")
    if len(set(column_names)) != len(column_names):
        raise ValueError("model frame contains duplicate column names")

    required_columns = list(spec.pk_columns)
    if spec.target_column is not None:
        required_columns.append(spec.target_column)
    if spec.weight_column is not None:
        required_columns.append(spec.weight_column)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError("model frame missing required columns: " + ", ".join(missing))

    if frame.loc[:, list(spec.pk_columns)].isna().any().any():
        raise ValueError("model frame primary key columns contain null values")
    if frame.duplicated(subset=list(spec.pk_columns)).any():
        raise ValueError("model frame primary key columns contain duplicate values")

    if (
        validation_split.stratify_column is not None
        and validation_split.stratify_column not in frame.columns
    ):
        raise ValueError(
            "validation_split.stratify_column is missing from model frame: "
            f"{validation_split.stratify_column}"
        )
    if validation_split.method == "kfold":
        n_splits = int(validation_split.n_splits or 5)
        if n_splits > len(frame):
            raise ValueError(
                f"validation_split.n_splits ({n_splits}) must not exceed row count ({len(frame)})"
            )
    split_column = validation_split_source_column(validation_split)
    if split_column is not None:
        if split_column in spec.pk_columns:
            raise ValueError("validation split column must not be a primary key column")
        if split_column == spec.target_column:
            raise ValueError("validation split column must not be the target column")
        if split_column == spec.weight_column:
            raise ValueError("validation split column must not be the weight column")
        validation_split_indices(frame, validation_split)


def create_model_frame_manifest_with_split(
    engine: Engine,
    *,
    frame: pd.DataFrame,
    spec: ModelFrameManifestSpec,
    manifest_id: str | None = None,
    validation_split: ValidationSplitConfig = ValidationSplitConfig.kfold(),
    validation_split_artifact_root: Path | None = None,
    created_by: str = "airflow",
) -> DatasetManifestResult:
    _validate_model_frame(frame, spec=spec, validation_split=validation_split)
    manifest_id = manifest_id or new_manifest_id(spec.dataset_name)
    manifest_df = pd.DataFrame(
        [
            {
                "manifest_id": manifest_id,
                "dataset_name": spec.dataset_name,
                "source_system": spec.source_system,
                "data_as_of_date": spec.data_as_of_date,
                "row_count": int(len(frame)),
                "pk_columns_json": json.dumps(list(spec.pk_columns)),
                "target_column": spec.target_column,
                "weight_column": spec.weight_column,
                "created_by": created_by,
            }
        ]
    )
    column_df = build_column_metadata(
        frame,
        manifest_id=manifest_id,
        pk_columns=spec.pk_columns,
        target_column=spec.target_column,
        weight_column=spec.weight_column,
        split_column=validation_split_source_column(validation_split),
    )
    split_set_id = split_set_id_for_validation_split(manifest_id, validation_split)
    split_artifact_uri = None
    split_artifact_sha256 = None
    if validation_split.materialize and split_set_id is not None:
        if validation_split_artifact_root is None:
            raise ValueError("validation_split_artifact_root is required when materialize=true")
        artifact_path = Path(validation_split_artifact_root) / f"{split_set_id}.npz"
        split_artifact_sha256 = write_validation_split_npz(
            frame,
            validation_split=validation_split,
            output_path=artifact_path,
        )
        split_artifact_uri = str(artifact_path)

    split_set_df = build_validation_split_set(
        frame,
        manifest_id=manifest_id,
        validation_split=validation_split,
        pk_columns=spec.pk_columns,
        created_by=created_by,
        artifact_uri=split_artifact_uri,
        artifact_sha256=split_artifact_sha256,
    )
    cv_fold_df = (
        build_validation_folds(
            frame,
            split_set_id=split_set_id,
            validation_split=validation_split,
        )
        if split_set_id is not None
        else pd.DataFrame()
    )
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        manifest_df.to_sql(
            "DATASET_MANIFEST",
            con,
            schema=schemas.pricing,
            if_exists="append",
            index=False,
        )
        column_df.to_sql(
            "DATASET_COLUMN",
            con,
            schema=schemas.pricing,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        if not split_set_df.empty:
            split_set_df.to_sql(
                "CV_SPLIT_SET",
                con,
                schema=schemas.pricing,
                if_exists="append",
                index=False,
            )
        if not cv_fold_df.empty:
            cv_fold_df.to_sql(
                "CV_FOLD",
                con,
                schema=schemas.pricing,
                if_exists="append",
                index=False,
            )

    return DatasetManifestResult(
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        split_artifact_uri=split_artifact_uri,
    )


def create_dataset_manifest_with_split(
    engine: Engine,
    *,
    dataset: DatasetSpec,
    manifest_id: str,
    validation_split: ValidationSplitConfig = ValidationSplitConfig.kfold(),
    validation_split_artifact_root: Path | None = None,
    created_by: str = "airflow",
) -> DatasetManifestResult:
    frame = pd.read_sql_query(text(dataset.manifest_sql), engine)
    return create_model_frame_manifest_with_split(
        engine,
        frame=frame,
        spec=ModelFrameManifestSpec(
            dataset_name=dataset.dataset_name,
            source_system=dataset.source_system,
            data_as_of_date=date.today(),
            pk_columns=dataset.pk_columns,
            target_column=dataset.target_column,
            weight_column=dataset.weight_column,
        ),
        manifest_id=manifest_id,
        validation_split=validation_split,
        validation_split_artifact_root=validation_split_artifact_root,
        created_by=created_by,
    )


def create_dataset_manifest(
    engine: Engine,
    *,
    dataset: DatasetSpec,
    manifest_id: str,
    n_splits: int = 5,
    random_state: int = 42,
    created_by: str = "airflow",
) -> str:
    return create_dataset_manifest_with_split(
        engine,
        dataset=dataset,
        manifest_id=manifest_id,
        validation_split=ValidationSplitConfig.kfold(
            n_splits=n_splits,
            random_state=random_state,
        ),
        created_by=created_by,
    ).manifest_id


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
