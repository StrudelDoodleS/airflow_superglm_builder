from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.data.manifest import runtime_dependency_metadata
from pricing_pipeline.data.row_identity import compute_row_order_sha256
from pricing_pipeline.data.split_artifacts import load_split_artifact_npz
from pricing_pipeline.data.split_artifacts import write_split_artifact_npz


@dataclass(frozen=True)
class CVSplitSet:
    split_set_id: str
    manifest_id: str
    split_mode: str
    splitter_class: str | None
    splitter_params_json: str | None
    row_order_sha256: str
    row_count: int
    fold_count: int
    artifact_uri: str | None
    artifact_sha256: str | None
    runtime_metadata_json: str | None = None


def _split_set_from_mapping(row) -> CVSplitSet:
    return CVSplitSet(
        split_set_id=row["split_set_id"],
        manifest_id=row["manifest_id"],
        split_mode=row["split_mode"],
        splitter_class=row["splitter_class"],
        splitter_params_json=row["splitter_params_json"],
        row_order_sha256=row["row_order_sha256"],
        row_count=int(row["row_count"]),
        fold_count=int(row["fold_count"]),
        artifact_uri=row["artifact_uri"],
        artifact_sha256=row["artifact_sha256"],
        runtime_metadata_json=row["runtime_metadata_json"],
    )


def fetch_split_set(engine: Engine, split_set_id: str) -> CVSplitSet:
    with engine.connect() as con:
        row = (
            con.execute(
                text(
                    """
                    SELECT
                        split_set_id,
                        manifest_id,
                        split_mode,
                        splitter_class,
                        splitter_params_json,
                        row_order_sha256,
                        row_count,
                        fold_count,
                        artifact_uri,
                        artifact_sha256,
                        runtime_metadata_json
                    FROM pricing.CV_SPLIT_SET
                    WHERE split_set_id = :split_set_id
                    """
                ),
                {"split_set_id": split_set_id},
            )
            .mappings()
            .one()
        )
    return _split_set_from_mapping(row)


def resolve_splitter(split_set: CVSplitSet):
    if split_set.split_mode != "REPLAYABLE":
        raise ValueError(
            f"split_set {split_set.split_set_id} is {split_set.split_mode}; "
            "load materialized artifact indices instead"
        )
    params = json.loads(split_set.splitter_params_json or "{}")
    if split_set.splitter_class == "sklearn.model_selection.KFold":
        return KFold(**params)
    if split_set.splitter_class == "sklearn.model_selection.train_test_split":
        return train_test_split
    raise ValueError(f"Unsupported splitter_class: {split_set.splitter_class}")


def _json_clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


def _values_equal(left, right) -> bool:
    return bool(_json_clean_value(left) == _json_clean_value(right))


def _deduplicate_values(values: list, *, field_name: str) -> None:
    for index, value in enumerate(values):
        if any(_values_equal(value, previous) for previous in values[:index]):
            raise ValueError(f"{field_name} must not contain duplicate values")


def _source_column_series(frame: pd.DataFrame, params: dict[str, object]) -> tuple[str, pd.Series]:
    column = params.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("source_column splitter params must include column")
    column = column.strip()
    if column not in frame.columns:
        raise ValueError(f"validation split column is missing from model frame: {column}")

    series = frame[column]
    if series.isna().any():
        raise ValueError(f"validation split column {column!r} contains null values")
    return column, series.map(_json_clean_value)


def _required_param_values(
    params: dict[str, object],
    *,
    field_name: str,
    min_length: int,
) -> list:
    value = params.get(field_name)
    if not isinstance(value, list) or len(value) < min_length:
        raise ValueError(
            f"source_column splitter params must include at least {min_length} {field_name}"
        )
    cleaned = [_json_clean_value(item) for item in value]
    _deduplicate_values(cleaned, field_name=field_name)
    return cleaned


def _unexpected_source_values(series: pd.Series, allowed_values: list) -> list:
    observed_values = [_json_clean_value(value) for value in pd.unique(series)]
    return sorted(
        [
            value
            for value in observed_values
            if not any(_values_equal(value, allowed) for allowed in allowed_values)
        ],
        key=lambda value: str(value),
    )


def _source_column_replay_folds(
    frame: pd.DataFrame,
    *,
    params: dict[str, object],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    method = params.get("method")
    _, series = _source_column_series(frame, params)

    if method == "column_kfold":
        fold_values = _required_param_values(params, field_name="fold_values", min_length=2)
        unexpected_values = _unexpected_source_values(series, fold_values)
        if unexpected_values:
            raise ValueError(
                "validation split column contains unexpected values: "
                + ", ".join(str(value) for value in unexpected_values)
            )

        folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for fold_no, fold_value in enumerate(fold_values, start=1):
            test_mask = series.eq(fold_value)
            train_mask = ~test_mask
            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError("source_column replay produced an empty train or test fold")
            folds[fold_no] = (train_idx, test_idx)
        return folds

    if method == "column_holdout":
        train_values = _required_param_values(params, field_name="train_values", min_length=1)
        test_values = _required_param_values(params, field_name="test_values", min_length=1)
        if any(
            _values_equal(train_value, test_value)
            for train_value in train_values
            for test_value in test_values
        ):
            raise ValueError("source_column train_values and test_values must not overlap")

        train_mask = series.isin(train_values)
        test_mask = series.isin(test_values)
        unexpected_values = _unexpected_source_values(series, [*train_values, *test_values])
        if unexpected_values:
            raise ValueError(
                "validation split column contains unexpected values: "
                + ", ".join(str(value) for value in unexpected_values)
            )

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0:
            raise ValueError("source_column replay produced no train rows")
        if len(test_idx) == 0:
            raise ValueError("source_column replay produced no test rows")
        return {1: (train_idx, test_idx)}

    raise ValueError(f"Unsupported source_column split method: {method!r}")


def _resolve_pk_columns(
    *,
    pk_column: str | None,
    pk_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if pk_columns is None:
        if pk_column is None:
            raise ValueError("pk_column or pk_columns is required")
        pk_columns = (pk_column,)
    if not pk_columns:
        raise ValueError("pk_column or pk_columns is required")
    return pk_columns


def replay_cv_folds(
    split_set: CVSplitSet,
    frame: pd.DataFrame,
    *,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pk_columns = _resolve_pk_columns(pk_column=pk_column, pk_columns=pk_columns)
    actual_hash = compute_row_order_sha256(frame, pk_columns=pk_columns)
    if actual_hash != split_set.row_order_sha256:
        raise ValueError(
            "row_order_sha256 mismatch for "
            f"{split_set.split_set_id}: expected {split_set.row_order_sha256}, got {actual_hash}"
        )
    if len(frame) != split_set.row_count:
        raise ValueError(
            f"row_count mismatch for {split_set.split_set_id}: "
            f"expected {split_set.row_count}, got {len(frame)}"
        )

    params = json.loads(split_set.splitter_params_json or "{}")
    if split_set.splitter_class == "sklearn.model_selection.train_test_split":
        stratify_column = params.pop("stratify_column", None)
        stratify = frame[stratify_column] if stratify_column else None
        train_idx, test_idx = train_test_split(
            np.arange(len(frame), dtype=np.int64),
            stratify=stratify,
            **params,
        )
        return {1: (np.asarray(train_idx), np.asarray(test_idx))}
    if split_set.splitter_class == "source_column":
        return _source_column_replay_folds(frame, params=params)

    cv = resolve_splitter(split_set)
    return {
        fold_no: (np.asarray(train_idx), np.asarray(test_idx))
        for fold_no, (train_idx, test_idx) in enumerate(cv.split(frame), start=1)
    }


def load_materialized_cv_folds(
    split_set: CVSplitSet,
    *,
    frame: pd.DataFrame | None = None,
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return load_split_artifact_npz(split_set, frame=frame, pk_columns=pk_columns)


def _validation_split_from_split_set(split_set: CVSplitSet):
    from pricing_pipeline.models.config import ValidationSplitConfig

    params = json.loads(split_set.splitter_params_json or "{}")
    if split_set.splitter_class == "sklearn.model_selection.KFold":
        return ValidationSplitConfig.kfold(
            n_splits=int(params.get("n_splits", split_set.fold_count)),
            random_state=params.get("random_state"),
            shuffle=bool(params.get("shuffle", True)),
            materialize=True,
        )
    if split_set.splitter_class == "sklearn.model_selection.train_test_split":
        return ValidationSplitConfig.train_test_split(
            test_size=float(params.get("test_size", 0.2)),
            random_state=params.get("random_state"),
            shuffle=bool(params.get("shuffle", True)),
            stratify_column=params.get("stratify_column"),
            materialize=True,
        )
    if split_set.splitter_class == "source_column" and params.get("method") == "column_kfold":
        return ValidationSplitConfig.column_kfold(
            column=str(params["column"]),
            materialize=True,
        )
    if split_set.splitter_class == "source_column" and params.get("method") == "column_holdout":
        return ValidationSplitConfig.column_holdout(
            column=str(params["column"]),
            train_values=tuple(params.get("train_values", ())),
            test_values=tuple(params.get("test_values", ())),
            materialize=True,
        )
    return ValidationSplitConfig.none()


def materialize_cv_folds(
    engine: Engine,
    split_set: CVSplitSet,
    frame: pd.DataFrame,
    *,
    output_path: Path,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> Path:
    pk_columns = _resolve_pk_columns(pk_column=pk_column, pk_columns=pk_columns)
    folds = replay_cv_folds(split_set, frame, pk_columns=pk_columns)
    artifact_sha256 = write_split_artifact_npz(
        folds,
        validation_split=_validation_split_from_split_set(split_set),
        pk_columns=pk_columns,
        row_count=len(frame),
        output_path=output_path,
    )

    with engine.begin() as con:
        con.execute(
            text(
                """
                UPDATE pricing.CV_SPLIT_SET
                SET
                    split_mode = 'MATERIALIZED',
                    artifact_uri = :artifact_uri,
                    artifact_sha256 = :artifact_sha256,
                    runtime_metadata_json = :runtime_metadata_json
                WHERE split_set_id = :split_set_id
                """
            ),
            {
                "split_set_id": split_set.split_set_id,
                "artifact_uri": str(output_path),
                "artifact_sha256": artifact_sha256,
                "runtime_metadata_json": runtime_dependency_metadata(),
            },
        )

    return output_path


def load_cv_folds(
    engine: Engine,
    split_set_id: str,
    *,
    dataset_loader,
    pk_column: str | None = "IDpol",
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    split_set = fetch_split_set(engine, split_set_id)
    pk_columns = _resolve_pk_columns(pk_column=pk_column, pk_columns=pk_columns)
    frame = dataset_loader(split_set.manifest_id)
    if split_set.split_mode == "MATERIALIZED":
        return load_materialized_cv_folds(split_set, frame=frame, pk_columns=pk_columns)
    return replay_cv_folds(split_set, frame, pk_columns=pk_columns)
