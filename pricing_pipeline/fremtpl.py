from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml
from sqlalchemy import text
from sqlalchemy.engine import Engine


FREMTPL_OPENML_ID = 41214
FREMTPL_DATASET_NAME = "freMTPL2freq"
FREMTPL_COLUMNS = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


def fetch_fremtpl() -> pd.DataFrame:
    dataset = fetch_openml(data_id=FREMTPL_OPENML_ID, as_frame=True)
    return dataset.frame.reset_index(drop=True)


def validate_fremtpl_raw(frame: pd.DataFrame) -> None:
    missing = [column for column in FREMTPL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"freMTPL raw data missing columns: {missing}")


def prepare_fremtpl_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_fremtpl_raw(frame)
    out = frame.loc[:, FREMTPL_COLUMNS].copy()
    out["IDpol"] = out["IDpol"].astype("int64")
    out["ClaimNb"] = out["ClaimNb"].astype("int64")
    return out


def _db_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def fremtpl_insert_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        tuple(_db_value(value) for value in row)
        for row in frame.loc[:, FREMTPL_COLUMNS].itertuples(index=False, name=None)
    ]


def _chunk_rows(rows: list[tuple[Any, ...]], chunksize: int):
    for index in range(0, len(rows), chunksize):
        yield rows[index : index + chunksize]


def bulk_insert_fremtpl_raw(
    engine: Engine,
    frame: pd.DataFrame,
    *,
    chunksize: int = 10000,
) -> int:
    if chunksize < 1:
        raise ValueError("chunksize must be greater than zero")

    prepared = prepare_fremtpl_raw_frame(frame)
    rows = fremtpl_insert_rows(prepared)
    if not rows:
        return 0

    columns = ", ".join(FREMTPL_COLUMNS)
    placeholders = ", ".join("?" for _ in FREMTPL_COLUMNS)
    sql = f"INSERT INTO pricing.FREMTPL_RAW ({columns}) VALUES ({placeholders})"

    connection = engine.raw_connection()
    cursor = None
    try:
        cursor = connection.cursor()
        if hasattr(cursor, "fast_executemany"):
            cursor.fast_executemany = True
        for chunk in _chunk_rows(rows, chunksize):
            cursor.executemany(sql, chunk)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None and hasattr(cursor, "close"):
            cursor.close()
        connection.close()

    return len(rows)


def load_fremtpl_raw(engine: Engine, *, replace: bool = False) -> int:
    with engine.begin() as con:
        existing_count = int(
            con.execute(text("SELECT COUNT_BIG(*) FROM pricing.FREMTPL_RAW")).scalar_one()
        )
        if existing_count and not replace:
            return existing_count
        if replace:
            con.execute(text("TRUNCATE TABLE pricing.FREMTPL_RAW"))

    frame = prepare_fremtpl_raw_frame(fetch_fremtpl())
    return bulk_insert_fremtpl_raw(engine, frame)
