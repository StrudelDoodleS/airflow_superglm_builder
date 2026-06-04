from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.orchestration.run_context import scoped_identifier


DATASET_NAME = "demo_custom_frequency_model_frame"
SOURCE_SYSTEM = "demo_sql_server_staging"
TARGET_COLUMN = "claim_count"
WEIGHT_COLUMN = "exposure"
PK_COLUMNS = ("policy_id",)
TRAINING_TABLE = "DEMO_CUSTOM_PUBLISH_TRAINING"
TRAINING_SQL_TEMPLATE = """
SELECT
    policy_id,
    territory,
    vehicle_age_band,
    driver_age,
    exposure,
    claim_count
FROM pricing_stg.{table_name}
ORDER BY policy_id
"""
TRAINING_SQL = TRAINING_SQL_TEMPLATE.format(table_name=TRAINING_TABLE)
FEATURE_COLUMNS = ("territory", "vehicle_age_band", "driver_age")
DEFAULT_OUTPUT_DIR = Path("state/demo_custom_publish")
DEFAULT_ROW_COUNT = 240
DEFAULT_SEED = 20260604


def training_table_for_run(run_key: object | None) -> str:
    return scoped_identifier(TRAINING_TABLE, run_key)


def training_sql_for_table(table_name: str) -> str:
    return TRAINING_SQL_TEMPLATE.format(table_name=table_name)


def build_demo_training_frame(
    *,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    territory = rng.choice(
        ["A", "B", "C", "D"],
        row_count,
        p=[0.25, 0.35, 0.25, 0.15],
    )
    vehicle_age_band = rng.choice(
        ["new", "mid", "old"],
        row_count,
        p=[0.20, 0.55, 0.25],
    )
    driver_age = rng.integers(18, 78, row_count)
    exposure = rng.uniform(0.15, 1.0, row_count)

    territory_factor = {"A": 0.80, "B": 1.00, "C": 1.20, "D": 1.45}
    vehicle_factor = {"new": 1.30, "mid": 1.00, "old": 0.90}
    linear_predictor = (
        -1.45
        + np.array([np.log(territory_factor[value]) for value in territory])
        + np.array([np.log(vehicle_factor[value]) for value in vehicle_age_band])
        + 0.006 * (driver_age - 45.0)
    )
    claim_count = rng.poisson(exposure * np.exp(linear_predictor))

    return pd.DataFrame(
        {
            "policy_id": np.arange(1, row_count + 1, dtype=np.int64),
            "territory": territory,
            "vehicle_age_band": vehicle_age_band,
            "driver_age": driver_age,
            "exposure": exposure,
            "claim_count": claim_count,
        }
    )


def write_training_frame(frame: pd.DataFrame, output_dir: str | Path) -> str:
    path = Path(output_dir) / "training_frame.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def read_training_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def materialize_training_source(
    engine,
    frame: pd.DataFrame,
    *,
    table_name: str = TRAINING_TABLE,
) -> int:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        frame.to_sql(
            table_name,
            con,
            schema=schemas.pricing_staging,
            if_exists="replace",
            index=False,
        )
    return int(len(frame))
