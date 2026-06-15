"""Read or stage freMTPL source data for the MTPL frequency model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pricing_pipeline.data.fremtpl import load_fremtpl_raw
from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.infra.schema import render_sql_schemas, schema_names_from_connectable


DEFAULT_OUTPUT_ROOT = Path("state/mtpl_frequency")


@dataclass(frozen=True)
class ModelFrameContract:
    """Static metadata the SQL manifest needs for this model's final frame."""

    dataset_name: str
    source_system: str
    pk_columns: tuple[str, ...]
    target_column: str
    weight_column: str | None = None

    def final_columns(self, feature_columns: Iterable[str]) -> tuple[str, ...]:
        columns = [*self.pk_columns, self.target_column]
        if self.weight_column is not None:
            columns.append(self.weight_column)
        columns.extend(feature_columns)
        return tuple(columns)

    def manifest_spec(self, data_as_of_date: str) -> ModelFrameManifestSpec:
        return ModelFrameManifestSpec(
            dataset_name=self.dataset_name,
            source_system=self.source_system,
            data_as_of_date=data_as_of_date,
            pk_columns=self.pk_columns,
            target_column=self.target_column,
            weight_column=self.weight_column,
        )


MODEL_FRAME = ModelFrameContract(
    dataset_name="freMTPL2freq_model_frame",
    source_system="freMTPL_raw_sql",
    pk_columns=("IDpol",),
    target_column="ClaimNb",
    weight_column="Exposure",
)

SOURCE_SQL = """
SELECT
    IDpol,
    ClaimNb,
    Exposure,
    Area,
    VehPower,
    VehAge,
    DrivAge,
    BonusMalus,
    VehBrand,
    VehGas,
    Density,
    Region
FROM pricing.FREMTPL_RAW
ORDER BY IDpol
"""


def prepare_source_data(
    engine,
    *,
    run_key: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_rows = load_fremtpl_raw(engine, replace=False)
    source_sql = render_sql_schemas(SOURCE_SQL, schema_names_from_connectable(engine))

    return {
        "run_key": run_key,
        "output_dir": str(output_path),
        "source_sql": source_sql,
        "source_row_count": raw_rows,
    }
