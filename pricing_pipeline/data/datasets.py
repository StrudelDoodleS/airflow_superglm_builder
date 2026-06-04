from __future__ import annotations

from pricing_pipeline.data.fremtpl import load_fremtpl_raw
from pricing_pipeline.data.manifest import (
    FREMTPL_DATASET_NAME,
    FREMTPL_RAW_SELECT_SQL,
    FREMTPL_SOURCE_SYSTEM,
)
from pricing_pipeline.models.spec import DatasetSpec


FREMTPL_DATASET_SPEC = DatasetSpec(
    dataset_name=FREMTPL_DATASET_NAME,
    source_system=FREMTPL_SOURCE_SYSTEM,
    manifest_sql=FREMTPL_RAW_SELECT_SQL,
    pk_columns=("IDpol",),
    target_column="ClaimNb",
    weight_column="Exposure",
    raw_loader=load_fremtpl_raw,
)
